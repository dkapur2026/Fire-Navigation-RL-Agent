"""Phase 4b: a pure deep-RL baseline (PPO) to compare against the planner agents. Needs ``pip install -e .[rl]``.

Purpose: answer "why not just use deep RL?" with data, and use a trained agent to stress-test the reward. Results,
setup and hyperparameters are in docs/RL.md.

What is here
------------
* ``train_ppo``: PPO with GAE, the clipped objective, a value loss and an entropy bonus, over vectorised envs (one
  subprocess per CPU), with a curriculum of phases (e.g. studio + dog first, then the ranch with novelties sampled
  per episode). Seeded and deterministic on CPU. Episodes draw their fire seeds from ``TRAIN_SEEDS`` only.
* ``evaluate_policy``: runs a trained policy through the SAME harness as baseline / oracle / adaptive
  (``evaluate.run_episode``), on ``EVAL_SEEDS`` only (disjoint from training, asserted).
* ``PPOAgent``: the ``Agent`` interface over a checkpoint. Evidence discipline: it rebuilds the policy input from the
  ``Observation`` and ``Event``s it is handed (``encode_observation``), never from the world, and gives every
  decision a plain-English reason.

The policy input is ``env.encode`` (wall, fire, burnt, agent, dependents, exit, door, bumped jam, time remaining)
without the smoke channel: smoke is not part of ``Observation`` (only the value at the rescuer's cell is), so an
agent playing through the harness could not reproduce it. It has no effect on the dynamics under full observation.
Smaller houses are padded with walls to the training canvas so one network spans the whole curriculum.

Variants (docs/RL.md, "Improvements tried"), each behind a ``PPOConfig`` flag that is off by default so the original
baseline stays reproducible bit for bit: ``collect_teacher`` / ``train_bc`` (behaviour cloning from an
evidence-disciplined planner on training seeds, as a warm start, with an optional annealed KL toward the cloned
policy during PPO), ``heard`` (an extra input plane from the rescuer's hearing events), ``mask_invalid`` (no moves
into walls) and ``recurrent`` (a GRU with BPTT over each rollout). ``load_policy`` rebuilds whichever variant a
checkpoint holds, and old checkpoints load unchanged.

torch and gymnasium are imported inside the functions that need them (core stays light).

Command line: ``python -m emberpath.rl_baseline --help`` (same as ``scripts/train_ppo.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from emberpath.blueprint import Blueprint
from emberpath.env import N_CHANNELS, TIME_CHANNEL
from emberpath.grid import MOVES, WAIT
from emberpath.planner import Decision
from emberpath.rules import key_cell, parse_key
from emberpath.world import FOLLOWING, WAITING, Event, Observation

TRAIN_SEEDS = range(0, 1000)
EVAL_SEEDS = range(1000, 1100)
assert not set(TRAIN_SEEDS) & set(EVAL_SEEDS), "train and eval seeds must be disjoint"

SMOKE_CHANNEL = 3
POLICY_CHANNELS = tuple(c for c in range(N_CHANNELS) if c != SMOKE_CHANNEL)
# Optional extra policy plane (variants with ``heard=True``), appended AFTER every ``env.encode`` channel so channels
# 0..10 and ``env.N_CHANNELS`` stay as they are (the Dreamer agent and old checkpoints rely on them): 1 on each hiding
# place whose most recent hearing event said someone is calling there. Built only from the rescuer's own ``Event``s
# (``occupied:<slot>`` trials), the same evidence the planner agents use, so nothing hidden leaks.
HEARD_CHANNEL = N_CHANNELS
HEARD_POLICY_CHANNELS = POLICY_CHANNELS + (HEARD_CHANNEL,)
# Optional world-model planes (variants with ``world_planes=True``), appended after the hearing plane: everything the
# planner agents (``HazardAwarePlanner`` inside ``AdaptiveAgent``) read, as input planes. Built from the public
# blueprint, the ``Observation`` and the agent's own ``Event``s only; the believed rules come from the SAME detector +
# repair search the adaptive agent runs, fed the same events (``WorldModelFeatures``). Scalars are broadcast over the
# whole canvas; "q8" planes are quantised to 1/255 so the stored teacher data decodes to them exactly.
WORLD_PLANES = (
    "burns_left",  # Observation.burns_left / 5 (5 = the most burn ticks any adult can take, world.BURN_TOLERANCE)
    "burns",  # Observation.burns / 5
    "child", "dog", "adult",  # waiting or following dependents by DependentView.kind (a found visitor is a child)
    "unknown",  # believed-occupied hiding places not yet searched or burnt: the planner's "unknown" stand-in people
    "carried", "panicked", "alone",  # DependentView.carried / .panicked / leader == own id (an adult acting alone)
    "follows_adult",  # following another adult (leader is someone else's id), not the rescuer
    "slow",  # pace > 1 (children walk at half pace; the only paces in the households are 1 and 2)
    "wood", "carpet", "tile",  # floor material (doors and exits already have planes 8 and 7)
    "door_outside",  # the floor cells beside each door on its outside (hallway) side: where a jam can be kicked in
    "broken_door",  # Observation.broken_doors
    "belief_spread",  # q8: believed spread probability of each cell's material
    "belief_jammed",  # believed-jammed doors not yet forced
    "belief_occupied",  # believed-occupied cells (rule occupied:<cell>)
    "follow_dog", "follow_spouse", "follow_child", "force_door",  # believed follow / force probabilities (scalars)
)
WORLD_BASE = HEARD_CHANNEL + 1
WORLD_CHANNELS = tuple(range(WORLD_BASE, WORLD_BASE + len(WORLD_PLANES)))
# Optional on top (``forecast=True``, an ablation): the planner's fire-arrival forecast under the believed spread rates
# (``HazardAwarePlanner.fire_eta``), as 1 / (1 + eta / 5), q8. A computed forecast, not a decision.
FORECAST_CHANNEL = WORLD_BASE + len(WORLD_PLANES)
WORLD_SCALAR_PLANES = ("burns_left", "burns", "follow_dog", "follow_spouse", "follow_child", "force_door")
WORLD_Q8_PLANES = ("belief_spread",)
MAX_BURN_TOLERANCE = 5
ACTION_WORDS = ("moving north", "moving south", "moving west", "moving east", "waiting")
assert len(MOVES) == 4 and WAIT == 4


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class Task:
    """One training setting. ``novelty`` None is the nominal world."""

    blueprint: str
    profile: str
    novelty: str | None = None

    @property
    def label(self) -> str:
        return f"{self.blueprint}/{self.profile}/{self.novelty or 'none'}"


@dataclass(frozen=True)
class Phase:
    """A curriculum phase: every episode picks one of ``tasks`` uniformly at random (and a training seed)."""

    tasks: tuple[Task, ...]
    steps: int


@dataclass(frozen=True)
class PPOConfig:
    n_envs: int = 32
    rollout: int = 128  # steps per env per update
    gamma: float = 0.99
    lam: float = 0.95  # GAE lambda
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 512
    lr: float = 3e-4  # linearly annealed to lr_final_frac * lr over the whole run
    lr_final_frac: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 0.1  # learning targets only; every reported return is unscaled
    max_steps: int = 150
    seed: int = 0
    workers: int = 0  # subprocesses stepping the envs; 0 = step them in this process
    torch_threads: int = 4
    # Variants (docs/RL.md, "Improvements tried"). The defaults reproduce the original baseline exactly.
    mask_invalid: bool = False  # forbid moves into a wall (read from the wall + agent planes of the input itself)
    heard: bool = False  # append the HEARD_CHANNEL plane (built from hearing events)
    recurrent: bool = False  # a GRU after the CNN; BPTT over each rollout, hidden state reset at episode starts
    hidden: int = 256  # GRU size (recurrent only)
    kl_coef: float = 0.0  # KL(reference || policy) on rollout states, e.g. toward the behaviour-cloned policy
    kl_anneal_frac: float = 0.5  # the KL coefficient falls linearly to 0 over this fraction of the run
    vf_warmup_updates: int = 0  # first updates train only the value head (the policy keeps its warm start)
    world_planes: bool = False  # append the WORLD_CHANNELS planes (the planner's world model, from evidence only)
    forecast: bool = False  # also the FORECAST_CHANNEL plane (needs world_planes)


# --------------------------------------------------------------------------- observation


def canvas_for(blueprints: Sequence[Blueprint]) -> tuple[int, int]:
    return max(b.height for b in blueprints), max(b.width for b in blueprints)


def pad_to(x: np.ndarray, canvas: tuple[int, int]) -> np.ndarray:
    """Place an (h, w, C) encoding at the top-left of the canvas; the margin is wall (and keeps the clock)."""
    h, w, c = x.shape
    H, W = canvas
    if h > H or w > W:
        raise ValueError(f"house is {h}x{w} but the policy was trained on a {H}x{W} canvas")
    if (h, w) == (H, W):
        return x
    out = np.zeros((H, W, c), dtype=np.float32)
    out[..., 0] = 1.0
    out[..., TIME_CHANNEL] = x[0, 0, TIME_CHANNEL]
    out[:h, :w] = x
    return out


def encode_observation(bp: Blueprint, obs: Observation, bumped: set, max_steps: int) -> np.ndarray:
    """``env.encode`` rebuilt from what an agent is handed: the blueprint (public), the ``Observation`` and the doors
    it has found jammed (from its own ``Event``s). Equal to ``env.encode(world)`` on every policy channel (tested);
    the smoke channel stays zero because ``Observation`` carries smoke only at the rescuer's cell."""
    x = np.zeros((bp.height, bp.width, N_CHANNELS), dtype=np.float32)
    for cell in bp.walls:
        x[cell + (0,)] = 1
    for cell in obs.burning:
        x[cell + (1,)] = 1
    for cell in obs.burnt:
        x[cell + (2,)] = 1
    x[obs.agent_pos + (4,)] = 1
    for d in obs.dependents:
        if d.status == WAITING:
            x[d.pos + (5,)] = 1
        elif d.status == FOLLOWING:
            x[d.pos + (6,)] = 1
    for cell in bp.exits:
        x[cell + (7,)] = 1
    for cell in bp.doors:
        x[cell + (8,)] = 1
    for cell in bumped - set(obs.broken_doors):
        x[cell + (9,)] = 1
    x[..., TIME_CHANNEL] = max(0, max_steps - obs.step) / max(1, max_steps)
    return x


def update_heard(heard: dict, events: Sequence[Event]) -> dict:
    """Fold hearing trials into ``heard`` ({slot: was someone calling there at the latest trial}). Only the
    rescuer's own events are read; a person who has been found stops calling, so their slot goes back to False."""
    for e in events:
        if parse_key(e.key)[0] == "occupied":
            heard[key_cell(e.key)] = bool(e.outcome)
    return heard


def with_heard(x: np.ndarray, heard: dict) -> np.ndarray:
    """Append the HEARD_CHANNEL plane to an (H, W, N_CHANNELS) encoding (padded or not)."""
    plane = np.zeros(x.shape[:2] + (1,), dtype=np.float32)
    for cell, calling in heard.items():
        if calling:
            plane[cell + (0,)] = 1
    return np.concatenate([x, plane], axis=-1)


def policy_channels(heard: bool = False, world: bool = False, forecast: bool = False) -> tuple[int, ...]:
    """The input planes a variant reads (smoke never)."""
    if forecast and not world:
        raise ValueError("the forecast plane needs the world-model planes")
    return (POLICY_CHANNELS + ((HEARD_CHANNEL,) if heard else ()) + (WORLD_CHANNELS if world else ())
            + ((FORECAST_CHANNEL,) if forecast else ()))


def _q8(v):
    return (np.rint(np.clip(v, 0.0, 1.0) * 255.0) / 255.0).astype(np.float32)


_STATIC: dict = {}


def _static_world_planes(bp: Blueprint, canvas: tuple[int, int]) -> np.ndarray:
    """The blueprint's part of the world planes (materials, kick-from cells), once per house and canvas."""
    key = (bp.rows, tuple(canvas))
    if key not in _STATIC:
        from emberpath.grid import neighbors4

        idx = {n: i for i, n in enumerate(WORLD_PLANES)}
        x = np.zeros(tuple(canvas) + (len(WORLD_PLANES),), np.float32)
        for cell, mat in bp.materials.items():
            if mat in ("wood", "carpet", "tile"):
                x[cell + (idx[mat],)] = 1
        for door, side in bp.door_outside.items():
            for n in neighbors4(door, bp.height, bp.width):
                if n in side:
                    x[n + (idx["door_outside"],)] = 1
        _STATIC[key] = x
    return _STATIC[key]


class WorldModelFeatures:
    """The world-model planes of one episode, from what an agent is handed (blueprint, Observation, its own Events).

    The believed rules are those of an ``AdaptiveAgent`` fed the same events: this class runs the agent's own detector
    and repair search (``AdaptiveAgent._repair``, the code the planner agents use, not a copy), and NEVER calls its
    planner's ``act`` (route search, pickup order): the policy gets the belief, not the planner's decisions. The only
    planner method used is ``fire_eta`` (the forecast), and only for ``forecast=True``. Call ``step`` exactly once per
    tick, with what ``Agent.act`` receives."""

    def __init__(self, bp: Blueprint, forecast: bool = False):
        from emberpath.agents import AdaptiveAgent

        self.bp, self.forecast = bp, forecast
        try:  # explain=False: no route comparison for the explanation (that would run the planner's search)
            self.model = AdaptiveAgent(bp, explain=False)  # never handed a world
        except TypeError:  # an AdaptiveAgent without explanations
            self.model = AdaptiveAgent(bp)
        self.searched: set = set()
        # Counterfactual experiments only (the belief-swap test): a function applied to the belief before it is drawn,
        # e.g. "forget the jams". None (always, in training and evaluation) draws the belief as it is.
        self.view = None

    @property
    def belief(self):
        return self.model.belief

    def observe(self, obs: Observation, events: Sequence[Event]) -> None:
        """AdaptiveAgent.act up to (not including) the planner call: same detector feed, same repair trigger."""
        m = self.model
        if events:
            m.all_events.extend(events)
            m.detector.observe(list(events))
            findings = m.detector.findings()
            if any(f.n_trials > m._seen.get(f.key, 0) for f in findings):
                _adaptive_repair(m, obs, findings)
        if obs.step > 0:  # as the planner: cells stood on or next to are searched
            from emberpath.grid import neighbors4

            self.searched.update(neighbors4(obs.agent_pos, self.bp.height, self.bp.width), (obs.agent_pos,))

    def step(self, obs: Observation, events: Sequence[Event], canvas: tuple[int, int]) -> np.ndarray:
        self.observe(obs, events)
        return self.planes(obs, canvas)

    def planes(self, obs: Observation, canvas: tuple[int, int], belief=None) -> np.ndarray:
        """The planes for ``obs`` under the current belief, or under ``belief`` if given (the swap test)."""
        idx = {n: i for i, n in enumerate(WORLD_PLANES)}
        b = belief if belief is not None else self.view(self.belief) if self.view else self.belief
        x = _static_world_planes(self.bp, canvas).copy()
        x[..., idx["burns_left"]] = min(1.0, max(0, obs.burns_left) / MAX_BURN_TOLERANCE)
        x[..., idx["burns"]] = min(1.0, obs.burns / MAX_BURN_TOLERANCE)
        for name, kind in (("follow_dog", "dog"), ("follow_spouse", "spouse"), ("follow_child", "child")):
            x[..., idx[name]] = b.follow_prob[kind]
        x[..., idx["force_door"]] = b.force_prob["door"]
        kinds = {"child": "child", "dog": "dog", "spouse": "adult"}
        for d in obs.dependents:
            if d.status not in (WAITING, FOLLOWING):
                continue
            x[d.pos + (idx[kinds.get(d.kind, "adult")],)] = 1
            if d.carried:
                x[d.pos + (idx["carried"],)] = 1
            if d.panicked:
                x[d.pos + (idx["panicked"],)] = 1
            if d.status == FOLLOWING and d.leader == d.id:
                x[d.pos + (idx["alone"],)] = 1
            elif d.status == FOLLOWING and d.leader is not None:
                x[d.pos + (idx["follows_adult"],)] = 1
            if d.pace > 1:
                x[d.pos + (idx["slow"],)] = 1
        waiting_at = {d.pos for d in obs.dependents if d.status == WAITING}
        for cell in b.occupied - self.searched - obs.burnt - waiting_at:  # planner._hidden_targets
            x[cell + (idx["unknown"],)] = 1
        for cell in b.occupied:
            x[cell + (idx["belief_occupied"],)] = 1
        for cell in obs.broken_doors:
            x[cell + (idx["broken_door"],)] = 1
        for cell in b.jammed_doors - obs.broken_doors:
            x[cell + (idx["belief_jammed"],)] = 1
        spread = {m: float(_q8(p)) for m, p in b.spread_prob.items()}
        for cell, mat in self.bp.materials.items():
            if mat in spread:
                x[cell + (idx["belief_spread"],)] = spread[mat]
        if not self.forecast:
            return x
        f = np.zeros(tuple(canvas) + (1,), np.float32)
        if b is self.model.planner.belief:
            planner = self.model.planner
        else:  # another belief's forecast (the swap test): a fresh planner, used only for fire_eta
            from emberpath.planner import HazardAwarePlanner

            planner = HazardAwarePlanner(self.bp, b)
        for cell, eta in planner.fire_eta(obs.burning, obs.burnt, step=obs.step).items():  # ticks, on the fire clock
            if math.isfinite(eta):
                f[cell + (0,)] = 1.0 / (1.0 + eta / 5.0)
        return np.concatenate([x, _q8(f)], axis=-1)


BELIEF_FAMILIES = ("jam", "spread", "occupied")


def nominal_part(belief, family: str):
    """``belief`` with one family of rules reset to the textbook value: "jam" (no doors jammed), "spread" (textbook
    spread rates), "occupied" (nobody unexpected). For the belief-swap test: same evidence, minus that repair."""
    from dataclasses import replace

    from emberpath.rules import nominal_rules

    nom = nominal_rules()
    if family == "jam":
        return replace(belief, jammed_doors=nom.jammed_doors)
    if family == "spread":
        return replace(belief, spread_prob=dict(nom.spread_prob))
    if family == "occupied":
        return replace(belief, occupied=nom.occupied)
    raise ValueError(family)


def _adaptive_repair(agent, obs: Observation, findings) -> None:
    """``AdaptiveAgent._repair`` (whichever signature this checkout has: the observation, or the older step)."""
    import inspect

    first = next(iter(inspect.signature(agent._repair).parameters))
    agent._repair(obs if first == "obs" else obs.step, findings)


def extend_input(x: np.ndarray, heard: dict | None, wm: "WorldModelFeatures | None", obs: Observation | None = None,
                 events: Sequence[Event] = ()) -> np.ndarray:
    """A padded (H, W, N_CHANNELS) encoding plus the hearing plane (always present when world planes follow, so the
    channel numbers stay fixed) and, with ``wm``, the world planes (``wm.step`` is called: once per tick)."""
    x = with_heard(x, heard or {})
    if wm is not None:
        x = np.concatenate([x, wm.step(obs, events, x.shape[:2])], axis=-1)
    return x


# --------------------------------------------------------------------------- vectorised envs


def _make_env(task: Task, max_steps: int):
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.env import EmberPathEnv

    return EmberPathEnv(load_blueprint(task.blueprint), HOUSEHOLDS[task.profile], task.novelty, max_steps)


class _Slots:
    """A few env slots stepped in one process. Each slot has its own RNG (seeded by run seed and slot index) for
    which task and which training seed its next episode gets, so a run is identical however slots are split
    across processes."""

    def __init__(self, slot_ids: Sequence[int], seed: int, canvas: tuple[int, int], max_steps: int,
                 heard: bool = False, world: bool = False, forecast: bool = False):
        self.ids, self.canvas, self.max_steps, self.heard = list(slot_ids), canvas, max_steps, heard
        self.world, self.forecast = world, forecast
        self.heard_maps: list[dict] = [{} for _ in self.ids]
        self.models: list[WorldModelFeatures | None] = [None for _ in self.ids]
        self.rngs = [np.random.default_rng([seed, i]) for i in self.ids]
        self.tasks: tuple[Task, ...] = ()
        self.envs: dict[Task, Any] = {}
        self.cur: list[Any] = [None] * len(self.ids)
        self.meta: list[dict] = [{} for _ in self.ids]

    def _reset(self, k: int) -> np.ndarray:
        rng = self.rngs[k]
        task = self.tasks[int(rng.integers(len(self.tasks)))]
        seed = int(TRAIN_SEEDS[int(rng.integers(len(TRAIN_SEEDS)))])
        if task not in self.envs:
            self.envs[task] = [_make_env(task, self.max_steps) for _ in self.ids]
        env = self.envs[task][k]
        obs, _ = env.reset(seed=seed)
        self.cur[k] = env
        self.meta[k] = {"task": task.label, "seed": seed, "comp": {}}
        self.heard_maps[k] = {}
        if self.world:
            self.models[k] = WorldModelFeatures(env.bp, self.forecast)
        return self._out(obs, k, [])

    def _out(self, obs: np.ndarray, k: int, events: Sequence[Event] = ()) -> np.ndarray:
        x = pad_to(obs, self.canvas)
        if self.world:  # built from the Observation the agent would be handed and this tick's events
            return extend_input(x, self.heard_maps[k], self.models[k], self.cur[k].world.observe(), events)
        return with_heard(x, self.heard_maps[k]) if self.heard else x

    def set_tasks(self, tasks: Sequence[Task]) -> np.ndarray:
        self.tasks = tuple(tasks)
        return np.stack([self._reset(k) for k in range(len(self.ids))])

    def step(self, actions: Sequence[int]):
        obs, rews, dones, finished = [], [], [], []
        for k, a in enumerate(actions):
            o, r, term, trunc, info = self.cur[k].step(int(a))
            if self.heard or self.world:
                update_heard(self.heard_maps[k], info["events"])
            comp = self.meta[k]["comp"]
            for name, v in info["reward_components"].items():
                comp[name] = comp.get(name, 0.0) + v
            done = term or trunc
            if done:
                finished.append({**{x: self.meta[k][x] for x in ("task", "seed")}, "outcome": info["outcome"],
                                 "return": info["episode_return"], "steps": self.cur[k].world.t,
                                 "components": dict(comp)})
                o = self._reset(k)
            else:
                o = self._out(o, k, info["events"])
            obs.append(o)
            rews.append(r)
            dones.append(float(done))
        return np.stack(obs), np.array(rews, np.float32), np.array(dones, np.float32), finished


def _worker(conn, slot_ids, seed, canvas, max_steps, heard=False, world=False,
            forecast=False) -> None:  # pragma: no cover (subprocess)
    slots = _Slots(slot_ids, seed, canvas, max_steps, heard, world, forecast)
    while True:
        cmd, arg = conn.recv()
        if cmd == "tasks":
            conn.send(slots.set_tasks(arg))
        elif cmd == "step":
            conn.send(slots.step(arg))
        else:
            conn.close()
            return


class VecEnv:
    """``n`` env slots, in this process (workers=0) or split over subprocesses. Auto-resets finished episodes."""

    def __init__(self, n: int, seed: int, canvas: tuple[int, int], max_steps: int, workers: int = 0,
                 heard: bool = False, world: bool = False, forecast: bool = False):
        self.n = n
        if workers <= 0:
            self.local, self.conns, self.procs = _Slots(range(n), seed, canvas, max_steps, heard, world, forecast), [], []
            return
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.local = None
        self.groups = [list(g) for g in np.array_split(np.arange(n), min(workers, n))]
        self.conns, self.procs = [], []
        for g in self.groups:
            a, b = ctx.Pipe()
            extra = (world, forecast) if world else ()
            p = ctx.Process(target=_worker, args=(b, [int(i) for i in g], seed, canvas, max_steps, heard, *extra),
                            daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def set_tasks(self, tasks: Sequence[Task]) -> np.ndarray:
        if self.local:
            return self.local.set_tasks(tasks)
        for c in self.conns:
            c.send(("tasks", tuple(tasks)))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, actions: np.ndarray):
        if self.local:
            return self.local.step(actions)
        for c, g in zip(self.conns, self.groups):
            c.send(("step", [int(actions[i]) for i in g]))
        parts = [c.recv() for c in self.conns]
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]), [e for p in parts for e in p[3]])

    def close(self) -> None:
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)


# --------------------------------------------------------------------------- network and PPO maths


def invalid_moves(x):
    """(B, 5) bool: which actions would walk into a wall, read from the input itself (wall plane 0 and the agent
    plane 4; the padded margin is wall). Waiting is never invalid, so at least one action always stays allowed.
    Doors are never masked: a jammed door can be kicked in from outside, so trying it is a real choice."""
    import torch
    import torch.nn.functional as F

    wall, agent = x[..., 0], x[..., 4]
    H, W = wall.shape[-2:]
    pw = F.pad(wall, (1, 1, 1, 1), value=1.0)
    shifted = (pw[:, 0:H, 1:W + 1], pw[:, 2:H + 2, 1:W + 1], pw[:, 1:H + 1, 0:W], pw[:, 1:H + 1, 2:W + 2])  # MOVES
    blocked = torch.stack([(agent * s).flatten(1).sum(1) > 0.5 for s in shifted], -1)
    return torch.cat([blocked, torch.zeros_like(blocked[:, :1])], -1)


def make_policy(canvas: tuple[int, int], n_in: int | None = None, *, channels: Sequence[int] = POLICY_CHANNELS,
                mask_invalid: bool = False, recurrent: bool = False, hidden: int = 256):
    """Small CNN: two 3x3 conv layers (the second with stride 2, for CPU speed), one hidden layer, 5 action logits and
    a value. Options (all off = the original baseline, bit for bit): ``channels`` to feed (add HEARD_CHANNEL for the
    hearing plane), ``mask_invalid`` (wall moves get logit -1e9), ``recurrent`` (a GRU cell of size ``hidden``
    between the hidden layer and the heads; use ``step`` / ``seq`` instead of ``forward``)."""
    import torch
    from torch import nn

    channels = tuple(channels)
    n_in = len(channels) if n_in is None else n_in

    class PolicyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            H, W = canvas
            self.canvas, self.n_in, self.channels = canvas, n_in, channels
            self.mask_invalid, self.recurrent, self.hidden = mask_invalid, recurrent, hidden
            self.heard = HEARD_CHANNEL in channels
            self.world = WORLD_CHANNELS[0] in channels
            self.forecast = FORECAST_CHANNEL in channels
            self.body = nn.Sequential(
                nn.Conv2d(n_in, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(), nn.Linear(32 * ((H + 1) // 2) * ((W + 1) // 2), 256), nn.ReLU(),
            )
            out = 256
            if recurrent:
                self.gru = nn.GRUCell(256, hidden)
                out = hidden
            self.pi, self.v = nn.Linear(out, 5), nn.Linear(out, 1)
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(m.weight, math.sqrt(2))
                    nn.init.zeros_(m.bias)
            nn.init.orthogonal_(self.pi.weight, 0.01)
            nn.init.orthogonal_(self.v.weight, 1.0)

        def _z(self, x: torch.Tensor) -> torch.Tensor:
            return self.body(x[..., list(self.channels)].permute(0, 3, 1, 2))

        def _heads(self, z: torch.Tensor, x: torch.Tensor):
            logits = self.pi(z)
            if self.mask_invalid:
                logits = logits.masked_fill(invalid_moves(x), -1e9)
            return logits, self.v(z).squeeze(-1)

        def forward(self, x: torch.Tensor):
            """x: (B, H, W, C) as ``env.encode`` (padded) gives it, plus the hearing plane for ``heard`` variants;
            channels not in ``self.channels`` (smoke) are dropped here."""
            if self.recurrent:
                raise TypeError("recurrent policy: use step(x, h) or seq(x, h0, starts)")
            return self._heads(self._z(x), x)

        def initial_state(self, batch: int) -> torch.Tensor:
            return torch.zeros(batch, self.hidden)

        def step(self, x: torch.Tensor, h: torch.Tensor):
            """One tick for a batch: (logits, value, next hidden state)."""
            h = self.gru(self._z(x), h)
            return (*self._heads(h, x), h)

        def seq(self, x: torch.Tensor, h0: torch.Tensor, starts: torch.Tensor):
            """BPTT over a (T, B, ...) chunk. ``starts[t, b]`` = 1 where step t begins a new episode: the hidden
            state is zeroed there, so no gradient or memory crosses an episode boundary. Returns (T, B) outputs."""
            T, B = x.shape[:2]
            z = self._z(x.reshape(T * B, *x.shape[2:])).reshape(T, B, -1)
            h, hs = h0, []
            for t in range(T):
                h = self.gru(z[t], h * (1.0 - starts[t]).unsqueeze(-1))
                hs.append(h)
            logits, v = self._heads(torch.stack(hs).reshape(T * B, -1), x.reshape(T * B, *x.shape[2:]))
            return logits.reshape(T, B, -1), v.reshape(T, B)

    return PolicyNet()


def make_policy_for(config: "PPOConfig", canvas: tuple[int, int]):
    """The network a config asks for."""
    channels = policy_channels(config.heard, config.world_planes, config.forecast)
    return make_policy(canvas, channels=channels,
                       mask_invalid=config.mask_invalid, recurrent=config.recurrent, hidden=config.hidden)


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: np.ndarray,
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimates for a (T, N) rollout. ``dones[t]`` = the episode ended with step t, so
    nothing is bootstrapped across it (every ending in this env is a true termination, never a truncation).
    ``last_value`` is V of the observation after the last step. Returns (advantages, value targets)."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float64)
    gae = np.zeros(rewards.shape[1:], dtype=np.float64)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        live = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * live - values[t]
        gae = delta + gamma * lam * live * gae
        adv[t] = gae
    return adv, adv + values


def ppo_loss(logits, values, actions, old_logp, adv, returns, clip: float, vf_coef: float, ent_coef: float):
    """Clipped surrogate + value loss - entropy bonus. Advantages are normalised per minibatch (a minibatch of one
    sample has no spread, so its centred advantage, 0, is used as is: dividing by its NaN std poisoned every weight)."""
    import torch

    dist = torch.distributions.Categorical(logits=logits)
    logp = dist.log_prob(actions)
    ratio = (logp - old_logp).exp()
    a = adv - adv.mean()
    if adv.numel() > 1:
        a = a / (adv.std() + 1e-8)
    pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
    vl = 0.5 * (values - returns).pow(2).mean()
    ent = dist.entropy().mean()
    stats = {"pg": pg.item(), "vf": vl.item(), "entropy": ent.item(),
             "clipfrac": float(((ratio - 1).abs() > clip).float().mean())}
    return pg + vf_coef * vl - ent_coef * ent, stats


# --------------------------------------------------------------------------- training


def _log(msg: str) -> None:
    print(msg, flush=True)


def train_ppo(phases: Sequence[Phase], config: PPOConfig = PPOConfig(), out: str | Path | None = None,
              init=None, log=_log, checkpoint_every: int = 0, ref=None) -> dict:
    """Train a PPO policy over the curriculum ``phases`` and return {"policy", "history", "episodes", ...}.

    Deterministic for a given ``config`` (CPU, fixed thread count). ``init`` continues from a checkpoint path or a
    network (the "keeps learning" experiment, or a behaviour-cloned warm start). ``ref``: a frozen network for the
    KL term (``config.kl_coef``). ``out``: a directory for ``final.pt``, intermediate ``step_<n>.pt`` checkpoints
    (every ``checkpoint_every`` env steps) and ``history.json``."""
    import copy

    import torch

    from emberpath.cli import load_blueprint

    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    gen = torch.Generator().manual_seed(config.seed)
    blueprints = {t.blueprint: load_blueprint(t.blueprint) for p in phases for t in p.tasks}
    if init is not None:
        net, meta = load_policy(init) if isinstance(init, (str, Path)) else (init, None)
        canvas = tuple(meta["canvas"]) if meta else tuple(net.canvas)
        net.train()
    else:
        canvas = canvas_for(list(blueprints.values()))
        net = make_policy_for(config, canvas)
    if (net.mask_invalid, net.heard, net.recurrent, getattr(net, "world", False), getattr(net, "forecast", False)) != (
            config.mask_invalid, config.heard, config.recurrent, config.world_planes, config.forecast):
        raise ValueError("config variant flags do not match the initial network")
    if ref is not None:
        ref = copy.deepcopy(ref).eval()
        for prm in ref.parameters():
            prm.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=config.lr, eps=1e-5)
    out = Path(out) if out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    n, T = config.n_envs, config.rollout
    total = sum(p.steps for p in phases)
    if config.world_planes:
        vec = VecEnv(n, config.seed, canvas, config.max_steps, config.workers, heard=config.heard, world=True,
                     forecast=config.forecast)
    else:
        vec = VecEnv(n, config.seed, canvas, config.max_steps, config.workers, heard=True) if config.heard else \
            VecEnv(n, config.seed, canvas, config.max_steps, config.workers)
    history: list[dict] = []
    episodes: list[dict] = []
    step, t0, next_ckpt, n_updates = 0, time.time(), checkpoint_every or math.inf, 0
    h = net.initial_state(n) if config.recurrent else None
    try:
        for pi, phase in enumerate(phases):
            obs = vec.set_tasks(phase.tasks)
            prev_done = np.ones(n, np.float32)  # every slot starts a fresh episode
            phase_end = step + phase.steps
            while step < phase_end:
                frac = 1.0 - step / total
                for g in opt.param_groups:
                    g["lr"] = config.lr * (config.lr_final_frac + (1 - config.lr_final_frac) * frac)
                kl_coef = config.kl_coef * max(0.0, 1.0 - (step / total) / max(1e-9, config.kl_anneal_frac))
                buf_o = np.zeros((T, n, *obs.shape[1:]), np.float32)
                buf_a = np.zeros((T, n), np.int64)
                buf_lp, buf_v, buf_r, buf_d = (np.zeros((T, n), np.float32) for _ in range(4))
                buf_s = np.zeros((T, n), np.float32)  # episode starts (recurrent)
                h0 = h.clone() if config.recurrent else None
                for t in range(T):
                    with torch.no_grad():
                        if config.recurrent:
                            h = h * (1.0 - torch.as_tensor(prev_done)).unsqueeze(-1)
                            logits, v, h = net.step(torch.as_tensor(obs), h)
                        else:
                            logits, v = net(torch.as_tensor(obs))
                    dist = torch.distributions.Categorical(logits=logits)
                    a = torch.multinomial(dist.probs, 1, generator=gen).squeeze(-1)
                    buf_o[t], buf_a[t], buf_lp[t], buf_v[t] = obs, a.numpy(), dist.log_prob(a).numpy(), v.numpy()
                    buf_s[t] = prev_done
                    obs, r, d, fin = vec.step(a.numpy())
                    buf_r[t], buf_d[t] = r * config.reward_scale, d
                    prev_done = d
                    step += n
                    for e in fin:
                        e.update(phase=pi, step=step)
                        episodes.append(e)
                with torch.no_grad():
                    if config.recurrent:
                        hl = h * (1.0 - torch.as_tensor(prev_done)).unsqueeze(-1)
                        last_v = net.step(torch.as_tensor(obs), hl)[1].numpy()
                    else:
                        last_v = net(torch.as_tensor(obs))[1].numpy()
                adv, ret = compute_gae(buf_r, buf_v, buf_d, last_v, config.gamma, config.lam)
                value_only = n_updates < config.vf_warmup_updates
                if config.recurrent:
                    stats = _recurrent_update(net, opt, config, gen, buf_o, buf_a, buf_lp, adv, ret, buf_s, h0,
                                              ref, kl_coef, value_only)
                else:
                    stats = _ff_update(net, opt, config, gen, buf_o, buf_a, buf_lp, adv, ret, ref, kl_coef, value_only)
                n_updates += 1
                recent = [e for e in episodes[-200:] if e["phase"] == pi][-100:]
                row = {"step": step, "phase": pi, "time": round(time.time() - t0, 1), **stats,
                       "train_success": float(np.mean([e["outcome"] == "success" for e in recent])) if recent else None,
                       "train_return": float(np.mean([e["return"] for e in recent])) if recent else None}
                history.append(row)
                if log and len(history) % 10 == 0:
                    log(f"phase {pi} step {step:>8}  success(last {len(recent)}) {row['train_success'] or 0:.2f}  "
                        f"return {row['train_return'] or 0:6.2f}  entropy {stats['entropy']:.2f}  ({row['time']:.0f}s)")
                if out and step >= next_ckpt:
                    save_policy(net, out / f"step_{step}.pt", canvas, config, phases)
                    next_ckpt += checkpoint_every
    finally:
        vec.close()
    touched = {e["seed"] for e in episodes}
    assert not touched & set(EVAL_SEEDS), "a training episode used an evaluation seed"
    result = {"policy": net, "history": history, "episodes": episodes, "canvas": canvas,
              "wall_clock": time.time() - t0, "steps": step}
    if out:
        save_policy(net, out / "final.pt", canvas, config, phases)
        (out / "history.json").write_text(json.dumps(
            {"config": asdict(config), "phases": _phases_json(phases), "wall_clock": result["wall_clock"],
             "history": history, "episodes": episodes}))
    return result


def _kl_to_ref(ref, x, logits, h0=None, starts=None):
    """KL(ref || policy), mean over samples; the reference sees the same inputs (and resets) as the policy."""
    import torch

    with torch.no_grad():
        ref_logits = ref.seq(x, h0, starts)[0] if starts is not None else ref(x)[0]
    p_ref = torch.softmax(ref_logits, -1)
    return (p_ref * (torch.log_softmax(ref_logits, -1) - torch.log_softmax(logits, -1))).sum(-1).mean()


def _apply(net, opt, config, loss) -> None:
    import torch

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
    opt.step()


def _ff_update(net, opt, config, gen, buf_o, buf_a, buf_lp, adv, ret, ref, kl_coef, value_only) -> dict:
    import torch

    T, n = buf_a.shape
    O = torch.as_tensor(buf_o.reshape(T * n, *buf_o.shape[2:]))
    A = torch.as_tensor(buf_a.reshape(-1))
    LP, ADV = torch.as_tensor(buf_lp.reshape(-1)), torch.as_tensor(adv.reshape(-1), dtype=torch.float32)
    RET = torch.as_tensor(ret.reshape(-1), dtype=torch.float32)
    stats: dict = {}
    for _ in range(config.epochs):
        for idx in torch.randperm(T * n, generator=gen).split(config.minibatch):
            logits, v = net(O[idx])
            if value_only:
                loss = config.vf_coef * 0.5 * (v - RET[idx]).pow(2).mean()
                stats = {"pg": 0.0, "vf": loss.item() / config.vf_coef, "entropy": float(
                    torch.distributions.Categorical(logits=logits.detach()).entropy().mean()), "clipfrac": 0.0}
            else:
                loss, stats = ppo_loss(logits, v, A[idx], LP[idx], ADV[idx], RET[idx],
                                       config.clip, config.vf_coef, config.ent_coef)
                if ref is not None and kl_coef > 0:
                    kl = _kl_to_ref(ref, O[idx], logits)
                    loss = loss + kl_coef * kl
                    stats["kl_ref"] = kl.item()
            _apply(net, opt, config, loss)
    return stats


def _recurrent_update(net, opt, config, gen, buf_o, buf_a, buf_lp, adv, ret, buf_s, h0, ref, kl_coef,
                      value_only) -> dict:
    """Minibatches are groups of whole env lanes, replayed over the full rollout from the hidden state they had at
    its start (truncated BPTT over the rollout; ``buf_s`` zeroes the state at every episode start inside it)."""
    import torch

    T, n = buf_a.shape
    per = max(1, min(n, config.minibatch // T))
    O, A, LP = torch.as_tensor(buf_o), torch.as_tensor(buf_a), torch.as_tensor(buf_lp)
    ADV, RET = torch.as_tensor(adv, dtype=torch.float32), torch.as_tensor(ret, dtype=torch.float32)
    S = torch.as_tensor(buf_s)
    stats: dict = {}
    for _ in range(config.epochs):
        for lanes in torch.randperm(n, generator=gen).split(per):
            logits, v = net.seq(O[:, lanes], h0[lanes], S[:, lanes])
            lg, vv = logits.reshape(-1, 5), v.reshape(-1)
            if value_only:
                loss = config.vf_coef * 0.5 * (vv - RET[:, lanes].reshape(-1)).pow(2).mean()
                stats = {"pg": 0.0, "vf": loss.item() / config.vf_coef, "entropy": float(
                    torch.distributions.Categorical(logits=lg.detach()).entropy().mean()), "clipfrac": 0.0}
            else:
                loss, stats = ppo_loss(lg, vv, A[:, lanes].reshape(-1), LP[:, lanes].reshape(-1),
                                       ADV[:, lanes].reshape(-1), RET[:, lanes].reshape(-1),
                                       config.clip, config.vf_coef, config.ent_coef)
                if ref is not None and kl_coef > 0:
                    kl = _kl_to_ref(ref, O[:, lanes], logits, h0[lanes], S[:, lanes])
                    loss = loss + kl_coef * kl
                    stats["kl_ref"] = kl.item()
            _apply(net, opt, config, loss)
    return stats


def _phases_json(phases: Sequence[Phase]) -> list:
    return [{"steps": p.steps, "tasks": [asdict(t) for t in p.tasks]} for p in phases]


def save_policy(net, path: str | Path, canvas, config: PPOConfig, phases: Sequence[Phase], extra=None) -> None:
    import torch

    arch = {"mask_invalid": net.mask_invalid, "recurrent": net.recurrent, "hidden": net.hidden}
    if getattr(net, "world", False):  # only the new variants carry these keys: older checkpoints stay as they were
        arch.update(world_planes=True, forecast=bool(net.forecast))
    torch.save({"state_dict": net.state_dict(), "canvas": list(canvas), "config": asdict(config),
                "phases": _phases_json(phases), "channels": list(net.channels), "arch": arch,
                **(extra or {})}, path)


def load_policy(path: str | Path):
    """(network, metadata) from a checkpoint written by ``train_ppo`` or ``train_bc``. Checkpoints from before the
    variants (no "arch" entry) load as the original feed-forward, unmasked network."""
    import torch

    meta = torch.load(path, map_location="cpu", weights_only=False)
    channels = tuple(meta["channels"])
    known = [policy_channels(h, w, f) for h in (False, True) for w in (False, True) for f in (False, True) if w or not f]
    if channels not in known:
        raise ValueError(f"{path}: trained on channels {meta['channels']}, this code feeds {POLICY_CHANNELS} "
                         f"(or {HEARD_POLICY_CHANNELS} with the hearing plane, or a world-planes variant)")
    arch = meta.get("arch", {})
    if bool(arch.get("world_planes", False)) != (WORLD_CHANNELS[0] in channels) or \
            bool(arch.get("forecast", False)) != (FORECAST_CHANNEL in channels):
        raise ValueError(f"{path}: arch {arch} does not match its channels")
    net = make_policy(tuple(meta["canvas"]), channels=channels, mask_invalid=arch.get("mask_invalid", False),
                      recurrent=arch.get("recurrent", False), hidden=arch.get("hidden", 256))
    net.load_state_dict(meta["state_dict"])
    net.eval()
    return net, meta


# --------------------------------------------------------------------------- behaviour cloning (warm start)

# Teachers must be evidence-disciplined: imitating them can only pass on what they inferred from observations and
# events. The oracle (handed the true rules) is refused, so its knowledge cannot leak into the student.
BC_TEACHERS = ("adaptive", "baseline", "naive_repair")


WORLD_SCALAR_CHANNELS = tuple(WORLD_BASE + WORLD_PLANES.index(n) for n in WORLD_SCALAR_PLANES)
WORLD_Q8_CHANNELS = tuple(WORLD_BASE + WORLD_PLANES.index(n) for n in WORLD_Q8_PLANES) + (FORECAST_CHANNEL,)


def _bc_binary_channels(channels: Sequence[int]) -> list[int]:
    return [c for c in channels if c != TIME_CHANNEL and c not in WORLD_SCALAR_CHANNELS and c not in WORLD_Q8_CHANNELS]


def _bc_scalar_channels(channels: Sequence[int]) -> list[int]:
    return [c for c in channels if c in WORLD_SCALAR_CHANNELS]


def _bc_q8_channels(channels: Sequence[int]) -> list[int]:
    return [c for c in channels if c in WORLD_Q8_CHANNELS]


def _pack_states(x: np.ndarray, channels: Sequence[int], max_steps: int) -> dict:
    """Bit-packed binary planes, ticks left, and (world variants) per-sample scalars and uint8 planes."""
    out = {"bits": np.packbits(x[..., _bc_binary_channels(channels)].astype(bool).reshape(len(x), -1), axis=1),
           "tleft": np.rint(x[:, 0, 0, TIME_CHANNEL] * max_steps).astype(np.uint8)}
    sc, q8 = _bc_scalar_channels(channels), _bc_q8_channels(channels)
    if sc or q8:
        out["scal"] = x[:, 0, 0, sc].astype(np.float32)
        out["q8"] = np.rint(x[..., q8] * 255.0).astype(np.uint8)
    return out


def _teacher_job(args) -> dict:
    """Run the teacher on (task, training seeds); record the policy input it would have produced, its action, the
    scaled reward, and episode starts. The teacher gets exactly what ``evaluate.run_episode`` hands an agent."""
    task, seeds, teacher, canvas, max_steps, heard = args[:6]
    world, forecast = (tuple(args[6:8]) + (False, False))[:2]
    from emberpath.agents import AGENTS

    cls = AGENTS[teacher]
    if getattr(cls, "privileged", False) or teacher not in BC_TEACHERS:
        raise ValueError(f"{teacher!r} is not an evidence-disciplined teacher")
    if set(seeds) - set(TRAIN_SEEDS):
        raise ValueError("behaviour-cloning data must come from training seeds only")
    env = _make_env(task, max_steps)
    xs, acts, rews, starts, eps = [], [], [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        agent = cls(env.bp)  # never handed the world
        o, events, hmap, first = env.world.observe(), [], {}, True
        wm = WorldModelFeatures(env.bp, forecast) if world else None
        while True:
            x = pad_to(obs, canvas)
            if world:
                xs.append(extend_input(x, hmap, wm, o, events))
            else:
                xs.append(with_heard(x, hmap) if heard else x)
            d = agent.act(o, events)
            obs, r, term, trunc, info = env.step(d.action)
            acts.append(d.action)
            rews.append(r)
            starts.append(first)
            eps.append(s)
            first = False
            events = info["events"]
            update_heard(hmap, events)
            if term or trunc:
                break
            o = env.world.observe()
    channels = policy_channels(heard, world, forecast)
    return {**_pack_states(np.stack(xs), channels, max_steps), "act": np.array(acts, np.int64),
            "rew": np.array(rews, np.float32), "start": np.array(starts, bool), "seed": np.array(eps, np.int64),
            "task": task.label}


def collect_teacher(tasks: Sequence[Task], episodes_per_task: int, teacher: str = "adaptive", seed: int = 0,
                    canvas: tuple[int, int] = (13, 24), max_steps: int = 150, heard: bool = False,
                    gamma: float = 0.99, reward_scale: float = 0.1, workers: int = 0, world: bool = False,
                    forecast: bool = False) -> dict:
    """(state, action, return) data from ``teacher`` on TRAINING seeds only (drawn per task with ``seed``). States
    are stored bit-packed (binary planes) plus ticks remaining, so a few hundred thousand fit in memory. Returns are
    discounted sums of the scaled reward, as targets for the value head."""
    if teacher not in BC_TEACHERS:
        raise ValueError(f"{teacher!r} is not an evidence-disciplined teacher (choose from {BC_TEACHERS})")
    rng = np.random.default_rng([seed, 7])
    jobs = []
    for task in tasks:
        picks = [int(TRAIN_SEEDS[i]) for i in rng.choice(len(TRAIN_SEEDS), episodes_per_task, replace=False)]
        for k in range(0, len(picks), 25):
            jobs.append((task, picks[k:k + 25], teacher, canvas, max_steps, heard)
                        + ((world, forecast) if world else ()))
    if workers > 0:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_teacher_job, jobs)
    else:
        parts = [_teacher_job(j) for j in jobs]
    channels = policy_channels(heard, world, forecast)
    keys = ("bits", "tleft", "act", "ret", "start", "seed", "task", "ep") + (("scal", "q8") if world else ())
    out: dict = {k: [] for k in keys}
    ep_id = 0
    for part in parts:
        n = len(part["act"])
        for k in ("bits", "tleft") + (("scal", "q8") if world else ()):
            out[k].append(part[k])
        ret = np.zeros(n, np.float32)
        g = 0.0
        for i in reversed(range(n)):  # each episode ends at the sample before the next start
            if i + 1 < n and part["start"][i + 1]:
                g = 0.0
            g = part["rew"][i] * reward_scale + gamma * g
            ret[i] = g
        out["ret"].append(ret)
        ids = np.cumsum(part["start"]) - 1 + ep_id
        ep_id = int(ids[-1]) + 1
        out["ep"].append(ids)
        for k in ("act", "start", "seed"):
            out[k].append(part[k])
        out["task"].extend([part["task"]] * n)
    data = {k: (np.concatenate(v) if k != "task" else np.array(v)) for k, v in out.items()}
    data.update(canvas=tuple(canvas), channels=tuple(channels), max_steps=max_steps, teacher=teacher)
    assert set(np.unique(data["seed"]).tolist()) <= set(TRAIN_SEEDS)
    return data


def load_teacher_data(path: str | Path) -> dict:
    """A dataset written by ``train_ppo.py --bc-teacher`` (np.savez of ``collect_teacher``'s output)."""
    z = np.load(path, allow_pickle=False)
    data = {k: z[k] for k in z.files}
    data.update(canvas=tuple(z["canvas"].tolist()), channels=tuple(z["channels"].tolist()),
                max_steps=int(z["max_steps"]), teacher=str(z["teacher"]))
    return data


def merge_teacher_data(parts: Sequence[dict]) -> dict:
    """Concatenate datasets of the same teacher, canvas, channels and horizon (episode ids renumbered)."""
    meta = ("canvas", "channels", "max_steps", "teacher")
    for d in parts[1:]:
        if any(tuple(np.atleast_1d(d[k])) != tuple(np.atleast_1d(parts[0][k])) for k in meta):
            raise ValueError("teacher datasets differ in teacher, canvas, channels or horizon")
    out = {k: parts[0][k] for k in meta}
    offset, eps = 0, []
    for d in parts:
        eps.append(d["ep"] + offset)
        offset += int(d["ep"].max()) + 1
    for k in parts[0]:
        if k not in meta:
            out[k] = np.concatenate(eps) if k == "ep" else np.concatenate([d[k] for d in parts])
    assert set(np.unique(out["seed"]).tolist()) <= set(TRAIN_SEEDS)
    return out


def decode_states(data: dict, idx: np.ndarray) -> np.ndarray:
    """Float input planes for samples ``idx`` (smoke left at zero: the network drops it)."""
    H, W = data["canvas"]
    channels = data["channels"]
    binary = _bc_binary_channels(channels)
    n_planes = max(channels) + 1
    bits = np.unpackbits(data["bits"][idx], axis=1, count=H * W * len(binary)).reshape(len(idx), H, W, len(binary))
    x = np.zeros((len(idx), H, W, n_planes), np.float32)
    x[..., binary] = bits
    x[..., TIME_CHANNEL] = (data["tleft"][idx].astype(np.float64) / data["max_steps"]).astype(np.float32)[:, None, None]
    sc, q8 = _bc_scalar_channels(channels), _bc_q8_channels(channels)
    if sc:
        x[..., sc] = data["scal"][idx][:, None, None, :]
    if q8:
        x[..., q8] = data["q8"][idx].astype(np.float32) / np.float32(255.0)
    return x


def train_bc(data: dict, config: PPOConfig, epochs: int = 4, lr: float = 1e-3, batch: int = 256,
             vf_coef: float = 0.5, seed: int = 0, net=None, log=_log, holdout: float = 0.05) -> dict:
    """Behaviour cloning: cross-entropy to the teacher's actions plus (``vf_coef``) a value regression to its
    discounted returns, so PPO fine-tuning starts with a critic that roughly fits a good policy. ``holdout`` of the
    episodes (still training seeds) is kept back to report accuracy. Recurrent networks train on whole episodes
    from a zero state. Returns {"policy", "history", "val_acc"}."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    torch.set_num_threads(config.torch_threads)
    canvas = tuple(data["canvas"])
    net = net or make_policy_for(config, canvas)
    if not set(net.channels) <= set(data["channels"]):  # a network may read fewer planes (an ablation)
        raise ValueError("the network reads channels the data does not have")
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    rng = np.random.default_rng([seed, 11])
    ep_ids = np.unique(data["ep"])
    val_eps = set(rng.choice(ep_ids, max(1, int(len(ep_ids) * holdout)), replace=False).tolist())
    is_val = np.isin(data["ep"], list(val_eps))
    act = torch.as_tensor(data["act"])
    ret = torch.as_tensor(data["ret"])
    history = []
    t0 = time.time()

    def episodes_of(mask):
        starts = np.flatnonzero(data["start"] & mask)
        ends = np.append(np.flatnonzero(data["start"])[1:], len(data["start"]))
        nxt = {int(s): int(ends[np.searchsorted(ends, s, side="right")]) for s in starts}
        return [np.arange(s, nxt[s]) for s in starts]

    def recurrent_batch(eps):
        L = max(len(e) for e in eps)
        idx = np.zeros((L, len(eps)), np.int64)
        valid = np.zeros((L, len(eps)), bool)
        for j, e in enumerate(eps):
            idx[:len(e), j], valid[:len(e), j] = e, True
        x = torch.as_tensor(decode_states(data, idx.reshape(-1))).reshape(L, len(eps), *canvas, -1)
        starts = torch.zeros(L, len(eps))
        starts[0] = 1.0
        logits, v = net.seq(x, net.initial_state(len(eps)), starts)
        m = torch.as_tensor(valid)
        I = torch.as_tensor(idx)
        return logits[m], v[m], act[I[m]], ret[I[m]]

    def loss_of(logits, v, a, r):
        ce = F.cross_entropy(logits, a)
        vl = 0.5 * (v - r).pow(2).mean()
        return ce + vf_coef * vl, ce, vl, (logits.argmax(-1) == a).float().mean()

    train_idx = np.flatnonzero(~is_val)
    train_eps, val_eps_list = (episodes_of(~is_val), episodes_of(is_val)) if config.recurrent else (None, None)
    for ep in range(epochs):
        net.train()
        sums = np.zeros(3)
        nb = 0
        if config.recurrent:
            order = rng.permutation(len(train_eps))
            per = max(1, batch // 32)  # whole episodes per step (about 60 ticks each)
            groups = [[train_eps[i] for i in order[k:k + per]] for k in range(0, len(order), per)]
        else:
            perm = rng.permutation(train_idx)
            groups = [perm[k:k + batch] for k in range(0, len(perm), batch)]
        for g in groups:
            if config.recurrent:
                logits, v, a, r = recurrent_batch(g)
            else:
                x = torch.as_tensor(decode_states(data, g))
                logits, v = net(x)
                a, r = act[g], ret[g]
            loss, ce, vl, acc = loss_of(logits, v, a, r)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sums += [ce.item(), vl.item(), acc.item()]
            nb += 1
        val = _bc_eval(net, data, is_val, val_eps_list, recurrent_batch, loss_of, config.recurrent)
        row = {"epoch": ep, "ce": sums[0] / nb, "vf": sums[1] / nb, "acc": sums[2] / nb, **val,
               "time": round(time.time() - t0, 1)}
        history.append(row)
        if log:
            log(f"bc epoch {ep}: train ce {row['ce']:.3f} acc {row['acc']:.3f}  val ce {val['val_ce']:.3f} "
                f"acc {val['val_acc']:.3f}  vf {row['vf']:.3f}  ({row['time']:.0f}s)")
    net.eval()
    return {"policy": net, "history": history, "val_acc": history[-1]["val_acc"] if history else None}


def _bc_eval(net, data, is_val, val_eps, recurrent_batch, loss_of, recurrent) -> dict:
    import torch

    net.eval()
    with torch.no_grad():
        if recurrent:
            outs = [recurrent_batch(val_eps[k:k + 8]) for k in range(0, len(val_eps), 8)]
            logits, v, a, r = (torch.cat([o[i] for o in outs]) for i in range(4))
        else:
            idx = np.flatnonzero(is_val)
            parts = [net(torch.as_tensor(decode_states(data, idx[k:k + 1024]))) for k in range(0, len(idx), 1024)]
            logits, v = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
            a, r = torch.as_tensor(data["act"][idx]), torch.as_tensor(data["ret"][idx])
        _, ce, vl, acc = loss_of(logits, v, a, r)
    return {"val_ce": ce.item(), "val_vf": vl.item(), "val_acc": acc.item()}


# --------------------------------------------------------------------------- the agent


class PPOAgent:
    """A trained policy behind the ``Agent`` interface. It sees only what ``env.encode`` shows (rebuilt from the
    observation and its own events), and explains each move with the policy's confidence in it."""

    def __init__(self, policy, blueprint: Blueprint, greedy: bool = True, max_steps: int = 150, seed: int = 0,
                 canvas: tuple[int, int] | None = None):
        self.policy, self.bp, self.greedy, self.max_steps = policy, blueprint, greedy, max_steps
        self.canvas = canvas or policy.canvas
        self.name = "ppo" if greedy else "ppo_sampled"
        self.rng = np.random.default_rng(seed)
        self.bumped: set = set()
        self.heard: dict = {}  # hearing plane (variants with heard=True): built from this agent's own events
        # world-model planes (variants with world_planes=True): the adaptive agent's belief, from this agent's events
        self.world_model = WorldModelFeatures(blueprint, getattr(policy, "forecast", False)) \
            if getattr(policy, "world", False) else None
        self.h = policy.initial_state(1) if getattr(policy, "recurrent", False) else None  # fresh per episode

    def features(self, obs: Observation, events: list[Event]) -> np.ndarray:
        for e in events:
            if e.outcome and parse_key(e.key)[0] == "door_jammed":
                self.bumped.add(key_cell(e.key))
        x = pad_to(encode_observation(self.bp, obs, self.bumped, self.max_steps), self.canvas)
        if self.world_model is not None:
            return extend_input(x, update_heard(self.heard, events), self.world_model, obs, events)
        if getattr(self.policy, "heard", False):
            x = with_heard(x, update_heard(self.heard, events))
        return x

    def act(self, obs: Observation, events: list[Event]) -> Decision:
        import torch

        x = self.features(obs, events)
        with torch.no_grad():
            if self.h is not None:
                logits, _, self.h = self.policy.step(torch.as_tensor(x[None]), self.h)
            else:
                logits, _ = self.policy(torch.as_tensor(x[None]))
        p = torch.softmax(logits[0], -1).numpy().astype(np.float64)
        a = int(p.argmax()) if self.greedy else int(self.rng.choice(5, p=p / p.sum()))
        how = "" if self.greedy else ", sampled"
        reason = f"Learned policy: {ACTION_WORDS[a]} ({p[a]:.0%} confident{how})."
        return Decision(a, reason)


def ppo_factory(policy, greedy: bool = True, max_steps: int = 150, seed: int = 0):
    """An ``Agent`` factory for ``evaluate.run_episode``: the world is never handed to the agent."""

    def make(bp: Blueprint, _world) -> PPOAgent:
        return PPOAgent(policy, bp, greedy=greedy, max_steps=max_steps, seed=seed)

    return make


def evaluate_policy(policy, blueprint: Blueprint, household, novelty: str | None,
                    seeds: Sequence[int] = EVAL_SEEDS, greedy: bool = True, max_steps: int = 150,
                    sample_seed: int = 0) -> list:
    """Run the policy through the shared harness on held-out seeds. Returns ``EpisodeResult``s (agent name
    "ppo" greedy, "ppo_sampled" sampled). Refuses any training seed. A sampled run is seeded per episode."""
    from emberpath.evaluate import run_episode

    seeds = list(seeds)
    leaked = set(seeds) & set(TRAIN_SEEDS)
    if leaked:
        raise ValueError(f"evaluate_policy uses held-out seeds only; got training seeds {sorted(leaked)[:5]}...")
    return [run_episode(blueprint, household, ppo_factory(policy, greedy, max_steps, sample_seed * 100_003 + s),
                        novelty, s, max_steps) for s in seeds]


def episode_components(agent, blueprint: Blueprint, household, novelty: str | None, seed: int,
                       max_steps: int = 150) -> dict:
    """One episode with the reward broken into components (summed) plus the path: for the exploit inspection."""
    from emberpath.reward import COMPONENTS, RewardTracker
    from emberpath.scenarios import NOVELTIES
    from emberpath.world import World

    world = World(blueprint, household, seed=seed, novelties=NOVELTIES[novelty](blueprint, household) if novelty else [],
                  max_steps=max_steps)
    tracker = RewardTracker(world)
    comp = dict.fromkeys(COMPONENTS, 0.0)
    obs, events, path, actions = world.observe(), [], [world.agent_pos], []
    while not world.done:
        d = agent.act(obs, events)
        res = world.step(d.action)
        for k, v in tracker.update(world).components.items():
            comp[k] += v
        obs, events = res.obs, res.events
        path.append(world.agent_pos)
        actions.append(d.action)
    return {"seed": seed, "outcome": world.outcome, "steps": world.t, "return": tracker.total, "components": comp,
            "path": path, "actions": actions, "counts": world.counts(), "bumped": sorted(world.bumped_jams)}


# --------------------------------------------------------------------------- command line


def default_phases(blueprint: str, profiles: Sequence[str], novelties: Sequence[str], steps: int,
                   warmup: int, mix_studio: bool = False) -> list[Phase]:
    """Warm-up on the studio with the dog (no novelty), then ``blueprint`` x profiles x novelties (plus the studio
    warm-up task, against forgetting, when ``mix_studio``)."""
    nov = [None if n == "none" else n for n in novelties]
    phases = [Phase((Task("studio", "single"),), warmup)] if warmup else []
    tasks = tuple(Task(blueprint, p, n) for p in profiles for n in nov)
    if mix_studio:
        tasks += (Task("studio", "single"),)
    phases.append(Phase(tasks, steps))
    return phases


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="train_ppo", description="Train the PPO baseline (docs/RL.md).")
    p.add_argument("--blueprint", default="ranch")
    p.add_argument("--profiles", nargs="+", default=["single", "kids"])
    p.add_argument("--novelties", nargs="+", default=["none", "door_jam", "carpet_flashover", "hidden_dependent"],
                   help="'none' for the nominal world; each episode samples one")
    p.add_argument("--steps", type=int, default=2_000_000, help="env steps on --blueprint (after the warm-up)")
    p.add_argument("--warmup-steps", type=int, default=150_000, help="env steps on studio/single first (0 = none)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/ppo", help="checkpoint directory (gitignored)")
    p.add_argument("--init", default=None, help="continue from this checkpoint")
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--threads", type=int, default=4, help="torch threads (fix it: results depend on it)")
    p.add_argument("--checkpoint-every", type=int, default=250_000)
    g = p.add_argument_group("variants (docs/RL.md, 'Improvements tried'); all off = the original baseline")
    g.add_argument("--mask-invalid", action="store_true", help="never move into a wall")
    g.add_argument("--heard", action="store_true", help="add the hearing plane (from hearing events)")
    g.add_argument("--recurrent", action="store_true", help="GRU policy, BPTT over each rollout")
    g.add_argument("--hidden", type=int, default=256)
    g.add_argument("--mix-studio", action="store_true", help="keep the studio warm-up task in the main phase")
    g.add_argument("--bc-teacher", default=None, choices=BC_TEACHERS,
                   help="behaviour-clone this evidence-disciplined agent first (training seeds only)")
    g.add_argument("--bc-episodes", type=int, default=400, help="teacher episodes per training setting")
    g.add_argument("--bc-epochs", type=int, default=4)
    g.add_argument("--bc-lr", type=float, default=1e-3)
    g.add_argument("--bc-data", nargs="+", default=None,
                   help="reuse teacher dataset(s) saved by earlier runs (.npz); several are merged")
    g.add_argument("--bc-only", action="store_true", help="stop after behaviour cloning (writes bc.pt)")
    g.add_argument("--collect-only", action="store_true",
                   help="stop after collecting (and saving) the teacher data; e.g. a superset of planes to reuse "
                        "with --bc-data for ablations that read fewer")
    g.add_argument("--kl-coef", type=float, default=0.0, help="KL toward the cloned policy during PPO")
    g.add_argument("--kl-ref", default=None, help="checkpoint for the KL reference (default: --init, e.g. a bc.pt)")
    g.add_argument("--kl-anneal", type=float, default=0.5, help="fraction of the run over which the KL fades out")
    g.add_argument("--vf-warmup", type=int, default=0, help="initial updates that train only the value head")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--ent-coef", type=float, default=0.01)
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--rollout", type=int, default=128)
    g.add_argument("--save-init", default=None,
                   help="also save the network as it is before any training (behaviour cloning or PPO) to this path; "
                        "it does not change the run (formal_eval's 0%% 'untrained' checkpoint)")
    g.add_argument("--world-planes", action="store_true",
                   help="add the world-model planes (burns, dependent kinds/states, materials, believed rules)")
    g.add_argument("--forecast", action="store_true", help="also the believed fire-arrival forecast plane")
    g.add_argument("--skip-inapplicable", action="store_true",
                   help="drop household x novelty combinations the house cannot host (instead of refusing)")
    g.add_argument("--extra-tasks", nargs="*", default=[], metavar="BLUEPRINT/PROFILE/NOVELTY",
                   help="extra settings sampled in the main phase alongside --blueprint's (e.g. ranch/kids/none)")
    g.add_argument("--canvas", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="policy canvas (default: the largest house trained on); smaller houses are padded into it")
    g.add_argument("--repeat-main", type=int, default=1,
                   help="with --extra-tasks: list --blueprint's settings this many times (sampling weight); 0 = only "
                        "the extra tasks (e.g. to collect teacher data for them alone)")
    g.add_argument("--seed-init", action="store_true",
                   help="no-op, kept for old command lines: every generator is now seeded with --seed before any "
                        "network is built")
    a = p.parse_args(argv)
    if a.forecast and not a.world_planes:
        raise SystemExit("--forecast needs --world-planes")
    if a.save_init and not a.bc_teacher:
        raise SystemExit("--save-init needs --bc-teacher (a PPO-only run builds its network inside train_ppo)")
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.scenarios import applicable

    bp = load_blueprint(a.blueprint)
    skipped = []
    for prof in a.profiles:
        for n in a.novelties:
            if not applicable(bp, HOUSEHOLDS[prof], None if n == "none" else n):
                if not a.skip_inapplicable:
                    raise SystemExit(f"{a.blueprint}: {prof} / {n} does not apply")
                skipped.append((prof, n))
    extra = []
    for spec in a.extra_tasks:
        b_, pr_, n_ = spec.split("/")
        if not applicable(load_blueprint(b_), HOUSEHOLDS[pr_], None if n_ == "none" else n_):
            raise SystemExit(f"--extra-tasks {spec} does not apply")
        extra.append(Task(b_, pr_, None if n_ == "none" else n_))
    if skipped:
        print(f"skipping inapplicable settings: {skipped}", flush=True)
    # Seed every generator before any network is built. The BC stage used to build its network from torch's default
    # generator, which is seeded randomly per process, so a BC (+PPO) run was not determined by --seed
    # (runs/formal_eval/PREREGISTRATION.md, deviation of 2026-11-09 01:45). train_ppo / train_bc re-seed as before.
    import random

    import torch

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    cfg = PPOConfig(n_envs=a.envs, seed=a.seed, workers=a.workers, torch_threads=a.threads,
                    mask_invalid=a.mask_invalid, heard=a.heard, recurrent=a.recurrent, hidden=a.hidden,
                    kl_coef=a.kl_coef, kl_anneal_frac=a.kl_anneal, vf_warmup_updates=a.vf_warmup, lr=a.lr,
                    ent_coef=a.ent_coef, gamma=a.gamma, rollout=a.rollout, world_planes=a.world_planes,
                    forecast=a.forecast)
    phases = default_phases(a.blueprint, a.profiles, a.novelties, a.steps, a.warmup_steps, a.mix_studio)
    if skipped or extra or a.repeat_main != 1:
        own = tuple(t for t in phases[-1].tasks if t.blueprint == a.blueprint
                    and (t.profile, t.novelty or "none") not in skipped)
        rest = tuple(t for t in phases[-1].tasks if t.blueprint != a.blueprint)
        main_tasks = own * a.repeat_main + rest + tuple(extra)
        if a.repeat_main == 0:
            phases = [Phase(tuple(extra), phases[-1].steps)]
        phases[-1] = Phase(main_tasks, phases[-1].steps)
    init, ref = a.init, None
    if a.kl_coef > 0 and not a.bc_teacher and (a.kl_ref or a.init):
        ref = load_policy(a.kl_ref or a.init)[0]
    if a.bc_teacher:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        tasks = list(dict.fromkeys(t for ph in phases for t in ph.tasks))
        canvas = canvas_for([load_blueprint(t.blueprint) for t in tasks])
        if a.canvas:
            if a.canvas[0] < canvas[0] or a.canvas[1] < canvas[1]:
                raise SystemExit(f"--canvas {a.canvas} is smaller than a training house {canvas}")
            canvas = tuple(a.canvas)
        t0 = time.time()
        tag = ("_heard" if cfg.heard else "") + ("_world" if cfg.world_planes else "") + ("_fc" if cfg.forecast else "")
        cache = out / f"bc_data_{a.bc_teacher}_{a.bc_episodes}_s{a.seed}{tag}.npz"
        if a.bc_data:
            cache = Path(a.bc_data[0])
        if a.bc_data and len(a.bc_data) > 1:
            data = merge_teacher_data([load_teacher_data(f) for f in a.bc_data])
            cache = Path("+".join(a.bc_data))
        if (a.bc_data and len(a.bc_data) > 1) or cache.exists():  # reuse (the file records teacher, tasks, seeds)
            if not (a.bc_data and len(a.bc_data) > 1):
                data = load_teacher_data(cache)
            if tuple(data["canvas"]) != tuple(canvas):
                raise SystemExit(f"{cache} was made on a {data['canvas']} canvas, this run needs {canvas}")
            if data["teacher"] != a.bc_teacher or not set(make_policy_for(cfg, canvas).channels) <= set(
                    data["channels"]):
                raise SystemExit(f"{cache} was made with another teacher or input")
        else:
            data = collect_teacher(tasks, a.bc_episodes, a.bc_teacher, a.seed, canvas, cfg.max_steps, cfg.heard,
                                   cfg.gamma, cfg.reward_scale, a.workers, world=cfg.world_planes,
                                   forecast=cfg.forecast)
            np.savez(cache, **{k: np.asarray(v) for k, v in data.items()})
        t_collect = time.time() - t0
        print(f"bc data: {len(data['act'])} samples from {len(np.unique(data['ep']))} {a.bc_teacher} episodes "
              f"({t_collect:.0f}s) -> {cache}", flush=True)
        if a.collect_only:
            return 0
        net0 = load_policy(a.init)[0].train() if a.init else make_policy_for(cfg, canvas)
        if a.save_init:  # a copy of the untouched weights; draws nothing from any RNG
            Path(a.save_init).parent.mkdir(parents=True, exist_ok=True)
            save_policy(net0, a.save_init, canvas, cfg, [Phase(tuple(tasks), 0)], extra={"untrained": True})
        bc = train_bc(data, cfg, epochs=a.bc_epochs, lr=a.bc_lr, seed=a.seed, net=net0)
        bc_phases = [Phase(tuple(tasks), 0)]
        save_policy(bc["policy"], out / "bc.pt", canvas, cfg, bc_phases,
                    extra={"bc": {"teacher": a.bc_teacher, "samples": int(len(data["act"])),
                                  "episodes_per_task": a.bc_episodes, "history": bc["history"],
                                  "collect_seconds": t_collect, "train_seeds_used": sorted(set(
                                      int(x) for x in np.unique(data["seed"])))}})
        print(f"bc: val acc {bc['val_acc']:.3f}, {time.time() - t0:.0f}s total -> {out / 'bc.pt'}", flush=True)
        if a.bc_only:
            return 0
        init = bc["policy"]
        ref = bc["policy"] if a.kl_coef > 0 else None
    res = train_ppo(phases, cfg, out=a.out, init=init, checkpoint_every=a.checkpoint_every, ref=ref)
    print(f"done: {res['steps']} env steps, {len(res['episodes'])} episodes, {res['wall_clock']:.0f}s -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
