"""Pre-registered evaluation of the HOUSE-trained policies (BC+PPO with the world-model planes) on held-out fires.

The protocol is fixed in runs/house_rl/PREREGISTRATION.md and read from runs/house_rl/protocol.json; this script only
executes it. It reuses scripts/formal_eval.py's tested statistics (paired / two-level bootstrap, Holm), its manifest
convention (``build_manifest`` with the run directory) and its lockstep harness, unchanged. Subcommands:

    python scripts/house_eval.py manifest --method bcppo --seed 0 [--run runs/house_rl/train/bcppo_s0]
    python scripts/house_eval.py eval --workers 7          # episodes (+ resistance-use behaviour), cached per job
    python scripts/house_eval.py lockstep --workers 7      # evidence lockstep on every evaluated checkpoint
    python scripts/house_eval.py exploits                  # top-return episodes, reward components
    python scripts/house_eval.py curve --workers 7         # manifest points on the house set, fires 1000..1019
    python scripts/house_eval.py stats                     # runs/house_rl/results.{json,md}
    python scripts/house_eval.py all --workers 7

Episodes go through the shared harness (``evaluate.run_episode``); learned agents are built only through
``rl_baseline.ppo_factory`` (never handed the world). Behaviour metrics (burn ticks, carrying) are read from the world
by the HARNESS after each tick, for measurement only: the agent never sees them beyond its own Observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import formal_eval as FE  # noqa: E402  (statistics, manifests and the lockstep pair, reused unchanged)

OUT = ROOT / "runs" / "house_rl"
TRAIN = OUT / "train"
CACHE = OUT / "cache"
PROTOCOL = OUT / "protocol.json"
TRAINING_FIRE_SEEDS = range(0, 1000)


def load_protocol(path: Path = PROTOCOL) -> dict:
    return json.loads(Path(path).read_text())


def fire_seeds(spec) -> list[int]:
    return list(range(spec[0], spec[1]))


def set_settings(proto: dict, name: str) -> list[tuple[str, str, str]]:
    return [tuple(x) for x in proto["eval_sets"][name]["settings"]]


def build_settings(blueprints: Sequence[str], profiles: Sequence[str], novelties: Sequence[str]) -> list[list[str]]:
    """Every applicable (blueprint, profile, novelty), in a fixed order."""
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.scenarios import applicable

    out = []
    for b in blueprints:
        bp = load_blueprint(b)
        for p in profiles:
            for n in novelties:
                if applicable(bp, HOUSEHOLDS[p], None if n == "none" else n):
                    out.append([b, p, n])
    return out


# --------------------------------------------------------------------------- agents and episodes


def agent_specs(proto: dict) -> dict[str, dict]:
    specs = {p: {"kind": "planner", "method": p, "seed": None, "ckpt": None} for p in proto["planners"]}
    for method, m in proto["methods"].items():
        for s in m["seeds"]:
            ck = ROOT / m["checkpoint"].format(seed=s)
            if ck.exists():
                specs[f"{method}_s{s}"] = {"kind": m["kind"], "method": method, "seed": s, "ckpt": str(ck),
                                           "sets": m.get("sets")}
    for label, ref in proto.get("fixed_checkpoints", {}).items():
        ck = ROOT / ref["path"]
        if ck.exists():
            specs[label] = {"kind": ref["kind"], "method": label, "seed": None, "ckpt": str(ck), "sets": ref.get("sets")}
    return specs


def _cache_file(label: str, spec: dict, setting, seeds: Sequence[int]) -> Path:
    stamp = FE.sha256(Path(spec["ckpt"]))[:16] if spec["ckpt"] else "planner"
    key = hashlib.sha256(repr((label, stamp, tuple(setting), seeds[0], seeds[-1], len(seeds), "v2")).encode()).hexdigest()
    return CACHE / f"{label}__{'_'.join(setting).replace(':', '-')}__{key[:12]}.json"


def play(bp, hh, factory_or_name, novelty, seed: int) -> dict:
    """One episode through the shared harness, with resistance-use measurements taken by the harness."""
    from emberpath.evaluate import run_episode
    from emberpath.grid import MOVES, WAIT
    from emberpath.world import RESCUER

    st = {"max_burns": 0, "carried_child": False, "children": 0, "door_ticks": 0, "bumps": 0, "known_bumps": 0,
          "prev": None}

    def record(world, decision) -> None:
        if decision is not None and st["prev"] is not None and decision.action != WAIT:
            pos, bumped, broken = st["prev"]
            tgt = (pos[0] + MOVES[decision.action][0], pos[1] + MOVES[decision.action][1])
            if tgt in world.blueprint.doors and tgt in world.rules.jammed_doors and tgt not in broken:
                st["bumps"] += 1  # the rescuer walked into a jammed door (measured by the harness)
                st["known_bumps"] += tgt in bumped  # ... one already known to be jammed
        st["prev"] = (world.agent_pos, frozenset(world.bumped_jams), frozenset(world.broken_doors))
        st["door_ticks"] += world.agent_pos in world.blueprint.doors and decision is not None
        st["max_burns"] = max(st["max_burns"], world.agent_burns)
        st["children"] = sum(d.profile.kind == "child" for d in world.dependents)
        if any(d.carried_by == RESCUER and d.profile.kind == "child" for d in world.dependents):
            st["carried_child"] = True

    forced = {}

    def rec(world, decision):
        record(world, decision)
        forced["n"] = len(world.broken_doors)

    r = run_episode(bp, hh, factory_or_name, novelty, seed, record=rec)
    alive = r.outcome != "agent_lost"
    return {"seed": r.seed, "score": r.score, "success": float(r.outcome == "success"),
            "died": float(r.outcome == "agent_lost"), "outcome": r.outcome, "steps": r.steps, "reward": r.reward,
            "rescued": r.rescued, "total": r.total, "burned": float(st["max_burns"] > 0),
            "burned_survived": float(st["max_burns"] > 0 and alive), "max_burns": st["max_burns"],
            "carried_child": float(st["carried_child"]), "has_children": float(st["children"] > 0),
            "door_ticks": st["door_ticks"], "bumps": st["bumps"], "known_bumps": st["known_bumps"],
            "forced": forced.get("n", 0)}


_MODELS: dict = {}


def _model(spec: dict):
    if spec["ckpt"] not in _MODELS:
        from emberpath.rl_baseline import load_policy

        _MODELS[spec["ckpt"]] = load_policy(spec["ckpt"])[0]
    return _MODELS[spec["ckpt"]]


def _episode_job(args) -> tuple[str, tuple, list]:
    label, spec, setting, seeds, cache = args
    import torch

    torch.set_num_threads(1)
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS

    bp_name, prof, nov = setting
    bp, hh = load_blueprint(bp_name), HOUSEHOLDS[prof]
    novelty = None if nov == "none" else nov
    if set(seeds) & set(TRAINING_FIRE_SEEDS):
        raise ValueError("evaluation must use held-out fire seeds only")
    if spec["kind"] == "planner":
        rows = [play(bp, hh, label, novelty, s) for s in seeds]
    elif spec["kind"] == "ppo":
        from emberpath.rl_baseline import ppo_factory

        net = _model(spec)
        rows = [play(bp, hh, ppo_factory(net, greedy=True, seed=s), novelty, s) for s in seeds]
    else:
        raise ValueError(spec["kind"])
    Path(cache).write_text(json.dumps(rows))
    return label, tuple(setting), rows


def _wants(spec: dict, set_name: str) -> bool:
    return spec.get("sets") is None or set_name in spec["sets"]


def plan_jobs(proto: dict, specs: dict, sets: Sequence[str] | None = None) -> list:
    jobs = []
    for set_name in sets or proto["eval_sets"]:
        seeds = fire_seeds(proto["eval_sets"][set_name]["fire_seeds"])
        for st in set_settings(proto, set_name):
            for label, spec in specs.items():
                if not _wants(spec, set_name):
                    continue
                f = _cache_file(label, spec, st, seeds)
                if not f.exists():
                    jobs.append((label, spec, st, seeds, str(f)))
    return jobs


def run_jobs(jobs, workers: int, fn=_episode_job) -> None:
    if not jobs:
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    order = {"ppo": 0, "planner": 1}
    jobs = sorted(jobs, key=lambda j: (order.get(j[1]["kind"], 2), j[0]))
    t0 = time.time()
    if workers <= 1:
        for j in jobs:
            fn(j)
        return
    with ProcessPoolExecutor(workers) as ex:
        for i, (label, st, _) in enumerate(ex.map(fn, jobs, chunksize=1)):
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                print(f"[{i + 1}/{len(jobs)}] {label} {'/'.join(st)} ({time.time() - t0:.0f}s)", flush=True)


def load_rows(label: str, spec: dict, setting, seeds) -> list[dict]:
    return json.loads(_cache_file(label, spec, setting, seeds).read_text())


def cube(proto: dict, specs: dict, labels: Sequence[str], set_name: str, metric: str = "score") -> np.ndarray:
    """(S, U, F): per training seed, per unit (house), per fire, the mean of ``metric`` over the unit's settings."""
    seeds = fire_seeds(proto["eval_sets"][set_name]["fire_seeds"])
    settings = set_settings(proto, set_name)
    units = list(dict.fromkeys(s[0] for s in settings))
    out = np.zeros((len(labels), len(units), len(seeds)))
    for i, lab in enumerate(labels):
        for j, u in enumerate(units):
            vals = []
            for st in (s for s in settings if s[0] == u):
                rows = sorted(load_rows(lab, specs[lab], st, seeds), key=lambda r: r["seed"])
                assert [r["seed"] for r in rows] == seeds
                vals.append([r[metric] for r in rows])
            out[i, j] = np.mean(vals, axis=0)
    return out


# --------------------------------------------------------------------------- lockstep


def _lockstep_job(args) -> dict:
    label, spec, proto_lock, cache = args
    import torch

    torch.set_num_threads(1)
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.scenarios import NOVELTIES, applicable

    model = _model(spec) if spec["kind"] == "ppo" else None
    pairs = ticks = 0
    leaks = []
    modes = (True, False) if spec["kind"] == "ppo" else (True,)
    for bp_name in proto_lock["blueprints"]:
        bp = load_blueprint(bp_name)
        for prof, hh in HOUSEHOLDS.items():
            for nov in sorted(NOVELTIES):
                if not applicable(bp, hh, nov):
                    continue
                for s in fire_seeds(proto_lock["fire_seeds"]):
                    for greedy in modes:
                        t, leaked = FE.lockstep_pair(spec, model, bp, hh, nov, s, greedy)
                        pairs += 1
                        ticks += t
                        if leaked:
                            leaks.append([bp_name, prof, nov, s, greedy, t])
    res = {"label": label, "ckpt": str(Path(spec["ckpt"]).relative_to(ROOT)) if spec["ckpt"] else None,
           "pairs": pairs, "ticks": ticks, "leaks": leaks}
    Path(cache).write_text(json.dumps(res))
    return res


def _lock_cache(label: str, spec: dict) -> Path:
    stamp = FE.sha256(Path(spec["ckpt"]))[:16] if spec["ckpt"] else spec["method"]
    return CACHE / f"lockstep__{label}__{stamp}.json"


def run_lockstep(proto: dict, specs: dict, workers: int) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(lab, sp, proto["lockstep"], str(_lock_cache(lab, sp))) for lab, sp in specs.items()]
    todo = [j for j in jobs if not Path(j[3]).exists()]
    if todo and workers <= 1:
        for j in todo:
            r = _lockstep_job(j)
            print(f"lockstep {r['label']}: {r['pairs']} pairs, {r['ticks']} ticks, {len(r['leaks'])} leaks", flush=True)
    elif todo:
        with ProcessPoolExecutor(max(1, workers)) as ex:
            for r in ex.map(_lockstep_job, todo):
                print(f"lockstep {r['label']}: {r['pairs']} pairs, {r['ticks']} ticks, {len(r['leaks'])} leaks",
                      flush=True)
    return [json.loads(Path(j[3]).read_text()) for j in jobs]


# --------------------------------------------------------------------------- exploit inspection


def exploits(proto: dict, specs: dict) -> list[dict]:
    """For each final bcppo checkpoint: the top-k return episodes on the main set, replayed with the reward broken
    into components, next to the adaptive agent on the same fire. A reward exploit would show as a high return with
    a low score (or a component the honest planner never earns)."""
    from emberpath.agents import AdaptiveAgent
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.rl_baseline import PPOAgent, episode_components

    ex = proto["exploit_inspection"]
    set_name, k = ex["set"], ex["top_k"]
    seeds = fire_seeds(proto["eval_sets"][set_name]["fire_seeds"])
    out = []
    for label, spec in specs.items():
        if spec["method"] not in ex["methods"]:
            continue
        eps = [(r["reward"], st, r) for st in set_settings(proto, set_name)
               for r in load_rows(label, spec, st, seeds)]
        eps.sort(key=lambda e: -e[0])
        for ret, st, r in eps[:k]:
            bp, hh = load_blueprint(st[0]), HOUSEHOLDS[st[1]]
            nov = None if st[2] == "none" else st[2]
            mine = episode_components(PPOAgent(_model(spec), bp), bp, hh, nov, r["seed"])
            ref = episode_components(AdaptiveAgent(bp), bp, hh, nov, r["seed"])
            out.append({"label": label, "setting": list(st), "seed": r["seed"], "return": mine["return"],
                        "outcome": mine["outcome"], "score": r["score"], "steps": mine["steps"],
                        "components": mine["components"], "counts": mine["counts"],
                        "adaptive": {"return": ref["return"], "outcome": ref["outcome"], "steps": ref["steps"],
                                     "components": ref["components"]},
                        "flag": bool(r["score"] < 1.0 or mine["outcome"] != "success")})
    return out


# --------------------------------------------------------------------------- learning curve (optional)

CURVE_SEEDS = list(range(1000, 1020))


def curve_specs(proto: dict) -> dict[str, dict]:
    out = {}
    for method in proto["curve"]["methods"]:
        for s in proto["methods"][method]["seeds"]:
            mf = TRAIN / f"{method}_s{s}" / "manifest.json"
            if not mf.exists():
                continue
            for pt in json.loads(mf.read_text())["points"]:
                tag = "p000_" + pt["label"] if pt["pct"] == 0 else f"p{pt['pct']:03d}"
                out[f"curve_{method}_s{s}_{tag}"] = {"kind": "ppo", "method": method, "seed": s,
                                                     "ckpt": str(ROOT / pt["path"]), "pct": pt["pct"],
                                                     "label": pt["label"], "step": pt["step"]}
    return out


def run_curve(proto: dict, workers: int) -> dict:
    specs = curve_specs(proto)
    settings = set_settings(proto, proto["curve"]["set"])
    jobs = [(lab, sp, st, CURVE_SEEDS, str(_cache_file(lab, sp, st, CURVE_SEEDS)))
            for lab, sp in specs.items() for st in settings]
    run_jobs([j for j in jobs if not Path(j[4]).exists()], workers)
    return curve_rows(proto)


def curve_rows(proto: dict) -> dict:
    specs = curve_specs(proto)
    settings = set_settings(proto, proto["curve"]["set"])
    rows = []
    for lab, sp in specs.items():
        if not all(_cache_file(lab, sp, st, CURVE_SEEDS).exists() for st in settings):
            continue
        r = [load_rows(lab, sp, st, CURVE_SEEDS) for st in settings]
        rows.append({"label": lab, "method": sp["method"], "seed": sp["seed"], "pct": sp["pct"], "point": sp["label"],
                     "step": sp["step"], "score": float(np.mean([np.mean([x["score"] for x in rr]) for rr in r])),
                     "success": float(np.mean([np.mean([x["success"] for x in rr]) for rr in r]))})
    return {"rows": rows, "fire_seeds": [CURVE_SEEDS[0], CURVE_SEEDS[-1] + 1]}


# --------------------------------------------------------------------------- belief swap: is the world model used?

FAMILY_NOVELTY = {"jam": "door_jam", "spread": "carpet_flashover", "occupied": "hidden_dependent"}


def _greedy(net, x: np.ndarray) -> int:
    import torch

    with torch.no_grad():
        return int(net(torch.as_tensor(x[None]))[0][0].argmax())


def _family_changed(belief, family: str) -> bool:
    from emberpath.rl_baseline import nominal_part

    return nominal_part(belief, family) != belief


class PolicyProbe:
    """A greedy world-planes PPO policy (exactly PPOAgent's input and action) that, at every tick where its belief in
    ``family`` differs from the textbook one, also records the action it would take with that family's planes drawn
    at the textbook belief, everything else identical. A policy without world planes gets a shadow belief (not fed to
    it) only to time the probe: its swapped input is its real input, so it cannot change (the negative control)."""

    def __init__(self, policy, bp, family: str):
        from emberpath.rl_baseline import PPOAgent, WorldModelFeatures

        self.inner = PPOAgent(policy, bp)
        self.family, self.name, self.log = family, "probe", []
        self.shadow = None if self.inner.world_model else WorldModelFeatures(bp)

    def act(self, obs, events):
        from emberpath.planner import Decision
        from emberpath.rl_baseline import WORLD_BASE, nominal_part

        a = self.inner
        x = a.features(obs, events)
        real = _greedy(a.policy, x)
        wm = a.world_model or self.shadow
        if self.shadow is not None:
            self.shadow.observe(obs, events)
        if _family_changed(wm.belief, self.family):
            if a.world_model is not None:
                xs = x.copy()
                xs[..., WORLD_BASE:] = wm.planes(obs, a.canvas, belief=nominal_part(wm.belief, self.family))
                alt = _greedy(a.policy, xs)
            else:
                alt = _greedy(a.policy, x)
            self.log.append(_probe_row(obs, real, alt, wm.belief, a.bp, self.family))
        return Decision(real, "probe")


class PlannerProbe:
    """The positive control: AdaptiveAgent (same detector, repair and planner), asking at the same kind of states what
    its planner would do with that family's belief at the textbook value (a copy of the planner, so its own state is
    untouched)."""

    def __init__(self, bp, family: str):
        from emberpath.agents import AdaptiveAgent

        self.agent, self.family, self.name, self.log, self.bp = AdaptiveAgent(bp, explain=False), family, "probe", [], bp

    def act(self, obs, events):
        import copy

        from emberpath.rl_baseline import _adaptive_repair, nominal_part

        ag = self.agent
        if events:  # AdaptiveAgent.act up to the planner call
            ag.all_events.extend(events)
            ag.detector.observe(events)
            findings = ag.detector.findings()
            if any(f.n_trials > ag._seen.get(f.key, 0) for f in findings):
                _adaptive_repair(ag, obs, findings)
        alt = None
        if _family_changed(ag.belief, self.family):
            p = copy.deepcopy(ag.planner)
            p.belief = nominal_part(ag.belief, self.family)
            alt = p.act(obs).action
        d = ag.planner.act(obs)
        if alt is not None:
            self.log.append(_probe_row(obs, d.action, alt, ag.belief, self.bp, self.family))
        return d


def _probe_row(obs, real: int, alt: int, belief, bp, family: str) -> dict:
    """Where each action leads, relative to what the swapped family is about: jam -> distance to the nearest
    believed-jammed door; occupied -> distance to the nearest believed-occupied cell; spread -> whether the next cell
    is carpet (the material the repaired spread rate is about)."""
    from emberpath.grid import MOVES, WAIT, bfs_multi

    def nxt(a):
        if a == WAIT:
            return obs.agent_pos
        c = (obs.agent_pos[0] + MOVES[a][0], obs.agent_pos[1] + MOVES[a][1])
        return c if bp.passable(c) else obs.agent_pos

    row = {"step": obs.step, "real": real, "alt": alt, "changed": real != alt}
    if family in ("jam", "occupied"):
        targets = (belief.jammed_doors - obs.broken_doors) if family == "jam" else belief.occupied
        if targets:
            d = bfs_multi(frozenset(targets), bp.passable, bp.height, bp.width)
            here = d.get(obs.agent_pos, math.inf)
            row["real_closer"] = d.get(nxt(real), math.inf) < here
            row["alt_closer"] = d.get(nxt(alt), math.inf) < here
            if family == "jam":  # standing on the outside (kick-in) side, next to a believed-jammed door
                row["kick_side"] = any(obs.agent_pos in bp.door_outside.get(j, ()) and
                                       abs(obs.agent_pos[0] - j[0]) + abs(obs.agent_pos[1] - j[1]) == 1
                                       for j in targets)
    else:
        row["real_carpet"] = bp.material(nxt(real)) == "carpet"
        row["alt_carpet"] = bp.material(nxt(alt)) == "carpet"
    return row


def probe_episode(kind: str, model, bp, hh, family: str, seed: int) -> list[dict]:
    from emberpath.evaluate import run_episode

    probe = PlannerProbe(bp, family) if kind == "planner" else PolicyProbe(model, bp, family)
    run_episode(bp, hh, lambda b, w: probe, FAMILY_NOVELTY[family], seed)
    return probe.log


def _make_branch_agent(kind: str, model, bp):
    """A fresh agent, its belief getter, and a switch that freezes its belief in one family at the textbook value from
    then on (the counterfactual branch). A policy without world planes has nothing to switch (negative control)."""
    from emberpath.agents import AdaptiveAgent
    from emberpath.rl_baseline import PPOAgent, WorldModelFeatures, nominal_part

    if kind == "planner":
        ag = AdaptiveAgent(bp, explain=False)
        real_act = ag.planner.act

        def swap(family):
            def act(obs):
                ag.planner.belief = nominal_part(ag.belief, family)  # the repair is kept, the planner is not told
                return real_act(obs)

            ag.planner.act = act

        return ag, (lambda: ag.belief), swap
    ag = PPOAgent(model, bp)
    if ag.world_model is not None:
        def swap(family):
            ag.world_model.view = lambda b: nominal_part(b, family)

        return ag, (lambda: ag.world_model.belief), swap
    shadow = WorldModelFeatures(bp)  # timing only (never fed to the policy)
    real = ag.act

    def act(obs, events):
        shadow.observe(obs, events)
        return real(obs, events)

    ag.act = act
    return ag, (lambda: shadow.belief), (lambda family: None)


def counterfactual(kind: str, model, bp, hh, family: str, seed: int) -> dict | None:
    """From the first tick at which the agent's belief in ``family`` has been repaired, continue the episode twice on
    the same fire seed: as is ("repaired"), and with that family frozen at the textbook belief ("nominal"). Both
    branches replay the identical deterministic history up to the swap tick (asserted), so they differ only in the
    belief drawn from then on. Returns None if the belief never changed in this episode."""
    from emberpath.grid import MOVES, WAIT
    from emberpath.scenarios import NOVELTIES
    from emberpath.world import World

    def world():
        return World(bp, hh, seed=seed, novelties=NOVELTIES[FAMILY_NOVELTY[family]](bp, hh))

    w = world()
    ag, belief, _ = _make_branch_agent(kind, model, bp)
    obs, events, t_swap = w.observe(), [], None
    while not w.done:  # pass 1: the swap tick = the first act after which the belief in this family differs
        d = ag.act(obs, events)
        if _family_changed(belief(), family):
            t_swap, ref = w.t, (w.agent_pos, w.burn.copy())
            break
        r = w.step(d.action)
        obs, events = r.obs, r.events
    if t_swap is None:
        return None
    out = {"t_swap": t_swap}
    for name, fam in (("repaired", None), ("nominal", family)):
        bw = world()
        bag, _, swap = _make_branch_agent(kind, model, bp)
        o, e = bw.observe(), []
        while bw.t < t_swap:
            res = bw.step(bag.act(o, e).action)
            o, e = res.obs, res.events
        assert bw.agent_pos == ref[0] and np.array_equal(bw.burn, ref[1]), "replay diverged"
        if fam:
            swap(fam)
        bumps = known = 0
        while not bw.done:
            d = bag.act(o, e)
            if d.action != WAIT:
                tgt = (bw.agent_pos[0] + MOVES[d.action][0], bw.agent_pos[1] + MOVES[d.action][1])
                if tgt in bp.doors and tgt in bw.rules.jammed_doors and tgt not in bw.broken_doors:
                    bumps += 1
                    known += tgt in bw.bumped_jams
            res = bw.step(d.action)
            o, e = res.obs, res.events
        c = bw.counts()
        out[name] = {"score": 0.0 if not bw.agent_alive else c["rescued"] / max(1, c["total"]),
                     "success": float(bw.outcome == "success"), "died": float(bw.outcome == "agent_lost"),
                     "steps": bw.t, "bumps": bumps, "known_bumps": known, "forced": len(bw.broken_doors)}
    return out


def _swap_job(args) -> dict:
    label, spec, setting, seeds, cache = args
    import torch

    torch.set_num_threads(1)
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS

    bp_name, prof, family = setting
    bp, hh = load_blueprint(bp_name), HOUSEHOLDS[prof]
    if set(seeds) & set(TRAINING_FIRE_SEEDS):
        raise ValueError("held-out fire seeds only")
    kind = "planner" if spec["kind"] == "planner" else "ppo"
    model = None if kind == "planner" else _model(spec)
    rows = []
    for s in seeds:
        rows.append({"seed": s, "probe": probe_episode(kind, model, bp, hh, family, s),
                     "cf": counterfactual(kind, model, bp, hh, family, s)})
    Path(cache).write_text(json.dumps(rows))
    return label, tuple(setting), rows


def swap_specs(proto: dict, specs: dict) -> dict:
    sw = proto["belief_swap"]
    return {k: v for k, v in specs.items() if v["method"] in sw["agents"]}


def run_swap(proto: dict, specs: dict, workers: int) -> None:
    sw = proto["belief_swap"]
    seeds = fire_seeds(sw["fire_seeds"])
    jobs = []
    for label, spec in swap_specs(proto, specs).items():
        for st in sw["settings"]:
            f = _cache_file("swap_" + label, spec, st, seeds)
            if not f.exists():
                jobs.append((label, spec, st, seeds, str(f)))
    print(f"{len(jobs)} belief-swap jobs", flush=True)
    run_jobs(jobs, workers, fn=_swap_job)


def _cluster_ci(num: list, den: list, B: int, rs: int) -> tuple[float, float, float]:
    """Ratio sum(num)/sum(den) with a bootstrap over episodes (the unit)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if den.sum() == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(rs)
    idx = rng.integers(0, len(num), (B, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float(num.sum() / den.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def swap_stats(proto: dict, specs: dict) -> dict:
    sw = proto["belief_swap"]
    B, rs = proto["bootstrap"]["n_boot"], proto["bootstrap"]["rng_seed"]
    seeds = fire_seeds(sw["fire_seeds"])
    out: dict = {}
    by_method: dict = {}
    for label, spec in swap_specs(proto, specs).items():
        for st in sw["settings"]:
            f = _cache_file("swap_" + label, spec, st, seeds)
            if f.exists():
                by_method.setdefault((spec["method"], st[2]), []).extend(json.loads(f.read_text()))
    for (meth, fam), eps in sorted(by_method.items()):
        probes = [e["probe"] for e in eps if e["probe"]]
        row = {"episodes": len(eps), "episodes_with_repair": len(probes),
               "states": int(sum(len(p) for p in probes))}
        row["change_rate"] = _cluster_ci([sum(r["changed"] for r in p) for p in probes], [len(p) for p in probes],
                                         B, rs)
        ch = [[r for r in p if r["changed"]] for p in probes]
        if fam in ("jam", "occupied"):
            for key in ("real_closer", "alt_closer"):
                row[key] = _cluster_ci([sum(bool(r.get(key)) for r in c) for c in ch], [len(c) for c in ch], B, rs)
        if fam == "jam":
            row["kick_side_changed"] = int(sum(bool(r.get("kick_side")) for c in ch for r in c))
        if fam == "spread":
            for key in ("real_carpet", "alt_carpet"):
                row[key] = _cluster_ci([sum(bool(r.get(key)) for r in c) for c in ch], [len(c) for c in ch], B, rs)
        cfs = [e["cf"] for e in eps if e["cf"]]
        row["counterfactual_pairs"] = len(cfs)
        if cfs:
            for m in ("score", "success", "died", "steps", "bumps", "known_bumps", "forced"):
                a = np.array([c["repaired"][m] for c in cfs], float)
                b = np.array([c["nominal"][m] for c in cfs], float)
                d = FE.paired_diff_bootstrap(a[None, None, :], b[None, None, :], B, rs, resample_seeds=False)
                row[f"cf_{m}"] = {"repaired": float(a.mean()), "nominal": float(b.mean()), "diff": d["diff"],
                                  "lo": d["lo"], "hi": d["hi"], "p": d["p"]}
        out.setdefault(fam, {})[meth] = row
    tests = []
    for fam, meth, metric in sw["tests"]:
        r = out.get(fam, {}).get(meth, {}).get(f"cf_{metric}")
        if r:
            tests.append((fam, meth, metric, r))
    for (fam, meth, metric, r), p in zip(tests, FE.holm([t[3]["p"] for t in tests])):
        r["p_holm"] = p
    out["_tests"] = [{"family": f, "method": m, "metric": k, **r} for f, m, k, r in tests]
    return out


# --------------------------------------------------------------------------- recovery


def _recovery_job(args) -> dict:
    label, spec, setting, seeds, cache = args
    import torch

    torch.set_num_threads(1)
    from emberpath.agents import AdaptiveAgent
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.evaluate import run_episode
    from emberpath.rl_baseline import PPOAgent
    from emberpath.rules import parse_key
    from emberpath.scenarios import NOVELTIES

    bp_name, prof, nov = setting
    bp, hh = load_blueprint(bp_name), HOUSEHOLDS[prof]
    def onset_of(seed: int) -> int:
        try:
            from emberpath.scenarios import novelty_specs

            return min(sp.step for sp in novelty_specs(nov, bp, hh, seed))
        except ImportError:  # before seeded novelties existed
            return min(sp.step for sp in NOVELTIES[nov](bp, hh))
    fam = {"door_jam": "door_jammed", "carpet_flashover": "spread_prob", "hidden_dependent": "occupied"}[nov]
    rows = []
    for s in seeds:
        holder = {}

        def make(b, w):
            ag = AdaptiveAgent(b, explain=False) if spec["kind"] == "planner" else PPOAgent(_model(spec), b)
            holder["a"] = ag
            return ag

        r = run_episode(bp, hh, make, nov, s)
        ag = holder["a"]
        model = ag if spec["kind"] == "planner" else (ag.world_model.model if ag.world_model else None)
        first = next((t for t, p in (model.patches if model else []) if parse_key(p.key)[0] == fam), None)
        rows.append({"seed": s, "score": r.score, "success": float(r.outcome == "success"),
                     "repair_tick": first, "ticks_to_repair": None if first is None else first - onset_of(s)})
    Path(cache).write_text(json.dumps(rows))
    return label, tuple(setting), rows


def run_recovery(proto: dict, specs: dict, workers: int) -> None:
    rc = proto["recovery"]
    seeds = fire_seeds(rc["fire_seeds"])
    jobs = []
    for label, spec in specs.items():
        if spec["method"] not in rc["agents"]:
            continue
        for st in rc["settings"]:
            f = _cache_file("recovery_" + label, spec, st, seeds)
            if not f.exists():
                jobs.append((label, spec, st, seeds, str(f)))
    print(f"{len(jobs)} recovery jobs", flush=True)
    run_jobs(jobs, workers, fn=_recovery_job)


def recovery_stats(proto: dict, specs: dict) -> dict:
    rc = proto["recovery"]
    seeds = fire_seeds(rc["fire_seeds"])
    out = {}
    for label, spec in specs.items():
        if spec["method"] not in rc["agents"]:
            continue
        for st in rc["settings"]:
            f = _cache_file("recovery_" + label, spec, st, seeds)
            if not f.exists():
                continue
            rows = json.loads(f.read_text())
            o = out.setdefault(spec["method"], {}).setdefault(st[2], {"rows": []})
            o["rows"].extend(rows)
    for meth, d in out.items():
        for nov, o in d.items():
            rows = o.pop("rows")
            t = [r["ticks_to_repair"] for r in rows if r["ticks_to_repair"] is not None]
            o.update(episodes=len(rows), repaired=len(t), median_ticks_to_repair=float(np.median(t)) if t else None,
                     mean_ticks_to_repair=float(np.mean(t)) if t else None, score=float(np.mean([r["score"] for r in rows])))
    return out


# --------------------------------------------------------------------------- jammed doors and behaviour

JAM_METRICS = ("score", "success", "steps", "door_ticks", "bumps", "known_bumps", "forced")


def cube_settings(specs: dict, labels: Sequence[str], settings, seeds, metric: str) -> np.ndarray:
    """(S, 1, F): per training seed, per fire, the mean of ``metric`` over ``settings``."""
    out = np.zeros((len(labels), 1, len(seeds)))
    for i, lab in enumerate(labels):
        vals = [[r[metric] for r in sorted(load_rows(lab, specs[lab], st, seeds), key=lambda r: r["seed"])]
                for st in settings]
        out[i, 0] = np.mean(vals, axis=0)
    return out


def jam_behaviour(proto: dict, specs: dict) -> dict:
    """door_jam minus none on the same households and fires, per method (paired over fires; two-level over seeds)."""
    jb = proto["jam_behaviour"]
    set_name = jb["set"]
    seeds = fire_seeds(proto["eval_sets"][set_name]["fire_seeds"])
    sts = set_settings(proto, set_name)
    jam = [st for st in sts if st[2] == "door_jam"]
    none = [(st[0], st[1], "none") for st in jam]
    B, rs = proto["bootstrap"]["n_boot"], proto["bootstrap"]["rng_seed"]
    labels = method_labels(proto, specs, set_name)
    out = {}
    for meth in jb["methods"]:
        if meth not in labels:
            continue
        labs = labels[meth]
        if not all(_cache_file(l, specs[l], st, seeds).exists() for l in labs for st in jam + none):
            continue
        row = {}
        for m in JAM_METRICS:
            a = cube_settings(specs, labs, jam, seeds, m)
            b = cube_settings(specs, labs, none, seeds, m)
            d = FE.paired_diff_bootstrap(a, b, B, rs, resample_seeds=True, pair_seeds=True)
            row[m] = {"door_jam": float(a.mean()), "none": float(b.mean()), "diff": d["diff"], "lo": d["lo"],
                      "hi": d["hi"]}
        out[meth] = row
    return out


# --------------------------------------------------------------------------- recovery through fine-tuning


def finetune_points(proto: dict) -> list[dict]:
    ft = proto["finetune_recovery"]
    run = ROOT / ft["run"]
    pts = [{"step": 0, "path": str(ROOT / ft["init"])}]
    for p in sorted(run.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        pts.append({"step": int(p.stem.split("_")[1]), "path": str(p)})
    if (run / "final.pt").exists():
        hist = json.loads((run / "history.json").read_text())
        last = int(hist["history"][-1]["step"]) if hist["history"] else 0
        if not pts or pts[-1]["step"] != last:
            pts.append({"step": last, "path": str(run / "final.pt")})
    return pts


def run_finetune(proto: dict, workers: int) -> None:
    ft = proto["finetune_recovery"]
    seeds = fire_seeds(ft["fire_seeds"])
    jobs = []
    for pt in finetune_points(proto):
        spec = {"kind": "ppo", "method": "finetune", "seed": 0, "ckpt": pt["path"]}
        label = f"ft_{pt['step']}"
        sts = ft["settings"] + (ft["reference_settings"] if pt["step"] == 0 else [])
        for st in sts:
            f = _cache_file(label, spec, st, seeds)
            if not f.exists():
                jobs.append((label, spec, st, seeds, str(f)))
    for ref in ft.get("zero_shot", {}).values():
        spec = {"kind": "ppo", "method": "zero_shot", "seed": 0, "ckpt": str(ROOT / ref)}
        for st in ft["settings"] + ft["reference_settings"]:
            f = _cache_file("zs_" + Path(ref).parent.name, spec, st, seeds)
            if (ROOT / ref).exists() and not f.exists():
                jobs.append(("zs_" + Path(ref).parent.name, spec, st, seeds, str(f)))
    print(f"{len(jobs)} fine-tune recovery jobs", flush=True)
    run_jobs(jobs, workers)


def finetune_stats(proto: dict) -> dict:
    ft = proto["finetune_recovery"]
    seeds = fire_seeds(ft["fire_seeds"])
    B, rs = proto["bootstrap"]["n_boot"], proto["bootstrap"]["rng_seed"]
    pts = finetune_points(proto)
    if not pts:
        return {}

    def mean_cube(label, ckpt, settings):
        spec = {"kind": "ppo", "ckpt": ckpt}
        if not all(_cache_file(label, spec, st, seeds).exists() for st in settings):
            return None
        return cube_settings({label: spec}, [label], settings, seeds, "score")

    base = mean_cube("ft_0", pts[0]["path"], ft["reference_settings"])
    if base is None:
        return {}
    pre = float(base.mean())
    hist = json.loads((ROOT / ft["run"] / "history.json").read_text()) if (ROOT / ft["run"] / "history.json").exists() else None
    eps_by_step = sorted(e["step"] for e in hist["episodes"]) if hist else []
    rows, recovered = [], None
    for pt in pts:
        c = mean_cube(f"ft_{pt['step']}", pt["path"], ft["settings"])
        if c is None:
            continue
        d = FE.paired_diff_bootstrap(c, base, B, rs, resample_seeds=False)
        n_eps = int(np.searchsorted(eps_by_step, pt["step"], side="right"))
        rows.append({"step": pt["step"], "episodes": n_eps, "score": float(c.mean()), "diff": d["diff"],
                     "lo": d["lo"], "hi": d["hi"], "ratio": float(c.mean()) / pre if pre else None})
        if recovered is None and pre and c.mean() >= ft["threshold"] * pre:
            recovered = {"step": pt["step"], "episodes": n_eps}
    zs = {}
    for name, ref in ft.get("zero_shot", {}).items():
        lab = "zs_" + Path(ref).parent.name
        a = mean_cube(lab, str(ROOT / ref), ft["settings"])
        b = mean_cube(lab, str(ROOT / ref), ft["reference_settings"])
        if a is not None and b is not None:
            zs[name] = {"door_jam": float(a.mean()), "none": float(b.mean()), "ratio": float(a.mean() / b.mean())
                        if b.mean() else None}
    return {"pre_novelty_score_none": pre, "threshold": ft["threshold"], "rows": rows, "recovered_at": recovered,
            "zero_shot": zs}


# --------------------------------------------------------------------------- statistics


def method_labels(proto: dict, specs: dict, set_name: str) -> dict[str, list[str]]:
    out = {p: [p] for p in proto["planners"]}
    for method, m in proto["methods"].items():
        labs = [f"{method}_s{s}" for s in m["seeds"] if f"{method}_s{s}" in specs
                and _wants(specs[f"{method}_s{s}"], set_name)]
        if labs:
            out[method] = labs
    for label in proto.get("fixed_checkpoints", {}):
        if label in specs and _wants(specs[label], set_name):
            out[label] = [label]
    return out


BEHAVIOUR = ("burned", "burned_survived", "carried_child")


def behaviour(proto: dict, specs: dict, labels: Sequence[str], set_name: str) -> dict:
    """Resistance use: share of episodes with >= 1 tick in flames, of those survived, and (episodes with children
    only) of those where the rescuer carried a child; pooled over the set's settings and the method's seeds."""
    seeds = fire_seeds(proto["eval_sets"][set_name]["fire_seeds"])
    rows = [r for lab in labels for st in set_settings(proto, set_name) for r in load_rows(lab, specs[lab], st, seeds)]
    kids = [r for r in rows if r["has_children"]]
    burned = [r for r in rows if r["burned"]]
    return {"episodes": len(rows), "burned": float(np.mean([r["burned"] for r in rows])),
            "burned_survived": float(np.mean([r["burned_survived"] for r in rows])),
            "survived_given_burned": float(np.mean([r["burned_survived"] for r in burned])) if burned else None,
            "mean_burn_ticks_given_burned": float(np.mean([r["max_burns"] for r in burned])) if burned else None,
            "carried_child_given_children": float(np.mean([r["carried_child"] for r in kids])) if kids else None,
            "child_episodes": len(kids)}


def compute_stats(proto: dict, specs: dict) -> dict:
    bs = proto["bootstrap"]
    B, rs = bs["n_boot"], bs["rng_seed"]
    res: dict = {"protocol_created_utc": proto["created_utc"], "methods": {}, "comparisons": [], "missing": [],
                 "behaviour": {}}
    for set_name in proto["eval_sets"]:
        labels = method_labels(proto, specs, set_name)
        res["methods"][set_name] = {}
        res["behaviour"][set_name] = {}
        for meth, labs in labels.items():
            if not all(_cache_file(l, specs[l], st, fire_seeds(proto["eval_sets"][set_name]["fire_seeds"])).exists()
                       for l in labs for st in set_settings(proto, set_name)):
                res["missing"].append([meth, None, set_name])
                continue
            row = {"labels": labs, "score": FE.seed_split(cube(proto, specs, labs, set_name), B, rs)}
            for metric in ("success", "died", "steps", "reward"):
                x = cube(proto, specs, labs, set_name, metric)
                row[metric] = {"mean": float(x.mean()), "per_seed": x.mean(axis=(1, 2)).tolist()}
            res["methods"][set_name][meth] = row
            res["behaviour"][set_name][meth] = behaviour(proto, specs, labs, set_name)
    fam = []
    for meth, ref, set_name in proto["comparisons"]:
        labels = method_labels(proto, specs, set_name)
        if meth not in res["methods"][set_name] or ref not in res["methods"][set_name]:
            res["missing"].append([meth, ref, set_name])
            continue
        a, b = cube(proto, specs, labels[meth], set_name), cube(proto, specs, labels[ref], set_name)
        pair = [meth, ref] in proto["paired_seed_comparisons"] and a.shape[0] == b.shape[0]
        prim = FE.paired_diff_bootstrap(a, b, B, rs, resample_seeds=True, pair_seeds=pair)
        sec = FE.paired_diff_bootstrap(a.mean(0, keepdims=True), b.mean(0, keepdims=True), B, rs, resample_seeds=False)
        succ = FE.paired_diff_bootstrap(cube(proto, specs, labels[meth], set_name, "success"),
                                        cube(proto, specs, labels[ref], set_name, "success"), B, rs,
                                        resample_seeds=True, pair_seeds=pair)
        per_seed = (a.mean(axis=(1, 2)) - b.mean(axis=(1, 2))).tolist() if pair else \
            (a.mean(axis=(1, 2))[:, None] - b.mean(axis=(1, 2))[None, :]).mean(1).tolist()
        fam.append({"method": meth, "ref": ref, "set": set_name, "n_seeds": [a.shape[0], b.shape[0]],
                    "means": [float(a.mean()), float(b.mean())], "primary": prim, "fire_only": sec,
                    "success_diff": succ, "per_seed_diff": per_seed, "paired_seeds": pair})
    adj = FE.holm([c["primary"]["p"] for c in fam])
    alpha = proto["multiple_testing"]["alpha"]
    for c, p in zip(fam, adj):
        c["p_holm"] = p
        c["verdict"] = ("no significant difference" if p >= alpha else
                        f"{c['method']} > {c['ref']}" if c["primary"]["diff"] > 0 else f"{c['method']} < {c['ref']}")
    res["comparisons"] = fam
    res["family_size"] = len(fam)
    # secondary (not in the family): same-seed ablation contrasts, fire-only bootstrap on seed 0
    res["secondary"] = []
    for a_lab, b_lab, set_name in proto.get("secondary_comparisons", []):
        if a_lab in specs and b_lab in specs and all(
                _cache_file(l, specs[l], st, fire_seeds(proto["eval_sets"][set_name]["fire_seeds"])).exists()
                for l in (a_lab, b_lab) for st in set_settings(proto, set_name)):
            a, b = cube(proto, specs, [a_lab], set_name), cube(proto, specs, [b_lab], set_name)
            r = FE.paired_diff_bootstrap(a, b, B, rs, resample_seeds=False)
            res["secondary"].append({"a": a_lab, "b": b_lab, "set": set_name, "means": [float(a.mean()), float(b.mean())],
                                     **r})
    return res


def _f(x, nd=3, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def _ci(r, nd=3, pct=False):
    if r is None or (isinstance(r, tuple) and not math.isfinite(r[0])):
        return "n/a"
    if isinstance(r, (tuple, list)):
        v, lo, hi = r
        return f"{v:.1%} [{lo:.1%}, {hi:.1%}]" if pct else f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
    return f"{r['diff']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]"


def render_world_model(res: dict, proto: dict) -> list[str]:
    L = []
    sw = res.get("belief_swap") or {}
    fams = [f for f in sw if not f.startswith("_")]
    if fams:
        bs = proto["belief_swap"]
        L += ["## Does the policy use the repaired world model? (belief-swap test)", "",
              f"Held-out house fires {bs['fire_seeds'][0]}..{bs['fire_seeds'][1] - 1}, every household with the "
              "family's novelty. At every state after the agent's own repair (its belief in that family differs from "
              "the textbook), the policy is run twice: with its real belief planes and with that family's planes drawn "
              "at the textbook belief, everything else identical. Counterfactual: from the first such tick the same "
              "fire is played out twice, with the repaired belief and with that family frozen at the textbook belief. "
              "Positive control: the adaptive planner (must change); negative control: the policy without world "
              "planes (cannot change). CIs: bootstrap over episodes.", ""]
        L += ["| family | agent | episodes with a repair | states | action-change rate [95% CI] | changed: real "
              "action closer to target | changed: swapped action closer | changed: real onto carpet | changed: "
              "swapped onto carpet |", "|---|---|---:|---:|---|---|---|---|---|"]
        for fam in fams:
            for meth, r in sw[fam].items():
                L.append(f"| {fam} | {meth} | {r['episodes_with_repair']}/{r['episodes']} | {r['states']} | "
                         f"{_ci(r['change_rate'], pct=True)} | {_ci(r.get('real_closer'), pct=True)} | "
                         f"{_ci(r.get('alt_closer'), pct=True)} | {_ci(r.get('real_carpet'), pct=True)} | "
                         f"{_ci(r.get('alt_carpet'), pct=True)} |")
        L += ["", "Targets: jam = the nearest believed-jammed door; occupied = the nearest believed-occupied cell.", "",
              "| family | agent | pairs | score repaired / nominal | score diff [CI] | success diff | steps diff | "
              "bumps diff | known-jam bumps diff | forced diff |", "|---|---|---:|---|---|---|---|---|---|---|"]
        for fam in fams:
            for meth, r in sw[fam].items():
                if not r.get("counterfactual_pairs"):
                    continue
                L.append(f"| {fam} | {meth} | {r['counterfactual_pairs']} | {r['cf_score']['repaired']:.3f} / "
                         f"{r['cf_score']['nominal']:.3f} | {_ci(r['cf_score'])} | {_ci(r['cf_success'], 2)} | "
                         f"{_ci(r['cf_steps'], 1)} | {_ci(r['cf_bumps'], 2)} | {_ci(r['cf_known_bumps'], 2)} | "
                         f"{_ci(r['cf_forced'], 2)} |")
        if sw.get("_tests"):
            L += ["", "Pre-registered counterfactual tests (repaired - nominal; Holm over these):", "",
                  "| family | agent | metric | difference [95% CI] | raw p | Holm p |", "|---|---|---|---|---:|---:|"]
            for t in sw["_tests"]:
                L.append(f"| {t['family']} | {t['method']} | {t['metric']} | {_ci(t)} | {t['p']:.4f} | "
                         f"{t['p_holm']:.4f} |")
        L.append("")
    rc = res.get("recovery") or {}
    if rc:
        L += ["## Recovery within an episode (no fine-tuning)", "",
              "Ticks from the novelty's onset to the agent's first repair of that rule family (its own belief), and "
              "the episode score.", "", "| agent | novelty | episodes | repaired | median ticks to repair | mean | "
              "score |", "|---|---|---:|---:|---:|---:|---:|"]
        for meth, d in rc.items():
            for nov, o in d.items():
                L.append(f"| {meth} | {nov} | {o['episodes']} | {o['repaired']} | "
                         f"{_f(o['median_ticks_to_repair'], 1, False)} | {_f(o['mean_ticks_to_repair'], 1, False)} | "
                         f"{o['score']:.3f} |")
        L.append("")
    ft = res.get("finetune_recovery") or {}
    if ft.get("rows"):
        fp = proto["finetune_recovery"]
        L += ["## Recovery through RL fine-tuning (policy without belief planes)", "",
              f"{fp['description']} Pre-novelty score (none, same households and fires): "
              f"{ft['pre_novelty_score_none']:.3f}; recovery = first checkpoint whose door_jam score reaches "
              f"{ft['threshold']:.0%} of it.", "",
              "| env steps of fine-tuning | training episodes | door_jam score | minus pre-novelty [fire CI] | "
              "ratio |", "|---:|---:|---:|---|---:|"]
        for r in ft["rows"]:
            L.append(f"| {r['step']} | {r['episodes']} | {r['score']:.3f} | {_ci(r)} | "
                     f"{_f(r['ratio'], 2, False)} |")
        rec = ft.get("recovered_at")
        L += ["", f"Recovered at: {rec['step']} env steps ({rec['episodes']} training episodes)." if rec else
              "Recovered at: not within the fine-tuning budget."]
        for name, z in ft.get("zero_shot", {}).items():
            L.append(f"Zero-shot (no fine-tuning) {name}: door_jam {z['door_jam']:.3f}, none {z['none']:.3f}, "
                     f"ratio {_f(z['ratio'], 2, False)}.")
        L.append("")
    jb = res.get("jam_behaviour") or {}
    if jb:
        L += ["## Jammed doors and behaviour (house, door_jam minus none, same households and fires)", "",
              "| agent | " + " | ".join(JAM_METRICS) + " |", "|---|" + "---|" * len(JAM_METRICS)]
        for meth, r in jb.items():
            L.append(f"| {meth} | " + " | ".join(f"{r[m]['none']:.2f} -> {r[m]['door_jam']:.2f} ({_ci(r[m], 2)})"
                                                  for m in JAM_METRICS) + " |")
        L += ["", "door_ticks: ticks the rescuer stood in a doorway; bumps: walks into a jammed, unforced door; "
                  "known: into one already known jammed; forced: doors kicked in.", ""]
    return L


def render_md(res: dict, proto: dict) -> str:
    L = ["# House evaluation: BC+PPO with the world-model planes", "",
         f"Pre-registered {proto['created_utc']} (runs/house_rl/PREREGISTRATION.md). Primary metric: pooled score "
         "(fraction rescued, 0 if the rescuer died). Greedy policies. Intervals: 95% percentile, "
         f"{proto['bootstrap']['n_boot']} paired resamples; primary = two-level bootstrap (training seeds, then houses, "
         f"then fires); Holm over the {res['family_size']} pre-registered comparisons at alpha "
         f"{proto['multiple_testing']['alpha']}.", ""]
    for set_name, es in proto["eval_sets"].items():
        L += [f"## {es['title']}", "",
              "| method | pooled score | per training seed | between-seed SD | fire SE | two-level SE | success | "
              "died | steps |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
        for meth, row in res["methods"][set_name].items():
            sc = row["score"]
            ps = ", ".join(f"{v:.3f}" for v in sc["per_seed"]) if len(sc["per_seed"]) > 1 else ""
            L.append(f"| {meth} | {sc['mean']:.3f} | {ps} | {_f(sc['between_seed_sd'], 3, False)} | "
                     f"{sc['fire_se']:.3f} | {sc['hier_se']:.3f} | {row['success']['mean']:.1%} | "
                     f"{row['died']['mean']:.1%} | {row['steps']['mean']:.0f} |")
        comps = [c for c in res["comparisons"] if c["set"] == set_name]
        if comps:
            L += ["", "| comparison | difference | 95% CI (two-level) | raw p | Holm p | verdict | fire-only CI | "
                      "success diff [CI] | per-seed diffs |", "|---|---:|---|---:|---:|---|---|---|---|"]
            for c in comps:
                p, s, sd = c["primary"], c["fire_only"], c["success_diff"]
                L.append(f"| {c['method']} - {c['ref']} | {_f(p['diff'])} | [{_f(p['lo'])}, {_f(p['hi'])}] | "
                         f"{p['p']:.4f} | {c['p_holm']:.4f} | {c['verdict']} | [{_f(s['lo'])}, {_f(s['hi'])}] | "
                         f"{_f(sd['diff'], 2)} [{_f(sd['lo'], 2)}, {_f(sd['hi'], 2)}] | "
                         f"{', '.join(_f(x, 3) for x in c['per_seed_diff'])} |")
        beh = res["behaviour"].get(set_name, {})
        if beh:
            L += ["", "Resistance use (harness measurements): share of episodes with >= 1 tick in flames, of those "
                      "survived, the share of burned episodes survived, and (episodes with children) the share where "
                      "the rescuer carried a child.", "",
                  "| method | episodes | >= 1 burn tick | burned and survived | survived given burned | "
                  "burn ticks given burned | carried a child (child episodes) |", "|---|---:|---:|---:|---:|---:|---:|"]
            for meth, b in beh.items():
                L.append(f"| {meth} | {b['episodes']} | {_pct(b['burned'])} | {_pct(b['burned_survived'])} | "
                         f"{_pct(b['survived_given_burned'])} | {_f(b['mean_burn_ticks_given_burned'], 2, False)} | "
                         f"{_pct(b['carried_child_given_children'])} ({b['child_episodes']}) |")
        L.append("")
    L += render_world_model(res, proto)
    if res.get("secondary"):
        L += ["## Secondary (not in the family): same-training-seed ablation contrasts, fire-only paired bootstrap", "",
              "| A - B | set | means | difference [95% CI] | p |", "|---|---|---|---|---:|"]
        for r in res["secondary"]:
            L.append(f"| {r['a']} - {r['b']} | {r['set']} | {r['means'][0]:.3f} vs {r['means'][1]:.3f} | "
                     f"{_f(r['diff'])} [{_f(r['lo'])}, {_f(r['hi'])}] | {r['p']:.4f} |")
        L.append("")
    if res.get("missing"):
        L += ["**Not run (method not trained or not evaluated):** "
              + "; ".join(f"{m}{' - ' + r if r else ''} ({s})" for m, r, s in res["missing"]), ""]
    lock = res.get("lockstep", [])
    if lock:
        lk = proto["lockstep"]
        L += ["## Lockstep evidence test (every evaluated checkpoint)", "",
              f"{', '.join(lk['blueprints'])}; every household x applicable novelty; fire seeds "
              f"{lk['fire_seeds'][0]}..{lk['fire_seeds'][1] - 1}; greedy and sampled.", "",
              "| agent | checkpoint | pairs | ticks compared | differing decisions |", "|---|---|---:|---:|---:|"]
        for r in lock:
            L.append(f"| {r['label']} | {r['ckpt'] or '-'} | {r['pairs']} | {r['ticks']} | {len(r['leaks'])} |")
        L.append("")
    ex = res.get("exploits", [])
    if ex:
        L += ["## Top-return episodes (reward exploit inspection)", "",
              "| agent | setting | seed | return | outcome | score | steps | adaptive return / outcome | "
              "components (agent) | flagged |", "|---|---|---:|---:|---|---:|---:|---|---|---|"]
        for e in ex:
            comp = ", ".join(f"{k} {v:+.1f}" for k, v in e["components"].items() if abs(v) > 0.05)
            L.append(f"| {e['label']} | {'/'.join(e['setting'])} | {e['seed']} | {e['return']:.2f} | {e['outcome']} | "
                     f"{e['score']:.2f} | {e['steps']} | {e['adaptive']['return']:.2f} / {e['adaptive']['outcome']} | "
                     f"{comp} | {'yes' if e['flag'] else 'no'} |")
        L.append("")
    cur = res.get("curve")
    if cur and cur["rows"]:
        L += [f"## Learning curves (house set, fire seeds {cur['fire_seeds'][0]}..{cur['fire_seeds'][1] - 1})", "",
              "| method | seed | point | env step | score | success |", "|---|---:|---|---:|---:|---:|"]
        for r in sorted(cur["rows"], key=lambda r: (r["method"], r["seed"], r["pct"], r["point"] != "untrained")):
            L.append(f"| {r['method']} | {r['seed']} | {r['point']} | {r['step']} | {r['score']:.3f} | "
                     f"{r['success']:.1%} |")
        L.append("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="write the checkpoint manifest of one house training run")
    m.add_argument("--method", default="bcppo")
    m.add_argument("--seed", type=int, required=True)
    m.add_argument("--run", default=None, help="run directory (default runs/house_rl/train/bcppo_s<seed>)")
    for name in ("eval", "lockstep", "exploits", "curve", "swap", "recovery", "finetune", "stats", "all"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--sets", nargs="*", default=None)
        p.add_argument("--labels", nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        run = Path(a.run) if a.run else TRAIN / f"bcppo_s{a.seed}"
        man = FE.build_manifest("bcppo", a.seed, run=run)  # bcppo layout: bc/bc.pt, ppo/, ckpt_p000_untrained.pt
        if a.method != "bcppo":
            man["method"], man["variant"] = "bcppo", a.method
            (run / "manifest.json").write_text(json.dumps(man, indent=1))
        errs = FE.validate_manifest(man)
        print(json.dumps({k: man[k] for k in ("method", "seed", "total_steps", "final_step")}),
              f"{len(man['points'])} points", "OK" if not errs else errs)
        return 1 if errs else 0
    proto = load_protocol(PROTOCOL)
    specs = agent_specs(proto)
    if a.labels:
        specs = {k: v for k, v in specs.items() if k in a.labels}
    if a.cmd in ("eval", "all"):
        jobs = plan_jobs(proto, specs, a.sets)
        print(f"{len(jobs)} episode jobs to run ({len(specs)} agents)", flush=True)
        run_jobs(jobs, a.workers)
    if a.cmd == "curve":
        run_curve(proto, a.workers)
        return 0
    if a.cmd in ("swap", "all"):
        run_swap(proto, specs, a.workers)
    if a.cmd in ("recovery", "all"):
        run_recovery(proto, specs, a.workers)
    if a.cmd in ("finetune", "all"):
        run_finetune(proto, a.workers)
    if a.cmd in ("lockstep", "all"):
        run_lockstep(proto, specs, a.workers)
    if a.cmd in ("exploits", "all"):
        (OUT / "exploits.json").write_text(json.dumps(exploits(proto, specs), indent=1))
    if a.cmd in ("stats", "all"):
        specs = agent_specs(proto)
        res = compute_stats(proto, specs)
        res["lockstep"] = [json.loads(_lock_cache(k, v).read_text()) for k, v in specs.items()
                           if _lock_cache(k, v).exists()]
        if (OUT / "exploits.json").exists():
            res["exploits"] = json.loads((OUT / "exploits.json").read_text())
        res["curve"] = curve_rows(proto)
        res["belief_swap"] = swap_stats(proto, specs)
        res["recovery"] = recovery_stats(proto, specs)
        res["finetune_recovery"] = finetune_stats(proto)
        res["jam_behaviour"] = jam_behaviour(proto, specs)
        (OUT / "results.json").write_text(json.dumps(res, indent=1))
        (OUT / "results.md").write_text(render_md(res, proto))
        print(render_md(res, proto))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
