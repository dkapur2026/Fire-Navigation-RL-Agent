"""Gymnasium wrapper, for the RL baseline (Phase 4). Requires `pip install -e .[rl]`.

Observation: float32 array (H, W, C) with channels
    0 wall  1 fire  2 burnt  3 smoke  4 agent  5 waiting dependent  6 following dependent
    7 exit  8 door  9 jammed door (only once the rescuer has bumped it: no hidden truth leaks)
    10 time remaining: (max_steps - t) / max_steps on every cell (the rescuer knows the clock; without it a
       deterministic policy cannot tell tick 3 from tick 90 in the same layout and loops, see docs/RL.md)
Actions: Discrete(5) = up, down, left, right, wait.
Reward: see ``emberpath.reward`` and docs/REWARD.md. In short: a cost for every tick, dense progress toward the
        exit, bonuses for rescues, penalties for dependents near or touching flames. ``info["reward_components"]``
        holds the per-component breakdown for every step.
Episode end: every ending (success, partial, agent_lost, timeout) is ``terminated``; ``truncated`` is always False,
        because the 150-tick horizon is part of the reward (see ``step``). ``info["outcome"]`` says which ending it was.
Info: nothing the rescuer could not perceive. Mid-episode ``info["total"]`` counts only dependents that are not hiding;
        the true counts, ``efficiency`` and ``episode_return`` are added once the episode is over.
"""

from __future__ import annotations

import numpy as np

from emberpath.blueprint import Blueprint
from emberpath.dependents import Household
from emberpath.reward import RewardConfig, RewardTracker
from emberpath.scenarios import NOVELTIES
from emberpath.world import FOLLOWING, WAITING, World

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None

N_CHANNELS = 11
TIME_CHANNEL = 10


def encode(world: World) -> np.ndarray:
    bp = world.blueprint
    x = np.zeros((bp.height, bp.width, N_CHANNELS), dtype=np.float32)
    for cell in bp.walls:
        x[cell + (0,)] = 1
    x[..., 1] = world.burn > 0
    x[..., 2] = world.burn < 0
    x[..., 3] = np.clip(world.smoke, 0, 1)
    x[world.agent_pos + (4,)] = 1
    for d in world.dependents:
        if d.status == WAITING and not d.hiding:  # a hiding dependent is unseen until found
            x[d.pos + (5,)] = 1
        elif d.status == FOLLOWING:
            x[d.pos + (6,)] = 1
    for cell in bp.exits:
        x[cell + (7,)] = 1
    for cell in bp.doors:
        x[cell + (8,)] = 1
    for cell in world.bumped_jams - world.broken_doors:
        x[cell + (9,)] = 1
    x[..., TIME_CHANNEL] = max(0, world.max_steps - world.t) / max(1, world.max_steps)
    return x


if gym is not None:

    class EmberPathEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            blueprint: Blueprint,
            household: Household,
            novelty: str | None = None,
            max_steps: int = 150,
            reward_config: RewardConfig | None = None,
        ):
            self.bp, self.household, self.novelty, self.max_steps = blueprint, household, novelty, max_steps
            self.reward_config = reward_config or RewardConfig()
            self.action_space = spaces.Discrete(5)
            self.observation_space = spaces.Box(0.0, 1.0, (blueprint.height, blueprint.width, N_CHANNELS), np.float32)
            self.world: World | None = None
            self.tracker: RewardTracker | None = None

        def reset(self, *, seed: int | None = None, options=None):
            super().reset(seed=seed)
            specs = NOVELTIES[self.novelty](self.bp, self.household) if self.novelty else []
            s = int(self.np_random.integers(0, 2**31 - 1)) if seed is None else seed
            self.world = World(self.bp, self.household, seed=s, novelties=specs, max_steps=self.max_steps)
            self.tracker = RewardTracker(self.world, self.reward_config)
            return encode(self.world), {}

        def step(self, action: int):
            w = self.world
            res = w.step(int(action))
            breakdown = self.tracker.update(w)
            # The horizon is part of the task, not an outside time limit: the reward charges contact for everyone
            # left inside at a timeout (and time to the horizon on a loss), and the world cannot step past it. So a
            # timeout TERMINATES; bootstrapping from it (as `truncated` would invite) would charge those people twice.
            terminated = w.done
            truncated = False
            info = {
                "events": res.events,
                "reward_components": breakdown.components,
                **res.info,
            }
            if not w.done:  # mid-episode, count only the people the rescuer could know about (as World.observe does)
                info["total"] = sum(not d.hiding for d in w.dependents)
            else:
                info["efficiency"] = self.tracker.efficiency(w)
                info["episode_return"] = self.tracker.total
            return encode(w), breakdown.total, terminated, truncated, info

else:

    class EmberPathEnv:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise ImportError("gymnasium is not installed: pip install -e .[rl]")
