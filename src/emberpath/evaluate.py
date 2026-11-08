"""Phase 7: the evaluation harness. Same seeds, same fires, different agents.

Because fire uses its own random stream, every agent faces *identical* fire for a given seed.
Differences in score are therefore caused by the agent, not by luck.

Score per episode = fraction of dependents rescued, or 0 if the rescuer died.

Intervals are paired bootstraps over seeds: every resample draws the same seeds for every agent,
so the shared fire (the main source of variance) cancels in differences such as ``gap_closed``.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from emberpath.agents import AdaptiveAgent, Agent, make_agent
from emberpath.blueprint import Blueprint
from emberpath.dependents import Household
from emberpath.reward import RewardTracker
from emberpath.rules import Rules, nominal_rules
from emberpath.scenarios import NOVELTIES
from emberpath.world import World


@dataclass(frozen=True)
class EpisodeResult:
    agent: str
    novelty: str
    seed: int
    outcome: str
    rescued: int
    total: int
    steps: int
    score: float
    reward: float = 0.0  # episode return under the default RewardConfig (see reward.py)
    efficiency: float | None = None  # fewest-possible steps / steps taken; successes only
    patches: tuple[tuple[int, str, float], ...] = ()  # (step, key, value) belief edits, adaptive agents only
    repair_calls: int = 0  # times the repair search ran (adaptive agents only)


def run_episode(
    blueprint: Blueprint,
    household: Household,
    agent_name: str | Callable[[Blueprint, World], Agent],
    novelty: str | None,
    seed: int,
    max_steps: int = 150,
    record=None,
) -> EpisodeResult:
    """Run one episode. `agent_name` is a registered name or an agent factory (e.g. a study's variant).

    `record(world, decision)` is called every tick if given (used for GIFs)."""
    specs = NOVELTIES[novelty](blueprint, household) if novelty else []
    world = World(blueprint, household, seed=seed, novelties=specs, max_steps=max_steps)
    if isinstance(agent_name, str):
        agent = make_agent(agent_name, blueprint, world)
    else:
        agent = agent_name(blueprint, world)
    tracker = RewardTracker(world)

    obs, events = world.observe(), []
    if record:
        record(world, None)
    while not world.done:
        decision = agent.act(obs, events)
        result = world.step(decision.action)
        tracker.update(world)
        obs, events = result.obs, result.events
        if record:
            record(world, decision)

    c = world.counts()
    score = 0.0 if not world.agent_alive else c["rescued"] / max(1, c["total"])
    return EpisodeResult(
        agent.name,
        novelty or "none",
        seed,
        world.outcome or "?",
        c["rescued"],
        c["total"],
        world.t,
        score,
        reward=tracker.total,
        efficiency=tracker.efficiency(world),
        patches=tuple((t, p.key, float(p.value)) for t, p in getattr(agent, "patches", ())),
        repair_calls=getattr(agent, "repairs", 0),
    )


def run_lifelong(
    blueprint: Blueprint,
    household: Household,
    novelty: str,
    episodes: int,
    seed: int = 0,
    vary_seeds: bool = False,
    max_steps: int = 150,
    record_for: Callable[[int], Callable | None] | None = None,
) -> list[tuple[EpisodeResult, Rules]]:
    """Run the adaptive agent for several episodes in the same changed house, carrying its beliefs forward.

    Episode 1 starts from the textbook rules and has to discover the change; each later episode starts from
    what the previous one learned (`AdaptiveAgent.lessons`). With the default `vary_seeds=False` every episode
    faces the same fire, so any difference between the first and last episode is down to what the agent knew.
    `record_for(i)` may return a per-tick recorder for episode i (used for GIFs). Returns, per episode, its
    result and the beliefs it started with."""
    belief, out = None, []
    for i in range(episodes):
        agents: list[AdaptiveAgent] = []

        def factory(bp: Blueprint, _world: World) -> AdaptiveAgent:  # the world is never handed over
            agents.append(AdaptiveAgent(bp, belief=belief))
            return agents[-1]

        rec = record_for(i) if record_for else None
        res = run_episode(blueprint, household, factory, novelty, seed + i * vary_seeds, max_steps, record=rec)
        out.append((res, belief or nominal_rules()))
        belief = agents[0].lessons()
    return out


def evaluate(
    blueprint: Blueprint,
    household: Household,
    agents: list[str],
    novelty: str | None,
    seeds: range | list[int],
    max_steps: int = 150,
) -> list[EpisodeResult]:
    return [
        run_episode(blueprint, household, a, novelty, s, max_steps)
        for a in agents
        for s in seeds
    ]


def summarize(results: list[EpisodeResult]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for agent in dict.fromkeys(r.agent for r in results):
        rows = [r for r in results if r.agent == agent]
        n = len(rows)
        effs = [r.efficiency for r in rows if r.efficiency is not None]
        out[agent] = {
            "episodes": n,
            "success_rate": sum(r.outcome == "success" for r in rows) / n,
            "mean_score": sum(r.score for r in rows) / n,
            "mean_return": sum(r.reward for r in rows) / n,
            "mean_efficiency": sum(effs) / len(effs) if effs else math.nan,  # over successful episodes only
            "mean_steps": sum(r.steps for r in rows) / n,
            "agent_lost_rate": sum(r.outcome == "agent_lost" for r in rows) / n,
            "timeout_rate": sum(r.outcome == "timeout" for r in rows) / n,
        }
    return out


def gap_closed(summary: dict[str, dict[str, float]], agent: str = "adaptive") -> float | None:
    """(agent - baseline) / (oracle - baseline) on mean score. None if undefined."""
    if not {"baseline", "oracle", agent} <= set(summary):
        return None
    base, orac, mine = (summary[k]["mean_score"] for k in ("baseline", "oracle", agent))
    return None if orac - base < 1e-9 else (mine - base) / (orac - base)


def paired_bootstrap(
    results: list[EpisodeResult],
    stat: Callable[[dict[str, np.ndarray]], np.ndarray],
    value: Callable[[EpisodeResult], float] = lambda r: r.score,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point, lo, hi): ``stat`` of per-agent means with a 95% paired-bootstrap interval over seeds.

    ``stat`` maps {agent: array of means} to an array (vectorised over resamples); it may return nan
    where it is undefined (e.g. a gap ratio with a zero denominator); those resamples are dropped."""
    by_agent: dict[str, dict[int, float]] = {}
    for r in results:
        by_agent.setdefault(r.agent, {})[r.seed] = value(r)
    seeds = sorted(next(iter(by_agent.values())))
    if any(sorted(v) != seeds for v in by_agent.values()):
        raise ValueError("paired bootstrap needs the same seeds for every agent")
    arrays = {a: np.array([v[s] for s in seeds]) for a, v in by_agent.items()}
    idx = np.random.default_rng(seed).integers(0, len(seeds), (n_boot, len(seeds)))
    point = float(stat({a: np.array([x.mean()]) for a, x in arrays.items()})[0])
    boot = stat({a: x[idx].mean(axis=1) for a, x in arrays.items()})
    boot = boot[np.isfinite(boot)]
    if not len(boot):
        return point, math.nan, math.nan
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def gap_ratio(agent: str = "adaptive") -> Callable[[dict[str, np.ndarray]], np.ndarray]:
    """``gap_closed`` as a vectorised bootstrap statistic (nan where the oracle gap is ~0)."""

    def stat(m: dict[str, np.ndarray]) -> np.ndarray:
        gap = m["oracle"] - m["baseline"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(gap > 1e-9, (m[agent] - m["baseline"]) / gap, np.nan)

    return stat


def write_csv(results: list[EpisodeResult], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0])))
        w.writeheader()
        w.writerows(asdict(r) for r in results)
