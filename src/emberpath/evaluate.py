"""Phase 7: the evaluation harness. Same seeds, same fires, different agents.

Because fire uses its own random stream, every agent faces *identical* fire for a given seed.
Differences in score are therefore caused by the agent, not by luck.

Score per episode = fraction of dependents rescued, or 0 if the rescuer died.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from emberpath.agents import make_agent
from emberpath.blueprint import Blueprint
from emberpath.dependents import Household
from emberpath.reward import RewardTracker
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


def run_episode(
    blueprint: Blueprint,
    household: Household,
    agent_name: str,
    novelty: str | None,
    seed: int,
    max_steps: int = 150,
    record=None,
) -> EpisodeResult:
    """Run one episode. `record(world, decision)` is called every tick if given (used for GIFs)."""
    specs = NOVELTIES[novelty](blueprint, household) if novelty else []
    world = World(blueprint, household, seed=seed, novelties=specs, max_steps=max_steps)
    agent = make_agent(agent_name, blueprint, world)
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
        agent_name,
        novelty or "none",
        seed,
        world.outcome or "?",
        c["rescued"],
        c["total"],
        world.t,
        score,
        reward=tracker.total,
        efficiency=tracker.efficiency(world),
    )


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


def write_csv(results: list[EpisodeResult], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0])))
        w.writeheader()
        w.writerows(asdict(r) for r in results)
