"""Reward-hacking audit: score deliberately degenerate policies under the LEGACY and the NEW reward.

    python scripts/audit_reward.py            # 20 seeds, ranch / single / door_jam
    python scripts/audit_reward.py --seeds 8  # quicker

A good reward ranks honest effort above every shortcut. A reward with a loophole lets a shortcut outscore
an honest-but-struggling agent, and a learning agent will find it. Rerun this after ANY change to reward.py
and add a policy here for every new exploit you think of.

Policies
  oracle     honest, knows the true rules (usually succeeds)
  baseline   honest, but its rulebook is stale (stuck at the jammed door, times out)
  stall      does nothing until the clock runs out
  suicide    walks into the nearest flame
  abandon    runs to the exit alone, never rescuing anyone
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from emberpath.agents import BaselineAgent, OracleAgent
from emberpath.blueprint import Blueprint
from emberpath.dependents import HOUSEHOLDS
from emberpath.grid import WAIT, action_toward, dijkstra, path_to
from emberpath.planner import HazardAwarePlanner
from emberpath.reward import RewardTracker
from emberpath.scenarios import NOVELTIES
from emberpath.world import World

ROOT = Path(__file__).resolve().parents[1]


class LegacyReward:
    """The reward env.py used before reward.py existed, kept here only so the audit can compare."""

    def __init__(self, world: World):
        self.prev, self.total = world.counts(), 0.0

    def update(self, world: World) -> None:
        c = world.counts()
        r = -0.01 + (c["rescued"] - self.prev["rescued"]) - 0.5 * (c["lost"] - self.prev["lost"])
        if world.outcome == "success":
            r += 2.0
        if world.outcome == "agent_lost":
            r -= 1.0
        self.prev, self.total = c, self.total + r


# ---------------------------------------------------------------- policies
def oracle(world, bp):
    agent = OracleAgent(bp, world)
    return lambda obs, ev: agent.act(obs, ev).action


def baseline(world, bp):
    agent = BaselineAgent(bp, world)
    return lambda obs, ev: agent.act(obs, ev).action


def stall(world, bp):
    return lambda obs, ev: WAIT


def suicide(world, bp):
    def act(obs, ev):
        if not obs.burning:
            return WAIT
        jammed = world.rules.jammed_doors  # an exploit policy is allowed to cheat
        dist, parent = dijkstra(
            obs.agent_pos,
            lambda c: 1.0 if bp.passable(c) and c not in jammed else float("inf"),
            bp.height,
            bp.width,
        )
        target = min((b for b in obs.burning if b in dist), key=lambda b: dist[b], default=None)
        if target is None or target == obs.agent_pos:
            return WAIT
        return action_toward(obs.agent_pos, path_to(parent, obs.agent_pos, target)[1])

    return act


def abandon(world, bp):
    planner = HazardAwarePlanner(bp, world.rules)

    def act(obs, ev):
        planner.belief = world.rules
        return planner.act(dataclasses.replace(obs, dependents=())).action

    return act


POLICIES = {"oracle": oracle, "baseline": baseline, "stall": stall, "suicide": suicide, "abandon": abandon}


def run(policy_name: str, bp: Blueprint, household, novelty: str | None, seed: int):
    specs = NOVELTIES[novelty](bp, household) if novelty else []
    world = World(bp, household, seed=seed, novelties=specs, max_steps=150)
    new, old = RewardTracker(world), LegacyReward(world)
    act = POLICIES[policy_name](world, bp)
    obs, events = world.observe(), []
    while not world.done:
        res = world.step(act(obs, events))
        new.update(world)
        old.update(world)
        obs, events = res.obs, res.events
    return old.total, new.total, world.outcome, world.counts()["rescued"] / max(1, world.counts()["total"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--profile", default="single")
    ap.add_argument("--novelty", default="door_jam")
    a = ap.parse_args()
    bp = Blueprint.from_file(ROOT / "blueprints" / "ranch.txt")
    hh = HOUSEHOLDS[a.profile]

    rows = {}
    for name in POLICIES:
        res = [run(name, bp, hh, a.novelty, s) for s in range(a.seeds)]
        n = len(res)
        outcomes = {}
        for r in res:
            outcomes[r[2]] = outcomes.get(r[2], 0) + 1
        med = lambda xs: sorted(xs)[len(xs) // 2]
        rows[name] = {
            "legacy": sum(r[0] for r in res) / n,
            "new": sum(r[1] for r in res) / n,
            "new_median": med([r[1] for r in res]),
            "rescued": sum(r[3] for r in res) / n,
            "outcomes": ", ".join(f"{k} {v}/{n}" for k, v in sorted(outcomes.items())),
        }

    lines = [
        f"# Reward audit: ranch / {a.profile} / {a.novelty} / {a.seeds} seeds",
        "",
        "Episode return. Higher is better. Honest effort should outrank every shortcut.",
        "Means are pulled down by the few failure episodes, which are penalized heavily on purpose;",
        "the median shows the typical episode.",
        "",
        "| policy | legacy (mean) | new (mean) | new (median) | fraction rescued | outcomes |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        lines.append(f"| {name} | {r['legacy']:.2f} | {r['new']:.2f} | {r['new_median']:.2f} | {r['rescued']:.2f} | {r['outcomes']} |")

    def ranking(key):
        return " > ".join(sorted(rows, key=lambda n: -rows[n][key]))

    lines += ["", f"Ranking under legacy reward: {ranking('legacy')}", f"Ranking under new reward:    {ranking('new')}", ""]
    for key, label in (("legacy", "legacy"), ("new", "new")):
        if rows["suicide"][key] > rows["baseline"][key]:
            lines.append(f"EXPLOIT under {label} reward: suicide ({rows['suicide'][key]:.2f}) outscores a struggling honest agent ({rows['baseline'][key]:.2f}).")
        else:
            lines.append(f"ok under {label} reward: suicide ({rows['suicide'][key]:.2f}) scores below a struggling honest agent ({rows['baseline'][key]:.2f}).")
        if rows["abandon"][key] > rows["oracle"][key]:
            lines.append(f"EXPLOIT under {label} reward: abandoning outscores honest rescue.")
    out = "\n".join(lines) + "\n"
    print(out)
    (ROOT / "docs" / "reward_audit.md").write_text(out)


if __name__ == "__main__":
    main()
