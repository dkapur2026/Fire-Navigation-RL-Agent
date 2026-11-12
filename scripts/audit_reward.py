"""Reward-hacking audit: score deliberately degenerate policies under the LEGACY and the NEW reward.

    python scripts/audit_reward.py            # 20 seeds, ranch / single / door_jam
    python scripts/audit_reward.py --seeds 8  # quicker

A good reward ranks honest effort above every shortcut. A reward with a loophole lets a shortcut outscore
an honest-but-struggling agent, and a learning agent will find it. Rerun this after ANY change to reward.py
and add a policy here for every new exploit you think of.

Policies
  oracle     honest, knows the true rules (usually succeeds)
  baseline   honest, but its rulebook is stale (stuck at the jammed door, times out)
  adaptive   honest, learns from what it sees (finds the jam by bumping it, then repairs its rulebook)
  stall      does nothing until the clock runs out
  suicide    walks into the nearest flame
  abandon    runs to the exit alone, never rescuing anyone
  reward_snoop  the baseline, plus any hidden person the reward's progress signal gives away: a stand-in for a learner
             that reads hints out of its reward. If the reward leaks nothing it plays exactly like baseline (try
             --novelty hidden_dependent); see docs/REWARD.md.
  door_shy   the baseline, but it never tries a door that is really jammed (it cheats to know which) and waits there
             instead, so the jam is never found. With an evidence-only potential an unfound jam keeps the potential
             optimistic, so this tests whether "not looking" pays: it must stay below every honest agent that looks
             and succeeds, and (since the distance terms of the potential are 0 at every ending) it must not beat
             the honest baseline stuck at the door either (docs/REWARD.md).
  jam_snoop  the baseline, plus any jammed door the reward's progress signal gives away before the rescuer has bumped
             it. It sees only what a learner fed its own rewards would: each tick's progress payment. It checks
             which "door d is jammed" hypothesis explains that payment from the observed positions, and once "no
             hidden jam" is ruled out it plans around the door. If the potential uses only jams the rescuer has
             found, "no hidden jam" is never ruled out and it plays exactly like baseline.
"""

from __future__ import annotations

import argparse
import dataclasses
import copy
from types import SimpleNamespace
from pathlib import Path

from emberpath.agents import AdaptiveAgent, BaselineAgent, OracleAgent
from emberpath.blueprint import Blueprint
from emberpath.dependents import HOUSEHOLDS
from emberpath.grid import MOVES, WAIT, action_toward, dijkstra, path_to
from emberpath.planner import HazardAwarePlanner
from emberpath.reward import COMPONENTS, RewardConfig, RewardTracker
from emberpath.rules import nominal_rules
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


def adaptive(world, bp):
    agent = AdaptiveAgent(bp)
    return lambda obs, ev: agent.act(obs, ev).action


def door_shy(world, bp):
    agent = BaselineAgent(bp, world)

    def act(obs, ev):
        a = agent.act(obs, ev).action
        if a != WAIT:
            dr, dc = MOVES[a]
            if (obs.agent_pos[0] + dr, obs.agent_pos[1] + dc) in world.rules.jammed_doors:  # an exploit may cheat
                return WAIT
        return a

    return act


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
        planner.belief = dataclasses.replace(world.rules, occupied=frozenset())  # nobody to collect, hidden or not
        return planner.act(dataclasses.replace(obs, dependents=())).action

    return act


def reward_snoop(world, bp):
    planner, tracker = HazardAwarePlanner(bp, nominal_rules()), RewardTracker(world)

    def phi_at(cell):  # the potential if the rescuer stood at `cell`
        here, world.agent_pos = world.agent_pos, cell
        phi = tracker.potential(world)
        world.agent_pos = here
        return phi

    def act(obs, ev):
        # The potential is highest where the "nearest waiting dependent" term vanishes: on someone the reward counts.
        phi = {s: phi_at(s) for s in bp.slots}
        top = max(phi.values())
        heard = {s for s, v in phi.items() if v == top} if min(phi.values()) < top else set()
        planner.belief = dataclasses.replace(nominal_rules(), occupied=frozenset(heard))
        return planner.act(obs).action

    return act


def jam_snoop(world, bp):
    """See the module docstring. Reads ONLY the progress payment (what a reward-fed learner sees) plus observable state
    (positions, its own bump events, broken doors); the hypothetical potentials are computed on a stand-in world whose
    jam set is the hypothesis, never on world.rules."""
    planner, paid, w = HazardAwarePlanner(bp, nominal_rules()), RewardTracker(world), RewardConfig().progress_weight
    probe = RewardTracker(world)
    found: set = set()
    candidates: set | None = None  # doors that could explain every hint so far; None = "no hidden jam" still fits
    prev = None

    def phi(snapshot, jams):
        pos, deps, _ = snapshot
        stand_in = SimpleNamespace(blueprint=bp, agent_pos=pos, dependents=deps, bumped_jams=set(jams), broken_doors=set(), done=False,
                                   rules=dataclasses.replace(nominal_rules(), jammed_doors=frozenset(jams)))
        return probe.potential(stand_in)

    def act(obs, ev):
        nonlocal candidates, prev
        for e in ev:
            if e.outcome and e.key.startswith("door_jammed:"):
                found.add(tuple(int(x) for x in e.key.split(":")[1].split(",")))
        known = frozenset(found - set(obs.broken_doors))
        now = (world.agent_pos, tuple(copy.copy(d) for d in world.dependents), known)  # observable in door_jam
        if prev is not None:
            got = paid.update(world).components["progress"]  # what the learner is paid this tick

            def fits(before: set, after: set) -> bool:  # payment if the hidden jam set went from `before` to `after`
                return abs(w * (phi(now, now[2] | after) - phi(prev, prev[2] | before)) - got) < 1e-9

            if candidates is None:
                if not fits(set(), set()):  # "no hidden jam" no longer explains the payment: some door just jammed
                    candidates = {d for d in bp.doors if d not in known and fits(set(), {d})}
            elif candidates:
                candidates = {d for d in candidates if fits({d}, {d})} or candidates
        prev = now
        planner.belief = dataclasses.replace(nominal_rules(), jammed_doors=frozenset(candidates or ()))
        return planner.act(obs).action

    return act


POLICIES = {
    "oracle": oracle,
    "baseline": baseline,
    "adaptive": adaptive,
    "stall": stall,
    "suicide": suicide,
    "abandon": abandon,
    "reward_snoop": reward_snoop,
    "door_shy": door_shy,
    "jam_snoop": jam_snoop,
}


def rollout(make_act, bp: Blueprint, household, novelty: str | None, seed: int, configs=(RewardConfig(),)):
    """One episode of policy `make_act(world, bp)`. Returns (legacy return, summed components per config, world):
    every config scores the same trajectory, so a weight sweep needs no extra episodes."""
    specs = NOVELTIES[novelty](bp, household) if novelty else []
    world = World(bp, household, seed=seed, novelties=specs, max_steps=150)
    trackers, old = [RewardTracker(world, c) for c in configs], LegacyReward(world)
    sums = [dict.fromkeys(COMPONENTS, 0.0) for _ in configs]
    act = make_act(world, bp)
    obs, events = world.observe(), []
    while not world.done:
        res = world.step(act(obs, events))
        for tracker, total in zip(trackers, sums):
            for k, v in tracker.update(world).components.items():
                total[k] += v
        old.update(world)
        obs, events = res.obs, res.events
    return old.total, sums, world


def run(policy_name: str, bp: Blueprint, household, novelty: str | None, seed: int):
    old, (new,), world = rollout(POLICIES[policy_name], bp, household, novelty, seed)
    c = world.counts()
    return old, sum(new.values()), world.outcome, c["rescued"] / max(1, c["total"])


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
        if rows["door_shy"][key] > min(rows["oracle"][key], rows["adaptive"][key]):
            lines.append(f"EXPLOIT under {label} reward: never finding the jam (door_shy) outscores honest agents that find it.")
        if label == "new":  # (the legacy reward is kept only to show its suicide loophole)
            for shortcut in ("stall", "door_shy"):
                if rows[shortcut][key] > rows["baseline"][key] + 1e-9:
                    lines.append(f"EXPLOIT under {label} reward: {shortcut} ({rows[shortcut][key]:.2f}) outscores the honest "
                                 f"agent stuck at the jammed door ({rows['baseline'][key]:.2f}).")
        if rows["abandon"][key] > rows["oracle"][key]:
            lines.append(f"EXPLOIT under {label} reward: abandoning outscores honest rescue.")
    snoop_ok = rows["reward_snoop"] == rows["baseline"]
    lines.append(("ok: reward_snoop plays exactly like baseline, so the new reward gives away no hidden person." if snoop_ok
                  else "LEAK: reward_snoop differs from baseline, so the new reward reveals where a hidden person is."))
    jam_ok = rows["jam_snoop"] == rows["baseline"]
    lines.append(("ok: jam_snoop plays exactly like baseline, so the new reward gives away no unfound jammed door." if jam_ok
                  else "LEAK: jam_snoop differs from baseline, so the new reward reveals a jammed door before it is bumped."))
    out = "\n".join(lines) + "\n"
    print(out)
    (ROOT / "docs" / "reward_audit.md").write_text(out)


if __name__ == "__main__":
    main()
