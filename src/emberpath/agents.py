"""Four agents, one interface.

baseline  plans with the textbook rules and never updates them. Hurt by novelty.
oracle    plans with the TRUE rules at every tick (cheating). Upper bound on what any
          adaptive agent could achieve with this planner.
adaptive  baseline + detector + repairer: learns the changed rules from evidence alone.
naive_repair  ablation: detector, but flagged rules jump to their posterior mean (no search).

The headline metric for the whole project:

    gap_closed = (adaptive - baseline) / (oracle - baseline)      (on mean score)
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Protocol

from emberpath.blueprint import Blueprint
from emberpath.explain import DecisionLog, Explanation, describe_patch, explain_adaptation, plan_with, technical_appendix
from emberpath.novelty import BetaBernoulliDetector, Finding, NoveltyDetector
from emberpath.planner import Decision, HazardAwarePlanner
from emberpath.repair import RepairResult, RepairSearch
from emberpath.rules import BOOLEAN_FAMILIES, Rules, RulePatch, key_cell, nominal_rules, parse_key
from emberpath.world import Event, Observation, World


class Agent(Protocol):
    name: str

    def act(self, obs: Observation, events: list[Event]) -> Decision:
        """`events` are the rule-trials observed during the previous tick."""
        ...


class BaselineAgent:
    name = "baseline"

    def __init__(self, blueprint: Blueprint, world: World | None = None):
        self.planner = HazardAwarePlanner(blueprint, nominal_rules())

    def act(self, obs: Observation, events: list[Event]) -> Decision:
        return self.planner.act(obs)


class OracleAgent:
    name = "oracle"
    privileged = True  # the only agent make_agent hands the true world to

    def __init__(self, blueprint: Blueprint, world: World):
        self.world = world
        self.planner = HazardAwarePlanner(blueprint, world.rules)

    def act(self, obs: Observation, events: list[Event]) -> Decision:
        self.planner.belief = self.world.rules  # peeks at the truth
        return self.planner.act(obs)


class Repairer(Protocol):
    def repair(self, belief: Rules, findings: list[Finding], events: list[Event]) -> RepairResult: ...


class PosteriorMeanRepair:
    """Ablation: no search. Set every flagged rule to the detector's posterior mean (booleans: round it)."""

    def repair(self, belief: Rules, findings: list[Finding], events: list[Event]) -> RepairResult:
        patches = [
            RulePatch(f.key, f.posterior_mean > 0.5 if parse_key(f.key)[0] in BOOLEAN_FAMILIES else f.posterior_mean)
            for f in findings
        ]
        return RepairResult(belief.with_patches(patches), patches, math.nan, math.inf)


class AdaptiveAgent:
    """Baseline planner + detector + repairer.

    Every tick: feed the new events to the detector; if it reports a *fresh* finding (a flagged key
    with evidence the repairer has not yet seen), repair the belief from the whole event history,
    hand the repaired belief to the same planner, and re-baseline the detector so it measures
    surprise against the new belief. Ticks without fresh findings cost one detector update.

    "Fresh" is per (key, amount of evidence), not per key: a rule that changes again, or changes
    back, is re-flagged by the detector after the re-baseline and gets repaired again.
    """

    name = "adaptive"

    def __init__(
        self,
        blueprint: Blueprint,
        world: World | None = None,
        detector: NoveltyDetector | None = None,
        repairer: Repairer | None = None,
        belief: Rules | None = None,  # start from beliefs learned in earlier episodes (see `lessons`)
        explain: bool = True,  # False: terse one-line notes, no route comparison (same actions either way)
    ):
        self.belief: Rules = belief or nominal_rules()
        self.planner = HazardAwarePlanner(blueprint, self.belief)
        self.detector = detector or BetaBernoulliDetector(self.belief)
        self.repairer = repairer or RepairSearch()
        self.all_events: list[Event] = []
        self.patches: list[tuple[int, RulePatch]] = []  # (step, patch) history
        self.repairs = 0  # repair calls made (cost accounting)
        self.log = DecisionLog()
        self._seen: dict[str, int] = {}  # key -> evidence count when the repairer last looked at it
        self._dismissed: set[tuple[str, float]] = set()  # (key, believed) surprises judged to be bad luck
        self.explain = explain
        self.explanations: list[Explanation] = []  # every change of mind, explained (plain text + appendix)
        self._tally: dict[str, list[int]] = {}  # key -> [times it happened, times it was tried], own evidence only

    def act(self, obs: Observation, events: list[Event]) -> Decision:
        note, pending = "", None
        if events:  # findings can only change when evidence arrives
            self.all_events.extend(events)
            for e in events:
                c = self._tally.setdefault(e.key, [0, 0])
                c[0] += e.outcome
                c[1] += 1
            self.detector.observe(events)
            findings = self.detector.findings()
            if any(f.n_trials > self._seen.get(f.key, 0) for f in findings):
                note, pending = self._repair(obs, findings)
        decision = self.planner.act(obs)
        if pending is not None:  # explain with both plans in hand: what I was doing, and what I do now
            before, result, old_plan, findings = pending
            evidence = {k: (s, n) for k, (s, n) in self._tally.items()}
            exp = explain_adaptation(self.planner.bp, obs, before, result, old_plan, decision, self.planner,
                                     findings, evidence)
            self.explanations.append(exp)
            note = exp.text
        if note:
            decision = replace(decision, reason=f"{note} {decision.reason}", explanation=note)
        self.log.add(obs.step, decision)
        return decision

    def _repair(self, obs: Observation, findings: list[Finding]):
        """Repair the belief from every current finding. Returns (note for the human, or "", and, when explaining,
        what explain_adaptation needs once the new plan is made: (old belief, result, old plan, findings))."""
        step = obs.step
        self.repairs += 1
        self._seen.update((f.key, f.n_trials) for f in findings)
        before = self.belief
        result = self.repairer.repair(before, findings, self.all_events)
        if not result.patches:  # say so once per (rule, belief), not on every further surprising tick
            new = {(f.key, f.believed) for f in findings} - self._dismissed
            self._dismissed |= new
            if not new:
                return "", None
            note = "Something surprised me, but it looks like bad luck; keeping my beliefs."
            if self.explain:
                evidence = {k: (s, n) for k, (s, n) in self._tally.items()}
                self.explanations.append(Explanation(step, note, appendix=technical_appendix(findings, result, evidence)))
            return note, None
        old_plan = plan_with(self.planner, before, obs) if self.explain else None
        self.belief = self.planner.belief = result.rules
        self.detector.reset_belief(self.belief)
        self.patches.extend((step, p) for p in result.patches)
        if self.explain:
            return "", (before, result, old_plan, findings)
        changed = " ".join(describe_patch(p, before) for p in result.patches)
        if not result.ambiguous:
            return f"Updating my beliefs: {changed}", None
        alt = " ".join(describe_patch(p, before) for p in result.runner_up) or "nothing has changed (bad luck)"
        return (
            f"I think this changed: {changed} But \"{alt}\" would explain what I saw almost as well "
            f"(margin {result.runner_up_margin:.1f}); acting on my best guess."
        ), None


    def lessons(self) -> Rules:
        """What to carry into the next episode in the same house: the final belief, except that a hiding place
        where someone was heard stays occupied. `occupied` means "someone hides here when the fire starts"; the
        silence after they have been found and led away says nothing against that, but the repair search reads
        it as the rule reverting."""
        heard = {key_cell(p.key) for _, p in self.patches if parse_key(p.key)[0] == "occupied" and p.value}
        return replace(self.belief, occupied=self.belief.occupied | heard)


class NaiveRepairAgent(AdaptiveAgent):
    """Ablation: detector, but no repair search (flagged rules jump to their posterior mean)."""

    name = "naive_repair"

    def __init__(self, blueprint: Blueprint, world: World | None = None, explain: bool = True):
        super().__init__(blueprint, repairer=PosteriorMeanRepair(), explain=explain)


AGENTS = {
    "baseline": BaselineAgent,
    "oracle": OracleAgent,
    "adaptive": AdaptiveAgent,
    "naive_repair": NaiveRepairAgent,
}


def make_agent(name: str, blueprint: Blueprint, world: World) -> Agent:
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}")
    cls = AGENTS[name]
    # Evidence discipline by construction: only privileged agents ever receive the world (and its true rules).
    return cls(blueprint, world if getattr(cls, "privileged", False) else None)
