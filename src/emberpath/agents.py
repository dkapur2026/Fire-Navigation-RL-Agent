"""Three agents, one interface.

baseline  plans with the textbook rules and never updates them. Hurt by novelty.
oracle    plans with the TRUE rules at every tick (cheating). Upper bound on what any
          adaptive agent could achieve with this planner.
adaptive  YOUR agent: baseline + detector + repairer. Closes the gap between the two.

The headline metric for the whole project:

    gap_closed = (adaptive - baseline) / (oracle - baseline)      (on mean score)
"""

from __future__ import annotations

from typing import Protocol

from emberpath.blueprint import Blueprint
from emberpath.explain import DecisionLog
from emberpath.novelty import BetaBernoulliDetector, NoveltyDetector
from emberpath.planner import Decision, HazardAwarePlanner
from emberpath.repair import RepairSearch
from emberpath.rules import Rules, RulePatch, nominal_rules
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

    def __init__(self, blueprint: Blueprint, world: World):
        self.world = world
        self.planner = HazardAwarePlanner(blueprint, world.rules)

    def act(self, obs: Observation, events: list[Event]) -> Decision:
        self.planner.belief = self.world.rules  # peeks at the truth
        return self.planner.act(obs)


class AdaptiveAgent:
    """Phase 5 wiring. Works as soon as the detector and repairer are implemented."""

    name = "adaptive"

    def __init__(
        self,
        blueprint: Blueprint,
        world: World | None = None,
        detector: NoveltyDetector | None = None,
        repairer: RepairSearch | None = None,
    ):
        self.belief: Rules = nominal_rules()
        self.planner = HazardAwarePlanner(blueprint, self.belief)
        self.detector = detector or BetaBernoulliDetector(self.belief)
        self.repairer = repairer or RepairSearch()
        self.all_events: list[Event] = []
        self.patches: list[tuple[int, RulePatch]] = []  # (step, patch) history
        self.log = DecisionLog()
        self._handled: set[str] = set()

    def act(self, obs: Observation, events: list[Event]) -> Decision:
        self.all_events.extend(events)
        self.detector.observe(events)
        fresh = [f for f in self.detector.findings() if f.key not in self._handled]
        if fresh:
            result = self.repairer.repair(self.belief, fresh, self.all_events)
            self.belief = result.rules
            self.planner.belief = self.belief
            self.detector.reset_belief(self.belief)
            self._handled.update(f.key for f in fresh)
            self.patches.extend((obs.step, p) for p in result.patches)
        decision = self.planner.act(obs)
        self.log.add(obs.step, decision)
        return decision


AGENTS = {"baseline": BaselineAgent, "oracle": OracleAgent, "adaptive": AdaptiveAgent}


def make_agent(name: str, blueprint: Blueprint, world: World) -> Agent:
    try:
        return AGENTS[name](blueprint, world)
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}") from None
