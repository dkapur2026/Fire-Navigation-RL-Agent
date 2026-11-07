"""The simulator: a burning house, a rescuer, and the people (or dog) being rescued.

This module is pure Python + numpy (no RL framework needed). ``env.py`` wraps it for RL.

One tick of ``World.step(action)``:
    1. scheduled novelties (rule changes) take effect
    2. the rescuer moves
    3. dependents react (bond with the rescuer, follow, or flee fire)
    4. fire spreads, smoke diffuses
    5. anyone standing in fire is lost; dependents standing on an exit are rescued

Every random trial that tests a rule is logged as an ``Event(key, outcome)``. These events
are the *evidence* a novelty detector learns from, with no hidden information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from emberpath.blueprint import Blueprint
from emberpath.dependents import DependentProfile, Household
from emberpath.grid import MOVES, WAIT, Cell, bfs_distances, manhattan, neighbors4
from emberpath.rules import Rules, RulePatch, door_key, nominal_rules

WAITING, FOLLOWING, RESCUED, LOST = "waiting", "following", "rescued", "lost"
BASE_VISIBILITY = 6


@dataclass(frozen=True)
class Event:
    """One Bernoulli trial of one rule: did the outcome happen?"""

    step: int
    key: str
    outcome: bool


@dataclass(frozen=True)
class NoveltySpec:
    """At tick `step`, the TRUE rules change by `patch`. The agent is not told."""

    step: int
    patch: RulePatch
    description: str = ""


@dataclass
class Dependent:
    id: int
    profile: DependentProfile
    pos: Cell
    status: str = WAITING


@dataclass(frozen=True)
class DependentView:
    id: int
    kind: str
    name: str
    pos: Cell
    status: str


@dataclass(frozen=True)
class Observation:
    step: int
    agent_pos: Cell
    alive: bool
    burning: frozenset[Cell]
    burnt: frozenset[Cell]
    dependents: tuple[DependentView, ...]
    smoke_at_agent: float
    visibility_radius: int | None  # None = full observability


@dataclass
class StepResult:
    obs: Observation
    events: list[Event]
    done: bool
    info: dict = field(default_factory=dict)


class World:
    def __init__(
        self,
        blueprint: Blueprint,
        household: Household,
        rules: Rules | None = None,
        seed: int = 0,
        novelties: tuple[NoveltySpec, ...] | list[NoveltySpec] = (),
        max_steps: int = 150,
    ) -> None:
        blueprint.validate(len(household.dependents))
        self.blueprint = blueprint
        self.household = household
        self.rules = rules or nominal_rules()
        self.novelties = tuple(sorted(novelties, key=lambda n: n.step))
        self.max_steps = max_steps
        self.t = 0

        # Independent random streams, so fire evolves identically for every agent given a seed.
        fire_ss, dep_ss = np.random.SeedSequence(seed).spawn(2)
        self._fire_rng = np.random.default_rng(fire_ss)
        self._dep_rng = np.random.default_rng(dep_ss)

        h, w = blueprint.height, blueprint.width
        self.burn = np.zeros((h, w), dtype=np.int32)  # 0 clear, >0 burning (ticks left), -1 burnt out
        self.smoke = np.zeros((h, w), dtype=np.float64)
        self._open = np.array(
            [[0.0 if (r, c) in blueprint.walls else 1.0 for c in range(w)] for r in range(h)]
        )
        for cell in blueprint.ignitions:
            self.burn[cell] = self.rules.burn_steps

        self.agent_pos: Cell = blueprint.agent_start
        self.agent_alive = True
        self.dependents = [
            Dependent(i, prof, blueprint.slots[i]) for i, prof in enumerate(household.dependents)
        ]
        self.novelty_log: list[tuple[int, RulePatch]] = []
        self.outcome: str | None = None

    # ----------------------------------------------------------------- queries
    @property
    def done(self) -> bool:
        return self.outcome is not None

    def is_burning(self, cell: Cell) -> bool:
        return bool(self.burn[cell] > 0)

    def burning_cells(self) -> frozenset[Cell]:
        return frozenset((int(r), int(c)) for r, c in zip(*np.nonzero(self.burn > 0)))

    def burnt_cells(self) -> frozenset[Cell]:
        return frozenset((int(r), int(c)) for r, c in zip(*np.nonzero(self.burn < 0)))

    def counts(self) -> dict[str, int]:
        total = len(self.dependents)
        rescued = sum(d.status == RESCUED for d in self.dependents)
        lost = sum(d.status == LOST for d in self.dependents)
        return {"total": total, "rescued": rescued, "lost": lost}

    def _walkable(self, cell: Cell) -> bool:
        """Static walkability under the TRUE rules (walls and jammed doors block)."""
        return self.blueprint.passable(cell) and cell not in self.rules.jammed_doors

    def observe(self, full: bool = True) -> Observation:
        radius = None
        if not full:
            radius = max(1, BASE_VISIBILITY - int(self.smoke[self.agent_pos] * 2))

        def visible(cell: Cell) -> bool:
            return radius is None or manhattan(cell, self.agent_pos) <= radius

        return Observation(
            step=self.t,
            agent_pos=self.agent_pos,
            alive=self.agent_alive,
            burning=frozenset(c for c in self.burning_cells() if visible(c)),
            burnt=frozenset(c for c in self.burnt_cells() if visible(c)),
            dependents=tuple(
                DependentView(d.id, d.profile.kind, d.profile.name, d.pos, d.status)
                for d in self.dependents
                if visible(d.pos) or d.status in (FOLLOWING, RESCUED)
            ),
            smoke_at_agent=float(self.smoke[self.agent_pos]),
            visibility_radius=radius,
        )

    # -------------------------------------------------------------------- tick
    def step(self, action: int, full_obs: bool = True) -> StepResult:
        if self.done:
            raise RuntimeError("episode is over")
        events: list[Event] = []

        # 1. novelties
        for spec in self.novelties:
            if spec.step == self.t:
                self.rules = self.rules.with_patch(spec.patch)
                self.novelty_log.append((self.t, spec.patch))

        # 2. rescuer moves
        if action != WAIT:
            dr, dc = MOVES[action]
            target = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if self.blueprint.passable(target):
                if target in self.blueprint.doors:
                    blocked = target in self.rules.jammed_doors
                    events.append(Event(self.t, door_key(target), blocked))
                    if not blocked:
                        self.agent_pos = target
                else:
                    self.agent_pos = target

        # 3. dependents
        self._update_dependents(events)

        # 4. environment
        self._spread_fire(events)
        self._update_smoke()

        # 5. consequences
        if self.burn[self.agent_pos] > 0:
            self.agent_alive = False
        for d in self.dependents:
            if d.status in (WAITING, FOLLOWING):
                if self.burn[d.pos] > 0:
                    d.status = LOST
                elif d.pos in self.blueprint.exits:
                    d.status = RESCUED

        self.t += 1
        self._check_done()
        return StepResult(
            obs=self.observe(full_obs),
            events=events,
            done=self.done,
            info={"outcome": self.outcome, **self.counts()},
        )

    # ------------------------------------------------------------- dependents
    def _update_dependents(self, events: list[Event]) -> None:
        active = [d for d in self.dependents if d.status in (WAITING, FOLLOWING)]
        if not active:
            return
        bp = self.blueprint
        for d in active:
            if d.status == WAITING and manhattan(d.pos, self.agent_pos) <= 1:
                d.status = FOLLOWING

        dist_to_agent = bfs_distances(
            self.agent_pos, self._walkable, bp.height, bp.width
        )
        burning = self.burning_cells()
        for d in active:
            if d.status == FOLLOWING:
                if self.t % d.profile.move_period != 0:
                    continue
                here = dist_to_agent.get(d.pos)
                if here is None or here == 0:
                    continue
                options = [
                    n
                    for n in neighbors4(d.pos, bp.height, bp.width)
                    if self._walkable(n) and n not in burning and dist_to_agent.get(n, 1 << 30) < here
                ]
                if not options:
                    continue
                succeeded = bool(self._dep_rng.random() < self.rules.follow_prob[d.profile.kind])
                events.append(Event(self.t, f"follow_prob:{d.profile.kind}", succeeded))
                if succeeded:
                    d.pos = min(options, key=lambda n: dist_to_agent[n])
            elif d.profile.flees_fire and burning:
                near = min(manhattan(d.pos, b) for b in burning)
                if near <= 2 and self._dep_rng.random() < d.profile.flee_prob:
                    options = [
                        n
                        for n in neighbors4(d.pos, bp.height, bp.width)
                        if self._walkable(n) and n not in burning
                    ]
                    if options:
                        d.pos = max(options, key=lambda n: min(manhattan(n, b) for b in burning))

    # -------------------------------------------------------------------- fire
    def _spread_fire(self, events: list[Event]) -> None:
        bp = self.blueprint
        burning = self.burning_cells()
        candidates: dict[Cell, int] = {}
        for b in burning:
            for n in neighbors4(b, bp.height, bp.width):
                if bp.passable(n) and self.burn[n] == 0 and n not in bp.exits:
                    candidates[n] = candidates.get(n, 0) + 1

        ignite: list[Cell] = []
        for cell in sorted(candidates):
            material = bp.material(cell)
            key = f"spread_prob:{material}"
            p = self.rules.spread_prob[material]
            caught = False
            for _ in range(candidates[cell]):  # one independent trial per burning neighbor
                hit = bool(self._fire_rng.random() < p)
                events.append(Event(self.t, key, hit))
                caught = caught or hit
            if caught:
                ignite.append(cell)

        for b in burning:  # burn down
            self.burn[b] -= 1
            if self.burn[b] == 0:
                self.burn[b] = -1
        for cell in ignite:
            self.burn[cell] = self.rules.burn_steps

    def _update_smoke(self) -> None:
        s = (self.smoke + 0.3 * (self.burn > 0)) * self._open
        p = np.pad(s, 1)
        nsum = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        q = np.pad(self._open, 1)
        ncnt = q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:]
        avg = np.where(ncnt > 0, nsum / np.maximum(ncnt, 1), 0.0)
        self.smoke = (0.6 * s + 0.4 * avg) * 0.97 * self._open

    # -------------------------------------------------------------- termination
    def _check_done(self) -> None:
        counts = self.counts()
        resolved = counts["rescued"] + counts["lost"] == counts["total"]
        if not self.agent_alive:
            self.outcome = "agent_lost"
        elif resolved and self.agent_pos in self.blueprint.exits:
            self.outcome = "success" if counts["lost"] == 0 else "partial"
        elif self.t >= self.max_steps:
            self.outcome = "timeout"
