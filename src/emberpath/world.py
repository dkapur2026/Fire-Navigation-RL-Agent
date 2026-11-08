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

An unexpected dependent (rule ``occupied:<cell>``, see ``scenarios.hidden_dependent``) is a real
person the household list does not mention. It is hidden (not in ``Observation.dependents``) until
it bonds with the rescuer; the only evidence before then is hearing: every tick, each unused
blueprint slot within ``HEARING_RANGE`` of the rescuer yields ``Event(occupied:<slot>, calling)``.
From tick ``REVEAL_STEP`` a person still hiding is heard from anywhere in the house (someone outside
shouts where they are), so the news always arrives, at a known time, whatever route the rescuer took.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from emberpath.blueprint import Blueprint
from emberpath.dependents import VISITOR, DependentProfile, Household
from emberpath.grid import MOVES, WAIT, Cell, bfs_distances, manhattan, neighbors4
from emberpath.rules import Rules, RulePatch, door_key, key_cell, nominal_rules, occupied_key, parse_key

WAITING, FOLLOWING, RESCUED, LOST = "waiting", "following", "rescued", "lost"
BASE_VISIBILITY = 6
# Manhattan distance, through walls, at which the rescuer hears someone calling from a hiding place.
HEARING_RANGE = 4
# From this tick on, a person still hiding is heard regardless of distance: late news, but news for certain.
REVEAL_STEP = 10
# An adult acting alone panics for PANIC_TICKS when flames come within PANIC_RADIUS or a door they try is jammed.
PANIC_TICKS = 10
PANIC_RADIUS = 2


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
    hidden: bool = False  # not in the household list and out of sight until it bonds with the rescuer
    leader: int | None = None  # whom a FOLLOWING dependent follows: None = the rescuer, own id = moving on its own
    panic_left: int = 0  # ticks of panic left (adults acting alone only)

    @property
    def hiding(self) -> bool:
        """Hidden and still waiting: the rescuer cannot see it (it can only be heard)."""
        return self.hidden and self.status == WAITING


@dataclass(frozen=True)
class DependentView:
    id: int
    kind: str
    name: str
    pos: Cell
    status: str
    pace: int = 1  # moves at most once every `pace` ticks (part of the household description, not a hidden rule)
    leader: int | None = None  # None = following the rescuer; own id = an adult acting alone; else whom it follows
    panicked: bool = False  # visibly panicking: moving erratically, not toward any goal


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
    broken_doors: frozenset[Cell] = frozenset()  # jammed doors the rescuer has forced open (seen happen)


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
        self._pending = list(self.novelties)
        self.max_steps = max_steps
        self.t = 0

        # Independent random streams, so fire evolves identically for every agent given a seed.
        # spawn(3) leaves the first two streams exactly as spawn(2) made them
        fire_ss, dep_ss, force_ss = np.random.SeedSequence(seed).spawn(3)
        self._force_rng = np.random.default_rng(force_ss)
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
        for d in self.dependents:
            if d.profile.autonomous:  # adults set off on their own from the first tick
                d.status, d.leader = FOLLOWING, d.id
        self._unused_slots = blueprint.slots[len(household.dependents):]
        self.novelty_log: list[tuple[int, RulePatch]] = []
        self.bumped_jams: set[Cell] = set()  # jammed doors someone has walked into (observable)
        self.broken_doors: set[Cell] = set()  # jammed doors the rescuer forced open: passable from then on
        self.outcome: str | None = None
        self._apply_due_novelties()  # step-0 novelties hold from the start (counts, observe, reward)

    # ----------------------------------------------------------------- queries
    @property
    def done(self) -> bool:
        return self.outcome is not None

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
        """Static walkability under the TRUE rules (walls and jammed, unbroken doors block)."""
        return self.blueprint.passable(cell) and (cell not in self.rules.jammed_doors or cell in self.broken_doors)

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
                DependentView(d.id, d.profile.kind, d.profile.name, d.pos, d.status, d.profile.move_period, d.leader, d.panic_left > 0)
                for d in self.dependents
                if (visible(d.pos) and not d.hiding)
                or d.status in (FOLLOWING, RESCUED)
            ),
            smoke_at_agent=float(self.smoke[self.agent_pos]),
            visibility_radius=radius,
            broken_doors=frozenset(self.broken_doors),
        )

    # -------------------------------------------------------------------- tick
    def step(self, action: int, full_obs: bool = True) -> StepResult:
        if self.done:
            raise RuntimeError("episode is over")
        events: list[Event] = []

        # 1. novelties
        self._apply_due_novelties()

        # 2. rescuer moves
        if action != WAIT:
            dr, dc = MOVES[action]
            target = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if self.blueprint.passable(target):
                if target in self.blueprint.doors and target not in self.broken_doors:
                    blocked = target in self.rules.jammed_doors
                    events.append(Event(self.t, door_key(target), blocked))
                    if blocked:
                        self.bumped_jams.add(target)
                        # From the outside (hallway) side every try is also an attempt to kick it in; from inside
                        # a room it cannot be forced. Own RNG stream, so the fire stays identical for every agent.
                        if self.agent_pos in self.blueprint.door_outside.get(target, ()):
                            forced = bool(self._force_rng.random() < self.rules.force_prob["door"])
                            events.append(Event(self.t, "force_prob:door", forced))
                            if forced:
                                self.broken_doors.add(target)
                    else:
                        self.agent_pos = target
                else:
                    self.agent_pos = target

        # 3. the rescuer listens, then dependents react
        self._listen(events)
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

    # ------------------------------------------------------------- novelties
    def _apply_due_novelties(self) -> None:
        while self._pending and self._pending[0].step <= self.t:
            spec = self._pending.pop(0)
            self.rules = self.rules.with_patch(spec.patch)
            self.novelty_log.append((self.t, spec.patch))
            if parse_key(spec.patch.key)[0] == "occupied" and spec.patch.value:
                cell = key_cell(spec.patch.key)
                if all(d.pos != cell for d in self.dependents):
                    self.dependents.append(Dependent(len(self.dependents), VISITOR, cell, hidden=True))

    def _listen(self, events: list[Event]) -> None:
        """One hearing trial per unused slot in range: is someone hiding there calling for help? After
        REVEAL_STEP, a slot where someone still hides is heard from anywhere (outcome True). Deterministic
        (no RNG), so it cannot disturb the fire stream."""
        calling = {d.pos for d in self.dependents if d.hiding}
        revealed = calling if self.t >= REVEAL_STEP else set()
        for slot in self._unused_slots:
            if slot in revealed or manhattan(slot, self.agent_pos) <= HEARING_RANGE:
                events.append(Event(self.t, occupied_key(slot), slot in calling))

    # ------------------------------------------------------------- dependents
    def _update_dependents(self, events: list[Event]) -> None:
        active = [d for d in self.dependents if d.status in (WAITING, FOLLOWING)]
        if not active:
            return
        bp = self.blueprint
        burning = self.burning_cells()
        adults = [d for d in active if d.leader == d.id]
        for a in adults:  # reached by the rescuer (before or after their own move): calm down and go with them
            if manhattan(a.pos, self.agent_pos) > 1:
                self._move_adult(a, burning, events)
            if manhattan(a.pos, self.agent_pos) <= 1:
                a.leader, a.panic_left = None, 0
        adults = [a for a in adults if a.leader == a.id]
        for d in active:
            if d.status != WAITING:
                continue
            if manhattan(d.pos, self.agent_pos) <= 1:
                d.status, d.leader = FOLLOWING, None
            elif not d.hidden and not d.profile.autonomous:  # an adult only takes the children they know about
                a = next((a for a in adults if a.status == FOLLOWING and manhattan(d.pos, a.pos) <= 1), None)
                if a:
                    d.status, d.leader = FOLLOWING, a.id
        for d in active:  # a follower whose adult is gone (out or lost) follows the rescuer instead
            if d.leader not in (None, d.id) and self.dependents[d.leader].status != FOLLOWING:
                d.leader = None

        leaders = {d.leader for d in active if d.status == FOLLOWING and d.leader != d.id}
        dist_to = {  # followers move toward their leader; nobody else needs a search
            ld: bfs_distances(self.agent_pos if ld is None else self.dependents[ld].pos, self._walkable, bp.height, bp.width)
            for ld in leaders
        }
        for d in active:
            if d.status == FOLLOWING and d.leader != d.id:
                if self.t % d.profile.move_period != 0:
                    continue
                dist_to_agent = dist_to[d.leader]
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

    def _move_adult(self, a: Dependent, burning: frozenset[Cell], events: list[Event]) -> None:
        """An adult acting alone walks toward the nearest waiting child they know about, else the nearest exit.
        They know the floor plan, not the hidden rules: a jammed door is found by walking into it, which the
        rescuer sees (a door Event, shared in `bumped_jams`). Deterministic, so it never touches the fire RNG."""
        if self.t % a.profile.move_period != 0:
            return
        bp = self.blueprint
        if not a.panic_left and any(manhattan(a.pos, b) <= PANIC_RADIUS for b in burning):
            a.panic_left = PANIC_TICKS
        if a.panic_left:  # erratic: stay or step anywhere open, goal forgotten (dependents' RNG, not the fire's)
            a.panic_left -= 1
            options = [a.pos] + [n for n in neighbors4(a.pos, bp.height, bp.width) if self._walkable(n) and n not in burning]
            a.pos = options[int(self._dep_rng.integers(len(options)))]
            return

        def open_(c: Cell) -> bool:
            return bp.passable(c) and c not in burning and (c not in self.bumped_jams or c in self.broken_doors)

        def clear(c: Cell) -> bool:  # keeps out of panic range of the flames
            return open_(c) and all(manhattan(c, b) > PANIC_RADIUS for b in burning)

        for ok in (clear, open_):  # stay clear of the fire if any goal allows it, else take the direct way
            here = bfs_distances(a.pos, ok, bp.height, bp.width)
            kids = [d.pos for d in self.dependents if d.status == WAITING and not d.hidden and d.pos in here]
            exits = [e for e in bp.exits if e in here]
            if kids or exits:
                break
        goal = min(kids, key=here.get) if kids else min(exits, key=here.get, default=None)
        if goal is None or here[goal] <= (1 if kids else 0):
            return  # nowhere to go, already beside the child (they bond below), or out
        to_goal = bfs_distances(goal, ok, bp.height, bp.width)
        step = min((n for n in neighbors4(a.pos, bp.height, bp.width) if to_goal.get(n, 1 << 30) < to_goal[a.pos]), key=to_goal.get)
        if step in bp.doors and step not in self.broken_doors:
            blocked = step in self.rules.jammed_doors
            events.append(Event(self.t, door_key(step), blocked))
            if blocked:
                self.bumped_jams.add(step)
                a.panic_left = PANIC_TICKS
                return
        a.pos = step

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
