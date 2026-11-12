"""Novelty presets: ways the world can change under the agent's feet.

A novelty is a ``RulePatch`` applied to the TRUE rules at a chosen tick. The agent is never
told. Presets are built per blueprint so they always leave the house solvable.

A preset returns ``NoveltySpec``s, or a ``SeededNovelty`` when what changes depends on the episode seed (the
multi-door jam: which doors jam). ``World`` resolves a ``SeededNovelty`` with its own seed, so callers keep calling
``NOVELTIES[name](bp, household)`` and passing the result to ``World(..., seed=seed)``. ``novelty_specs`` gives the
concrete changes for a seed.
"""

from __future__ import annotations

import functools
import heapq
import itertools
import math
import os
from typing import Callable

import numpy as np

from emberpath.blueprint import Blueprint
from emberpath.dependents import Household
from emberpath.grid import Cell, neighbors4
from emberpath.rules import DEFAULT_FORCE, RulePatch, door_key, occupied_key
from emberpath.world import NoveltySpec, SeededNovelty, resolve_novelties


def _shortest_path(bp: Blueprint, src: Cell, dst: Cell, jammed=frozenset()) -> list[Cell]:
    dist = bp.reachable(src, jammed)
    if dst not in dist:
        return []
    path = [dst]
    while path[-1] != src:
        cur = path[-1]
        path.append(
            min(n for n in neighbors4(cur, bp.height, bp.width) if dist.get(n) == dist[cur] - 1)
        )
    return path[::-1]


def _mission_route(bp: Blueprint) -> list[Cell]:
    """Shortest route: rescuer -> first dependent -> nearest exit (ignoring fire)."""
    first = bp.slots[0]
    from_first = bp.reachable(first)
    exit_cell = min((e for e in bp.exits if e in from_first), key=lambda e: from_first[e])
    return _shortest_path(bp, bp.agent_start, first) + _shortest_path(bp, first, exit_cell)


def door_jam_single(bp: Blueprint, household: Household, step: int = 1) -> list[NoveltySpec]:
    """LEGACY: the `door_jam` of every study before 2026-11-09, including the pre-registered ranch formal study in
    runs/formal_eval. ONE door, the first on the rescuer's mission route (to the first dependent, then out), jams
    shut, the same door on every seed. Chosen so that an alternative route still exists (the house stays solvable)."""
    n = len(household.dependents)
    for door in _mission_route(bp):
        if door not in bp.doors:
            continue
        try:
            bp.validate(n, frozenset({door}))
        except ValueError:
            continue
        return [
            NoveltySpec(
                step,
                RulePatch(door_key(door), True),
                f"The door at {door} jams shut.",
            )
        ]
    raise ValueError(f"{bp.name}: no door can be jammed while keeping the house solvable")


# ----------------------------------------------------------------------------------------- multi-door jam
# Spawn key of the jam draw's own random stream. World spawns keys 0..3 of SeedSequence(seed) (fire, dependents,
# forcing, burn tolerance); this key is far from those, so the draw never shares or shifts a world stream (the fire
# stays identical), and it depends on the seed alone, never on anything an agent does.
JAM_STREAM = 0xD00A
_REJECTION_TRIES = 64  # random draws before enumerating every valid set (the distribution is the same either way)


def jam_count_range(bp: Blueprint) -> tuple[int, int]:
    """How many doors jam, by house size: uniform in [max(1, doors // 6), max(1, doors // 3)].

    studio (2 doors): 1; ranch (6 doors): 1-2; house (12 doors): 2-4; a procedural house in proportion to its doors.
    Fewer only if no set of the drawn size keeps the house solvable, which never happens on the three hand-drawn
    houses (tests/test_door_jam_multi.py)."""
    d = len(bp.doors)
    return max(1, d // 6), max(1, d // 3)


def jam_rng(seed: int) -> np.random.Generator:
    """The jam draw's random stream for this seed (see JAM_STREAM)."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(JAM_STREAM,)))


def _within(bp: Blueprint, cell: Cell, steps: int) -> set[Cell]:
    """Cells a walker starting at `cell` can stand on after `steps` ticks, before any jam (walls only)."""
    return {c for c, d in bp.reachable(cell).items() if d <= steps}


def _step_off(bp: Blueprint, cells: set[Cell], jammed: frozenset[Cell]) -> set[Cell]:
    """Someone standing on a door as it jams can only step off it, to either side: replace such cells by their open
    neighbours (each must work, since we cannot know which side they pick)."""
    out = {c for c in cells if c not in jammed}
    for c in cells & jammed:
        out |= {n for n in neighbors4(c, bp.height, bp.width) if bp.passable(n) and n not in jammed}
    return out


def jam_set_ok(bp: Blueprint, n: int, jammed: frozenset[Cell], step: int = 1, forcing_never_pays: bool = True) -> bool:
    """Does the house stay solvable with these doors jammed, WITHOUT forcing any of them?

    1. A rescue exists: the rescuer's jam-free region at the start holds an exit and every place one of the first `n`
       dependents can be standing when the jam lands, `step` ticks in (someone on an exit is already out). The
       region is closed under everyone's movement (nobody crosses a jammed door until it is broken, and breaking one
       only opens more), so the dependents stay inside it.
    2. Never sealed in: wherever the rescuer can be standing when the jam lands, including on a door that jams under
       it (it can step off to either side), its jam-free region holds an exit. A rescuer that walked into a door at
       tick 0 may thereby cut itself off from someone; it can always get out.

    3. Forcing never pays: from the start to each dependent, and from each dependent to its nearest exit, the way
       round the jams is no longer than the best way that kicks jammed doors in from the hallway side (each costing
       its expected 1 / force_prob = 10 tries).

    Deliberately stronger than "solvable by forcing doors from the hallway side". The planner forces a door only to
    reach an exit, never to reach a person, and only from where it stands. So a person reachable only by forcing (1)
    or much faster by forcing (3) makes the oracle, which knows the jam, take the long way round, while the baseline
    walks into the door and kicks it in: on the ranch, jamming the hallway door of the dog's room (8,4) gave oracle
    0.00 vs baseline 0.60 on `single` (first draft, 2026-11-09). The oracle-baseline gap would then measure that
    planner limit rather than the value of knowing the jam."""
    need = _step_off(bp, set().union(*(_within(bp, s, step) for s in bp.slots[:n])), jammed) - set(bp.exits)
    start = bp.reachable(bp.agent_start, jammed)
    if not need <= start.keys() or not any(e in start for e in bp.exits):
        return False
    for p in _step_off(bp, _within(bp, bp.agent_start, step), jammed):
        if not any(e in bp.reachable(p, jammed) for e in bp.exits):
            return False
    # 3. Forcing never beats the detour (see the docstring's last paragraph): from the start to each dependent and from
    #    each dependent to the nearest exit, the way round is no longer than the way through, kicking doors in.
    forcible = {d: bp.door_outside[d] for d in jammed if d in bp.door_outside}
    if not forcible or not forcing_never_pays:
        return True
    exits = set(bp.exits)
    for slot in bp.slots[:n]:
        if start[slot] > _forcing_dist(bp, jammed, forcible, bp.agent_start, {slot}):
            return False
        around = bp.reachable(slot, jammed)
        if min(around.get(e, math.inf) for e in exits) > _forcing_dist(bp, jammed, forcible, slot, exits):
            return False
    return True


def _forcing_dist(bp: Blueprint, jammed, forcible: dict, src: Cell, goals: set[Cell]) -> float:
    """Ticks from `src` to the nearest goal when a jammed door may be kicked in from its hallway side (expected
    1 / force_prob tries, nominal rules: what the world allows; walking into it from inside a room never works)."""
    force = 1.0 / DEFAULT_FORCE["door"]
    best, todo = {src: 0.0}, [(0.0, src)]
    while todo:
        d, c = heapq.heappop(todo)
        if c in goals:
            return d
        if d > best[c]:
            continue
        for nb in neighbors4(c, bp.height, bp.width):
            if not bp.passable(nb):
                continue
            if nb in jammed:
                if c not in forcible.get(nb, ()):
                    continue
                step = 1.0 + force
            else:
                step = 1.0
            if d + step < best.get(nb, math.inf):
                best[nb] = d + step
                heapq.heappush(todo, (d + step, nb))
    return math.inf


def _route_doors(bp: Blueprint) -> frozenset[Cell]:
    return frozenset(c for c in _mission_route(bp) if c in bp.doors)


_VALID: dict[tuple, tuple[tuple[Cell, ...], ...]] = {}


def _valid_sets(bp: Blueprint, n: int, k: int, step: int, strict: bool = True) -> tuple[tuple[Cell, ...], ...]:
    """Every valid k-door jam set, in a fixed order (cached per house: its rows determine everything)."""
    key = (bp.rows, n, k, step, strict)
    if key not in _VALID:
        route = _route_doors(bp)
        _VALID[key] = tuple(
            s for s in itertools.combinations(sorted(bp.doors), k)
            if route.intersection(s) and jam_set_ok(bp, n, frozenset(s), step, forcing_never_pays=strict)
        )
    return _VALID[key]


def draw_jammed_doors(bp: Blueprint, n: int, seed: int, step: int = 1) -> tuple[Cell, ...]:
    """Which doors jam on this seed: deterministic in (house, household size, seed), drawn from `jam_rng(seed)` only.

    k ~ uniform on `jam_count_range`; then a uniformly random k-set of doors, redrawn until it (a) includes at least
    one door on the mission route, so the jam is in the way as the legacy single jam always was, and (b) passes
    `jam_set_ok`. After `_REJECTION_TRIES` misses, uniformly among all valid k-sets; if there are none, k - 1.

    If no set at all passes `jam_set_ok`'s condition 3 (on about 5% of procedural houses forcing beats the detour for
    every door on the mission route), the same search without it: conditions 1-2 still hold, as they did for the
    legacy single jam, but there the oracle may take the long way round (the planner limit in `jam_set_ok`). Never
    on the hand-drawn houses (tests/test_door_jam_multi.py)."""
    rng = jam_rng(seed)
    lo, hi = jam_count_range(bp)
    doors, route = sorted(bp.doors), _route_doors(bp)
    k = min(int(rng.integers(lo, hi + 1)), len(doors))
    for _ in range(_REJECTION_TRIES):
        pick = frozenset(doors[i] for i in rng.choice(len(doors), size=k, replace=False))
        if route & pick and jam_set_ok(bp, n, pick, step):
            return tuple(sorted(pick))
    for strict in (True, False):
        for kk in range(k, 0, -1):
            valid = _valid_sets(bp, n, kk, step, strict)
            if valid:
                return valid[int(rng.integers(len(valid)))]
    raise ValueError(f"{bp.name}: no door can be jammed while keeping the house solvable")


def door_jam_multi(bp: Blueprint, household: Household, step: int = 1, seed: int | None = None) -> list:
    """Several doors jam shut at tick `step`: how many by house size (`jam_count_range`), which ones drawn per seed
    (`draw_jammed_doors`), always leaving the house solvable (`jam_set_ok`).

    With `seed`: the concrete specs, one per door. Without: one `SeededNovelty` that `World` resolves with its own
    seed. Raises ValueError (checked eagerly, seed-free) if no door on the mission route can jam alone, since then
    no seed has a valid draw."""
    n = len(household.dependents)
    if seed is not None:
        return [NoveltySpec(step, RulePatch(door_key(d), True), f"The door at {d} jams shut.")
                for d in draw_jammed_doors(bp, n, seed, step)]
    if not _valid_sets(bp, n, 1, step, strict=False):
        raise ValueError(f"{bp.name}: no door can be jammed while keeping the house solvable")
    lo, hi = jam_count_range(bp)
    return [SeededNovelty(step, functools.partial(_draw_specs, bp, household, step),
                          f"{lo}-{hi} doors jam shut; which ones depends on the seed.")]


def _draw_specs(bp: Blueprint, household: Household, step: int, seed: int) -> list[NoveltySpec]:
    return door_jam_multi(bp, household, step, seed=seed)


def carpet_flashover(bp: Blueprint, household: Household, step: int = 2) -> list[NoveltySpec]:
    """Carpet suddenly burns faster: 45% per tick instead of 16% (think: synthetic fibres). 60% made carpeted rooms
    on the house unsurvivable for every agent; 40% barely separated baseline from oracle; 50% was tried too.

    In the ranch the fire starts on living-room carpet beside the front exit, so this changes
    which exit is the safe one: a decision the agent can only get right if its belief is right."""
    return [
        NoveltySpec(
            step,
            RulePatch("spread_prob:carpet", 0.45),
            "Carpet now flashes over: fire races across it.",
        )
    ]


def hidden_dependent(bp: Blueprint, household: Household, step: int = 0) -> list[NoveltySpec]:
    """Someone the household list does not mention (a visiting child) is hiding in the house.

    Placed in the first blueprint slot the household does not use that the rescuer can reach, from
    the start. Semi-structural: a new *person*, but expressed in the rule grammar as `occupied:<cell>`."""
    n = len(household.dependents)
    bp.validate(n)
    reachable = bp.reachable(bp.agent_start)
    for slot in bp.slots[n:]:
        if slot in reachable:
            return [
                NoveltySpec(
                    step,
                    RulePatch(occupied_key(slot), True),
                    f"A visiting child is hiding at {slot}.",
                )
            ]
    raise ValueError(f"{bp.name}: no spare reachable dependent slot for a hidden dependent")


def compound_multi(bp: Blueprint, household: Household) -> list:
    """Two different novelties at different times: the multi-door jam at tick 1, carpet flashover at tick 10."""
    return door_jam_multi(bp, household, step=1) + carpet_flashover(bp, household, step=10)


def compound_single(bp: Blueprint, household: Household) -> list[NoveltySpec]:
    """LEGACY compound (before 2026-11-09): the single-door jam at tick 1, carpet flashover at tick 10."""
    return door_jam_single(bp, household, step=1) + carpet_flashover(bp, household, step=10)


# Which jam `door_jam` and `compound` mean: "multi" from 2026-11-09, "single" before. EMBERPATH_DOOR_JAM=single
# restores the old presets under their old names, so a study run before the change (the pre-registered ranch formal
# study in runs/formal_eval used the single jam) re-runs unchanged. The legacy presets are also importable as
# `door_jam_single` / `compound_single` (`LEGACY_NOVELTIES`). They are not NOVELTIES keys: every loop over NOVELTIES
# (the dataset's job residues, the lockstep tests, the exploit search) would otherwise change.
DEFAULT_DOOR_JAM = "multi"
DOOR_JAM_VERSION = os.environ.get("EMBERPATH_DOOR_JAM", DEFAULT_DOOR_JAM)
if DOOR_JAM_VERSION not in ("single", "multi"):
    raise ValueError(f"EMBERPATH_DOOR_JAM must be 'single' or 'multi', not {DOOR_JAM_VERSION!r}")

if DOOR_JAM_VERSION == "multi":
    door_jam, compound = door_jam_multi, compound_multi
else:
    door_jam, compound = door_jam_single, compound_single
NOVELTIES: dict[str, Callable[[Blueprint, Household], list]] = {
    "door_jam": door_jam,
    "carpet_flashover": carpet_flashover,
    "compound": compound,
    "hidden_dependent": hidden_dependent,
}
LEGACY_NOVELTIES: dict[str, Callable[[Blueprint, Household], list]] = {
    "door_jam_single": door_jam_single,
    "compound_single": compound_single,
}


def novelty_specs(name: str, bp: Blueprint, household: Household, seed: int) -> list[NoveltySpec]:
    """The concrete rule changes preset `name` makes on this seed (what `World(..., seed=seed)` applies)."""
    return resolve_novelties(NOVELTIES[name](bp, household), seed)


def applicable(bp: Blueprint, household: Household, novelty: str | None = None) -> bool:
    """Can this house host this household (enough slots, solvable) and this novelty? A larger household or a
    novelty that needs a spare slot does not fit every house; callers skip those combinations rather than fail."""
    try:
        bp.validate(len(household.dependents))
        if novelty:
            NOVELTIES[novelty](bp, household)
    except ValueError:
        return False
    return True
