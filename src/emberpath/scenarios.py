"""Novelty presets: ways the world can change under the agent's feet.

A novelty is a ``RulePatch`` applied to the TRUE rules at a chosen tick. The agent is never
told. Presets are built per blueprint so they always leave the house solvable.
"""

from __future__ import annotations

from typing import Callable

from emberpath.blueprint import Blueprint
from emberpath.dependents import Household
from emberpath.grid import Cell, neighbors4
from emberpath.rules import RulePatch, door_key, occupied_key
from emberpath.world import NoveltySpec


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


def door_jam(bp: Blueprint, household: Household, step: int = 1) -> list[NoveltySpec]:
    """A door on the rescuer's mission route (to the first dependent, then out) jams shut.

    Chosen so that an alternative route still exists (the house stays solvable)."""
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


def carpet_flashover(bp: Blueprint, household: Household, step: int = 2) -> list[NoveltySpec]:
    """Carpet suddenly burns several times faster (think: synthetic fibres).

    In the ranch the fire starts on living-room carpet beside the front exit, so this changes
    which exit is the safe one: a decision the agent can only get right if its belief is right."""
    return [
        NoveltySpec(
            step,
            RulePatch("spread_prob:carpet", 0.60),
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


def compound(bp: Blueprint, household: Household) -> list[NoveltySpec]:
    """Two different novelties at different times."""
    return door_jam(bp, household, step=1) + carpet_flashover(bp, household, step=10)


NOVELTIES: dict[str, Callable[[Blueprint, Household], list[NoveltySpec]]] = {
    "door_jam": door_jam,
    "carpet_flashover": carpet_flashover,
    "compound": compound,
    "hidden_dependent": hidden_dependent,
}


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
