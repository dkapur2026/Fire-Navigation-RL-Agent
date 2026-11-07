"""Households: who the rescuer is trying to get out.

A household is just a labelled list of dependents. The *labels* ("single", "married",
"kids") are for humans; to the simulator each dependent is a point with a few dynamics:
how fast it moves, whether it bolts from fire, and (via the rules) how reliably it follows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DependentProfile:
    kind: str  # "dog" | "spouse" | "child" (keys into Rules.follow_prob)
    name: str
    glyph: str  # one letter for renders
    move_period: int = 1  # may move once every N ticks (children are slower)
    flees_fire: bool = False  # waiting dependents may run away from nearby fire
    flee_prob: float = 0.0


@dataclass(frozen=True)
class Household:
    label: str
    description: str
    dependents: tuple[DependentProfile, ...]


HOUSEHOLDS: dict[str, Household] = {
    "single": Household(
        "single",
        "Rescue the dog. Fast, but easily spooked: it may run from fire and follows unreliably.",
        (DependentProfile("dog", "Rex", "P", move_period=1, flees_fire=True, flee_prob=0.5),),
    ),
    "married": Household(
        "married",
        "Rescue your spouse. Normal speed and follows reliably.",
        (DependentProfile("spouse", "Sam", "S"),),
    ),
    "kids": Household(
        "kids",
        "Rescue two children. They move at half speed, so the order you collect them matters.",
        (
            DependentProfile("child", "Ada", "K", move_period=2),
            DependentProfile("child", "Max", "K", move_period=2),
        ),
    ),
}
