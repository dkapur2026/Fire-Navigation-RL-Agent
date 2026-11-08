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
    autonomous: bool = False  # an adult who acts alone: fetches the nearest child and leads them out, else heads out;
    # panics (moves erratically for PANIC_TICKS) near flames or on finding a jammed door; joins the rescuer when reached


@dataclass(frozen=True)
class Household:
    label: str
    description: str
    dependents: tuple[DependentProfile, ...]


DOG = DependentProfile("dog", "Rex", "P", move_period=1, flees_fire=True, flee_prob=0.5)
SPOUSE = DependentProfile("spouse", "Sam", "S", autonomous=True)
KIDS = (DependentProfile("child", "Ada", "K", move_period=2), DependentProfile("child", "Max", "K", move_period=2))
_DOG = "The dog is fast but easily spooked: it may run from fire and follows unreliably."
_SPOUSE = ("Your spouse does not wait to be rescued: they go for the nearest child and lead them out, or head for the"
           " exit, but panic near flames or at a jammed door. If you reach them, they follow you reliably.")
_KIDS = "The children move at half speed, so the order you collect them matters."

# Who starts where follows this order: dependent i starts on blueprint slot i + 1.
HOUSEHOLDS: dict[str, Household] = {
    "single": Household("single", f"Rescue the dog. {_DOG}", (DOG,)),
    "married": Household("married", f"Your spouse. {_SPOUSE}", (SPOUSE,)),
    "kids": Household("kids", f"Rescue two children. {_KIDS}", KIDS),
    "kids_dog": Household("kids_dog", f"Two children and the dog. {_KIDS} {_DOG}", (*KIDS, DOG)),
    "family": Household("family", f"Your spouse and two children. {_SPOUSE} {_KIDS}", (SPOUSE, *KIDS)),
    "family_dog": Household("family_dog", f"Spouse, two children and the dog. {_SPOUSE} {_KIDS} {_DOG}", (SPOUSE, *KIDS, DOG)),
}

# The unexpected dependent of the `hidden_dependent` novelty: a neighbour's child on a sleepover, hiding
# (closet, under a bed). Moves at a child's pace; does not bolt, so it stays where it hid until found.
VISITOR = DependentProfile("child", "Lily", "V", move_period=2)
