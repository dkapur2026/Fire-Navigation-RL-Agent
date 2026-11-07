"""Rules: the "physics" of the burning house, and what the agent believes about them.

One small data structure plays two roles:

* The world holds the TRUE rules.
* The agent holds BELIEVED rules (what it assumes when it plans).

A *novelty* is a ``RulePatch`` applied to the truth. A *repair* is a ``RulePatch``
applied to the belief. Detecting novelty means noticing the belief and the truth have
drifted apart; repairing means finding the patch that brings them back together.

Every rule is addressed by a string key and behaves like a Bernoulli probability, so the
same evidence format works for every rule:

    spread_prob:<material>   chance fire jumps from a burning cell to a neighbor of this material, per tick
    follow_prob:<kind>       chance a bonded dependent (dog/spouse/child) manages to follow, per allowed tick
    door_jammed:<r>,<c>      1.0 if that door cannot be opened, else 0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

MATERIALS = ("wood", "carpet", "tile", "door")
DEPENDENT_KINDS = ("dog", "spouse", "child")

# Tuned so a competent agent usually wins but a stale rulebook costs lives (see docs/ARCHITECTURE.md).
DEFAULT_SPREAD = {"wood": 0.10, "carpet": 0.16, "tile": 0.02, "door": 0.07}
DEFAULT_FOLLOW = {"dog": 0.70, "spouse": 0.95, "child": 0.90}
DEFAULT_BURN_STEPS = 10


def parse_key(key: str) -> tuple[str, str]:
    """Validate a rule key and split it into (family, name)."""
    if ":" not in key:
        raise ValueError(f"bad rule key {key!r}")
    family, name = key.split(":", 1)
    if family == "spread_prob" and name in MATERIALS:
        return family, name
    if family == "follow_prob" and name in DEPENDENT_KINDS:
        return family, name
    if family == "door_jammed":
        try:
            r, c = (int(x) for x in name.split(","))
        except ValueError as exc:
            raise ValueError(f"bad door key {key!r}") from exc
        return family, f"{r},{c}"
    raise ValueError(f"unknown rule key {key!r}")


def door_key(cell: tuple[int, int]) -> str:
    return f"door_jammed:{cell[0]},{cell[1]}"


def door_cell(key: str) -> tuple[int, int]:
    family, name = parse_key(key)
    if family != "door_jammed":
        raise ValueError(f"{key!r} is not a door key")
    r, c = (int(x) for x in name.split(","))
    return (r, c)


@dataclass(frozen=True)
class RulePatch:
    """One change to one rule."""

    key: str
    value: float | bool

    def __post_init__(self) -> None:
        family, _ = parse_key(self.key)
        if family == "door_jammed":
            if not isinstance(self.value, (bool, int)) or self.value not in (0, 1):
                raise ValueError("door_jammed patches take True/False")
        else:
            if isinstance(self.value, bool) or not 0.0 <= float(self.value) <= 1.0:
                raise ValueError(f"{self.key} needs a probability in [0, 1]")


@dataclass(frozen=True)
class Rules:
    spread_prob: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SPREAD))
    follow_prob: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FOLLOW))
    jammed_doors: frozenset[tuple[int, int]] = frozenset()
    burn_steps: int = DEFAULT_BURN_STEPS

    def prob(self, key: str) -> float:
        """The probability this rule currently assigns to a 'success' outcome."""
        family, name = parse_key(key)
        if family == "spread_prob":
            return self.spread_prob[name]
        if family == "follow_prob":
            return self.follow_prob[name]
        return 1.0 if door_cell(key) in self.jammed_doors else 0.0

    def with_patch(self, patch: RulePatch) -> "Rules":
        family, name = parse_key(patch.key)
        if family == "spread_prob":
            return replace(self, spread_prob={**self.spread_prob, name: float(patch.value)})
        if family == "follow_prob":
            return replace(self, follow_prob={**self.follow_prob, name: float(patch.value)})
        cell = door_cell(patch.key)
        doors = set(self.jammed_doors)
        doors.add(cell) if patch.value else doors.discard(cell)
        return replace(self, jammed_doors=frozenset(doors))

    def with_patches(self, patches: list[RulePatch] | tuple[RulePatch, ...]) -> "Rules":
        out = self
        for patch in patches:
            out = out.with_patch(patch)
        return out

    def diff(self, other: "Rules", doors: list[tuple[int, int]] | None = None) -> list[RulePatch]:
        """Patches that turn `self` into `other` (door keys limited to `doors` if given)."""
        patches: list[RulePatch] = []
        for m in MATERIALS:
            if self.spread_prob[m] != other.spread_prob[m]:
                patches.append(RulePatch(f"spread_prob:{m}", other.spread_prob[m]))
        for k in DEPENDENT_KINDS:
            if self.follow_prob[k] != other.follow_prob[k]:
                patches.append(RulePatch(f"follow_prob:{k}", other.follow_prob[k]))
        check = doors if doors is not None else sorted(self.jammed_doors | other.jammed_doors)
        for d in check:
            if (d in self.jammed_doors) != (d in other.jammed_doors):
                patches.append(RulePatch(door_key(d), d in other.jammed_doors))
        return patches


def nominal_rules() -> Rules:
    """What a freshly-deployed agent believes: the textbook rules of a house fire."""
    return Rules()
