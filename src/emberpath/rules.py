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
    force_prob:door          chance the rescuer breaks open a jammed door, per tick spent forcing it
    occupied:<r>,<c>         1.0 if an UNEXPECTED dependent (not in the declared household) is at that cell, else 0.0

door_jammed and occupied are the BOOLEAN_FAMILIES: near-deterministic, patched with True/False.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

MATERIALS = ("wood", "carpet", "tile", "door")
DEPENDENT_KINDS = ("dog", "spouse", "child")

# Tuned so a competent agent usually wins but a stale rulebook costs lives (see docs/ARCHITECTURE.md).
DEFAULT_SPREAD = {"wood": 0.10, "carpet": 0.16, "tile": 0.02, "door": 0.07}
DEFAULT_FOLLOW = {"dog": 0.70, "spouse": 0.95, "child": 0.90}
DEFAULT_FORCE = {"door": 0.10}  # about 10 ticks to force a jammed door: a real cost to weigh against a detour
DEFAULT_BURN_STEPS = 10
BOOLEAN_FAMILIES = ("door_jammed", "occupied")


def parse_key(key: str) -> tuple[str, str]:
    """Validate a rule key and split it into (family, name)."""
    if ":" not in key:
        raise ValueError(f"bad rule key {key!r}")
    family, name = key.split(":", 1)
    if family == "spread_prob" and name in MATERIALS:
        return family, name
    if family == "follow_prob" and name in DEPENDENT_KINDS:
        return family, name
    if family == "force_prob" and name in DEFAULT_FORCE:
        return family, name
    if family in BOOLEAN_FAMILIES:
        try:
            r, c = (int(x) for x in name.split(","))
        except ValueError as exc:
            raise ValueError(f"bad cell key {key!r}") from exc
        return family, f"{r},{c}"
    raise ValueError(f"unknown rule key {key!r}")


def door_key(cell: tuple[int, int]) -> str:
    return f"door_jammed:{cell[0]},{cell[1]}"


def occupied_key(cell: tuple[int, int]) -> str:
    return f"occupied:{cell[0]},{cell[1]}"


def key_cell(key: str) -> tuple[int, int]:
    """The cell a boolean-family key refers to."""
    family, name = parse_key(key)
    if family not in BOOLEAN_FAMILIES:
        raise ValueError(f"{key!r} is not a cell key")
    r, c = (int(x) for x in name.split(","))
    return (r, c)


def door_cell(key: str) -> tuple[int, int]:
    if parse_key(key)[0] != "door_jammed":
        raise ValueError(f"{key!r} is not a door key")
    return key_cell(key)


@dataclass(frozen=True)
class RulePatch:
    """One change to one rule."""

    key: str
    value: float | bool

    def __post_init__(self) -> None:
        family, _ = parse_key(self.key)
        if family in BOOLEAN_FAMILIES:
            if not isinstance(self.value, (bool, int)) or self.value not in (0, 1):
                raise ValueError(f"{family} patches take True/False")
        else:
            if isinstance(self.value, bool) or not 0.0 <= float(self.value) <= 1.0:
                raise ValueError(f"{self.key} needs a probability in [0, 1]")


@dataclass(frozen=True)
class Rules:
    spread_prob: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SPREAD))
    follow_prob: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FOLLOW))
    force_prob: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FORCE))
    jammed_doors: frozenset[tuple[int, int]] = frozenset()
    occupied: frozenset[tuple[int, int]] = frozenset()  # cells holding an unexpected dependent
    burn_steps: int = DEFAULT_BURN_STEPS

    def prob(self, key: str) -> float:
        """The probability this rule currently assigns to a 'success' outcome."""
        family, name = parse_key(key)
        if family == "spread_prob":
            return self.spread_prob[name]
        if family == "follow_prob":
            return self.follow_prob[name]
        if family == "force_prob":
            return self.force_prob[name]
        cells = self.jammed_doors if family == "door_jammed" else self.occupied
        return 1.0 if key_cell(key) in cells else 0.0

    def with_patch(self, patch: RulePatch) -> "Rules":
        family, name = parse_key(patch.key)
        if family == "spread_prob":
            return replace(self, spread_prob={**self.spread_prob, name: float(patch.value)})
        if family == "follow_prob":
            return replace(self, follow_prob={**self.follow_prob, name: float(patch.value)})
        if family == "force_prob":
            return replace(self, force_prob={**self.force_prob, name: float(patch.value)})
        field_name = "jammed_doors" if family == "door_jammed" else "occupied"
        cells = set(getattr(self, field_name))
        cell = key_cell(patch.key)
        cells.add(cell) if patch.value else cells.discard(cell)
        return replace(self, **{field_name: frozenset(cells)})

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
        for k in DEFAULT_FORCE:
            if self.force_prob[k] != other.force_prob[k]:
                patches.append(RulePatch(f"force_prob:{k}", other.force_prob[k]))
        check = doors if doors is not None else sorted(self.jammed_doors | other.jammed_doors)
        for d in check:
            if (d in self.jammed_doors) != (d in other.jammed_doors):
                patches.append(RulePatch(door_key(d), d in other.jammed_doors))
        for c in sorted(self.occupied ^ other.occupied):
            patches.append(RulePatch(occupied_key(c), c in other.occupied))
        return patches


def nominal_rules() -> Rules:
    """What a freshly-deployed agent believes: the textbook rules of a house fire."""
    return Rules()
