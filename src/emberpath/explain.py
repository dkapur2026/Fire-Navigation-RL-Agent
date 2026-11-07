"""Phase 6: turn what the agent knows and did into words a person can read.

Working now: ``describe_patch`` (a rule change in plain English) and ``DecisionLog``.
TODO (Phase 6): ``explain_adaptation`` for the full "what changed, how sure, what I did" report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from emberpath.planner import Decision
from emberpath.rules import RulePatch, Rules, door_cell, parse_key

_MATERIAL_WORDS = {"wood": "Wood floors", "carpet": "Carpet", "tile": "Tile floors", "door": "Doors"}
_KIND_WORDS = {"dog": "The dog", "spouse": "Your spouse", "child": "The children"}


def describe_patch(patch: RulePatch, before: Rules | None = None) -> str:
    """One rule change, in plain English, e.g. 'Carpet now catches fire ... (was 45%).'"""
    family, name = parse_key(patch.key)
    if family == "spread_prob":
        was = f" (was {before.spread_prob[name]:.0%})" if before else ""
        return (
            f"{_MATERIAL_WORDS[name]} now catch fire {float(patch.value):.0%} of the time "
            f"next to a flame{was}."
        )
    if family == "follow_prob":
        was = f" (was {before.follow_prob[name]:.0%})" if before else ""
        return f"{_KIND_WORDS[name]} now follows only {float(patch.value):.0%} of the time{was}."
    cell = door_cell(patch.key)
    return f"The door at {cell} is {'jammed shut' if patch.value else 'working again'}."


@dataclass
class DecisionLog:
    """An audit trail: every decision with its stated reason."""

    entries: list[tuple[int, Decision]] = field(default_factory=list)

    def add(self, step: int, decision: Decision) -> None:
        self.entries.append((step, decision))

    def narrate(self, collapse_repeats: bool = True) -> list[str]:
        lines: list[str] = []
        last = None
        for step, d in self.entries:
            if collapse_repeats and d.reason == last:
                continue
            lines.append(f"t={step:>3}  {d.reason}")
            last = d.reason
        return lines


def explain_adaptation(*args, **kwargs) -> str:
    """PHASE 6 — build the end-of-episode report.

    Input: the detector findings, the repair patches (with before/after beliefs), and the
    DecisionLog. Output: a short paragraph a non-expert can read, plus a technical appendix
    (posterior, log Bayes factor, number of trials). See docs/ROADMAP.md, Phase 6.
    """
    raise NotImplementedError("Phase 6: explain_adaptation is not built yet.")
