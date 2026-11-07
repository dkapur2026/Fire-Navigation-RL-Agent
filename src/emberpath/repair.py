"""Phase 5: work out *what* changed and patch the agent's beliefs.

STATUS: interface only. Build this after the detector (see docs/ROADMAP.md, Phase 5).

The detector says "something is off with these rules". The repairer answers "what is the
smallest edit to my rulebook that explains what I saw?" That is a tiny program-synthesis
problem: search a space of rule edits and keep the one with the best score.

Suggested design
----------------
* Hypothesis space: single ``RulePatch`` edits, then pairs if singles do not explain the data.
  - spread_prob / follow_prob: candidate values on a grid (e.g. 0.05 steps) or the posterior mean.
  - door_jammed: toggle for each door that has bump evidence.
* Score: log-likelihood of ALL stored events under the patched rules, minus a complexity
  penalty per edit (Occam: prefer the fewest, smallest changes). Keep the top-k for diagnostics.
* Return the patched ``Rules`` and the list of ``RulePatch`` es, plus the margin over the
  runner-up hypothesis (this is the agent's confidence).
* Edge cases worth testing: two simultaneous novelties, a novelty that *reverts*, evidence that
  is ambiguous between two explanations (agent should say so, not guess).
"""

from __future__ import annotations

from dataclasses import dataclass

from emberpath.novelty import Finding
from emberpath.rules import RulePatch, Rules
from emberpath.world import Event


@dataclass(frozen=True)
class RepairResult:
    rules: Rules
    patches: list[RulePatch]
    log_likelihood_gain: float
    runner_up_margin: float


class RepairSearch:
    """TODO(Phase 5): implement per the module docstring."""

    def __init__(self, max_edits: int = 2, complexity_penalty: float = 2.0):
        self.max_edits = max_edits
        self.complexity_penalty = complexity_penalty

    def repair(self, belief: Rules, findings: list[Finding], events: list[Event]) -> RepairResult:
        raise NotImplementedError("Phase 5: RepairSearch.repair")
