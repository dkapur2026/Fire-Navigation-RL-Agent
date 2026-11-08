"""Phase 5: work out *what* changed and patch the agent's beliefs.

The detector says "something is off with these rules". The repairer answers "what is the
smallest edit to my rulebook that explains what I saw?" That is a tiny program-synthesis
problem: search a space of rule edits and keep the one with the best score.

Design
------
* Hypothesis space. Only flagged keys that have evidence are edited.
  - spread_prob / follow_prob: every value on a 0.05 grid, plus the detector's posterior mean.
  - BOOLEAN_FAMILIES (door_jammed, occupied): the toggle of the current belief.
  A hypothesis is a set of edits on distinct keys: the empty set (keep the belief), singles, and,
  only when the best single leaves another key still surprising, sets of up to ``max_edits``.
* Score = log-likelihood of every stored event for the edited keys under the patched rules,
  minus ``complexity_penalty`` per edit (Occam: one strong patch beats a pile of weak ones).
  Events enter only through per-key (successes, trials) counts, aggregated once per call, so
  scoring a hypothesis costs O(#keys) however long the episode was. Keys are independent
  Bernoulli coins, so the score is a sum of per-key gains and each key's best value does not
  depend on which other keys are edited.
* Confidence. ``runner_up_margin`` is the score gap to the best hypothesis that edits a
  *different set of keys* (neighbouring grid values for the same key are the same explanation,
  only less precise; the finding's interval already says how precise). When that gap is below
  ``ambiguity_margin`` the result is ``ambiguous``: the agent should say it is unsure rather than
  present a guess as fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from emberpath.novelty import Finding
from emberpath.rules import BOOLEAN_FAMILIES, RulePatch, Rules, parse_key
from emberpath.world import Event

_EPS = 1e-3  # keeps log-likelihoods finite for p in {0, 1} (doors, hearing, a grid value of 0)
_GRID = tuple(round(0.05 * i, 2) for i in range(21))


@dataclass(frozen=True)
class RepairResult:
    rules: Rules
    patches: list[RulePatch]
    log_likelihood_gain: float  # raw log-likelihood improvement of the chosen edits (no penalty)
    runner_up_margin: float  # score gap to the best different explanation; inf if there is none
    ambiguous: bool = False  # True when that gap is too small to call
    runner_up: tuple[RulePatch, ...] = ()  # the edits of that best different explanation (() = "nothing changed")


def _log_lik(successes: int, trials: int, p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return successes * np.log(p) + (trials - successes) * np.log1p(-p)


def _best_edit(belief: Rules, finding: Finding, s: int, n: int) -> tuple[RulePatch, float]:
    """The single edit of this key that most improves the log-likelihood, and that gain."""
    believed = belief.prob(finding.key)
    boolean = parse_key(finding.key)[0] in BOOLEAN_FAMILIES
    if boolean:
        values = np.array([1.0 - believed])
    else:
        values = np.array([v for v in (*_GRID, finding.posterior_mean) if v != believed])
    gains = _log_lik(s, n, values) - _log_lik(s, n, believed)
    i = int(np.argmax(gains))
    return RulePatch(finding.key, bool(values[i]) if boolean else float(values[i])), float(gains[i])


class RepairSearch:
    def __init__(self, max_edits: int = 2, complexity_penalty: float = 2.0, ambiguity_margin: float = 1.0):
        self.max_edits = max_edits
        self.complexity_penalty = complexity_penalty
        self.ambiguity_margin = ambiguity_margin

    def repair(self, belief: Rules, findings: list[Finding], events: list[Event]) -> RepairResult:
        counts: dict[str, list[int]] = {}
        for e in events:
            c = counts.setdefault(e.key, [0, 0])
            c[0] += e.outcome
            c[1] += 1

        # One best edit per flagged key with evidence (findings order, duplicates dropped).
        edits: dict[str, tuple[RulePatch, float]] = {}
        for f in findings:
            if f.key in counts and f.key not in edits:
                edits[f.key] = _best_edit(belief, f, *counts[f.key])

        def score(keys: tuple[str, ...]) -> float:
            return sum(edits[k][1] for k in keys) - self.complexity_penalty * len(keys)

        hypotheses: list[tuple[str, ...]] = [(), *((k,) for k in edits)]
        best_single = max(hypotheses, key=score)
        if any(score((k,)) > 0 for k in edits if (k,) != best_single):  # residual surprise
            for size in range(2, self.max_edits + 1):
                hypotheses.extend(combinations(edits, size))

        ranked = sorted(hypotheses, key=score, reverse=True)  # stable: on a tie the simpler one wins
        best = ranked[0]
        margin = score(best) - score(ranked[1]) if len(ranked) > 1 else math.inf
        patches = [edits[k][0] for k in best]
        return RepairResult(
            rules=belief.with_patches(patches),
            patches=patches,
            log_likelihood_gain=sum(edits[k][1] for k in best),
            runner_up_margin=margin,
            ambiguous=margin < self.ambiguity_margin,
            runner_up=tuple(edits[k][0] for k in ranked[1]) if len(ranked) > 1 else (),
        )
