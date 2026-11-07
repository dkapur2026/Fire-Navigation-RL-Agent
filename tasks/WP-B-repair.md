# WP-B: Rule repair search  (A2)

**Goal.** Given suspect rules and the evidence, find the *smallest edit to the rulebook that explains what was seen*, and say
how sure you are. This is a small program-synthesis problem: search a space of `RulePatch` edits and keep the best by score.

## Read first
`src/emberpath/repair.py` (design in its docstring), `rules.py` (`RulePatch`, `Rules.with_patch`, `Rules.diff`), `novelty.py`
(`Finding`), `tests/test_acceptance_future.py` (repair tests), `docs/REWARD.md` only for tone.

## Deliverables
1. `RepairSearch.repair` in `repair.py`.
   * Hypothesis space: single `RulePatch` edits for flagged keys. `spread_prob` / `follow_prob`: candidate values on a grid
     (0.05 steps) plus the posterior mean. `door_jammed`: toggle each door that has bump evidence.
   * Score = log-likelihood of **all** stored events for the edited keys under the patched rules, minus
     `complexity_penalty` per edit. Likelihood of a Bernoulli outcome under probability `p` (clip away from 0 and 1).
   * If single edits leave strong residual surprise, search pairs (bounded by `max_edits`).
   * Return the patched `Rules`, the list of patches, the log-likelihood gain, and the margin over the runner-up hypothesis.
   * Ambiguity: when two hypotheses are within a small margin, say so. Add an optional `ambiguous: bool = False` field to
     `RepairResult` (allowed: optional field with a default) and make the agent's explanation use it.
2. Tests, building on the acceptance ones: single novelty recovered with the right value (within tolerance); **two simultaneous
   novelties both recovered**; a no-op when findings are empty; ambiguity flagged for deliberately confounded evidence; the
   smallest explanation wins (one patch, not several).
3. Property test: applying the returned patches to the nominal rules gives rules whose `prob(key)` is within tolerance of the
   truth that generated the events.

## Done when
```bash
python -m pytest tests/test_acceptance_future.py -q -k "repair"   # xfail markers removed; pass
python -m pytest -q
```
You can test with synthetic events (Bernoulli draws from a patched `Rules`), so this package does not need WP-A finished.

## Constraints
Pure Python + numpy. Deterministic (no unseeded randomness). Do not change `Finding`; it is WP-A's interface.

## Docs
"Repair" section in `docs/ARCHITECTURE.md` with the hypothesis space and scoring rule. Flip A2 in `docs/SPEC.md`.
