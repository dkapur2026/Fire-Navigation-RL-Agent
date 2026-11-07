# WP-F: Wire the adaptive agent and run the evaluation study  (A3, A4, E4, E5, E6)

**Goal.** Make `adaptive` work end to end and prove it with numbers: how much of the baseline-to-oracle gap it closes, how
fast, at what false-alarm cost.

## Read first
`src/emberpath/agents.py` (`AdaptiveAgent` skeleton is already wired), `evaluate.py`, `scripts/make_results.py`, `docs/results.md`,
`docs/ARCHITECTURE.md`. Needs WP-A and WP-B merged.

## Deliverables
1. `AdaptiveAgent` working with the real detector and repairer. Fix whatever integration issues appear (belief hand-off to the
   planner, re-baselining the detector after a repair, handling a second novelty).
2. Study, over **at least 100 seeds** per cell, for every household x novelty (`none`, `door_jam`, `carpet_flashover`, `compound`):
   * mean score, success rate, `gap_closed`, each with a **bootstrap 95% confidence interval**;
   * **recovery curve:** score as a function of observations since the novelty;
   * **false-alarm rate** on the `none` scenario and **detection delay** on novel ones;
   * **ablations:** no detector (= baseline); detector but no repair search (only flip flagged rules to their posterior mean);
     full pipeline.
3. Extend `scripts/make_results.py` so `docs/results.md` includes the adaptive column and `gap_closed` with intervals.
4. The "does no harm" check: on `none`, adaptive is within 0.05 of baseline.

## Done when
```bash
python -m pytest tests/test_acceptance_future.py -q -k "adaptive"   # remove xfail markers; pass
emberpath eval --profile single --novelty door_jam --agents baseline oracle adaptive --seeds 30   # gap closed >= 50%
make results      # regenerates docs/results.md with adaptive; every "none" row still shows oracle gap +0.00
python -m pytest -q
```
Target: `gap_closed >= 50%` on `door_jam` for all three households. If you cannot reach it, report the honest number and the
diagnosis. Do not tune the scenarios to make it look better.

## Constraints
Do not change the novelty presets or default rules to help the agent. If a preset is genuinely flawed, treat that as a benchmark
bug: document it in `docs/ARCHITECTURE.md` like the `wood_flashover` note, and rerun the oracle-gap table.

## Docs
README results table gets the adaptive column. Flip A3, A4, E4, E5, E6 in `docs/SPEC.md`.
