# WP-J: Procedural houses with graded, measured difficulty  (W8)

**Goal.** Two hand-drawn houses are not a benchmark. Generate many, guarantee each is solvable, and *measure* difficulty rather
than guessing it, so training and evaluation can cover a controlled range.

## Read first
`src/emberpath/blueprint.py` (the ASCII format and `validate`), `scenarios.py` (novelty presets must work on generated houses),
`evaluate.py`, `docs/ARCHITECTURE.md` (the calibration notes explain why measured difficulty matters).

## Deliverables
1. `src/emberpath/procgen.py`*: `generate(seed, tier, ...) -> Blueprint`.
   * Room-and-corridor layouts with controllable size, number of rooms, doors, loops (alternative routes), floor-material mix,
     number of exits, and ignition distance from the nearest exit.
   * Deterministic for a given seed. Output is valid ASCII that `Blueprint.parse` accepts.
   * **Solvability guarantee:** `validate` passes for every household up to three dependents, and `door_jam` finds a door to jam
     that leaves the house solvable (generate loops where needed, or reject and resample).
   * Reject degenerate houses (rescuer next to an exit, fire adjacent to the rescuer, dependents unreachable).
2. **Measured difficulty.** For each generated house compute: route length, door count, fire-to-exit distance, and the **oracle's
   empirical success rate over 30 seeds**. Define tiers (1 easy to 5 hard) by thresholds on those measurements, chosen by looking
   at the data, and show that mean oracle success decreases with tier.
3. CLI: `emberpath gen --tier 3 --seed 7` prints a house; `emberpath run --blueprint gen:3:7 ...` accepts it.
4. `scripts/coverage_report.py`*: diversity of a generated set (histograms of the metrics, a pairwise-distance summary, how many
   distinct topologies), so coverage is a number.

## Done when
```bash
python -m pytest -q -k procgen     # 200 seeds x all tiers: parse, validate, deterministic, novelty presets still work
python scripts/coverage_report.py  # prints coverage and the oracle-success-by-tier table (must be monotone non-increasing within tolerance)
```

## Constraints
No new core dependencies. Generation must be fast (hundreds per second). If tiers do not separate cleanly, fix the generator or
the thresholds; do not just relabel.

## Docs
A "Procedural houses" section in `docs/ARCHITECTURE.md` with the tier table. Flip W8 in `docs/SPEC.md`.
