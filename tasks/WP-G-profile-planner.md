# WP-G: Profile and speed up the planner  (P7)

**Goal.** Evaluation is slow (`make results` takes about 4 minutes on 2 cores) and almost all of it is the planner. Find where the
time goes, fix the worst offenders, and prove nothing changed.

## Read first
`src/emberpath/planner.py`, `grid.py`, `world.py` (`_update_dependents`, `_spread_fire`), `scripts/make_results.py`.

## Deliverables
1. **Before touching anything**, freeze the behaviour: write `tests/golden/trajectories.json` containing, for a fixed grid of
   (blueprint, household, novelty, agent, seed), the outcome, step count, and a hash of the action sequence. Add a test that
   compares against it. This is your safety net; commit it first.
2. `scripts/profile_planner.py`*: run `cProfile` (and a line-level look at the top function) over a representative episode set;
   write the top hotspots and timings to `docs/ARCHITECTURE.md`.
3. Fix the worst two or three. Candidates, to be confirmed by the profile rather than assumed: fire-arrival field recomputed
   from scratch every tick (cache or update incrementally), Dijkstra per waiting dependent per tick inside the permutation loop,
   repeated closures and dict lookups in `_enter_cost`, `bfs_distances` called several times per tick for the same source.
4. Report speedup per fix and overall.

## Done when
```bash
python -m pytest -q                      # golden-trajectory test passes: behaviour bit-identical
python scripts/profile_planner.py        # prints before/after timings
make results                             # docs/results.md is IDENTICAL to before, except it now runs at least 2x faster
```

## Constraints
No behaviour change of any kind, including tie-breaking order: if two routes tie, the same one must still win. If a speedup
requires changing results, it is not part of this package. Do not add compiled dependencies to the core install.

## Docs
A "Performance" section in `docs/ARCHITECTURE.md` with the profile and the measured speedups. Flip P7 in `docs/SPEC.md`.
