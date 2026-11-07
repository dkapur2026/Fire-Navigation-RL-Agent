# WP-L: Post-training validation study  (D4)

**Goal.** Answer the question that makes a data pipeline worth having: *does training on this data actually make a model better,
and which data helps?* Run the experiment, report what it shows, including null results.

## Read first
`dataset.py` and `docs/DATA.md` (WP-K), `procgen.py` (WP-J), `env.py` (`encode`), `evaluate.py`, `docs/REWARD.md`. Needs WP-J
and WP-K. Needs the `rl` extra (torch).

## Deliverables
1. `scripts/post_training_study.py`*: behaviour cloning of a small policy network on **slices** of the dataset:
   * (a) successful oracle episodes only; (b) baseline episodes only; (c) mixed successes and failures; (d) filtered by the
     quality score from WP-K; (e) data-quantity sweep (10%, 30%, 100%); optionally (f) reward-weighted cloning.
2. Evaluate each trained policy on **held-out houses and held-out novelties** (never seen in training): success rate, mean return,
   efficiency, and `gap_closed` against baseline and oracle. At least 5 training seeds per condition, with bootstrap 95% intervals.
3. Control conditions: a randomly initialised policy and a scripted baseline, so the effect sizes have a reference.
4. `docs/DATA_STUDY.md`*: the results table, the figure(s), and an honest reading: which slices helped, which did not, whether
   quality filtering paid off, how performance scales with data, where models fail (read some failure episodes). State clearly if
   the differences are within noise.

## Done when
```bash
python scripts/post_training_study.py --quick    # minutes; small grid; proves the pipeline end to end
python scripts/post_training_study.py            # full grid, seeds >= 5
python -m pytest -q -k study                     # tests the slicing, splitting (no leakage), and metric code, not the training itself
```

## Constraints
No peeking: hyper-parameters are chosen on a validation split, then the test split is touched once. Seeded. CPU-feasible. If the
data does not help the model at all, that is the finding; say so and say what you would try next.

## Docs
`docs/DATA_STUDY.md`; link it from `README.md`. Flip D4 in `docs/SPEC.md`.
