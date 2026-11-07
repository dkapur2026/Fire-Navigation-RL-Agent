# WP-K: Trajectory dataset and data-quality validators  (D1-D3)

**Goal.** Turn episodes into a clean, versioned dataset and prove it is clean. Data quality is the product here: a model is only as
good as the trajectories it learns from.

## Read first
`src/emberpath/evaluate.py` (`run_episode`), `reward.py` (`RewardBreakdown`), `world.py` (`Event`, `Observation`), `agents.py`,
`docs/REWARD.md`. Better with WP-J (many houses), but works on the two shipped ones.

## Deliverables
1. `src/emberpath/dataset.py`*: a versioned schema (`schema_version`) for an episode record: blueprint name or hash, household,
   novelty, seed, agent, per-step (compact observation, action, reason, reward components, events), outcome, return, efficiency.
   JSONL read/write (optional gzip). Keep it dependency-free.
2. `scripts/build_dataset.py`*: collect episodes from several agents and scripted policies across blueprints, households,
   novelties, and seeds. **Split by (blueprint, seed) group**, not by episode, to prevent leakage between train and test.
3. **Validators** (`scripts/validate_dataset.py`*, exit code non-zero on failure):
   * schema and types;
   * **replay determinism:** re-running the recorded actions with the recorded seed reproduces the recorded outcome and return
     (a strong consistency check that catches corrupted or inconsistent records);
   * return equals the sum of the recorded reward components; outcome matches the final state;
   * duplicate detection (hash of blueprint + seed + actions);
   * split leakage: no (blueprint, seed) group appears in two splits;
   * outliers: abnormally long episodes, impossible reward values.
4. A **statistics and balance report**: counts by tier, novelty, household, outcome; return distributions; how many failed
   episodes; a simple per-episode quality score you can threshold on.
5. Tests that **inject corruption** (flip an action, edit a return, duplicate a record, leak a seed across splits) and assert the
   validator catches each one.

## Done when
```bash
python scripts/build_dataset.py --out /tmp/ds --episodes 200
python scripts/validate_dataset.py /tmp/ds            # exit 0 on clean data
python -m pytest -q -k dataset                        # includes the corruption-injection tests
```

## Constraints
Records must be small (no full-grid dumps per step unless compact). Deterministic builds: same arguments, byte-identical file.
Never silently drop records; report them.

## Docs
`docs/DATA.md`* describing the schema, the validators, and what each one catches. Flip D1-D3 in `docs/SPEC.md`.
