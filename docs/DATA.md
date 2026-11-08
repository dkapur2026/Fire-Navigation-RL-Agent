# The trajectory dataset

**In one line:** every episode becomes one JSON line you can replay exactly, and a validator proves the file is
clean before anyone trains on it.

Code: `src/emberpath/dataset.py` (stdlib + the simulator; no pandas/pyarrow). Scripts:

```bash
python scripts/build_dataset.py --out /tmp/ds --episodes 200 [--agents baseline oracle adaptive random] [--workers 8] [--gzip]
python scripts/validate_dataset.py /tmp/ds [--no-replay] [--report stats.md]   # exit 1 on any failure
python -m pytest -q -k dataset                                                  # corruption-injection tests
```

## What gets collected
A *job* is one (house, household, novelty, seed). Every agent in `--agents` plays it. Job `j` uses
blueprint `j mod B`, household `j mod 3` and novelty `j mod 5` (`none` + every preset in `scenarios.NOVELTIES`).
With the default 7 blueprints (`ranch`, `studio`, `gen:1` ... `gen:5`; 7 is coprime with 15) all 105 triples appear
in 105 jobs, and every pair far sooner (all blueprint x novelty pairs in 35 jobs). `gen:<tier>` draws a fresh
procedural house every pass (`procgen.generate(j // B, tier)`); `gen:<tier>:<seed>` pins one. The fire seed is `j`.
`random` is a scripted, seeded random walk: a deliberately bad policy, so the data holds failures and not just experts.

Nothing is dropped silently. `manifest.json` lists:
* `skipped_jobs`: preset not applicable to the house (only `studio/kids/hidden_dependent`: the studio has no spare slot);
* `dropped_duplicates`: an episode whose trajectory equals one already written (same house, household, novelty, seed
  and actions). This is common and expected: with no novelty, baseline and oracle are identical by design (the
  control invariant), and the adaptive agent often never needs to change its mind. In a 600-record build 236
  duplicates were dropped (most of them oracle/adaptive copies of baseline).

Same arguments give byte-identical files, for any `--workers` and any `PYTHONHASHSEED` (checked: two 200-record builds,
serial and 4 workers, `diff -r` identical; gzip is written with `mtime=0`).

## The record (`schema_version` 1)

| field | meaning |
|---|---|
| `id` | hash of (blueprint hash, household, novelty, seed, agent) |
| `content_hash` | hash of (blueprint hash, household, novelty, seed, actions): the duplicate key |
| `split` | `train` / `val` / `test`, from a hash of (blueprint hash, seed), ~80/10/10 |
| `blueprint`, `blueprint_hash`, `tier` | house name (`ranch`, `gen:3:7`), sha256 of its ASCII, measured tier (procgen's `tier_of(measure(bp))` for shipped houses too: ranch 3, studio 2) |
| `household`, `seed`, `agent`, `max_steps` | episode parameters |
| `privileged` | `true` for `oracle`: its actions and reasons were computed from the true rules |
| `labels` | **hidden truth, episode level only**: `novelty` and `true_patches` = `[[step, key, value], ...]` |
| `steps[t]` | `obs` (what the rescuer saw before acting), `action` (0-4), `reason`, `reward` (7 floats in `reward.COMPONENTS` order), `events` (this tick's rule trials as `{key: [trials, successes]}`) |
| `final` | the terminal `obs` plus `rescued`, `lost`, `total` |
| `outcome`, `n_steps`, `return`, `score`, `efficiency` | as in `evaluate.run_episode` (a test checks they match it) |
| `quality` | `score x (0.5 + 0.5 x efficiency)`, 0 for failures: threshold it to keep good demonstrations |

`obs` is a compacted `World.observe()`: `pos`, `alive`, `deps` (`[id, kind, r, c, status]`, exactly the dependents the
world shows), `smoke` (at the rescuer, 4 decimals), and fire **delta-coded**: `fire_on` / `fire_off` are flat cell
indices (`r * width + c`) that started / stopped burning since the previous observation; burnt cells are therefore
every cell that has stopped burning. Events are tallied per key because the evidence is exchangeable
(Beta-Bernoulli): order inside a tick carries no information. Size: about 320 bytes per step uncompressed, about 13
bytes per step gzipped (600 episodes, 45 088 steps: 580 KB gzipped).

Evidence discipline: the per-step side holds only `World.observe()` and the `Event` log. A hidden dependent is absent
from `deps` while it waits (the only trace is `occupied:<cell>` hearing events, which are real evidence); a jammed door
appears only as a `door_jammed:` event on the tick the rescuer bumps it; which novelty happened is in `labels` only.

## Validators and what each catches

| check | catches |
|---|---|
| `schema` | missing or unexpected fields, wrong types, unknown household/outcome/novelty/split, wrong `schema_version`, `id`/`content_hash` not matching the record |
| `replay` | re-runs the recorded actions with the recorded seed and compares every step (obs, events, reward components) and the outcome, length, final state, return and efficiency. Catches a flipped action, an edited observation, or a record from a different code version |
| `leak` | evidence discipline, from the record alone: a hidden dependent shown as `waiting`, a `door_jammed:` event without the rescuer having just walked into that door, an `occupied:` event out of hearing range, an extra observation field (e.g. `jammed`) |
| `return` | `return` != sum of all recorded components |
| `outcome` | `outcome` contradicts the final state (alive, rescued + lost = total, length = `max_steps` for a timeout); `score`/`quality` inconsistent |
| `duplicate` | two records with the same `content_hash` |
| `split_leakage` | a (blueprint, seed) group in two splits, or a record stored in a file other than its `split` |
| `outlier` | longer than `max_steps` or `n_steps` inconsistent; impossible values: action outside 0-4, `time` != -step cost, `rescue`/`fire_contact` not a whole number of bonuses, `success` or `agent_death` before the last step, proximity outside [-0.1 x dependents, 0], absurd `progress` |
| `manifest` | the number of records on disk differs from the manifest |

`tests/test_dataset.py` injects each corruption (flip an action, edit a return, duplicate a record, leak a seed
across splits, impossible reward, too long, wrong outcome, dropped field, bad type, a waiting hidden dependent in
`obs`, an unbumped jammed door as an event and as an obs field) and asserts the right check fires.

## Numbers (M-series laptop, 8 cores)
* 200 records, serial: 26.6 s (7.5 records/s, 10.5 episodes/s counting dropped duplicates). 4 workers: 8.6 s (23 records/s).
* 600 records, 8 workers, gzip: 20.0 s (30 records/s). Validation with replay: 35.5 s for 600 (serial).
* The 600-record build covers every house kind x household x novelty triple except the inapplicable
  `studio/kids/hidden_dependent`. Outcomes: random 4/209 successes, baseline 81/209, oracle 77/107, adaptive 53/75.

## Limits
* Full observability only (`visibility_radius` is not recorded); a partial-observation dataset needs `obs` to carry the radius.
* Oracle records are `privileged`: imitating them teaches a policy to act on information it cannot see (it walks
  straight to a hidden child). Filter on `privileged` for evidence-only training.
* Dropping duplicates makes per-agent counts unequal; compare agents with `evaluate.py`, not by counting records.
