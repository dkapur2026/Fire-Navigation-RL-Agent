# WP-M: Reward robustness and automated exploit search  (R11)

**Goal.** The reward audit so far covers exploits we thought of. Build tools that find exploits we did *not* think of, and show how
sensitive conclusions are to the reward's weights.

## Read first
`src/emberpath/reward.py`, `docs/REWARD.md` (especially "Limitations"), `scripts/audit_reward.py`, `tests/test_reward.py`.
Better with WP-C (a trained agent is the strongest exploit finder), but this package does not need it.

## Deliverables
1. **Automated exploit search** (`scripts/find_exploits.py`*): define a small parameterized family of scripted policies, for
   example weights for "head to the nearest dependent", "head to the exit", "stay away from fire", "stay near fire", "wait", and
   an exploration noise level. Search that space (random search, or a simple evolution strategy) to **maximize return under the
   reward**. Report any policy whose return is at least that of honest rescue while its *success rate* is clearly lower. That
   gap between reward and true success is a loophole. Print the policy parameters and the component breakdown that explains it.
2. **Weight sweep** (`scripts/reward_sweep.py`*): perturb each `RewardConfig` weight across a sensible range and measure
   **ranking stability**: does honest rescue still outrank every degenerate policy from `audit_reward.py`? Report Kendall's tau
   between rankings across settings and the weight ranges where the ranking breaks.
3. **Sparse vs shaped ablation:** compare a sparse reward (rescue and success only) with the full one on how quickly a learner
   improves and on how much return variance there is across seeds. Use PPO if WP-C is done, otherwise a simple tabular or
   evolution-strategy learner on `studio`.
4. **Signal reliability:** across seeds, how noisy is the return of a fixed good policy? Report the standard deviation and the
   number of seeds needed to tell two policies apart.
5. Fold every new exploit into `audit_reward.py` and `tests/test_reward.py`, fix the reward if justified, and record the story in
   `docs/REWARD.md`.

## Done when
```bash
python scripts/find_exploits.py --budget small       # finds the known suicide loophole if you run it against a copy of the legacy reward
python scripts/reward_sweep.py                       # prints the stability table
python -m pytest -q                                  # new tests for the search harness
make audit                                           # still passes after any reward change
```
A useful sanity check: run the exploit search against the **legacy reward** (it lives in `scripts/audit_reward.py`); it should
rediscover the suicide loophole. If it cannot, the search is too weak to trust against the new reward.

## Constraints
Searching and sweeping must be seeded and reproducible. Do not change reward defaults just to make a sweep look stable; if you
find a better setting, justify it with the audit and tests. Report negative results: "searched N policies, found no exploit" is a
result, but say how large the search was.

## Docs
Update `docs/REWARD.md` (findings, sweep ranges, caveats). Flip R11 in `docs/SPEC.md`.
