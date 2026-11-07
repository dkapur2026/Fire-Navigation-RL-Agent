# WP-C: PPO baseline and trained-agent reward audit  (N1, R10)

**Goal.** Answer "why not just use deep RL?" with data, and use a learning agent to stress-test the reward: a trained
agent is the best exploit finder there is.

## Read first
`src/emberpath/rl_baseline.py` (design in its docstring), `env.py`, `reward.py`, `docs/REWARD.md` (especially "Limitations"),
`scripts/audit_reward.py`.

## Deliverables
1. PPO in PyTorch (about 150 lines; or stable-baselines3 behind the `rl` extra). Policy: small CNN over the `(H, W, 10)` grid
   to 5 action logits plus a value head. CPU-feasible: first iterate on `studio`, then `ranch`. Add `matplotlib` to the `rl` extra
   for plots.
2. Two experiments, seeded (train on seeds 0..999, evaluate on 1000..1099):
   * **Frozen:** train on the nominal world, evaluate on a world that changes mid-episode (`door_jam`, `carpet_flashover`).
   * **Keeps learning:** continue training after the change. Record environment steps and episodes until the policy recovers to 90%
     of its pre-novelty score.
3. Plots saved to `docs/media/`: success vs environment steps for both settings.
4. **Reward audit of the trained agent (R10).** Print the per-component breakdown (`info["reward_components"]`) for its
   best-return episodes and for episodes where it scores high but does not succeed. Look for behaviour the reward pays for that a
   person would not want (hovering near a dependent without finishing, exploiting the death charge, farming progress, dawdling).
   For every exploit found: add a degenerate policy to `scripts/audit_reward.py`, add a test, fix the reward, rerun `make audit`.
   If you find none, say so and say how hard you looked.
5. A paragraph in `README.md` answering "why not just RL?" using your numbers, whatever they show.

## Done when
```bash
pip install -e ".[rl]"
python -m emberpath.rl_baseline --help          # or the entry point you add; document it
python -m pytest -q                               # PPO smoke test (a few updates on studio) included
make audit                                        # still passes
```

## Constraints
Keep torch and gymnasium out of the core install; import lazily. Do not edit reward defaults to flatter the learner; if a change
is justified, it needs a test and an audit row. Report if PPO fails to learn at all in a reasonable budget; that is a result.

## Docs
`docs/REWARD.md` (what the trained agent revealed), README paragraph, flip N1 and R10 in `docs/SPEC.md`.
