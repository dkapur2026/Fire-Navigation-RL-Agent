# Work packages

Each file is a self-contained brief for one work package (WP). Run one with `/work-package <id>` in Claude Code.
Requirement IDs (W1, A2, ...) refer to `docs/SPEC.md`.

## Index

| ID | Package | Track | Owns (new files marked *) | Needs | Size |
|---|---|---|---|---|---|
| A | [Bayesian novelty detector](WP-A-detector.md) | A | `novelty.py`, `scripts/false_alarms.py`* | nothing | M |
| B | [Rule repair search](WP-B-repair.md) | A | `repair.py` | `Finding` shape only | M |
| C | [PPO baseline + trained-agent reward audit](WP-C-ppo-baseline.md) | A+B | `rl_baseline.py`, `env.py` | nothing | L |
| D | [Neural perception front-end](WP-D-perception.md) | A | `perception.py`* | nothing | L |
| E | [Explanations](WP-E-explanations.md) | A | `explain.py`, `cli.py` banners | Finding / RepairResult shapes | M |
| F | [Wire the adaptive agent + evaluation study](WP-F-adaptive-agent-and-eval.md) | A | `agents.py`, `evaluate.py`, `scripts/` | A, B | M |
| G | [Profile and speed up the planner](WP-G-profile-planner.md) | A | `planner.py`, `grid.py` | nothing | S |
| H | [Technical report](WP-H-report.md) | A | `docs/REPORT.md`* | F | S |
| I | [LLM-agent interface + benchmark](WP-I-llm-interface.md) | B | `llm_env.py`*, `scripts/eval_llm.py`* | nothing (better with J) | M |
| J | [Procedural houses with graded difficulty](WP-J-procgen.md) | B | `procgen.py`* | nothing | M |
| K | [Trajectory dataset + quality validators](WP-K-dataset.md) | B | `dataset.py`*, `scripts/build_dataset.py`*, `scripts/validate_dataset.py`* | nothing (better with J) | M |
| L | [Post-training validation study](WP-L-posttraining-validation.md) | B | `scripts/post_training_study.py`*, `docs/DATA_STUDY.md`* | J, K | L |
| M | [Reward robustness + automated exploit search](WP-M-reward-robustness.md) | B | `scripts/find_exploits.py`*, `scripts/reward_sweep.py`* | nothing (better with C) | M |
| S | [Stretch items](S-stretch.md) | A+B | varies | varies | varies |

Sizes: S about half a day, M about a day, L two or more days.

## Dependencies

```
 A ──► F ──► H            J ──► K ──► L
 B ──►─┘                  J ─ ─► I          (I works on studio/ranch without J)
 E  (needs only the Finding / RepairResult shapes)
 C ─ ─► M                 G  (independent; do before heavy evaluation)
 D  (independent)         ── hard dependency      ─ ─ soft: better with, not blocked by
```

## Suggested orders

* **Track A, shortest complete story:** A → B → F → E → H. Add G before F if evaluation feels slow, C or D for depth.
* **Track B, shortest complete story:** J → K → I → M → L. Add C any time; it also audits the reward (R10).
* **Both, one person:** A, B, F, J, K, I, M, E, H, then L and C.

## Running in parallel
Packages with disjoint "Owns" columns can run side by side in separate git worktrees (see `START_HERE.md`). Do not run
two packages that own the same file at once (A/B/E overlap only through shared interfaces, which must not change).

## Every package ends the same way
1. The "Done when" commands in its file pass.
2. `python -m pytest -q` is green (xfail markers removed for features you built).
3. `docs/SPEC.md` and `ROADMAP.md` statuses updated; anything the file names under "Docs" updated.
4. Negative or surprising results are written down, not hidden.
