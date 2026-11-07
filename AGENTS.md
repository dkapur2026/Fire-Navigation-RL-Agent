# AGENTS.md: guide for coding agents working on emberpath

## Mission
Build an agent that notices when the true rules of a burning-house world change, repairs its own
rulebook, and explains what it did. Success = the `gap_closed` metric in README.md.

## Orientation (read in this order)
1. `README.md` and `ROADMAP.md`: what and why.
2. `src/emberpath/rules.py`: the `Rules` / `RulePatch` vocabulary everything is built on.
3. `src/emberpath/world.py`: the simulator and the `Event` evidence log.
4. `src/emberpath/planner.py` and `agents.py`: how beliefs become decisions.
5. `tests/test_acceptance_future.py`: executable specs for the features you are asked to build.

## Commands
```bash
pip install -e ".[dev]"            # core (numpy, pillow). torch/gymnasium/pyro are optional extras.
python -m pytest -q                # ~20 s. Add -m "not slow" while iterating.
emberpath run --profile single --novelty door_jam --agent adaptive --gif /tmp/x.gif
emberpath eval --profile kids --novelty door_jam --agents baseline oracle adaptive --seeds 20
make results                       # regenerates docs/results.md (about 4 minutes)
make audit                         # reward-hacking audit, about 20 seconds
```

## Invariants (do not break these)
1. **Evidence discipline.** Only `OracleAgent` may read `world.rules`. Every other agent learns from
   `Observation`s and `Event`s it was handed. If your agent needs more information, add it to the
   observation deliberately and say so in your PR; never reach into the world.
2. **Fire has its own RNG stream.** For a given seed every agent must face identical fire. Do not draw
   from `_fire_rng` in a way that depends on the agent's actions. (`test_fire_is_independent_of_what_the_agent_does`.)
3. **Rules are the single source of truth.** New rules extend the key grammar in `rules.py` (and its
   tests). A rule is a Bernoulli probability; a trial is an `Event(step, key, outcome)`.
4. **The control must stay at zero.** With no novelty, baseline and oracle must be identical
   (`test_oracle_equals_baseline_when_nothing_changes`). If you touch dynamics, planner or defaults, run
   `make results` and confirm every `none` row still shows an oracle gap of `+0.00`.
5. **Every decision carries a human-readable `reason`.** Never return a bare action.
6. **Core stays light.** `import emberpath` must work with only numpy and pillow. Import torch / pyro /
   gymnasium lazily, inside the feature that needs them, and declare them in the matching extra.
7. **Determinism.** Everything is seeded. A test that is flaky is a bug in the test or the code, not noise.
8. **The reward is audited, not assumed.** Any change to `reward.py` needs a test for each new exploit you can think of and a new degenerate policy in `scripts/audit_reward.py`. Honest effort must still outrank every shortcut (`make audit`). Read `docs/REWARD.md` first.
9. **Report negative results.** If something does not help, write it down in `docs/ARCHITECTURE.md` (see the
   `wood_flashover` note). Do not tune a benchmark until it flatters the method.

## Work packages (can run in parallel; file ownership avoids merge conflicts)

| WP | Task | Owns | Depends on | Done when |
|---|---|---|---|---|
| A | Bayesian detector | `novelty.py` | nothing | detector tests in `test_acceptance_future.py` pass, xfail removed |
| B | Rule repair search | `repair.py` | `Finding` dataclass only (use synthetic events) | repair tests pass |
| C | PPO baseline + eval | `rl_baseline.py`, `env.py` | nothing | success-vs-steps plot saved to `docs/media/`; best episodes' reward components inspected for exploits |
| D | Neural perception | new `perception.py` | `render.py`, `env.encode` | accuracy reported; planner can consume it |
| E | Explanations | `explain.py`, CLI banners | Finding / RepairResult shapes | Phase 6 criteria in ROADMAP |
| F | Wire + evaluate adaptive agent | `agents.py`, `scripts/` | A, B | gap-closed >= 50% on `door_jam` |
| G | Profile the planner | `planner.py`, `grid.py` | nothing | `make results` identical before/after, faster |
| H | Report | `docs/REPORT.md` | F | 2 pages, every number reproducible from a command |
| I | LLM-agent interface + benchmark | `llm_env.py`, `scripts/eval_llm.py` | nothing | offline tests pass with a scripted client; no hidden-rule leakage |
| J | Procedural houses, graded difficulty | `procgen.py` | nothing | 200 seeds solvable and deterministic; oracle success falls with tier |
| K | Trajectory dataset + validators | `dataset.py`, `scripts/build_dataset.py`, `scripts/validate_dataset.py` | nothing | validator catches every injected corruption |
| L | Post-training validation study | `scripts/post_training_study.py`, `docs/DATA_STUDY.md` | J, K | held-out results with intervals, honest reading |
| M | Reward robustness + exploit search | `scripts/find_exploits.py`, `scripts/reward_sweep.py` | nothing | search rediscovers the legacy suicide loophole |

Paths in the *Owns* column that do not exist yet are created by that package.

Full briefs for every package are in `tasks/` (start with `tasks/README.md`). In Claude Code, run `/work-package <id>`.

Rules for parallel work: stay inside the files you own; if you need a change elsewhere, make the smallest
possible edit and mention it. Interfaces that must not change without agreement: `Rules`, `RulePatch`,
`Event`, `Finding`, `RepairResult`, `Decision`, `Agent.act(obs, events)`.

## Definition of done for any change
- Tests added or updated; `python -m pytest -q` is green.
- Docs updated (README table, ROADMAP status, ARCHITECTURE if a design decision changed).
- `make results` regenerated if dynamics, planner or defaults changed, and the diff explained.
- No new required dependency in the core install.

## Style
Type hints on public functions. Docstrings explain *why*, not *what*. Prefer small pure functions. Keep
user-facing text (reasons, explanations) free of jargon: write for someone who has never seen the code.
