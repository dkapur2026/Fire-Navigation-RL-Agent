# emberpath

A burning-house rescue environment plus an agent that notices when the world's rules change, repairs its own rulebook,
and explains what it did. It also serves as a trustworthy RL environment: audited reward, difficulty control, data
pipeline, LLM-agent interface.

Read in this order: `docs/SPEC.md` (what to build, with status), `ROADMAP.md`, `docs/ARCHITECTURE.md`,
`docs/REWARD.md`, then the task file for what you were asked to do (`tasks/`).

## Commands
```bash
pip install -e ".[dev]"      # core only: numpy + pillow. Extras: rl (torch, gymnasium), bayes (pyro), llm (anthropic)
python -m pytest -q          # ~30 s; 49 pass, 7 xfail (acceptance tests for features not built yet)
python -m pytest -q -m "not slow"
make audit                   # reward-hacking audit, ~20 s
make results                 # benchmark table, ~4 min. Run after ANY change to dynamics, planner, or default rules/reward
emberpath run --profile single --novelty door_jam --agent oracle --gif /tmp/x.gif
emberpath eval --profile kids --novelty door_jam --agents baseline oracle --seeds 20
```

## How to work here
* **One work package at a time.** Use `/work-package <id>` (see `tasks/README.md`). Stay inside the files that
  package owns; if you must touch another file, make the smallest edit and say so.
* **Plan first** when a change touches more than one file or any shared interface.
* **Tests are the spec.** `tests/test_acceptance_future.py` holds `xfail` tests for features not built yet. When you
  implement one, remove its `xfail` marker. Never weaken or delete a test to make it pass; if a test is wrong, say why.
* **Verify before claiming done.** Run the tests and whatever command the task file lists. Report what you ran and
  what it showed, including failures.
* **Be honest in docs.** If something does not work or does not help, write that down (see the `wood_flashover`
  note in `docs/ARCHITECTURE.md`). Never tune a benchmark until it flatters the method. Never mark a requirement ✅
  in `docs/SPEC.md` until its acceptance command passes.
* **Update status** in `docs/SPEC.md` and `ROADMAP.md` when a package is done.

## Non-negotiables
@AGENTS.md

## Interfaces that must not change without agreement
`Rules`, `RulePatch`, `Event`, `Finding`, `RepairResult`, `Decision`, `Agent.act(obs, events)`,
`RewardTracker.update(world) -> RewardBreakdown`. Adding an optional field with a default is fine.

## Secrets
Never write an API key into a file. The `llm` extra reads `ANTHROPIC_API_KEY` from the environment. Tests must never
call the network.
