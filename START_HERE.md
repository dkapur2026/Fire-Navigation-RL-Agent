# Start here: building emberpath with Claude Code

```bash
# 1. unzip, then
cd emberpath
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make test                  # expect: 49 passed, 7 xfailed   (the 7 are specs for features not built yet)

# 2. give Claude Code a git history to diff against (and worktrees to parallelize with)
git init && git add -A && git commit -m "Initial emberpath scaffold"

# 3. open Claude Code in this folder. It reads CLAUDE.md automatically.
claude
```

Then, inside Claude Code:

| Command | What it does |
|---|---|
| `/work-package A` | Implements work package A (the Bayesian detector): reads the task file, plans, builds, tests, updates status |
| `/check` | Runs the test suite, and the reward audit if the reward changed, and reports honestly |
| `/status` | Compares `docs/SPEC.md` against the repo and tells you what is really done |

## What to build, in order

**Shortest complete story (Track A, about a weekend):** `A` detector → `B` repair → `F` wire the adaptive agent and
evaluate → `E` explanations → `H` write-up. At the end `emberpath eval --agents baseline oracle adaptive` shows the
adaptive agent closing most of the gap, with a plain-English account of what it noticed.

**RL-environment story (Track B):** `J` procedural houses → `K` trajectory dataset with quality checks → `I` LLM-agent
interface → `M` reward robustness and exploit search → `L` post-training validation study. Start `C` (PPO) any time;
it also tests the reward.

Everything is in `tasks/README.md` with the dependency graph, sizes, and which files each package owns.

## Running packages in parallel
Packages that own different files can run side by side in separate git worktrees:

```bash
git worktree add ../emberpath-wp-a -b wp-a
git worktree add ../emberpath-wp-j -b wp-j
# open `claude` in each folder and run /work-package A  and  /work-package J
```
Merge one at a time; run `make test` after each merge.

## What is already done
A simulator (fire, smoke, dependents, jammed doors), a symbolic planner, an audited reward, baseline / oracle agents,
a same-seed evaluation harness, GIF rendering, a Gymnasium env, and 49 tests. `docs/SPEC.md` lists every requirement
with its status. `docs/EXPLAINING_IT.md` explains the project to technical and non-technical audiences.

## A note on permissions
`.claude/settings.json` pre-approves running the tests, `make`, the `emberpath` CLI, and the project scripts so
Claude Code does not stop to ask each time. Read it before first use and trim it if you prefer prompts.
