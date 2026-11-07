---
description: Run the tests (and the reward audit if the reward changed) and report honestly
---

Verify the current state of the repo.

1. Run `python -m pytest -q`. Report passed / failed / xfailed counts. The 7 acceptance tests in
   `tests/test_acceptance_future.py` are expected to xfail until their features are built; an unexpected *XPASS* means a
   feature now works and its `xfail` marker should be removed; any *failure* is a real problem.
2. If `src/emberpath/reward.py`, `env.py`, or `scripts/audit_reward.py` changed (check `git status` / `git diff --stat`), also
   run `make audit` and confirm honest effort still ranks above every shortcut and suicide is below a struggling honest agent.
3. If dynamics (`world.py`), the planner, or default rules changed, tell me `make results` (about 4 minutes) should be rerun,
   and that every "none" row must still show an oracle gap of +0.00. Run it only if I ask.
4. Say plainly what you ran and what it showed. Do not summarize failures away.
