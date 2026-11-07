---
description: Compare docs/SPEC.md against the repo and say what is really done
---

Audit project status without changing any files.

1. Read `docs/SPEC.md` and `ROADMAP.md`.
2. For every requirement marked ✅, check there is code and a passing test behind it. For every ⬜ or ◻, check whether
   it was actually built (look for implementations replacing `NotImplementedError`, new files named in `tasks/`, and
   removed `xfail` markers in `tests/test_acceptance_future.py`).
3. Run `python -m pytest -q` and note the counts.
4. Report: (a) requirements whose status in the docs is wrong, in either direction; (b) the next package to work on,
   given the dependency graph in `tasks/README.md`; (c) anything that looks half-done. Propose the doc edits but do not
   make them unless I say so.
