---
description: Implement one work package from tasks/ (for example /work-package A)
argument-hint: <package id: A-M, or S for stretch items>
---

Implement work package **$ARGUMENTS** of this project.

1. Read `CLAUDE.md`, then the task file in `tasks/` whose name starts with `WP-$ARGUMENTS` (or `S-` for stretch).
   Read every file listed under "Read first". If no task file matches, list `tasks/` and ask which one I meant.
2. State a short plan before writing code: the files you will create or change, the shared interfaces you will
   respect, and the tests you will add. If the plan changes a shared interface, stop and ask me. Otherwise proceed.
3. Build it, staying inside the files this package owns. Make the smallest possible edit to anything else and mention it.
4. Run the commands listed under "Done when" in the task file. Fix failures at the cause; never weaken a test to
   make it pass. If an acceptance test's expectation is wrong, tell me why instead of editing it silently.
5. Update status: flip the matching rows in `docs/SPEC.md` and `ROADMAP.md`, and update any docs the task file names.
   Only mark something done if its acceptance command passed.
6. Finish with: what you built, the exact commands you ran and what they showed (including anything that failed or
   did not help), and what is left. Report negative results plainly.
