# WP-E: Explanations  (X1, X2)

**Goal.** Every adaptation ends with a short account that a non-expert can read and an expert can audit: what changed, how
sure the agent is, what it did about it, and when it noticed versus when the truth actually changed.

## Read first
`src/emberpath/explain.py` (`describe_patch`, `DecisionLog` already work; `explain_adaptation` is the stub), `novelty.py`
(`Finding`), `repair.py` (`RepairResult`), `agents.py` (`AdaptiveAgent.patches`, `.log`), `cli.py`, `render.py`.
You only need the *shapes* of `Finding` and `RepairResult`; you can build against hand-made instances before WP-A and WP-B land.

## Deliverables
1. `explain_adaptation(...)` returning two layers:
   * **Plain English paragraph.** Example shape: "At step 4 I noticed the door at (5, 3) would not open three times in a row. I now
     assume it is jammed and am taking the study route. I am fairly sure, but a stuck latch could also explain it." No jargon:
     a test must assert words like *posterior*, *Bayes*, *likelihood*, *Beta* never appear in this layer.
   * **Technical appendix.** Rule key, believed vs inferred value, posterior mean and interval, log Bayes factor, number of
     trials, runner-up hypothesis and margin, steps from change to detection.
2. The CLI: `emberpath run --agent adaptive` ends with both layers. Add `--explain technical|plain|both`.
3. GIF banner: show "NOVELTY DETECTED" from the step the agent noticed, alongside the existing "truth changed at t=..." banner,
   so viewers see the delay.
4. Handle the uncomfortable cases honestly in the text: no novelty found, several rules changed, an ambiguous repair, a false alarm
   (flagged but no real change). Each gets a golden-string test.

## Done when
```bash
python -m pytest -q -k explain
emberpath run --agent adaptive --novelty door_jam --gif /tmp/x.gif   # needs WP-A and WP-B; before then, test with fixtures
```

## Constraints
Deterministic output (golden strings, no timestamps). Plain layer under about 120 words. Do not invent facts the agent did not
infer; the text must be generated from the structured results, not from templates that assume the answer.

## Docs
Show one real output in `README.md` once the adaptive agent works. Flip X1 and X2 in `docs/SPEC.md`.
