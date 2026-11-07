# Stretch items

Pick by interest, after the core packages. Each is small enough to brief in a paragraph; turn one into a full task file by copying the
structure of `WP-A-detector.md` before you start. Run with `/work-package S` and name the item.

## S1. Partial observability and a belief tracker  (P6)
`world.observe(full=False)` already hides what smoke hides. Build a tracker that remembers where dependents and fire *were* and
how stale that memory is, and make the planner plan under it (for example, search for a dependent last seen near a door).
*Done when:* an agent using partial observability plus the tracker beats one that forgets, on `studio` and `ranch`, and a test
shows the tracker never reveals information the observation did not.

## S2. Triage  (P5)
When fire makes it impossible to save everyone, choose who to save by expected survivors under uncertainty, and explain the choice
in plain language. *Done when:* in a constructed scenario where only one of two children can be saved, the agent picks the higher
survival odds and says why; the reward audit is extended so triage cannot be gamed by abandoning everyone.

## S3. Human on-the-loop supervisor  (X3)
The agent proposes a plan with a confidence; a scripted supervisor can veto or redirect; the agent asks when evidence is ambiguous
(from WP-B's ambiguity flag). *Done when:* a run shows a veto changing the route and the log records who decided what.

## S4. Change-point inference  (A5)
Infer *when* a rule changed, and discount evidence from before the change. Bayesian online change-point detection (Pyro or NumPyro
behind the `bayes` extra, or a closed-form version). *Done when:* recovery time on late novelties (`compound`) improves measurably
over the no-forgetting detector, with intervals.

## S5. Structural novelty  (W9)
A new *kind* of thing appears that no existing rule key describes, for example a window that works as an exit. Parameter repair cannot
express this. *Done when:* the benchmark includes the scenario, the baseline and the parameter-repair agent fail it, and you have
either a working approach or a clear write-up of why the current representation cannot capture it.

## S6. Amortized repair  (A6)
A small network proposes likely `RulePatch`es from the evidence; the symbolic search verifies and scores them. *Done when:* it
matches the pure search's accuracy with fewer candidate evaluations, with numbers.

## S7. Two rescuers
Two agents coordinate with shared beliefs about the rules. *Done when:* two rescuers beat one on the `kids` household and they
agree on which rule changed.
