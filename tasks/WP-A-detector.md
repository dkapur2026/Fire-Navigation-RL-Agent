# WP-A: Bayesian novelty detector  (A1)

**Goal.** Notice when the true rules have drifted from the agent's belief, from nothing but the `Event` log. Say *which* rules
look off, how confident, and how far.

## Read first
`src/emberpath/novelty.py` (the design is in its docstring), `src/emberpath/rules.py`, the `Event` class in `world.py`,
`tests/test_acceptance_future.py` (the three detector tests), `docs/ARCHITECTURE.md`.

## Deliverables
1. `BetaBernoulliDetector` in `novelty.py`, replacing every `NotImplementedError`.
   * Per rule key keep (successes, trials). Believed probability comes from `Rules.prob(key)`.
   * Score each key with a log Bayes factor: marginal likelihood of the data under a free `p ~ Beta(1, 1)` minus its
     likelihood under the point null `p = belief`. Use `math.lgamma`; no scipy.
   * Flag when the factor exceeds `threshold` and `trials >= min_trials`.
   * Posterior mean and 90% interval: a grid posterior over `p` (about 1001 points) is exact enough and needs only numpy.
   * Door keys are near-deterministic: a single bump on a believed-working door is already strong evidence. Handle with a
     sensible prior, not a special case in the test.
   * Sequential testing: the detector is peeked at every tick, so a fixed threshold inflates false alarms. Use a
     correction (for example a mixture sequential probability ratio, or a threshold that grows with log trials) and justify it.
   * `reset_belief(new_belief)` re-baselines surprise after a repair. Keep evidence, change the null.
   * Optional exponential forgetting (`decay`) so stale pre-change evidence does not dilute a late novelty. Test it with the
     `compound` scenario (second novelty at tick 10).
2. `scripts/false_alarms.py`*: false-alarm rate against threshold on quiet worlds (events sampled from the nominal rules)
   and detection delay against threshold on novel worlds. Print a table; write `docs/media/detector_roc.png` if you add matplotlib to a
   dev extra, else a text table.
3. Extra unit tests for monotonicity (more contrary evidence never lowers the score), `min_trials`, interval coverage on
   simulated data, door handling, and `reset_belief`.

## Done when
```bash
python -m pytest tests/test_acceptance_future.py -q -k "detector"   # remove the xfail markers; all pass
python scripts/false_alarms.py                                       # <= 5% false alarms at the default threshold
python -m pytest -q
```

## Constraints
Core install stays numpy + pillow. Do not change `Finding` or the `NoveltyDetector` protocol (adding optional fields is fine).
Events from fire spread arrive several per tick (one independent trial per burning neighbour); that is valid Bernoulli data.

## Docs
Add a short "Detector" section to `docs/ARCHITECTURE.md` (design choices, the false-alarm/delay trade-off with real numbers).
Flip A1 in `docs/SPEC.md`.
