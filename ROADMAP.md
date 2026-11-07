# Roadmap

Each phase lists what to build, the files it touches, and **how you know it is done**. Phases 1-3 are
finished; their acceptance criteria are the passing tests in `tests/`.

Legend: done · next · stretch

## Finished

| Phase | Deliverable | Evidence |
|---|---|---|
| 1 | World: blueprint parser, fire as a cellular automaton, smoke, rule-trial `Event` log, partial-observability view | `tests/test_world.py` |
| 2 | Households (single / married / kids) as profiles; jammed doors; scheduled novelties (`RulePatch`) | `tests/test_blueprint_and_rules.py` |
| 3 | Hazard-aware symbolic planner (fire-arrival forecast, brute-force ordering of pickups, pacing for followers) with a plain-English reason for every move | `tests/test_planner_and_scenarios.py` |
| 3b | `baseline` / `oracle` agents, same-seed evaluation, GIF renderer, Gymnasium env | `docs/results.md`, `make demo` |
| 3c | Reward: per-tick time cost, potential-based progress, rescue/success bonuses, fire proximity and contact penalties, death charge; reward-hacking audit | `tests/test_reward.py`, `docs/REWARD.md`, `make audit` |

## Phase 4: perceive the mismatch  (next)

**4a. Bayesian novelty detector** · `src/emberpath/novelty.py`
Per-rule Beta posterior over the rule's true probability; flag a rule when the observed trials are far
less likely under the belief than under the data (log Bayes factor), with a minimum trial count and a
sequential-testing correction. Output `Finding`s with posterior mean, 90% interval, trial count.
*Done when:* the 3 detector tests in `tests/test_acceptance_future.py` pass (remove their
`xfail` marker), including at most 5% false alarms on quiet worlds.

**4b. Deep RL baseline** · `src/emberpath/rl_baseline.py`, `env.py`
PPO in PyTorch over the `(H, W, 10)` grid, using the reward in `reward.py`. Then read the per-component breakdown of the trained agent's best episodes: the reward audit so far covers exploits we thought of, and a learning agent is the real test. Train on the nominal world, then evaluate frozen vs. keep-learning
after a novelty. *Done when:* a plot of success vs. environment steps exists for both settings, and the
README answers "why not just RL?" with those numbers (whatever they show).

**4c. Neural perception front-end**  (the literal "neuro" in neuro-symbolic)
Train a small CNN (PyTorch) that maps a smoke-degraded *rendered image* of the house to the symbolic state
(burning cells, dependent positions). The planner then plans from the CNN's output instead of ground truth.
*Done when:* perception accuracy is reported, and the gap-closed metric is re-measured with imperfect
perception. This is where deep-learning training, tuning, debugging and profiling skills show up.

## Phase 5: repair the rulebook  (next)

**5a. Rule repair search** · `src/emberpath/repair.py`
Given findings and all stored events, search over `RulePatch` edits (singles, then pairs) scoring by
log-likelihood minus a per-edit complexity penalty; return the patched rules, the patches, and the margin
over the runner-up. Report ambiguity honestly when two explanations tie.
*Done when:* `test_repair_*` pass; two simultaneous novelties are both recovered.

**5b. Wire the adaptive agent** · `src/emberpath/agents.py` (skeleton already wired)
*Done when:* `emberpath eval --agents baseline oracle adaptive --novelty door_jam` reports gap-closed >= 50%
and the "does not hurt when nothing changes" test passes.

## Phase 6: explain  (next)

`src/emberpath/explain.py::explain_adaptation`, CLI wiring, GIF banners.
Two layers for every adaptation: a **plain-English paragraph** ("At step 4 I noticed the door at (5, 3)
wouldn't open three times in a row. I now assume it is jammed and am using the study route") and a
**technical appendix** (posterior, Bayes factor, trials, runner-up hypothesis).
*Done when:* every `emberpath run --agent adaptive` ends with both layers, and the GIF banner shows
"NOVELTY DETECTED" at the step the agent noticed, next to the step the truth actually changed.

## Phase 7: evaluate and write up  (next)

`scripts/`, `docs/REPORT.md`, `experiments/`
* Table: gap-closed for every household x novelty, with confidence intervals over >= 100 seeds.
* Curves: score vs. number of observations after the novelty (recovery time); false-alarm rate vs. threshold.
* Ablations: no detector / no repair search / perfect-perception vs. CNN perception.
* A 2-page tech-report-style write-up (problem, method, results, limitations, next steps).
* **Profile the planner**: the full results table takes ~4 min on 2 cores. Find the hot spots (`cProfile`),
  fix the worst two (e.g. cache fire-arrival fields, replan only on change) *without changing any result*:
  `make results` before and after must match to the digit.

## Track B: RL environment engineering  (next)

The same simulator, reward and harness, pointed at a different question: *is this a trustworthy environment to train and
evaluate agents in?* Full briefs in `tasks/`.

| Phase | Package | What | Done when |
|---|---|---|---|
| B1 | WP-J | Procedural houses with measured difficulty tiers and a solvability guarantee | 200 seeds solvable; oracle success falls with tier |
| B2 | WP-K | Versioned trajectory dataset with quality validators (replay determinism, label consistency, leakage) | validator catches every injected corruption |
| B3 | WP-I | Text / tool-call interface for LLM agents, offline-testable, no leaked hidden rules | scripted-client tests pass; benchmark script runs |
| B4 | WP-M | Automated exploit search, reward weight sweeps, sparse-vs-shaped ablation | search rediscovers the legacy suicide loophole |
| B5 | WP-L | Post-training study: does training on slices of this data improve a model on held-out houses and novelties? | table with intervals and an honest reading |

## Depth tracks  (stretch; pick by interest)

| Track | What | Why it is interesting |
|---|---|---|
| **Partial observability** | `world.observe(full=False)` already hides what smoke hides. Add a belief tracker so the agent remembers where people were and how long ago. | Real sensors are blind sometimes; memory + uncertainty. |
| **Triage** | When not everyone can be saved, choose who, by expected survivors under uncertainty. | Combinatorial optimization; decisions with real stakes need explaining. |
| **Human on-the-loop** | The agent proposes a plan with a confidence; a scripted supervisor can veto; the agent asks when evidence is ambiguous. | Matches how autonomy is actually fielded: humans supervise, not drive. |
| **Change-point inference** | Infer *when* the rule changed, not just that it did (Bayesian online change-point detection, Pyro/NumPyro). | Recovery speed depends on forgetting stale evidence. |
| **Structural novelty** | A new kind of thing (a window you can exit through) that no existing rule key describes. | Parameter repair is the easy case; this is the research frontier. |
| **Amortized repair** | A small network proposes likely patches; the symbolic search verifies them. | Neural proposes, symbolic disposes. |
| **Two rescuers** | Coordinate two agents with shared beliefs. | Multi-agent teaming. |

## Suggested order if time is short

4a detector -> 5a repair -> 5b adaptive agent -> 6 explanations -> 7 table + write-up.
That is a complete, defensible story. Then add one of: 4b (RL comparison) or 4c (neural perception).
