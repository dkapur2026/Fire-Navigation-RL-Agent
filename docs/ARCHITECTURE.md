# Architecture and design decisions

## Data flow

```
                    truth (hidden from agent)                    agent
              ┌──────────────────────────────┐         ┌─────────────────────────────┐
 RulePatch ──►│ World (Rules = TRUE rules)   │         │ Planner (Rules = BELIEVED)  │
 (novelty,    │  fire · smoke · dependents   │         │  forecast fire → price cells │
  scheduled)  └───────┬───────────────┬──────┘         │  → order pickups → move      │
                      │ Observation   │ Events         └──────▲───────────────┬──────┘
                      │ (what it sees)│ (rule trials)         │ patched       │ Decision(action, reason)
                      ▼               ▼                       │ beliefs       ▼
                   ┌─────────────────────────┐        ┌───────┴───────┐   back into World
                   │ Detector (novelty.py)   │──────► │ Repair        │
                   │ "which rules look off?" │ Finding│ "smallest edit│
                   └─────────────────────────┘        │  that explains│
                                                      │  the data?"   │
                                                      └───────────────┘
```

## Key decisions

**One vocabulary: `RulePatch`.** A novelty is a patch applied to the truth. A repair is a patch applied to the
belief. The agent succeeds exactly when the patch it infers matches the one the world applied. That gives a
clean, testable definition of "adapted correctly" and a single concept to explain to any audience.

**Every rule is a Bernoulli probability.** Spread rates, follow reliability and "is this door jammed" all
reduce to "a coin with bias p". So the evidence format is uniform (`Event(step, key, outcome)`) and the
detector and repairer need no special cases per rule family.

**The event log is the only evidence channel.** The agent never sees the rulebook. It sees only the flips of
the coins. This keeps the project honest: detection is statistical inference, not peeking.

**Separate random streams.** Fire draws from its own seeded stream, so for a given seed every agent faces
the *identical* fire. Differences between agents are then caused by the agent, not luck, and comparisons need
far fewer seeds.

**Oracle and control.** The oracle is the same planner given the true rules at every tick: the ceiling for
any belief-repair method on this planner. With no novelty it must equal the baseline (a built-in control
that catches bugs in the harness).

**Dependents are labelled points in the state space.** "Single / married / kids" are profiles that set
movement speed, flee behaviour and how many people there are (plus follow reliability via `Rules`). The
simulator has no notion of "dog" versus "spouse" beyond those numbers, which keeps the model honest. The
labels exist so a human can follow the story.

**The planner is deliberately symbolic and transparent.** It forecasts when fire reaches each cell using the
*believed* spread rates (`1 / p` expected ticks per cell crossed), compares that with when the rescuer can
get there, and prices cells accordingly. Wrong beliefs make wrong forecasts, which makes wrong routes. This is the
mechanism by which belief repair translates into better decisions.

## Calibration notes (what we tried, what we found)

* **Spread rates.** First-guess rates (carpet 0.45 per neighbor per tick) made fire cross the whole studio in
  about eight ticks, so nothing was survivable. A sweep chose wood 0.10, carpet 0.16, tile 0.02, door 0.07 and
  10 burn ticks, giving a competent baseline 80-95% success with the kids household hardest.
* **Negative result: `wood_flashover`.** The first spread-rate novelty made wood floors burn faster. It produced
  *no* oracle gap: knowing the true rate did not change any decision, because in the ranch every exit route
  crosses the hallway, so a fast hallway fire is deadly whatever you believe. A novelty only measures adaptation
  if a correct belief enables a *different, better decision*.
* **Making spread rates actionable.** Two changes: (1) the planner now forecasts fire arrival from believed rates
  instead of using a myopic distance penalty; (2) the ranch's first dependent moved into the living room, beside
  the fire and the front exit, so "front exit or back exit?" is a real choice. `carpet_flashover` then flips that
  choice, and shows an oracle gap for the kids household (0.65 vs 0.84).
* **Still no gap for one dependent.** For single and married households the best route is unchanged by
  `carpet_flashover`, so the oracle gap is 0. It is kept as a *false-alarm* probe: the detector may flag the
  change, but acting on it should not hurt.
* **`compound` is capped by physics.** A jammed door plus fast carpet is brutal; even the oracle only reaches
  0.1-0.4. Gap-closed on this scenario should be read against that low ceiling.
* **Cost.** Planning dominates run time (~1.0 ms per tick after the trim; detector plus repair add ~10-15 us). A trim
  pass (cached blueprint geometry, a neighbour table, BFS only while someone follows, a fire-crossing cache) made
  evaluation 1.9x faster with every episode byte-identical; `make results` now takes ~40 s on 8 cores. Full profiling
  is still WP-G.

## Phase 1-3 verification and the `hidden_dependent` novelty

Phases 1-3 were re-verified against their tests (world, blueprint/rules, planner/scenarios, reward: 50 passed, 1 skipped)
and the reward audit. Adding the novelty left every pre-existing benchmark episode bit-identical (1440 baseline/oracle
episodes, HEAD vs. new code), so every `none` row still shows an oracle gap of +0.00.

* **Rule.** A new boolean family `occupied:<r>,<c>` sits beside `door_jammed` in `rules.BOOLEAN_FAMILIES`. The nominal
  belief is "nobody unexpected" (p = 0). The novelty is `RulePatch(occupied:<slot>, True)` at tick 0 on a blueprint slot
  the household does not use (ranch: (2, 10) for single/married, (3, 19) for kids; `ValueError` if the house has no
  spare slot, e.g. studio/kids).
* **A real life, hidden.** The extra person (a child, half speed) counts in `world.counts()`, the reward and the score.
  While waiting it is absent from `Observation.dependents` and the `env.encode` channel; it appears once found.
* **Evidence.** Each tick, for every unused slot within Manhattan distance 4 (`HEARING_RANGE`) of the rescuer, the world
  logs `Event(occupied:<slot>, someone is waiting there)`. No random draws (fire-RNG invariant holds), and in worlds
  without the novelty the events are all `False`, which changes no decision.
* **Planner.** Each cell in `belief.occupied` is an extra pickup target ("I heard someone at (r, c)") until the rescuer
  reaches it or it burns. Baseline never hears it; oracle goes from tick 0; adaptive goes once the detector flags it.
* **Late reveal.** With hearing alone, about 23% of baseline-route episodes never came within range, capping gap
  closed near 77%. Widening the range would make the novelty trivial, so instead, from tick `REVEAL_STEP` = 10, a
  person still hiding is heard from anywhere (someone outside shouts where they are; no RNG). The news now always
  arrives, at a fixed late time, and the benchmark measures adaptation to late information rather than route luck.
  Effect (60 seeds, adaptive only; baseline/oracle cannot read events and are unchanged): single 0.83 -> 0.93,
  married 0.86 -> 0.92, kids 0.79 -> 0.87. Adaptive now beats the oracle on single and married, because the oracle,
  going for the visitor from tick 0, dies more often (see "oracle is not an upper bound").
* Fixed along the way: ASCII render drew a lost dependent as alive.

## Detector (`novelty.py`, SPEC A1)

* **Score.** Per rule key, the log Bayes factor of a free `p` against the point null `p = belief`. Stochastic keys
  use `Beta(1, 1)`; boolean keys (`door_jammed`, `occupied`) use a U-shaped `Beta(0.1, 0.1)` alternative and a
  near-deterministic null (contrary outcome likelihood `noise = 1e-4`). One contrary outcome scores 8.5 and flags at
  once; agreeing outcomes can only lower the score. No door-specific code: everything goes through `Rules.prob`.
* **Sequential peeking without a growing threshold.** The Bayes factor against a point null is a mixture SPRT, so
  Ville's inequality bounds P(ever >= threshold) by e^-threshold per key however often it is checked: 0.25% per key at
  the default 6.0. `min_trials = 5` guards stochastic keys only.
* **Posterior.** Mean and 90% interval from a lazily built 1001-point grid; scores are cached. Cost ~15 us per tick.
* **`reset_belief`** keeps the evidence and changes the null after a repair. Optional exponential `decay` exists but
  is off: it trades accuracy for flag flapping and loses the Ville guarantee.
* **Trade-off** (`scripts/false_alarms.py`; ranch, baseline's trajectories, 3 households x 30 seeds, peeking every tick):

  | threshold | 2 | 3 | 4 | **6 (default)** | 8 | 10 |
  |---|---|---|---|---|---|---|
  | false alarms, quiet worlds | 10.0% | 4.4% | 3.3% | **0.0%** | 0.0% | 0.0% |

  Median ticks from change to flag at 6.0: door_jam 4 (equal to the first time the door is tried, a floor no
  detector can beat), carpet_flashover 2, compound 4 and 4, hidden_dependent 10 (the reveal at `REVEAL_STEP`;
  0% missed; before the reveal existed it was 25 with 23% never heard). At threshold 10 the occupied rule needs two
  calls, so the flag comes one tick later (11).

## Repair (`repair.py`, SPEC A2)

* **Hypothesis space.** Only flagged keys with at least one event. Spread and follow probabilities: a 0.05 grid plus
  the detector's posterior mean. Boolean families: the toggle `1 - belief`. Pairs (up to `max_edits = 2`) only when the
  best single edit leaves a flagged key still surprising.
* **Score.** Sum of per-key Bernoulli log-likelihood gains over all stored events (p clipped to [1e-3, 1 - 1e-3])
  minus `complexity_penalty = 2.0` nats per edit, so the smallest explanation wins.
* **Confidence.** `runner_up_margin` is the gap to the best hypothesis editing a *different* key set (inf if none);
  below `ambiguity_margin = 1.0` nat the result is `ambiguous`, and the agent says so ("I think X changed, but Y
  would explain it almost as well").
* **Tested.** 37 unit tests: values recovered within tolerance, two simultaneous novelties both recovered, no-op on
  empty findings, ambiguity on confounded evidence, one patch beats several, patched `prob` near the generating truth.
* **Negative result.** On every shipped preset the search returns exactly what the naive ablation (flip flagged rules to
  their posterior mean) returns: each preset changes one key per flag, so there is nothing to search. The search earns
  its keep only on confounded or multi-key evidence, which the presets do not yet contain (`naive_repair` and `adaptive`
  columns are identical in `docs/adaptive_study.md`).

## Spouses who act alone, and forcing doors (`world.py`, `planner.py`)

* **Forcing.** Walking into a jammed door from its hallway side is also an attempt to kick it in (`force_prob:door`,
  default 10%, a rule like any other, so the detector can learn it; drawn from a third RNG stream so fire stays
  identical). From inside a room it cannot be forced (doors swing inward). "Outside" is the side whose door-bounded area
  touches more doors, i.e. the hallway. The planner prices a believed-jammed door at 1/force_prob ticks when it stands
  on the outside, else as a wall; a door seen forced open is open. **Negative result first:** with forcing allowed from
  both sides, any rate (even 2%) erased the `door_jam` gap, because the baseline's stubborn bumping became forcing (at
  10%, kids: baseline 0.89 vs oracle 0.70). Outside-only restores it exactly on ranch (single 0.00 vs 0.82).
* **Spouses.** Married and family spouses act alone: nearest known waiting child first (the child then follows them),
  else the nearest exit, avoiding cells within 2 of flames when any route allows. Flames within 2, or a jammed door
  they walk into (which the rescuer sees as a door event), start a 10-tick panic: random steps from the dependents'
  RNG. Reached by the rescuer, they calm down and follow. The planner treats a spouse as a pickup when they are
  panicking, within 3 of flames, or have no believed route out; it leaves a child to a spouse who will reach it first.
* **Effect on ranch** (60 seeds): married door_jam gap gone (0.90 / 0.90; she walks out herself), married
  hidden_dependent 0.73 / 0.88 / adaptive 0.93; family (spouse + 2 kids, uses slot 4 added to ranch for the visitor)
  0.93 with no novelty, and only `hidden_dependent` separates the agents (0.70 / 0.88 / 0.88). Single and kids rows
  are unchanged.

## Survival-first pickup order (`planner.py`)

The planner used to collect people in the order with the cheapest route. In the kids ranch the fire starts in the south
room beside one child, so it always fetched the safe child in the north first, and the other burned while waiting (or on
the way out) in a third of fires with no novelty at all. Now, for each order, it estimates when it reaches each person
(at the slowest follower's pace once someone follows: `DependentView.pace`, part of the household description) and
whether that is before the fire. It picks the order that reaches the most people in time, then the cheapest; the reason
says so ("The fire will reach them first, so they go before the closer pickup").

* **The forecast had to be made pessimistic.** `fire_eta` chains mean crossing times along one path, but real fire tries
  every burning neighbour each tick: over 200 simulated fires, the 10th-percentile arrival was 0.44-0.59x the forecast at
  five ranch cells (median about 0.65x). With the raw forecast the new ordering changed almost nothing; it now judges
  "in time" against `save_margin` = 0.5x the forecast. Route costs still use the raw forecast.
* **Effect, ranch/kids, 60 seeds, share of fires where every child gets out** (old order -> new): no novelty 67% -> 92%;
  door_jam adaptive 12% -> 43%; carpet_flashover 43-73% -> 92% for every agent; hidden_dependent adaptive 72% -> 90%;
  compound 0% -> 0% (even the oracle cannot save both). Rescuer deaths are about the same (5-7%). Single and married are
  unchanged (one declared person, so the order rarely matters).
* **Negative results.** (1) Fast carpet no longer creates an oracle gap for kids (all agents 0.96), so
  `carpet_flashover` no longer separates the agents on ranch; `test_carpet_flashover_has_an_oracle_gap_for_kids` is a
  strict xfail recording that. (2) Going first to the endangered child can cost the rescuer too (kids/hidden_dependent
  seed 0: every agent now dies at t = 48). (3) `save_margin` was set from the forecast's measured error, not by sweeping
  the benchmark, but it is one number tuned on one house.

## Adaptive agent (`agents.py`, SPEC A3, A4, E4-E6)

* **Loop.** One planner reused across ticks. The detector runs only when events arrived; repair runs only on fresh
  findings (more trials than last seen for that key); after an edit the detector is re-baselined with `reset_belief`.
  Every edit becomes the decision reason ("Updating my beliefs: ..."), with the runner-up when ambiguous. A rejected
  hypothesis is reported once per (key, value).
* **Results** (`make results`, 60 seeds; `docs/adaptive_study.md`, 100 seeds, paired bootstrap): door_jam gap closed
  100% / 96% / 61% for single / married / kids (100 seeds: 97% / 98% / 65%); hidden_dependent 122% / 117% / 87% with the tick-10 reveal (100 seeds: 116% / 114% / 84%);
  every `none` row +0.00 with 0 false alarms in 300 episodes and 0% wrong edits on novel ones.
* **Information-limited cells.** kids/carpet_flashover (4%) and married/compound (0%) are not agent bugs. The
  "oracle told k ticks late" curve shows the value of the knowledge is gone within 2-3 ticks (kids/carpet: 0.84 at
  k = 0, 0.68 at k = 2), while carpet evidence needs ~3 ticks to accumulate. kids/door_jam is slower (median delay 21
  ticks) because two slow children reach the jammed door late.
* **The oracle is not an upper bound.** kids/compound closes 115% (100 seeds; 117% at 60): an oracle that knows the true carpet rate routes
  more cautiously from tick 10 than an agent that learns it a few ticks later, and here caution loses.
* **Repeat repairs.** After the hidden person is found, `occupied` events flip back to False and the agent toggles the
  rule back: cheap, correct, but extra repair calls.

## Lifelong adaptation (`evaluate.run_lifelong`, `emberpath improve`, SPEC A8)

The adaptive agent normally starts every episode from the textbook rules. `run_lifelong` runs it several times in the
same changed house and starts each episode from what the last one learned (`AdaptiveAgent.lessons`: the final belief,
except that a hiding place where someone was heard stays occupied, because the silence after they are found is not
evidence that nobody hid there). By default every episode replays the same fire, so first vs last isolates knowledge.

**Result: knowing sooner is not always better.** Ranch, 3 episodes, same fire, 30 seeds, mean score:

| household | novelty | first | last | last better / worse (seeds) |
|---|---|---|---|---|
| single | door_jam | 0.80 | 0.80 | 2 / 2 |
| single | carpet_flashover | 0.87 | 0.87 | 0 / 0 |
| single | compound | 0.17 | 0.17 | 1 / 1 |
| single | hidden_dependent | 0.90 | **0.75** | 1 / 7 |
| married | door_jam | 0.83 | 0.90 | 4 / 2 |
| married | carpet_flashover | 0.93 | 0.93 | 0 / 0 |
| married | compound | 0.00 | 0.13 | 4 / 0 |
| married | hidden_dependent | 0.90 | **0.78** | 1 / 6 |
| kids | door_jam | 0.60 | 0.68 | 9 / 5 |
| kids | carpet_flashover | 0.95 | 0.95 | 0 / 0 |
| kids | compound | 0.40 | 0.42 | 2 / 1 |
| kids | hidden_dependent | 0.93 | 0.93 | 0 / 0 |

The last episode lands on (or near) the oracle's score, because it starts with the oracle's knowledge. Where the oracle
is worse than the adaptive agent (single/married hidden_dependent), the extra knowledge hurts the same way:
the planner goes for the hidden person from tick 0 and gets caught by fire more often. This is a planner limitation (it is
not risk-aware enough to use early knowledge well), not a belief error; we report it rather than pick flattering cells.

## Procedural houses (`procgen.py`, SPEC W8)

`generate(seed, tier)` builds a room-and-corridor house as ASCII and parses it with `Blueprint.parse`; seeded per
(tier, seed); `--blueprint gen:<tier>:<seed>` and `emberpath gen` expose it. Acceptance per house: `validate` passes for
three dependents, `door_jam` has a door to jam, the start is more than 3 from every exit, fire more than 2 from the
rescuer and more than 1 from every slot, and 3 slots (one spare for `hidden_dependent`).

**Measured, not guessed.** Each house is measured on the hardest household (kids): `p_lost` (fire-only Monte Carlo
chance the fire reaches a dependent before the earliest pickup) and `margin` (largest safety margin of any escape plan
against the planner's fire forecast). Tier 5 if p_lost > 0.25, tier 4 if p_lost > 0, else tier 3 / 2 / 1 for margin
<= 0 / <= 45 / > 45. Thresholds were picked from 400 exploratory houses on a separate seed stream; the generator
resamples until the measured tier matches. `scripts/coverage_report.py` (100 houses, 20 per tier, oracle over 30 fires):

| tier | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| oracle success | 0.99 ±0.01 | 0.85 ±0.08 | 0.76 ±0.12 | 0.47 ±0.11 | 0.21 ±0.09 |

77 / 100 distinct topologies; pairwise distance (mean |z| over metrics) 1.05. All four rule-change presets build on every
household on 1000/1000 houses (generator author's check). Negatives: route length and fire-race margin barely predict success (p_lost does,
r = -0.75); generation takes ~2 ms for any valid house but ~20-30 ms tier-targeted, below the "hundreds per second" goal.

## Trajectory dataset (`dataset.py`, SPEC D1-D3; details in `docs/DATA.md`)

Schema v1, one JSON line per episode: house name/hash, household, novelty, seed, agent, per-step compact observation,
action, reason, reward components and events, outcome, return, efficiency. Splits are by (house, seed) group. Validators:
schema, replay determinism, return = sum of components, outcome vs final state, duplicates, split leakage, outliers, and
leak checks (hidden person shown waiting, door event without a bump, hearing beyond range). Verified: a 200-record build
(4 workers, 4.7 s) validates with 0 issues and is byte-identical to a 2-worker build; 17 tests inject each corruption.
Caveats: duplicates are dropped at build and listed in `manifest.json` (80 in that build: with no novelty, baseline and
oracle are identical by design); `--episodes 200` covers every pair of (house, household, novelty) but not every triple
(10 coverage gaps listed); observations are full, not smoke-limited.

## Reward robustness (`scripts/find_exploits.py`, `scripts/reward_sweep.py`, SPEC R11; details in `docs/REWARD.md`)

* **Search.** A 5-parameter scripted policy family (weights for dependent / exit / fire / wait, plus noise), random
  search plus a simple evolution strategy, confirmed on fresh seeds. Against the legacy reward the small budget (240
  policies, 960 episodes) rediscovers the suicide loophole (dies 92% of the time, return -1.45 vs. baseline's -1.58).
  Against the current reward: 0 confirmed loopholes at small (960 episodes) and medium (4800) budgets.
* **Fixed.** A hidden waiting person no longer counts in the pickup or fire-proximity terms (the reward leaked its
  location at up to 0.39 per tick, which an RL learner could exploit); `partial` endings are charged time to the
  horizon. Outcomes, scores and `docs/results.md` are unchanged; only returns moved.
* **Not fixed: door-jam potential leak.** The progress potential uses true distances, so with a jammed door it differs
  from a no-novelty control by up to 0.36 per tick while observations are identical. A learner could read the jam from
  the reward. Left for a decision (fix: compute the potential on the believed map, which changes the reward contract).
* **Sweep.** Honest-first and dying-never-pays hold at default weights on all four presets; they break at progress
  weight x4 (dying pays on door_jam) and proximity weight x4 (honest not first on compound). Mean Kendall tau across
  settings >= 0.85 (min 0.40 at the progress break).
* **Sparse vs shaped** (ES on studio/married): shaped reaches 0.42 success, sparse 0.03 (107 of 120 generations flat).
* **Signal reliability.** Returns are noisy: ~380 paired seeds to detect a 0.5 return difference on door_jam (sd 3.5).
  Small audits like `make audit` (20 seeds) are a smoke test, not a measurement.

## Why this is not just "A* with fire"

A* with a fire penalty solves the *planning* problem when the rules are known. The project is about what
happens when they are not: detecting that the rulebook is wrong from sparse, noisy trials, localizing which
rule, correcting it, and doing so with calibrated uncertainty and a human-readable explanation. The planner is
the stable part; the research is the loop around it.
