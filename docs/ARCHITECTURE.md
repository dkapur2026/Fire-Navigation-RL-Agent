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
* **Trade-off** (`scripts/false_alarms.py`; ranch, baseline's trajectories, 6 households x 30 seeds, peeking every
  tick; rerun in the final verification after the script was fixed to skip households the ranch cannot host, which had
  made it crash):

  | threshold | 2 | 3 | 4 | **6 (default)** | 8 | 10 |
  |---|---|---|---|---|---|---|
  | false alarms, quiet worlds | 11.1% | 5.0% | 3.9% | **0.0%** | 0.0% | 0.0% |

  (The earlier 3-household run gave 10.0 / 4.4 / 3.3 / 0.0%.) Median ticks from change to flag at 6.0 (current run):
  door_jam 4 (equal to the first time the door is tried, a floor no detector can beat; 31% of baseline episodes never
  try it), carpet_flashover 4, compound 4 and 6, hidden_dependent 5. Earlier run: carpet 2, compound 4 and 4,
  hidden_dependent 10 (the reveal at `REVEAL_STEP`;
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

## Carrying, exits in time, and a gentler carpet novelty

* **Carrying** (`world.py`). With flames within `CARRY_RADIUS` = 3, a parent (the rescuer, or a spouse acting alone)
  picks up the child they lead that is nearest the fire (children only: see below). They share a cell from then on (rescued or lost
  together) and the carrier skips every 4th tick (25% slower). Followers keep heading to their leader's position, so
  they leave by the same exit; a spouse's followers switch to the rescuer only if she is lost or her exit is cut off.
* **Exit in time** (`planner.py`). Leading people out, the rescuer takes the cheapest exit whose route the group (at the
  slowest walker's pace, 4/3 while carrying) covers before the pessimistic fire forecast at every cell; if none is
  safe, the cheapest exit. The reason says when a nearer exit was passed over.
* **`carpet_flashover` is now 45%** (60% made carpeted rooms on the house unsurvivable for every agent; 40% barely
  separated baseline from oracle; 50% was tried). **Tile now spreads at 25%** (was 2%, then 12%): this changes every
  fire; the mostly-tiled house becomes much more dangerous.
* **2026-11-08: tile back down to 15%** (history 0.02 -> 0.12 -> 0.25 -> 0.15). Every fire changes again. 60 seeds,
  mean over the 29 rows of `make results` / `make results-house` (baseline / oracle / adaptive): ranch 0.754 / 0.865 /
  0.856 -> 0.749 / 0.869 / 0.858 (flat); house 0.529 / 0.577 / 0.584 -> 0.627 / 0.678 / 0.677 (+0.10). Every `none`
  row still has oracle gap +0.00. Most moved: house single compound 0.12 -> 0.68, single carpet_flashover
  0.22 -> 0.57, single none and door_jam 0.27 -> 0.52 (all agents), house family carpet_flashover baseline
  0.72 -> 0.88 (oracle 0.72 -> 0.82). Worse: ranch married none 0.98 -> 0.93 (a slower fire scores lower here,
  not investigated); house family carpet_flashover oracle gap -0.01 -> -0.07; gap closed falls on house
  family_dog door_jam (62% -> 44%, interval now includes 0), family door_jam (107% -> 77%) and kids
  hidden_dependent (105% -> 78%). House single compound now scores above single none (0.68 vs 0.52).
* **Pickup priority.** Orders are ranked by: (1) which children are reached before the pessimistic forecast, the most
  endangered child first (saving the child the fire reaches soonest outranks saving a safer one); (2) no lower-priority
  person before a higher one (children, then adults, then the dog: the dog never delays a child); (3) everyone reached;
  (4) *keep heading for the same person* (the rescuer used to flip between equal plans each tick, e.g. Max then Rex);
  (5) whoever the fire reaches soonest (the forecast follows walls and doors); (6) route cost. Before (1)-(2), a plan
  saving the dog plus a safe child beat a plan saving the endangered child, so the rescuer went for the dog while Max
  burned (house family_dog seed 0). Measured with the adaptive agent, 60 seeds, kids_dog and family_dog on both houses
  x {none, door_jam, carpet}: children out 64% -> 73%, dog out 79% -> 65% (vs. the same planner with the dog ranked
  like a child). Largest gains: ranch kids_dog door_jam 51% -> 77%, house family_dog door_jam 44% -> 78% of children.
  Cost: house family_dog carpet_flashover rescuer deaths 5% -> 38% (the endangered child is in the burning carpet
  room); the score, which counts the dog as a full life, falls slightly on the house (0.559 -> 0.552) and rises on
  ranch (0.816 -> 0.829).
* **Carrying is triggered by flames near the child** (within 3 walking steps; fire does not cross walls), not near the
  parent; the parent picks the child up once beside them. The rescuer turns back for a following child who has
  fallen behind with flames that close (`go_back_for_children`).
* **Burn tolerance.** The rescuer and a spouse draw a tolerance of 2-5 ticks in flames (own RNG stream; fire
  unchanged) and fall when their total burns reach it. Children and the dog have none; a carried child is shielded
  and is lost only if the carrier falls. This is what makes going back worth it: without tolerance, going back
  roughly broke even and doubled rescuer deaths in house carpet fires (family 18% -> 42%); with it, the summed
  baseline + oracle score over 24 child cells rises 36.29 -> 36.77 (house kids carpet 0.42 -> 0.57), though house
  family_dog carpet still loses (0.55 -> 0.47, rescuer deaths 25% -> 47%).
* **Negative result:** married/hidden_dependent on ranch has no oracle gap left (0.92 vs 0.93, 60 seeds): the spouse
  walks out alone and the adults now survive brief contact with fire. Recorded as a test that asserts the tie.
* **Caught in flames** (`fire_escape`, `out_when_burned`). A rescuer standing in a burning cell used to see every
  route as walled off (flames are walls, even underfoot), stand still and burn to death. Now it first steps to the
  nearest cell at least 2 steps from any flame (stepping through flames if needed), and once burned it makes no new
  pickups and heads out with whoever is with it. Measured over every household x novelty on both houses (3 agents,
  40 seeds): ranch 0.816 -> 0.820 with rescuer deaths 3.5% -> 2.1%; house 0.558 -> 0.556, deaths 5.9% -> 5.1%;
  worst cell house family carpet_flashover (-0.11). The "straight out once burned" part alone changes nothing
  measurable; it is kept because it was asked for.
* **Exits are safe places, and carried groups may cross a little fire.** A rescuer caught in flames heads for an exit
  when that is at least as near as any cell clear of the fire (fire never enters an exit); it used to step to the
  nearest clear cell, which could be deeper in the burning room (house family carpet_flashover seed 0: died at t=24
  with the child two steps from the exit; now 3/3 out). With no fire-free route and *nobody walking behind* (everyone
  with the rescuer is carried), it crosses to the exit through the fewest burning cells, if those are fewer than its
  remaining burns. Effect: house 0.552 -> 0.564 (family carpet 0.52/0.54/0.52 -> 0.72/0.72/0.69, family_dog carpet
  0.41/0.50/0.47 -> 0.53/0.61/0.55), ranch 0.829 -> 0.825 (no cell moved by 0.04 or more).
* **Rejected: crossing flames as a general fallback** when no fire-free route exists. It cost 0.02 on ranch and up to
  0.12 per cell: the rescuer crossed, its walking children could not follow and burned, whereas holding still lets
  the flames burn out (10 ticks) and then everyone walks out (ranch kids seed 0: success without it, 0/2 with it).
* **Effect, measured by switching each change off in turn (baseline agent, ranch, 60 seeds):**
  - Carrying the dog cost lives: family_dog with no novelty 0.89 -> 0.83, rescuer deaths 8% -> 15% (the fast dog slows
    the rescuer near the flames). Children walk at half pace, so carrying them (4/3) is faster: carrying is now
    children only, and that cell is back to 0.86.
  - Exit in time: single compound oracle 0.20 -> 0.58, kids door_jam adaptive 0.67 -> 0.75; but married with no
    novelty 0.95 -> 0.92 (a longer "safe" route is not always safer).
  - Carpet rate on the house (baseline / oracle / adaptive): at 40%, kids_dog 0.69/0.74/0.74 and family 0.51/0.55/0.55
    (survivable, but almost no oracle gap); at 50%, kids_dog 0.41/0.42/0.42, family 0.24/0.20/0.18, family_dog
    0.31/0.54/0.38 (a real gap again for family_dog, adaptive closes about a third of it, but carpet rooms are deadly
    again). On ranch the novelty opens no gap at either rate.

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

## Fire distances follow walls everywhere (`planner.py`, `world.py`; `tests/test_fire_forecast.py`)

Real fire moves only between 4-neighbouring floor cells, never into an exit or an already burnt cell, and burns a jammed
door like any door. The forecast (`fire_eta`), the carry trigger (`fire_steps`, the planner's `near`), the escape
check and the reward's proximity term already measured that way. Four places used straight-line (Manhattan) distance
through walls instead, and one ignored burnt cells. All now use walking distance (BFS over floor):

* **Forecast firebreaks.** `fire_eta(burning, burnt)` no longer routes the fire through burnt-out cells (they cannot
  catch again), so burnt ground carries no fire risk in route costs.
* **"Fire nearby, don't wait for stragglers"** (`_maybe_wait_for_followers`) was Manhattan <= 3: a flame in the next
  room through a wall stopped the rescuer waiting for a child.
* **Spouse panic and "keep clear of flames"** (`World._move_adult`) were Manhattan <= `PANIC_RADIUS`; the planner's
  "the fire is close to them" (`_needs_help`, radius + 1) mirrored it. Both are now walking distance, so they still agree.
* **The dog bolting** (flee within 2, step to the neighbour farthest from fire) is now walking distance.
* **Left as is, on purpose:** hearing (`HEARING_RANGE`, sound carries through walls) and the partial-observation
  visibility radius (`BASE_VISIBILITY`, Manhattan through walls; only used with `full_obs=False`). Adjacency checks
  (`manhattan(...) <= 1`) are not distances through walls.
* **Effect** (60 seeds, every household x novelty, tile at 0.15, mean score before -> after): ranch baseline 0.756 ->
  0.749, oracle 0.872 -> 0.868, adaptive 0.859 -> 0.858; house baseline 0.626 -> 0.627, oracle 0.677 -> 0.677, adaptive
  0.676 -> 0.677; control gaps stay +0.00. Most of the ranch change is the planner part (0.756 -> 0.751 baseline).
  **Mild negative result:** the through-wall Manhattan check was a lucky proxy for "fire will come round soon". In ranch
  married (no novelty 0.97 -> 0.93) the rescuer used to turn back for a spouse in the hallway above the burning room;
  now it leads out, and the spouse later walks into the fire that has come round through the doors (seed 14). The
  correct geometry is kept; a forecast-based "the fire will reach them soon" would be the honest fix, not Manhattan.

## Adaptive agent (`agents.py`, SPEC A3, A4, E4-E6)

* **Loop.** One planner reused across ticks. The detector runs only when events arrived; repair runs only on fresh
  findings (more trials than last seen for that key); after an edit the detector is re-baselined with `reset_belief`.
  Every edit becomes the decision reason ("Updating my beliefs: ..."), with the runner-up when ambiguous. A rejected
  hypothesis is reported once per (key, value).
* **Results** (current `docs/results.md`, 60 seeds, and `docs/adaptive_study.md`, regenerated in the final
  verification: 100 seeds, all six households, paired bootstrap): door_jam gap closed single 102% (100 seeds: 95%),
  kids 77% (81%), kids_dog 81% (86%); married, family and family_dog have an oracle gap under 0.05 (at 100 seeds
  family_dog is 100%). hidden_dependent 108% / 100% / 100% / 100% for single / kids / kids_dog / family (100 seeds:
  117% / 100% / 100% / 100%). Every `none` row has oracle gap +0.00. False alarms: 1 in 600 quiet study episodes
  (family_dog, seed range 0..99; its score is unchanged at two decimals), 0 in the other five households; 0% wrong
  edits on novel episodes.
* **Detection delay** (study, median ticks from the change to the right edit): jammed door 5 for every household
  whose rescuer tries that door (single, kids, kids_dog, family_dog; an earlier version of this note said 21 for kids,
  from before the dynamics changes); the married and family rescuers mostly never try it (the spouse leads out another
  way), so the jam is found in only 8% / 0% of episodes, which costs nothing. Fast carpet 6 at t=2, 8.5 when it
  arrives at t=10 in `compound`. Hidden visitor 11 (single, kids: the tick-10 reveal) or 4-6 when its slot is within
  earshot early.
* **Information-limited cells.** `carpet_flashover` now has no oracle gap on the ranch at all (survival-first ordering
  already covers it). Where knowledge does matter, its value can vanish within a few ticks: an oracle told about the
  `compound` changes 3 ticks late drops from 0.57 to 0.42 on single, while the carpet evidence takes 6-9 ticks to
  accumulate. single / compound is the weakest door-jam-family cell (56% at 100 seeds, 62% at 60).
* **The oracle is not an upper bound.** kids/compound closes 113% at 60 seeds (no agent gets both children out): an
  oracle that knows the true carpet rate routes more cautiously from tick 10 than an agent that learns it a few ticks
  later, and here caution loses.
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
  policies, 960 episodes) rediscovered the suicide loophole (dies 92% of the time, return -1.45 vs. baseline's -1.58).
  With today's dynamics it still finds it in-sample (the slow test passes), but the fresh-seed recheck no longer
  confirms it (final verification; `docs/REWARD.md`).
  Against the current reward: 0 confirmed loopholes at small (960 episodes) and medium (4800) budgets.
* **Fixed.** A hidden waiting person no longer counts in the pickup or fire-proximity terms (the reward leaked its
  location at up to 0.39 per tick, which an RL learner could exploit); `partial` endings are charged time to the
  horizon. Outcomes, scores and `docs/results.md` are unchanged; only returns moved.
* **Fixed later: door-jam potential leak.** The progress potential used true distances, so with a jammed door it
  differed from a no-novelty control by up to 0.36 per tick while observations were identical, and `jam_snoop` read the
  jam from its own payments. Distances now use only bumped jams (`reward.known_jams`).
* **Fixed in the final verification: partial credit at the end.** The potential was not zeroed at the end of an
  episode, so a failure kept its final distances, and with evidence-only jams `stall` and `door_shy` out-scored the
  honest baseline stuck at the jam (-14.36 and -14.24 vs -14.66), against AGENTS.md invariant 8. The distance terms
  are now 0 at every terminal state (a fixed charge per loss to fire stays, so letting someone burn is still worse than
  an unresolved timeout). `make results` / `make results-house` are byte-identical (the planners never read the
  reward); only returns moved. Test: `test_exploit_no_ending_earns_partial_shaping_credit`.
* **Sweep.** Honest-first and dying-never-pays hold at default weights on all four presets; they break at progress
  weight x4 (dying pays on door_jam) and proximity weight x4 (honest not first on compound). Mean Kendall tau across
  settings >= 0.85 (min 0.40 at the progress break).
* **Sparse vs shaped** (ES on studio/married): shaped reaches 0.42 success, sparse 0.03 (107 of 120 generations flat).
* **Signal reliability.** Returns are noisy: ~380 paired seeds to detect a 0.5 return difference on door_jam (sd 3.5).
  Small audits like `make audit` (20 seeds) are a smoke test, not a measurement.

## PPO baseline (`rl_baseline.py`, SPEC N1, R10; details in `docs/RL.md`)
* **Negative result.** PPO with a small CNN, 2.2 M steps (30 CPU minutes), reaches 42% (dog) and 7% (kids) held-out
  success on the nominal ranch against the planner's 89% and 96%, and never learns `door_jam` or `hidden_dependent`
  in 1 M steps with novelties sampled per episode. It is a budget result (one seed, no tuning, still rising slowly).
* **A time-remaining channel was appended to `env.encode` (channel 10).** Without it a greedy policy cannot tell tick 3
  from tick 90 in the same layout and loops (probe: 0% greedy vs 85% sampled on studio). Channels 0-9 are unchanged.
* **`PPOAgent` rebuilds `encode` from `Observation` + its own events** (tested equal on every policy channel), so it
  goes through the same harness as the planner agents without touching the world. The smoke channel is dropped from
  the policy input because `Observation` carries smoke only at the rescuer's cell.
* **The progress-potential door-jam leak** showed no measurable effect on this policy (see docs/RL.md); it was fixed
  later in `reward.py`, after these checkpoints were trained.
* **BC+PPO.** Cloning the adaptive agent on training fires (plus a policy-side hearing plane and a wall mask) and
  fine-tuning with PPO for 600k steps reaches 0.930 pooled on the 8 trained ranch settings against 0.903 for the
  adaptive agent (+0.026 [+0.006, +0.046]) and 0.327 for PPO from scratch; one training seed, held-out = new fires in
  the same houses, and below the planner on the unseen `compound` (dog: -0.10 [-0.17, -0.04]). A GRU clone was
  0.03 worse than the feed-forward one. All learned checkpoints predate the terminal-potential fix and were not
  retrained.

## DreamerV3 (`dreamer.py`; details in `docs/DREAMER.md`): a negative result
A compact DreamerV3 (RSSM world model, imagined actor-critic) was tried because PPO lagged the planner. After 82 CPU-minutes
(~4,600 updates, 1-2 orders of magnitude below the paper) it rescued nobody on any held-out ranch fire and failed the studio
sanity check; its reward head never learned the large terminal rewards and it never reliably tracked the rescuer. No bug was
found (`tests/test_dreamer.py`, 30 tests, pass). It is kept, tested and runnable, but on CPU it is the weakest agent
in the project, and the Dreamer work was stopped.
Planner guidance did not change that. Seeding its replay with 82k transitions of the adaptive agent (training seeds,
tapered from 50% of each batch to 0 by 60% of the run, autonomous at test time) still gave 0% everywhere. A probe shows
why: the same encoder without the RSSM learns the success reward from those demos in 400 updates, while the world
model's reward head stays constant. The latent ignores the input, so the bottleneck is the representation, not the
data (`docs/DREAMER.md`, "Planner-guided Dreamer").

## 2026-11-09: `door_jam` jams several doors (`scenarios.py`; `tests/test_door_jam_multi.py`)
Asked for: "jam 2-4 random doors on the house and 1-2 on the ranch, so we can definitively see whether the jammed
doors affect the RL agent's behaviour". Before, `door_jam` jammed ONE door, the first on the mission route, the same
on every seed: (5,3) on the ranch, (5,4) on the house, where it was off the route the planner actually takes for most
households (house door_jam oracle gap +0.00 for single, married, kids, kids_dog).

* **How many.** `jam_count_range(bp)`: uniform in [max(1, doors // 6), max(1, doors // 3)]. studio (2 doors) 1, ranch
  (6) 1-2, house (12) 2-4, procedural houses in proportion. The drawn size is always achieved on the three hand-drawn
  houses for every household (tested); elsewhere the draw falls back to fewer doors only if no set of that size is valid.
* **Which.** `draw_jammed_doors` uses `jam_rng(seed)` = `SeedSequence(seed, spawn_key=(0xD00A,))`, a stream of its
  own: World spawns keys 0-3 (fire, dependents, forcing, burn tolerance), so the fire and every other stream are
  untouched and the draw depends on the seed alone, never on an action. A uniformly random set of that size, redrawn
  (from the same stream, so deterministically) until it passes the checks below; after 64 misses, uniformly among
  all valid sets (`_valid_sets`, enumerated and cached per house). At least one jammed door is on the mission route,
  as the single jam always was.
* **Seeded presets.** The draw needs the seed but presets are called as `NOVELTIES[name](bp, household)` everywhere
  (and by files other agents own). So `door_jam` returns a `world.SeededNovelty` and `World.__init__` resolves it with
  its own seed (`resolve_novelties`); `scenarios.novelty_specs(name, bp, hh, seed)` gives the concrete patches. Every
  agent built on a seed faces the same jams, with no caller passing the seed twice.
* **Solvable** (`jam_set_ok`), without forcing anything: (1) the start's jam-free region holds an exit and everywhere a
  dependent can stand when the jam lands (tick 1); it is closed under movement, so nobody drifts out of reach;
  (2) wherever the rescuer can stand at tick 1, including on a door that jams under it, an exit is reachable, so it is
  never sealed in; (3) forcing never pays: start -> each dependent and each dependent -> exit, the way round is no
  longer than the way through with hallway-side kicks at their expected 10 tries each. Tested on house, ranch, studio
  and 12 procedural houses (40 seeds x every household), plus a random walker that must always have an exit open.
* **Negative result that shaped (3).** The first draft had only (1)-(2). On the ranch it could jam (8,4), the hallway
  door of the dog's room. The oracle, knowing, took the 38-step way round past the fire; the baseline walked into the
  door from the hallway and kicked it in (12 + ~10 ticks). ranch/single, the 15 of 40 seeds that jammed (8,4) alone: oracle 0.00,
  baseline 0.60. The planner forces a door only to reach an exit, and only from where it stands, never to reach a
  person (`planner._enter_cost`, `_best_order` timing excludes jammed doors), so a known jam can make the oracle
  worse than ignorance. (3) removes those draws rather than changing the planner (not this package's file, and the
  adaptive agent is house-rl's BC teacher). Cost: on the ranch only (5,3) can jam alone and the 2-door sets are
  {(5,3),(5,18)} and {(5,3),(8,17)}: the ranch's door_jam is the old jam plus, on half the seeds, a second one.
  On about 5% of procedural houses (240 of 5,000 house x household cells, 200 seeds x 5 tiers) every route door fails (3); there the draw
  falls back to (1)-(2) only, the guarantee the legacy jam had, and the oracle can be hurt the same way.
  Fixing the planner to price forcing on pickup legs would let (3) go.
* **Legacy.** `door_jam_single` / `compound_single` (`scenarios.LEGACY_NOVELTIES`) are the old presets, and
  `EMBERPATH_DOOR_JAM=single` puts them back under the old names. **The pre-registered ranch formal study
  (`runs/formal_eval`) used the single jam**; re-run it with `EMBERPATH_DOOR_JAM=single`. The legacy names are not
  NOVELTIES keys: the dataset's job residues (`dataset.job_cell`) and every loop over NOVELTIES would change.
  `procgen.generate` still filters on the single jam, so every procedural house is unchanged.
* **Adaptive agent with several jams.** Jams are found one bump at a time and each is a new `door_jammed` key, so the
  detector flags it at once and the repair search patches it (tested). House, every household, 60 seeds: 360
  episodes, the agent bumped 0 / 1 / 2 / 3 jams in 170 / 124 / 55 / 11 of them, and its final belief held every
  bumped jam and no unjammed door in all 360 (ranch: 112 / 244 / 4 episodes with 0 / 1 / 2 bumps, same result).
  Known gap, not fixed: `RepairSearch(max_edits=2)` patches at most two keys per call, so three keys surprising in
  the same tick (rescuer and spouse bumping two jams while carpet flashover is flagged) leave one unpatched until it
  is tried again. A fix that adds "every key that pays for itself" broke `test_max_edits_bounds_the_number_of_patches`
  and `test_confounded_evidence_is_flagged_ambiguous` (max_edits is a hard bound by spec), so it was reverted. Only
  one adult acts alone in any household, so at most two jams can be bumped in one tick.
* **Results** (60 seeds; before -> after; `docs/results.md`, `docs/results_house.md`; every `none` row still +0.00).
  Ranch door_jam barely moves (single gap +0.82 -> +0.82, kids +0.40 -> +0.39, kids_dog +0.27 -> +0.24, gap closed
  102% / 79% / 82%), as expected from (3). The house now separates the agents where it did not: door_jam oracle gap
  single +0.00 -> +0.07, kids +0.00 -> +0.14, kids_dog +0.01 -> +0.08 (adaptive closes 100%+ of each), compound
  single +0.00 -> +0.12, kids -0.01 -> +0.14, kids_dog -0.00 -> +0.08, family -0.02 -> +0.11, family_dog -0.10 -> +0.12
  (compound previously had a negative oracle gap on family_dog; now positive everywhere but married). But house
  door_jam **family** fell from +0.29 (gap closed 77%) to +0.08 (36% [-31%, 93%]) and family_dog from +0.10 to +0.02:
  the old fixed jam (5,4) sat squarely on the spouse's and the rescuer's route for those households; random draws
  often do not, and the spouse bumping a jam is the evidence that mattered there. Honest reading: on the house the
  multi-jam is a broader but, for the family households, weaker test than the old one.
* Smaller edits outside this package: `world.py` (`SeededNovelty`, `resolve_novelties`; behaviour unchanged for plain
  specs), `procgen.py` (imports `door_jam_single`), `dataset.py` (`true_patches` from `world.novelties`),
  `scripts/ppo_results.py` (its door probe uses `door_jam_single`; episodes it re-runs under `door_jam` now use the
  multi jam, so that column describes only the first legacy door). The CLI banner (`cli._banner`, owned by the
  explanation package) shows only the last patch, i.e. one of the jammed doors.
* **Stale walkthrough.** docs/OVERVIEW.md and README.md point at `gifs/house/family_door_jam_{adaptive,baseline}_seed0.gif`
  as "the agent finds the jam at (5, 4) and gets all three out; the baseline loses one". That was the single jam.
  After `make gifs` with the multi jam (seed 0 jams (5, 19), (8, 10), (8, 20), (8, 29)) baseline, oracle and adaptive all rescue 2/3
  there; the old story reproduces with `EMBERPATH_DOOR_JAM=single make gifs`. Those docs were left alone (not this
  package's to rewrite beyond a pointer line).

## 2026-11-09: the fire starts slowly (`world.FireClock`; `tests/test_fire_forecast.py`)
Asked for: "slow early spread and ramp up fire spread after 10 ticks", because on the house the kids / kids_dog
households were close to impossible for every agent (oracle all-out 2-7%): the fire reached Ada's room (family room,
5 cells from the ignition) at about tick 20-30, before the rescuer could (29 ticks to get there from the bedroom).

* **Design: fewer fire turns, not lower probabilities.** On a *fire tick* the fire takes one whole turn exactly as
  before (every burning cell tries each 4-neighbour once at the rule's own `spread_prob`, logged as an `Event`; then
  every burning cell burns down by one). On other ticks the fire does nothing and draws nothing from its stream; smoke
  still drifts every tick. Lowering p early would have told the detector "carpet burns slower than believed" (false
  alarms) and confounded `carpet_flashover`; with fewer turns each trial is still a Bernoulli trial at the true p, so
  the evidence keeps its meaning. Burn-down pauses with the spread, so the fire runs on its own clock: **after n fire
  turns the house burns exactly as the old fire did after n ticks** (`test_the_clocked_fire_is_the_old_fire_warped_in_time`,
  bit-identical burn grids on house and ranch). The schedule depends on the tick alone (no RNG, no action), so
  invariant 2 holds unchanged; it is public physics, not a rule: novelties never change it and it is not in `Rules`.
* **Schedule** (`world.DEFAULT_FIRE_CLOCK = FireClock(slow_every=3, ramp_start=10, ramp_ticks=20)`): fire-tick rate
  1/3 for ticks 0-9, rising linearly to 1 over ticks 10-29, every tick from 30. Tick t is a fire tick when the
  cumulative rate crosses an integer during it: ticks 2, 5, 8, 11, 14, 16, 17, 19, 20, 22, 23, 24, 26, 27, 28, 29, then
  all (16 turns in the first 30 ticks instead of 30). `EMBERPATH_FIRE_CLOCK=off` restores the old fire (and reproduces
  the old results tables exactly, checked on all 35 sweep settings); `EMBERPATH_FIRE_CLOCK=every,start,ramp` was the
  sweep's knob.
* **Every forecast follows the clock.** `HazardAwarePlanner.fire_rounds` is the old forecast (expected fire *turns*,
  Dijkstra over 1/p); `fire_eta(burning, burnt, step=..., margin=1)` times it with `world.fire_ticks_for_rounds(step,
  turns)`, which is exact for whole turns (the j-th turn after `step` is over at its tick + 1 - step) and linear in
  between, and the identity from tick 30 on (so late-episode decisions are unchanged). `step` is a required keyword so
  no caller can forget it. The save margin now scales turns, then the result is timed (`eta_safe`): since the true fire
  is an exact monotone warp of the old one, quantiles commute with the warp, and "the 10th-percentile arrival is about
  half the mean forecast" keeps exactly the meaning it had. `_exit_in_time` / `_best_order` take that `eta_safe`.
  Also updated: `explain.fire_reach_chance` (the simulated fires take turns only on fire ticks counted from the
  observation's step; `test_risk_follows_the_fire_clock_in_the_slow_start`), `procgen` (`measure` forecasts from step 0,
  `_first_fire` runs on the clock with burn-out in turns; so procedural houses' measured tiers, and hence which house a
  `gen:<tier>:<seed>` is, can change), and the RL forecast plane (`rl_baseline.WorldModelFeatures.planes`, one line:
  `step=obs.step`). The reward has no fire forecast (its proximity term is a walking distance), nor does the dreamer.
* **Calibration** (fire alone, 100 seeds, from ticks 0 and 5; `test_the_tick_forecast_is_calibrated_through_the_slow_start`).
  The forecast follows one chain of mean crossing times, so the real fire arrives sooner everywhere, by the same factor
  as before the clock: median true/forecast arrival 0.69-0.74 for cells the fire reaches during the slow phase, 0.67-0.76
  for later cells (no clock, steady state: 0.6-0.7). Reading fire turns as ticks would be 1.05-1.44 for the early cells
  (expecting them far too soon). Median 10th-percentile arrival / pessimistic forecast: 0.86-1.00 (before the clock,
  from tick 0: 0.86-1.03). Rank correlation of forecast and true arrival order > 0.8 on both houses.
* **Sweep** (training seeds 0-59; planners only; house kids / kids_dog / family and ranch kids / kids_dog / family / single,
  x {none, door_jam, carpet_flashover, compound, hidden_dependent}; o = oracle, a = adaptive; gap = mean oracle -
  baseline score over the four novelties; every `none` gap was +0.00 in every config):

  | clock (every, start, ramp) | house kids all-out o/a: none; novelties | house kids_dog | house family | house gap | ranch oracle all-out | ranch gap |
  |---|---|---|---|---|---|---|
  | off (before) | 7%/7%; 3%/3% | 3%/3%; 1%/1% | 88%/88%; 52%/59% | +0.074 | 75% | +0.202 |
  | 2, 10, 0 | 13%/13%; 7%/5% | 3%/3%; 2%/1% | 92%/92%; 70%/70% | +0.086 | 77% | +0.227 |
  | 2, 10, 5 | 18%/18%; 9%/7% | 8%/8%; 2%/2% | 95%/95%; 74%/71% | +0.091 | 80% | +0.249 |
  | 3, 10, 0 | 18%/18%; 10%/7% | 8%/8%; 2%/1% | 100%/100%; 79%/78% | +0.094 | 79% | +0.232 |
  | 3, 10, 5 | 25%/25%; 11%/10% | 7%/7%; 3%/2% | 100%/100%; 80%/78% | +0.090 | 79% | +0.244 |
  | 4, 10, 5 | 25%/25%; 11%/10% | 5%/5%; 2%/2% | 100%/100%; 80%/78% | +0.098 | 80% | +0.248 |
  | 5, 10, 5 | 25%/25%; 11%/10% | 5%/5%; 2%/2% | 100%/100%; 80%/77% | +0.100 | 80% | +0.248 |
  | 3, 10, 10 | 25%/25%; 12%/10% | 5%/5%; 2%/2% | 100%/100%; 80%/78% | +0.089 | 80% | +0.250 |
  | **3, 10, 20 (chosen)** | **52%/52%; 27%/24%** | 10%/10%; 4%/4% | 100%/100%; 85%/82% | +0.099 | 84% | +0.270 |
  | 4, 10, 20 | 60%/60%; 29%/26% | 10%/10%; 4%/4% | 100%/100%; 86%/82% | +0.093 | 84% | +0.269 |
  | 3, 10, 30 | 68%/68%; 35%/31% | 10%/10%; 4%/4% | 100%/100%; 88%/83% | +0.086 | 87% | +0.279 |
  | 4, 10, 40 | 70%/70%; 48%/39% | 13%/13%; 6%/5% | 100%/100%; 90%/87% | +0.112 | 88% | +0.309 |

  Reading: slowing only the first 10 ticks saturates at 25% (ten ticks of fire are only ever worth ~7 ticks of delay);
  what matters is how gradually the ramp comes in. Chosen 3,10,20: house kids all-out 7% -> 52% with no novelty and
  ~27% with one, the house still separates the agents (mean gap +0.074 -> +0.099; kids door_jam +0.20, compound +0.19),
  the ranch stays far from trivial (oracle all-out 75% -> 84%), and full speed arrives at tick 30, close to the user's
  "after 10 ticks". Longer ramps (30, 40) make kids easier still but flatten novelty gaps on the house kids rows
  (3,10,30: hidden_dependent +0.01, carpet +0.01) and keep the fire slow until tick 40-50. Slower than 1/3 adds nothing.
* **Novelty timing.** `carpet_flashover` lands at tick 2 and compound's carpet change at tick 10, the ramp's start.
  Not confounded: the detector sees trials at the true p whatever the clock, so the ramp itself raises no alarm (quiet
  worlds, threshold 6: 0.0% false alarms on ranch and house, before and after; `scripts/false_alarms.py`, now with
  `--blueprint`). It costs delay in ticks, because there are fewer trials per tick early on (median ticks change -> flag,
  default detector): ranch carpet_flashover 4 -> 9, compound carpet 6 -> 10; house carpet_flashover 6 -> 14 with 3% -> 15%
  of episodes never flagging (the baseline's episode ends before the fire has tried carpet enough), compound carpet
  7 -> 11 (missed 17% -> 17%). Novelty steps left unchanged. A side effect, recorded in `tests/test_detector.py`: with
  little pre-change evidence the plain detector's estimate of the tick-10 carpet change lands near the true 0.45, and
  forgetting (decay 0.9) no longer helps (it overshoots and drops the flag on 2 of 10 seeds); the old test claimed the
  opposite against a stale 0.60 target.
* **Results** (60 seeds, before -> after; `docs/results.md`, `docs/results_house.md`; every `none` row +0.00 on both).
  House oracle all-out, none / door_jam / carpet / compound / hidden: kids 7/7/2/2/3% -> 52/40/38/30/0%; kids_dog
  3/3/0/0/2% -> 10/5/5/3/2%; family 88/53/70/38/45% -> 100/80/100/67/95%; single 52/52/57/67/42% -> 85/82/93/83/55%.
  House oracle gaps: single door_jam +0.07 -> +0.25, compound +0.12 -> +0.23; kids door_jam +0.14 -> +0.20, compound
  +0.14 -> +0.19, **hidden_dependent +0.23 -> +0.08** (baseline score 0.36 -> 0.49 while the oracle's stays 0.58; nobody gets
  all three out; cause not investigated); kids_dog door_jam +0.08 -> +0.16; family hidden +0.10 -> +0.24, **compound +0.11 -> +0.04**;
  family_dog door_jam +0.02 -> +0.09, **compound +0.12 -> +0.03**; family carpet -0.07 -> +0.00. Ranch: single door_jam
  +0.82 -> +0.88, compound +0.57 -> +0.90; kids door_jam +0.39 -> +0.50, compound +0.07 -> +0.29 (oracle all-out
  0% -> 50%); kids_dog compound +0.00 -> +0.18; **married hidden_dependent +0.04 -> +0.50** (the spouse no longer needs
  help, so the baseline no longer walks past the hiding visitor; `tests/test_hidden_dependent.py` records the flip).
* **Negatives.** (1) House kids_dog stays nearly impossible (oracle all-out 3% -> 10%): its third pickup, the dog in
  the living room beside the ignition, comes after tick 55, deep in the full-speed fire, which no early slowdown
  reaches. (2) Ranch **single** got worse without novelty (all-out 88% -> 78%; carpet_flashover 93% -> 83%): not
  investigated (a guess, unverified: the south exit now stays open longer and the route to it passes the
  living-room fire). (3) Ranch married door_jam / compound: baseline all-out 85% -> 18%, 80% ->
  27% with score still ~1.0: everyone is out but the baseline rescuer times out inside (in the episode inspected it stood
  at (4,3), in the bedroom behind the jammed door (5,3), for the rest of the episode). (4) Detection delays in ticks roughly double
  (above). (5) **`make audit` now prints `EXPLOIT under new reward`** on its scenario (ranch / single / door_jam, 20
  seeds): suicide -13.91 vs the stuck baseline -14.03 (before: -15.20 vs -14.03). This is the documented "death charge
  assumes a static hazard" limitation (docs/REWARD.md) surfacing in the mean: suicide dies at tick ~27, when the slow
  fire is not yet near the dog, so its death charge uses a proximity rate of ~0, while the stuck baseline later pays
  1.0-1.7 of proximity as the fire reaches the dog. Before, the fast fire often burned the dog during the suicide
  episode too (returns -14.9 to -22.9 instead of -13.37 on 9 of 20 seeds; now 2). Suicide's *median* was already above the baseline's
  before (-13.37 vs -14.19). Checked 24 settings (ranch and house x single / kids / family / married x none / door_jam /
  compound, 20 seeds): suicide's mean beats the stuck baseline only there, and beats `stall` on ranch / kids / door_jam
  (-23.51 vs -24.09; before -25.20 vs -24.12); never on the house; honest agents (oracle, adaptive) still rank first
  everywhere, and `scripts/find_exploits.py --budget small` confirms 0 loopholes (legacy reward: 2, as it should).
  Not fixed here: it needs a reward change (`reward.py`, invariant 8; REWARD.md's own suggestion is to charge a death at
  the worst-case proximity rate). (6) Learned checkpoints (PPO, Dreamer, house-rl) and the dataset were made under the old fire and are
  obsolete for these dynamics; README headline numbers quoted from the old `docs/results.md` are stale until refreshed.
* **Tests changed honestly** (not weakened): `test_the_spouse_takes_some_fire_but_a_child_does_not` (Sam panicked off
  his burning cell; the old pass relied on the tick-0 spread re-igniting the cell he stepped to; now flames surround
  him); `test_exit_choice_skips_...` passes the pessimistic forecast explicitly; `test_exploit_abandoning_the_dependent_...`
  puts the dog in the far side room (its slot is beside the south exit route, which now stays open, so "abandoning"
  bonded with it on the way out); the dataset `wrong_outcome` corruption relabels a timeout as a death (a timeout
  with everyone out relabelled "success" is caught only by replay: the static check has no exit test, a validator gap
  recorded here, not fixed); the risk-formula test is evaluated past the slow start, plus a new slow-start test.
* Files outside this package touched minimally: `rl_baseline.py` (one line, the forecast plane's `step`),
  `procgen.py`, `scripts/false_alarms.py` (crashed on `SeededNovelty`: now resolves per seed; `--blueprint`), tests above.

## Why this is not just "A* with fire"

A* with a fire penalty solves the *planning* problem when the rules are known. The project is about what
happens when they are not: detecting that the rulebook is wrong from sparse, noisy trials, localizing which
rule, correcting it, and doing so with calibrated uncertainty and a human-readable explanation. The planner is
the stable part; the research is the loop around it.
