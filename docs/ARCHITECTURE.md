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
                   │ Detector (Phase 4)      │──────► │ Repair (Ph. 5)│
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
* **Cost.** Planning dominates run time (the full results table takes about 4 minutes on 2 cores). Profiling it is
  a deliberate work package (WP-G).

## Why this is not just "A* with fire"

A* with a fire penalty solves the *planning* problem when the rules are known. The project is about what
happens when they are not: detecting that the rulebook is wrong from sparse, noisy trials, localizing which
rule, correcting it, and doing so with calibrated uncertainty and a human-readable explanation. The planner is
the stable part; the research is the loop around it.
