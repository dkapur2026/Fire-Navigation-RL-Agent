# Explaining emberpath to different audiences

## 30 seconds, anyone

> "I built a rescue robot for a burning house. It plans a route using what it *thinks* it knows about fire and
> doors. Then I secretly change the rules: a door jams, or the carpet starts burning faster. Most planners keep
> trusting their old assumptions and fail. Mine notices that reality has stopped matching its expectations,
> figures out which assumption broke, fixes just that one, and tells you what it did, in plain English."

## 2 minutes, non-technical, with the demo

Play the two GIFs side by side (`docs/media/`).

1. "Same house, same fire, same moment the door jams. The left robot never finds out. It keeps trying the door."
2. "The right one is cheating: it's been told the door is jammed. That's the best any robot could do with this
   planner."
3. "My goal is a robot that behaves like the right one **without being told**. The score is how much of the
   gap between left and right it closes."
4. "Why a dog, a spouse, or kids? Different people behave differently: a dog bolts, children move slowly. Each
   is just a different set of numbers to the simulator, but it makes the stakes easy to picture."
5. "And it has to *explain itself*, because a rescue system people can't question is one nobody will trust."

## 5 minutes, technical

* **Setup.** Grid world from ASCII blueprints. Fire is a stochastic cellular automaton (per-material spread
  probabilities, burn-out timers); smoke diffuses and shrinks visibility. Dependents have follow reliability,
  speed and flee behavior.
* **Representation.** Every rule is a Bernoulli probability addressed by a string key. The simulator logs each
  trial as `Event(step, key, outcome)`. That log is the agent's *only* evidence.
* **Novelty = patch.** A novelty is a `RulePatch` applied to the *true* rules at a scheduled tick; the agent isn't told.
* **Agent.** Symbolic planner over a *believed* `Rules`. It forecasts fire arrival (expected ticks `1/p` per
  cell), prices cells by whether fire beats the rescuer there, and brute-forces the pickup order. Bad
  beliefs produce bad routes.
* **Adaptation loop (built: `novelty.py`, `repair.py`, `AdaptiveAgent`).** (1) Beta-Bernoulli posterior per rule with
  a log Bayes factor test (a mixture SPRT, valid however often it is checked), flags suspect rules with calibrated
  intervals. (2) A search over minimal rule edits, scored by
  likelihood minus a complexity penalty, picks the patch that best explains *all* the evidence. (3) Replan with
  the patched rules.
* **Evaluation.** Same-seed comparison of baseline / oracle / adaptive. Headline metric: fraction of the
  baseline-to-oracle gap closed. Built-in control: with no novelty, baseline and oracle are identical.
* **Explanations (built: `explain.py`, `places.py`).** Every move carries a plain-English reason. On the tick the
  agent repairs its beliefs, the reason starts with an explanation in four parts: what changed, how the plan changes,
  what that costs (steps and an estimated added fire risk), and how sure it is. `emberpath run` ends with an
  explanation log and a technical appendix (detector scores, repair margin, the risk numbers). The GIF banner shows
  the explanation from the tick the agent *noticed*, next to the tick the world really changed. Details below.
* **Neural side.** Built: a PPO baseline and a BC+PPO policy for the "why not just RL" question (`docs/RL.md`), and a
  DreamerV3 world-model agent that was tried and abandoned (`docs/DREAMER.md`). Not built: a CNN perception front-end
  from degraded images, and a learned proposal distribution over patches.

## What the agent says when it changes its mind

### Names people can understand
* **Doors are numbered** 1, 2, 3, ... in reading order (top row first, left to right). The number depends only on the
  floor plan, so "door 4" is the same door in every run, and the GIF draws the number on the door.
* **Rooms are named** by optional metadata at the end of a blueprint file. Lines starting with `# ` (hash, space) are
  never part of the grid (no grid row can start that way), so older files and the loader's grid are unaffected:

  ```
  # room 1,9: kitchen        <- the room containing row 1, column 9 is "the kitchen"
  ```

  `house`, `ranch` and `studio` carry names (the grids are byte-for-byte unchanged). A house without names, such as
  a procedural one, gets them from the floor plan: the room with clearly the most doors (at least 4) is "the
  hallway", the rest are named by where they sit ("the north-west room", "the north room 2").
* Exits are named by the room they open from ("the hallway exit", "the south exit of the family room").

### The four parts of an explanation
| Part | Example | Where it comes from |
|---|---|---|
| What changed | "Door 4 (between the main bedroom and the hallway) no longer opens (it was stuck when tried)." Several jams found together: "Door 1 (...) and door 4 (...) no longer open." "Carpet is catching fire faster than I expected (about 1 in 3 per step next to a flame, instead of about 1 in 6). I saw it catch 32 of the 91 times a flame was next to it." "Someone not on the household list is hiding in the study: I can hear them calling." | the repair's edits, the belief before it, and the agent's own tally of what it saw |
| How the plan changes | "Rerouting via the kitchen." "Forcing door 3 from the bathroom (about 10 kicks)." "Going for Max first." "My plan stays the same." | the planner asked what it would do this tick with the old belief and with the new one (`plan_with`), and the two routes compared |
| What it costs | "Compared with the way I had planned, this adds about 18 steps, with an estimated 12% added fire risk (the chance the fire gets to my path before I do goes from 0% to 12%)." | route lengths, kicks, and the fire-risk estimate below |
| How sure | "I am not sure about this: it could also be that nothing has changed and I was just unlucky. My best guess is only about 1.5 times more believable, so I am acting on it and will keep checking." | only when the repair search's runner-up is within its ambiguity margin; the margin (a log ratio) is told as "N times" |

The plain text never contains rule keys, coordinates, or words such as "posterior", "Bayes factor", "patch" or
"likelihood" (`explain.JARGON`, checked on every explanation of 18 real runs in `tests/test_explain.py`). Those
numbers are in the technical appendix, for an expert.

### The fire-risk number, exactly
"The chance the fire gets to my path before I do" for a route is estimated from 200 simulated fires
(`explain.fire_reach_chance`). Each starts from the flames the agent can see and spreads by the simulator's rule
with the **believed** rates: each tick every burning cell tries once to ignite each neighbouring floor cell with the
believed spread probability `p` of that cell's material (k burning neighbours: `1 - (1 - p)^k`), never into an exit
or a burnt-out cell, and a cell burns for the believed burn time and then burns out (cells already burning are
assumed to have their whole burn ahead: the agent cannot see how long they have burnt). The rescuer reaches route
cell `i` at tick `i x pace` (the slowest follower's pace) plus `1 / force_prob` ticks for every door on the route it
believes jammed and must kick in. Then

    risk(route)       = share of the simulated fires in which some route cell i catches fire at or before arrival(i)
    added fire risk   = risk(new route) - risk(old route), in percentage points

Both routes are scored on the same simulated fires under the **new** belief, and the old route is timed as it was
planned (a door then believed open costs no kicks), so for a jam the number compares the detour with the way the
agent had meant to go. A route is the path to the current target priced by the planner's own cell costs, then the
shortest way from there to an exit. The simulation has its own fixed seed and never touches the world's fire stream.
When a belief about fire speed changes but the route does not, the explanation instead compares the route's risk
under the old and new belief ("is now about 40%, not the 10% I expected"). Limits: it is a model estimate, only as
good as the believed rates; it ignores that the rescuer would react to the fire; it is the risk of the route as
planned now, not of the whole rescue. `tests/test_explain.py::test_risk_formula_on_a_hand_built_case` checks it on a
corridor where the answer is known (certain spread: exactly 0 or 1; p = 1/2: 1/2 with one burning neighbour, 3/4
with two).

### Evidence discipline
The explainer is never handed the world: it gets the blueprint, the agent's observation, its belief before and
after, the repair result, both plans and the agent's own tallies. A lockstep test runs two worlds that differ only in
a jam the agent has not tried yet and checks the explanations are identical while its input is. Explaining never
changes what the agent does: `AdaptiveAgent(explain=False)` gives identical actions
(`test_explaining_never_changes_what_the_agent_does`), and identical episode results in every ranch and house cell of
the results tables (household x novelty, 60 seeds each), checked in one process so both variants ran the same code.

### Example: house, family, `door_jam`, held-out seed 1000
`make explain-gif` (seed 1000 is the first held-out seed, fixed in advance). Two doors jam at tick 1 (door 4 and
door 6); the agent only ever tries door 4, so door 6 is never mentioned: it explains what it has evidence for.

```
t=  6  Heading to Max (child), 9 steps away.
t=  7  Door 4 (between the main bedroom and the hallway) no longer opens (it was stuck when tried). Rerouting via
       the kitchen. Compared with the way I had planned, this adds about 18 steps, with an estimated 12% added fire
       risk (the chance the fire gets to my path before I do goes from 0% to 12%). Heading to Max (child), 27 steps away.
t=  8  Heading to Max (child), 26 steps away.
...
Explanation log (what I noticed and what I did about it):
  t=  7  Door 4 (between the main bedroom and the hallway) no longer opens (it was stuck when tried). Rerouting via
         the kitchen. Compared with the way I had planned, this adds about 18 steps, with an estimated 12% added fire
         risk (the chance the fire gets to my path before I do goes from 0% to 12%).
Technical details:
  t=  7  door_jammed:5,4: believed 0.00, seen 1/1, posterior mean 0.92 (90% interval 0.30-1.00), log Bayes factor 8.5, flagged at t=6
  t=  7  repair: door_jammed:5,4 -> True; log-likelihood gain 6.91; runner-up: no change (score margin 4.91)
  t=  7  fire risk (200 simulated fires, believed rates): old route 0.000 -> new route 0.120; old route under the old belief 0.000

family / adaptive / novelty=door_jam / seed=1000
outcome: SUCCESS   rescued 3/3   steps 71   score 1.00
```

The GIF is `docs/media/house_family_door_jam_adaptive.gif`: from tick 7 its banner reads "NOTICED at t=7 (the world
really changed at t=1): Door 4 (between the main bedroom and the hallway) no longer opens ... Rerouting via the
kitchen." Before that it shows only that the truth changed, which the agent does not know yet.

Other explanations from real runs (same commands, `--agent adaptive`):
* house, kids, `door_jam`, seed 1001, t=22: "Door 3 (between the bathroom and the dining room) no longer opens (it
  was stuck when tried). Forcing door 3 from the bathroom (about 10 kicks). Compared with the way I had planned, this
  adds about 10 steps, with an estimated 2% added fire risk (...from 0% to 2%)."
* house, kids, `door_jam`, seed 1000, t=22: "Door 6 (between the bathroom and the hallway) no longer opens (it was
  stuck when tried). Going for Max first. ..."
* house, kids, `compound`, seed 2, t=18: "Carpet is catching fire faster than I expected (about 1 in 3 per step next
  to a flame, instead of about 1 in 6). I saw it catch 32 of the 91 times a flame was next to it. Going for Max
  first. Compared with the way I had planned, this takes about as long, and cuts the fire risk by about 98% (the
  chance the fire gets to my path before I do goes from 100% to 2%)."
* ranch, single, `hidden_dependent`, seed 1, t=12: "Someone not on the household list is hiding in the study: I can
  hear them calling. Going to find whoever is hiding in the study. Compared with the way I had planned, this adds
  about 8 steps, with an estimated 17% added fire risk (the chance the fire gets to my path before I do goes from 7%
  to 24%)."

Honest notes: ambiguous repairs are rare in real runs (the examples of that wording come from the tests), and two
jams are found on the same tick only when two doors are tried together; usually each jam gets its own explanation
when it is found. The "added steps" compare whole routes (to the target, then out), so they can differ from the
change in the "N steps away" of the planner's reason, which counts only the way to the target.

## The questions you will get, and honest answers

**"Isn't this just A*?"** A* solves the planning once the rules are known. The research here is the loop around
it: detecting that the rules changed, finding *which* one, and fixing it from a handful of observations, with
calibrated uncertainty. The planner is deliberately simple so the adaptation is what is measured.

**"Why not just retrain a deep RL agent?"** We measured it (`docs/RL.md`). PPO from scratch, after 2.2 M steps, got
everyone out of 42% of held-out fires with the dog against the planner's 89%, and never learned a jammed door or a
hidden visitor in more than 2,000 training episodes of each; the adaptive agent repairs the right rule 5 to 11 ticks
after the change, inside the episode. A PPO policy that first imitated the adaptive planner and was then fine-tuned
did match it and edge past it (0.93 vs 0.90 pooled, one training run), but it learned from the planner, was tested
on new fires in the same houses, and fell behind on a combination of changes it never trained on; and it cannot say
what changed. A DreamerV3 world model learned nothing on our budget.

**"What is neuro-symbolic about it?"** Symbolic: the rulebook and planner. Probabilistic: the Bayesian inference
over rules. Neural: the RL comparison (PPO, BC+PPO, DreamerV3: built), perception from images (Phase 4c) and the optional
learned patch proposer (not built). The adaptive agent itself is symbolic and probabilistic; the docs say so.

**"Is the fire realistic?"** No, and it isn't meant to be. The claim is about the *adaptation loop*, which is
independent of fire physics. Swapping in a better simulator should not change the method.

**"What was the hardest part?"** (Real answer, from the build.) Making the benchmark measure what it claims. My first
spread-rate novelty showed *zero* benefit from knowing the truth, because no decision depended on it. Fixing
that meant making the planner forecast fire from its beliefs and redesigning the house so the exit choice
actually hinges on how fast the carpet burns. Details are in `docs/ARCHITECTURE.md`.

## How it connects to Charles River Analytics' public work

Framed as "same family of problems", not "a copy of anything they build":

| Their public themes | Where it shows up here |
|---|---|
| Adaptable / open-world AI that combines probabilistic programming with symbolic reasoning to handle novelty | The whole project: detect novelty, repair the rulebook |
| AI-assisted planning with neuro-symbolic methods | The symbolic planner; triage and ordering (combinatorial optimization) |
| Explainable AI: every output with a human-understandable reason | `Decision.reason`, Phase 6 explanations |
| Autonomy that earns trust; human on-the-loop control | Supervisor interface (depth track) |
| Rescue / casualty-evacuation style robotics; autonomy that keeps working when conditions degrade (e.g. limited communications) | The burning-house scenario; smoke-limited visibility as the analogue |
| Rigorous test and evaluation of autonomy | Same-seed harness, oracle ceiling, control experiment, false-alarm testing |

## Questions worth asking in an interview, prompted by the project

* How do you decide a rule has *really* changed versus noise, in your fielded systems?
* Which novelties matter most operationally: parameter drift, new object types, or adversary behavior?
* How do human operators currently learn to trust an adaptation they didn't witness?
