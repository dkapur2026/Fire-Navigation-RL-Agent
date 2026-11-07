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
* **Adaptation loop (to build).** (1) Beta-Bernoulli posterior per rule with a log Bayes factor test, sequentially
  corrected, flags suspect rules with calibrated intervals. (2) A search over minimal rule edits, scored by
  likelihood minus a complexity penalty, picks the patch that best explains *all* the evidence. (3) Replan with
  the patched rules.
* **Evaluation.** Same-seed comparison of baseline / oracle / adaptive. Headline metric: fraction of the
  baseline-to-oracle gap closed. Built-in control: with no novelty, baseline and oracle are identical.
* **Neural side (to build).** A CNN perception front-end from degraded images, a PPO baseline for the "why not
  just RL" question, and optionally a learned proposal distribution over patches.

## The questions you will get, and honest answers

**"Isn't this just A*?"** A* solves the planning once the rules are known. The research here is the loop around
it: detecting that the rules changed, finding *which* one, and fixing it from a handful of observations, with
calibrated uncertainty. The planner is deliberately simple so the adaptation is what is measured.

**"Why not just retrain a deep RL agent?"** That is the baseline to beat, and the roadmap measures it (Phase 4b).
The hypothesis is that explicit structure lets the agent adapt from dozens of observations rather than thousands
of episodes, and gives a human-readable reason. If the data says otherwise, report that.

**"What is neuro-symbolic about it?"** Symbolic: the rulebook and planner. Probabilistic: the Bayesian inference
over rules. Neural: perception from images (Phase 4c), the RL comparison, and the optional learned patch proposer.
Today the scaffold is symbolic and probabilistic; the neural parts are scoped but not built, and the docs say so.

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
