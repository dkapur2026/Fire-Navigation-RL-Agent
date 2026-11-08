# emberpath: specification

Status: ✅ built and tested · ⬜ to build · ◻ stretch. Requirement IDs are cited by the task files in `tasks/`.
When you finish a work package, flip its rows here and in `ROADMAP.md`.

## 1. Purpose

One codebase, two connected goals.

* **Track A: adaptive AI.** An agent that notices when the rules of its world have changed, works out which rule,
  repairs its own rulebook, replans, and explains what it did. (Neuro-symbolic: symbolic planner, Bayesian
  inference, search over rule edits, learned perception.)
* **Track B: RL environment engineering.** A trustworthy environment to train and evaluate agents in: a clear task,
  a reward that resists hacking, controllable difficulty, a data pipeline with quality checks, an interface for
  LLM agents, and evidence about whether the environment's data actually improves a trained model.

Both tracks share the simulator, the reward, and the evaluation harness. Everything must be explainable to a
non-technical audience (watch the GIF, read the plain-English reasons) and defensible to a technical one (numbers
reproducible from one command).

## 2. The scenario

A house is on fire. A rescuer starts somewhere in a floor plan and must bring dependents to an exit, fast, keeping
them away from flames. Households are labelled points in the state space: **single** (a dog: fast, spooked by fire,
unreliable follower), **married** (a spouse: normal speed, reliable), **kids** (two children: half speed). Mid-episode
the *true rules change without warning* (a door jams, carpet burns faster). The agent is judged on how much of the
gap between a stale-rulebook baseline and an all-knowing oracle it closes.

## 3. Requirements

### World (W)
| ID | Requirement | Status |
|---|---|---|
| W1 | Parse ASCII floor plans: walls, floor materials, doors, exits, rescuer start, dependent slots, ignition | ✅ |
| W2 | Fire as a stochastic cellular automaton: per-material spread probability, burn-out timers | ✅ |
| W3 | Smoke diffusion that shrinks the visibility radius (partial-observability view) | ✅ |
| W4 | Households as parameterized profiles (speed, flee behaviour, count); follow reliability lives in the rules | ✅ |
| W5 | Jammed doors; scheduled novelties as `RulePatch`es applied to the true rules | ✅ |
| W6 | Every rule trial logged as `Event(step, key, outcome)`; this is the agent's only evidence | ✅ |
| W7 | Determinism: seeded; fire uses its own random stream so every agent faces identical fire | ✅ |
| W8 | Procedurally generated houses with graded, measured difficulty and a solvability guarantee | ✅ `procgen.py`, `scripts/coverage_report.py` (generation ~20-30 ms/house, below the WP-J speed target) |
| W9 | Structural novelty (a new kind of object, e.g. a window exit) that no existing rule key describes | ◻ (W11 is semi-structural: a new person, but expressed as an `occupied` rule key) |
| W10 | Two shipped blueprints: `studio` (tiny, for tests) and `ranch` (main, calibrated) | ✅ |
| W12 | Forcing jammed doors (`force_prob:door`, 10% a try, own RNG stream), only from the hallway side (`Blueprint.door_outside`) | ✅ `tests/test_spouse_and_doors.py` |
| W13 | Adults who act alone (married, family): fetch children, avoid flames, panic for 10 ticks near fire or at a jammed door, join the rescuer when reached; planner helps them when needed | ✅ `tests/test_spouse_and_doors.py` |
| W14 | `house` blueprint: wider, 8 rooms, 3 exits (front, back door, bedroom window), 4 slots | ◻ draft: balance not yet acceptable (see ARCHITECTURE) |
| W11 | `hidden_dependent` novelty: an unexpected person (rule `occupied:<r>,<c>`) hides at a spare slot; heard within 4 cells, and from anywhere after tick 10 (`REVEAL_STEP`); unseen until found | ✅ `tests/test_hidden_dependent.py` |

### Planner (P)
| ID | Requirement | Status |
|---|---|---|
| P1 | Symbolic hazard-aware planner that forecasts fire arrival from *believed* spread rates | ✅ |
| P2 | Every decision carries a plain-English reason | ✅ |
| P3 | Optimal-order pickup of multiple dependents (brute force, up to 3) | ✅ |
| P4 | Pacing: wait for lagging followers when safe | ✅ |
| P5 | Triage when not everyone can be saved (maximize expected survivors, explain the choice) | ◻ |
| P6 | Belief tracker for hidden dependents / fire under limited visibility | ◻ |
| P7 | Planner at least 2x faster with results bit-identical | ⬜ WP-G |

### Reward (R)
| ID | Requirement | Status |
|---|---|---|
| R1 | Per-tick time cost (be quick) | ✅ |
| R2 | Potential-based progress shaping toward the exit (cannot change the optimal policy; cannot be farmed) | ✅ |
| R3 | Rescue and success bonuses | ✅ |
| R4 | Smooth penalty for a dependent *near* flames | ✅ |
| R5 | Large penalty for a dependent *touching* flames (lost) | ✅ |
| R6 | Death charge so failing early is never cheaper than carrying on | ✅ |
| R7 | Per-component reporting every step (`info["reward_components"]`) | ✅ |
| R8 | Efficiency metric: fewest-possible steps / steps taken (successes only) | ✅ |
| R9 | Reward-hacking audit: tests per exploit + degenerate-policy script comparing legacy vs new reward | ✅ |
| R10 | Audit of a *trained* agent: read the component breakdown of its best episodes | ⬜ WP-C |
| R11 | Automated exploit search, weight sweeps, sparse-vs-shaped ablation, ranking stability | ✅ `scripts/find_exploits.py`, `scripts/reward_sweep.py` |

### Adaptation (A)
| ID | Requirement | Status |
|---|---|---|
| A1 | Bayesian detector: per-rule posterior, log Bayes factor, calibrated interval, <= 5% false alarms | ✅ |
| A2 | Repair: search over minimal `RulePatch` edits, complexity penalty, runner-up margin, reports ambiguity | ✅ |
| A3 | Adaptive agent wired end to end; closes >= 50% of the oracle gap on `door_jam` for every household | ✅ |
| A4 | Does no harm when nothing changes | ✅ |
| A5 | Change-point inference: *when* did it change, and forget stale evidence | ◻ (detector has optional exponential `decay`, off by default) |
| A6 | Amortized repair: a small network proposes patches, symbolic search verifies | ◻ |
| A7 | Evidence discipline enforced: only privileged agents get the world; lockstep test shows non-oracle agents act identically while their inputs match; `env.encode` hides unbumped jams | ✅ `tests/test_evidence_discipline.py` |
| A8 | Lifelong adaptation: adaptive agent carries learned beliefs across episodes; `emberpath improve` writes first/last-episode GIFs | ✅ mechanism (`tests/test_lifelong.py`); helps 3 of 12 cells, hurts 2 (see ARCHITECTURE "Lifelong adaptation") |

### Neural (N)
| ID | Requirement | Status |
|---|---|---|
| N1 | Deep-RL baseline (PPO, PyTorch): frozen vs keep-learning after a novelty; sample-cost comparison | ⬜ WP-C |
| N2 | Neural perception front-end: CNN maps a smoke-degraded image to symbolic state; planner consumes it | ⬜ WP-D |

### Explanation (X)
| ID | Requirement | Status |
|---|---|---|
| X1 | `explain_adaptation`: plain-English paragraph plus technical appendix (posterior, Bayes factor, trials, runner-up) | ⬜ WP-E |
| X2 | GIF banner showing when the agent noticed versus when the truth changed | ⬜ WP-E |
| X3 | Human on-the-loop: agent proposes plan + confidence, a scripted supervisor can veto, agent asks when ambiguous | ◻ |

### Evaluation (E)
| ID | Requirement | Status |
|---|---|---|
| E1 | Same-seed harness comparing `baseline` / `oracle` / `adaptive` | ✅ |
| E2 | Headline metric `gap_closed = (adaptive - baseline) / (oracle - baseline)` | ✅ |
| E3 | Benchmark table script (`make results`) with a control row that must be zero | ✅ |
| E4 | Confidence intervals over >= 100 seeds | ✅ `docs/adaptive_study.md` |
| E5 | Recovery curves (score vs observations after novelty); false-alarm rate vs threshold | ✅ `scripts/adaptive_study.py`, `scripts/false_alarms.py` |
| E6 | Ablations: no detector / no repair search / perfect vs CNN perception | ✅ no detector, no repair search · ⬜ CNN perception (WP-D) |
| E7 | Two-page technical report; every number reproducible from a command | ⬜ WP-H |
| E8 | ASCII and GIF rendering | ✅ |
| E9 | Gymnasium environment (passes `check_env`) | ✅ (`check_env` passes when run with the `rl` extra; the env test in the suite skips without gymnasium) |

### Data (D)  *(Track B)*
| ID | Requirement | Status |
|---|---|---|
| D1 | Trajectory recorder with a versioned JSONL schema (observations, actions, reasons, reward components, events, outcome) | ✅ `dataset.py`, `docs/DATA.md` |
| D2 | Quality validators: schema, duplicates, replay-determinism, label consistency, return = sum of components, split leakage | ✅ `scripts/validate_dataset.py` |
| D3 | Dataset statistics and balance report (by tier, novelty, outcome) | ✅ |
| D4 | Post-training validation study: does training on slices of this data improve a model on held-out houses and novelties? | ⬜ WP-L |

### LLM-agent interface (L)  *(Track B)*
| ID | Requirement | Status |
|---|---|---|
| L1 | Text / tool-call interface to the world (map, status, event narration; move / wait tools) that leaks no hidden rules | ⬜ WP-I |
| L2 | Scripted fake client so tests never touch the network | ⬜ WP-I |
| L3 | Optional Anthropic client behind the `llm` extra; key from the environment only | ⬜ WP-I |
| L4 | LLM-agent benchmark by difficulty tier; invalid actions handled and counted, transcripts saved | ⬜ WP-I |

### Non-functional (Q)
| ID | Requirement | Status |
|---|---|---|
| Q1 | Deterministic: everything seeded; flaky tests are bugs | ✅ |
| Q2 | Core install needs only numpy and pillow; torch / gymnasium / pyro / anthropic are optional extras, imported lazily | ✅ |
| Q3 | Dockerfile that runs a demo with no extras | ✅ |
| Q4 | Test suite green; acceptance tests for unbuilt features exist as `xfail` and flip to pass when built | ✅ |
| Q5 | Docs for two audiences (`docs/EXPLAINING_IT.md`) | ✅ |
| Q6 | Docs state plainly what is built and what is not | ✅ |

## 4. Out of scope
Realistic fire physics; 3D or multi-floor houses; real robots; training large models. The claim is about the
adaptation loop and the environment engineering around it, not the fidelity of the fire.

## 5. Traceability to the two roles this project is meant to support

### Applied scientist, neuro-symbolic AI (autonomy R&D)
| The role asks for | Where it shows up |
|---|---|
| Neuro-symbolic integration, probabilistic programming and Bayesian reasoning | A1, A2, N2; `rules.py`, `novelty.py`, `repair.py` |
| Program-synthesis flavored search, combinatorial optimization | A2 (search over rule edits), P3, P5 |
| Designing and debugging complex AI systems; profiling | P7 (WP-G), the benchmark-validity story in `docs/ARCHITECTURE.md` |
| Writing reports and papers | E7 (WP-H) |
| Python, Linux, Git, Docker, one ML library | the whole repo; N1, N2 in PyTorch |
| Deployable solutions, strategic long-lived codebase | Q1-Q6, `AGENTS.md`, tests as executable specs |
| Explainable, trustworthy autonomy; human on-the-loop | P2, X1-X3 |

### RL environment data engineer / researcher
| The role asks for | Where it shows up |
|---|---|
| Design and improve RL training environments | W1-W8, E9 |
| Task objectives, reward functions, evaluation standards | R1-R8, E1-E3 |
| Prevent reward hacking, find loopholes | R9-R11 (`docs/REWARD.md`, `scripts/audit_reward.py`) |
| Collect, clean, structure and evaluate data | D1-D3 |
| Validation environments for post-training data effectiveness | D4 (WP-L), R10 |
| Difficulty and coverage | W8 (WP-J), tiers |
| LLM agents, benchmark construction, automated evaluation | L1-L4, E1-E6 |
| Translate real-world tasks into trainable, measurable environments | the scenario and `docs/REWARD.md` |
