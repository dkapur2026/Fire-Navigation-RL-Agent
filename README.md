# emberpath

**A rescue agent that notices when the rules of its world change mid-run, works out what changed, and adapts, while
explaining itself in plain English.**

A house is on fire, and the rescuer has to get a dog, a spouse, children, or a whole family out. Partway through, the
world changes without warning: a door jams, the carpet starts burning three times faster, or the rescuer hears a child
nobody mentioned, hiding upstairs. A normal planner keeps trusting its old rulebook and fails. This agent spots the
mismatch from evidence alone, rewrites the one rule that broke, re-plans, and says why:

```
t=  8  Updating my beliefs: The door at (5, 4) is jammed shut. Heading to Max (child), 27 steps away.
t= 12  Updating my beliefs: Someone not on the household list is hiding at (3, 29). I heard someone at (3, 29); going to find them, 30 steps away.
```

**Full project breakdown (how it works, for technical and non-technical readers, results, history, limitations):
[docs/OVERVIEW.md](docs/OVERVIEW.md).**

## The one number that matters

```
gap_closed = (adaptive - baseline) / (oracle - baseline)
```

`baseline` plans with textbook rules and never updates them. `oracle` is always told the true rules (the same planner
with perfect knowledge). `adaptive` starts like the baseline and has to work out the change from what it sees. All
three face identical fires on identical seeds.

## Current results

Ranch house, 60 fires per cell, from [docs/results.md](docs/results.md). Score = fraction of dependents rescued (0 if
the rescuer dies). Brackets are 95% paired-bootstrap intervals.

| household | novelty | baseline | oracle | adaptive | gap closed |
|---|---|---|---|---|---|
| single (dog) | none (control) | 0.88 | 0.88 | 0.88 | n/a (gap +0.00, as required) |
| single (dog) | door jams | 0.02 | 0.83 | 0.85 | 102% [91%, 116%] |
| kids (two) | door jams | 0.47 | 0.87 | 0.78 | 77% [64%, 90%] |
| kids + dog | door jams | 0.60 | 0.87 | 0.82 | 81% [65%, 98%] |
| single (dog) | jam, then faster carpet | 0.00 | 0.57 | 0.35 | 62% [45%, 78%] |
| single (dog) | hidden visitor | 0.44 | 0.87 | 0.90 | 108% [89%, 133%] |
| kids (two) | hidden visitor | 0.63 | 0.86 | 0.86 | 100% |
| family | hidden visitor | 0.70 | 0.93 | 0.93 | 100% |

* Every no-novelty row, for all six households, shows an oracle gap of +0.00 and an adaptive agent identical to the
  baseline: no harm when nothing changes. (Over 100 seeds the detector raised one false alarm in 600 quiet episodes,
  family_dog, without changing the score: [docs/adaptive_study.md](docs/adaptive_study.md).)
* Averaged over all 29 ranch rows: baseline 0.749, oracle 0.869, adaptive 0.858.
* On the wider `house` blueprint ([docs/results_house.md](docs/results_house.md)): family door jam 77%, married
  hidden visitor 97% (0.50 -> 0.98), family hidden visitor 200% (the oracle is handicapped there, see OVERVIEW).
* Read honestly: faster carpet alone no longer separates the agents on either house, some house rows have wide
  intervals (family + dog door jam: 44% [-25%, 110%]), and gap closed above 100% means the oracle is a yardstick, not
  an optimum. All negative results are listed in [docs/OVERVIEW.md](docs/OVERVIEW.md#54-negative-results-read-honestly).

Watch one: `gifs/house/family_door_jam_adaptive_seed0.gif` against `gifs/house/family_door_jam_baseline_seed0.gif`
(run `make gifs` first; the `gifs/` folder is generated and not tracked by git).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core needs only numpy + pillow; add ",rl" for gymnasium/torch
make test                          # full test suite
make results                       # ranch benchmark -> docs/results.md   (make results-house for the house)
make gifs                          # a GIF per household x novelty x agent -> gifs/house/
make audit                         # try to hack the reward with degenerate policies

emberpath list                     # households, novelties, blueprints
emberpath run --blueprint house --profile married --novelty hidden_dependent --agent adaptive --seed 0 --gif out.gif
emberpath eval --profile kids --novelty door_jam --agents baseline oracle adaptive --seeds 30
emberpath improve --profile kids --novelty door_jam --episodes 5 --gif-dir gifs/   # keeps what it learned; first/last GIFs
```

Or in Docker: `docker build -t emberpath . && docker run --rm emberpath`.

## What is built

| part | status |
|---|---|
| World: blueprints (`studio`, `ranch`, `house`), fire (starts slowly, then ramps up: docs/SPEC.md W18), smoke, six households, spouse with panic, carrying, burn tolerance, door forcing, hearing | done |
| Novelties: `door_jam`, `carpet_flashover`, `compound`, `hidden_dependent` | done |
| Hazard-aware symbolic planner with a plain-English reason for every move | done |
| Bayesian novelty detector (Beta-Bernoulli Bayes factors, sequentially valid) | done |
| Rule repair search with runner-up and ambiguity reporting; the adaptive agent | done |
| Evaluation harness, ablation study, steps-vs-oracle metric, GIFs | done |
| Audited reward, automated exploit search, weight sweeps | done |
| Evidence discipline: lockstep leak test, no memory between episodes | done |
| Procedural houses with measured difficulty; validated trajectory dataset; Gymnasium env | done |
| PPO baseline, trained-agent reward audit | done. From scratch it is far below the planner. Warm-started by cloning the adaptive agent and then fine-tuned, it matches the planner and beats it with a hidden visitor (one seed; [docs/RL.md](docs/RL.md#improvements-tried)) |
| DreamerV3 (world model with memory) vs PPO and the planner | tried and abandoned, a negative result: 0% held-out success after 82 CPU-minutes, below PPO. Seeding its replay with planner demos (tapered to zero) did not help either: still 0%, because the reward signal is lost in its RSSM latent (the same reward head without the RSSM learns it) ([docs/DREAMER.md](docs/DREAMER.md#planner-guided-dreamer-demo-seeded-replay-tapered-to-zero)) |
| Explanations: numbered doors and named rooms, "what changed, how the plan changes, what it costs (steps, estimated fire risk), how sure", an end-of-run explanation log with a technical appendix, GIF banner at the tick the agent noticed ([docs/EXPLAINING_IT.md](docs/EXPLAINING_IT.md#what-the-agent-says-when-it-changes-its-mind)) | done |
| CNN perception, LLM-agent interface | not built |

Since 2026-11-09 `door_jam` jams several doors drawn per seed (house 2-4, ranch 1-2); see docs/SPEC.md W17.

## Why not just use deep RL?

We tried ([docs/RL.md](docs/RL.md)). A PPO policy (small CNN over the grid, 2.2 M environment steps, about 30
minutes on 8 CPU cores, curriculum from the studio to the ranch with novelties sampled per episode) gets everyone
out of 42% of held-out ranch fires with the dog and 7% with two children; the planner gets 89% and 96% on the same
fires. When a door jams PPO drops to 3% with the dog, where the adaptive agent keeps 86%. After 1 M more steps with
novelties in the mix PPO still had not recovered on a jammed door or a hidden visitor, while the adaptive agent
fixes its rulebook 5 to 11 ticks after the change, inside the episode. Reading the trained agent's reward
components found no exploit. Compare it yourself: `emberpath eval --agents baseline adaptive ppo --checkpoint
runs/ppo_novelty/final.pt --seed-start 1000 --seeds 100` after `python scripts/train_ppo.py` (commands in docs/RL.md).
Update: PPO warm-started by behaviour cloning of the adaptive agent (training fires only), then fine-tuned for 600k
steps, reaches a pooled held-out score of 0.930 against 0.903 for the adaptive agent (+0.026 [+0.006, +0.046]). It
matches the planner in every trained setting and beats it on the dog with a hidden visitor, but falls behind it on the
unseen compound novelty ([docs/RL.md, Improvements tried](docs/RL.md#improvements-tried)). Read it with the caveats:
one training seed, about 33 comparisons, held-out means new fires in the same houses, it learned by imitating the
planner first, and part of its hidden-visitor edge is the oracle's "unknown person first" handicap. Try it:
`emberpath run --profile single --novelty hidden_dependent --agent ppo --checkpoint runs/bcppo/final.pt --seed 1000`
(checkpoints live in the gitignored `runs/`).

## Where to read next

* [docs/OVERVIEW.md](docs/OVERVIEW.md): the full breakdown.
* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): design decisions and every measured change, including negative results.
* [docs/REWARD.md](docs/REWARD.md): the reward and its exploit audit. [docs/DATA.md](docs/DATA.md): the dataset.
* [docs/SPEC.md](docs/SPEC.md) and [ROADMAP.md](ROADMAP.md): requirements and status. [tasks/](tasks/README.md): work
  packages. [AGENTS.md](AGENTS.md) / [CLAUDE.md](CLAUDE.md): rules for coding agents working here.

## Layout

```
blueprints/        ASCII floor plans: studio (tiny, tests), ranch (benchmark), house (wider, 6 exits)
src/emberpath/
  rules.py         Rules + RulePatch (every rule a Bernoulli coin)   world.py      simulator + Event log
  blueprint.py     floor-plan parser                                 dependents.py the six households
  scenarios.py     novelty presets                                   planner.py    hazard-aware planner
  novelty.py       Bayesian detector                                 repair.py     rule repair search
  agents.py        baseline / oracle / adaptive / naive_repair       evaluate.py   same-seed harness
  reward.py        the audited reward                                env.py        Gymnasium wrapper
  procgen.py       procedural houses                                 dataset.py    trajectory dataset
  explain.py       plain-English explanations of each repair         render.py     ASCII + GIF
  places.py        door numbers and room names for explanations
  rl_baseline.py   PPO, BC+PPO and PPOAgent                          cli.py        the emberpath command
  dreamer.py       DreamerV3 (abandoned negative result)
scripts/           results, GIFs, audits, exploit search, sweeps, studies, dataset build/validate
tests/             unit, acceptance and leak tests
```

## Limitations

* A 2D toy world: the claim is about the adaptation loop, not fire physics.
* Novelties are changes to existing rules. Structural novelty (new kinds of object or action) is out of scope.
* The oracle is the same planner given the truth, not a proven optimum, and it sometimes loses to the adaptive agent.
* The learned agents are one training run each on two fixed houses, and their checkpoints predate the final reward fix
  (the end-of-episode potential, [docs/REWARD.md](docs/REWARD.md)); they were not retrained.
* See [docs/OVERVIEW.md](docs/OVERVIEW.md#9-limitations-and-future-work) for the full list.
