# The reward

**In one line:** get everyone out as fast as possible, and keep the people you are rescuing away from the flames.

The reward lives in `src/emberpath/reward.py`. It is built to be *read*: every step reports each component
separately (`info["reward_components"]` in the Gym env), so you can always see what the agent was paid for.

## What it rewards and penalizes

| Component | Formula (defaults) | What it is for |
|---|---|---|
| `time` | `-0.02` every tick; to the horizon if the episode ends with someone lost | Pressure to be quick. Every wasted step costs, and only a full success stops the clock early. |
| `progress` | `0.03 x (potential(next) - potential(now))` | A dense "you are getting closer" signal (details below). |
| `rescue` | `+1.0` per dependent that reaches an exit | The point of the exercise. |
| `success` | `+5.0` when everyone is out and the rescuer is at an exit | Getting *everyone* out is worth far more than the sum of single rescues. |
| `fire_proximity` | `-0.10 x closeness` per waiting or following dependent the rescuer knows about, per tick | **Near flames.** Closeness is 0.75 at one step from fire, 0.5 at two, 0.25 at three, and 0 at four or more. Measured along walkable paths, since fire doesn't jump walls. |
| `fire_contact` | `-10.0` per dependent lost to flames, and per dependent still inside at a timeout | **A lost life dwarfs every other term** (about 130x the worst single-tick proximity charge). Losing one of two children costs more than the whole rescue is worth. |
| `agent_death` | see "Failing early" below | The rescuer dying must never be a good option. |

**Efficiency** is reported but not directly rewarded: `fewest possible steps / steps taken`, for successful
episodes only (`info["efficiency"]`, and a column in `emberpath eval`). The "fewest possible" ignores fire and
how slowly children move, so 1.0 is a ceiling that real runs rarely reach.

### A real episode
Ranch, one dog, a door jams, an agent that knows the true rules (seed 1). 54 steps against a geometric minimum of
21, so efficiency 0.39 (the jam forces a detour):

```
time -1.08   progress +0.63   rescue +1.00   success +5.00   fire_proximity 0.00   fire_contact 0.00   TOTAL +5.55
```

## Progress shaping, and why it is safe
Rewarding "being closer" is the classic way to speed up learning and also the classic way to create a loophole.
The potential is

```
potential = - (steps each waiting/following dependent still needs to reach an exit)
            - (steps from the rescuer to the nearest dependent still waiting that it knows about)
            - 2 x (longest distance in the house) for each dependent already lost
```

and `progress` is the *change* in potential. Because it is a difference of a function of the state, summed over
an episode it collapses to `potential(end) - potential(start)`. So walking in circles earns exactly nothing, and
the shaping cannot change which behavior is best; it only makes good behavior easier to find (Ng, Harada &
Russell, 1999). The lost-dependent term matters: if losing someone simply deleted their distance from the sum,
the agent would be *paid* for letting them burn.

"That it knows about" matters for the `hidden_dependent` novelty: a hidden visitor who is still waiting is left out
of the pickup term and of `fire_proximity` (`reward.perceived`, mirroring `World.observe`). Their exit distance stays
in the potential, constant while they wait, so finding them causes no jump; they count in full in every outcome term
(rescue, success, contact, death charge).

## Exploits we tried (and the defence for each)

| Exploit | Why a naive reward pays for it | Defence | Test |
|---|---|---|---|
| **Suicide.** Walk into the flames. | The old reward charged `-1` for dying but `-0.01 x 150 = -1.5` for timing out, so an agent that thought it would fail preferred dying. | If the rescuer dies the meter keeps running: the current per-tick cost until the horizon, plus contact for everyone left behind, plus a flat `1.0`. | `test_exploit_suicide_is_never_cheaper_than_stalling` |
| **Let them burn.** | Losing a dependent removes their distance from a naive "distance left" signal. | Lost dependents are held at a fixed value worse than any reachable distance. | `test_exploit_losing_a_dependent_never_raises_the_potential` |
| **Farm progress by looping.** | Rewarding "moved closer" pays every time you step closer again. | Potential-based shaping telescopes; loops net to zero. | `test_exploit_progress_reward_cannot_be_farmed_by_looping`, `test_exploit_back_and_forth_pays_nothing` |
| **Abandon them.** Run out alone. | Time cost ends sooner if you stop trying. | Lose rescue and success bonuses, pay contact if they burn, and the potential charges you for walking away from the nearest waiting dependent. | `test_exploit_abandoning_the_dependent_scores_worse_than_rescuing` |
| **Dawdle.** | None, if speed isn't priced. | Per-tick cost, so a slower success scores lower than a faster one. | `test_faster_rescue_scores_higher` |
| **Read the hiding place out of the reward** (found by `find_exploits.py`'s leak probe). | The pickup term counted a hidden visitor, so after the dog was collected the progress signal pointed straight at the hiding place (per-tick progress up to 0.39 that differed from a no-novelty control in 12/12 probe episodes). | Dense terms only see dependents the rescuer can perceive. | `test_exploit_progress_and_proximity_do_not_reveal_a_hidden_dependent`, `test_reward_snoop_learns_nothing_about_hidden_people` |
| **Let them burn to stop the clock** (found by `reward_sweep.py` at the default weights). | An episode ends once everyone is out or lost, so a loss stopped the time cost. In the hopeless `compound` fire, `abandon` (-2.83) outscored the oracle (-3.02, married, 60 seeds) although the oracle rescued 12% and abandon nobody. | A `partial` ending is charged time to the horizon, like a timeout. | `test_exploit_letting_them_burn_does_not_stop_the_clock`, `test_abandoning_does_not_outscore_honest_rescue_in_the_hopeless_compound_fire` |
| **Leave the endangered one behind** (opened by the heavy loss penalty). | With `-10` per loss to flames but nothing for someone still inside at a timeout, never reaching a child the fire is closing on beats trying and failing. | A timeout charges `contact_penalty` for everyone still inside, exactly like a loss. | `test_exploit_abandoning_the_dependent_scores_worse_than_rescuing`, audit `abandon` / `stall` rows |
| **Treat near-fire as free.** | A binary "touched fire" penalty gives no reason to stay clear until it's too late. | Smooth proximity penalty, with contact costing far more than any amount of nearness. | `test_fire_proximity_grows_as_a_dependent_nears_the_flames`, `test_touching_costs_far_more_than_being_near` |

### Audit with real policies
`python scripts/audit_reward.py` runs deliberately degenerate policies under the old and the new reward
(ranch, one dog, jammed door, 20 seeds). Full output: [reward_audit.md](reward_audit.md).

| policy | old reward | new reward (mean) | new reward (median) | rescued |
|---|---|---|---|---|
| oracle (honest, correct rules) | 1.84 | 2.05 | 5.47 | 0.85 |
| baseline (honest, stale rules) | -1.66 | -14.73 | -14.55 | 0.00 |
| stall | -1.82 | -14.79 | -14.55 | 0.00 |
| **suicide** | **-1.24** | **-15.20** | -14.09 | 0.00 |
| abandon | -1.24 | -14.68 | -14.55 | 0.00 |
| reward_snoop | -1.66 | -14.73 | -14.55 | 0.00 |

(Weights since raised to `success_bonus` 5 and `contact_penalty` 10 with the timeout charge; the ranking checks still
hold and `find_exploits.py --budget small` finds 0 loopholes in 960 episodes.)

`reward_snoop` is the baseline plus any hidden person the reward's potential gives away (it probes the potential at
every blueprint slot). It must play exactly like the baseline; the audit prints `LEAK` if it does not. With
`--novelty hidden_dependent` the old potential let it rescue 0.80 of dependents on 10 seeds (8/10 successes) against the
baseline's 0.40 (0/10); now it is identical to the baseline.

* **Old reward:** suicide (-1.24) outscored an honest agent that was struggling (-1.66). That is the loophole,
  confirmed with a real policy and not just arithmetic.
* **New reward:** suicide is the worst policy. Honest rescue (oracle) ranks first under both.
* **About `abandon` ranking above `baseline`:** not an exploit. The baseline is stuck at a jammed door and
  earns nothing; abandoning at least gets out. Since a loss keeps the clock running the two are 0.05 apart. The
  meaningful comparison is abandon versus a *successful* honest agent: about 20 points apart.
* **About the oracle mean of -0.00:** 3 of 20 oracle episodes fail in this hot-fire scenario, and failures are
  charged heavily on purpose. The median (1.47) is the typical successful episode.

My first version of the audit had a bug worth recording: the suicide policy planned with plain wall geometry,
bumped the jammed door forever, and "failed to die" in 18 of 20 episodes. A broken exploit policy proves nothing
about the reward, so the audit now makes it jam-aware and prints outcomes so this kind of mistake is visible.

## Automated search and sweeps (WP-M)
All seeded; numbers below are from the commands shown, on an 8-core laptop.

**Exploit search** (`python scripts/find_exploits.py --budget small|medium`). A 5-parameter family of scripted
policies (weights on closeness to the nearest visible waiting dependent, closeness to the exit, distance from fire, a
wait bonus, and exploration noise; routes avoid visible flames and doors found jammed), searched by random search
plus a small evolution strategy to maximize return, on ranch / single for the control and every novelty preset. A
loophole = return at least an honest agent's (oracle or baseline, same seeds) while success is lower by 0.25 or more, or
deaths higher by 0.25 or more without more rescues; candidates are re-checked on fresh seeds.
* Legacy reward, small budget (240 policies x 4 seeds, 24 s): **rediscovers the suicide loophole** on `door_jam`,
  confirmed on 12 fresh seeds: `fire=-1.00` (walk into the flames), died 0.92 vs 0.00, return -1.45 vs the stuck
  baseline's -1.58. Its component breakdown under the new reward shows why the new reward refuses it: `agent_death -6.04`.
* New reward: no confirmed loophole. Small budget: 240 policies, 960 episodes, 21 s. Medium budget: 800 policies x 6
  seeds, 4800 episodes, 83 s. One medium candidate on `hidden_dependent` (return -0.54 vs oracle -0.84, success 0.50
  vs 0.83 on 6 seeds) did not hold up on 20 fresh seeds. This is a negative result over a small, hand-made family; it is
  not a proof.
* **Leak probe** (same script). The novelty world and a no-novelty control are stepped with the same actions while
  observations and evidence are identical; any reward difference there tells a learner something it could not see.
  `hidden_dependent`: progress leaked up to 0.39 per tick in 12/12 episodes before the fix, none after.
  `carpet_flashover`: none. `door_jam` and `compound`: **still leak** (see Limitations).

**Weight sweep** (`python scripts/reward_sweep.py --part sweep`, 40 seeds, about 2 min). Each `RewardConfig` weight
x0, x0.25, x0.5, x2, x4 (radius 1, 2, 3, 6, 8), one at a time, ranking stall, suicide, abandon, baseline and oracle on
the four presets. At the defaults both checks hold everywhere ("honest first": oracle above stall, suicide and abandon;
"dying never pays": suicide below stall and baseline). Mean Kendall tau against the default ranking is 0.85 or more for
every setting. The checks break only at `progress_weight` x4 (0.12: suicide edges above a struggling agent on
`door_jam`) and `proximity_weight` x4 (0.4: abandon above oracle on `compound`). Before the "partial" fix, the
default row itself failed "honest first" on `compound`. With only 20 seeds, many settings appeared to break on
`compound`; that was seed noise (see reliability).

**Signal reliability** (same script, 40 seeds). Returns are heavy-tailed because failures are charged hard. The
oracle's return sd is 3.9 on `door_jam`, 3.1 on `carpet_flashover`, 3.6 on `compound` and 6.6 on `hidden_dependent`,
against medians of +1.45, +1.54, -5.35 and +1.90. Telling two policies 0.5 apart in mean return (5% test, 80% power)
takes about 940 seeds per policy unpaired on `door_jam`, or 380 paired (same seeds, identical fire). Compare
policies on paired seeds, and report medians and success rates next to mean return; a 20-seed audit ranks policies
that differ by several points, not by tenths.

**Sparse vs shaped** (`--part ablation`, about 20 s). An evolution strategy over the same family on studio /
married, 6 learner seeds x 20 generations, success measured on 10 held-out seeds. The full reward reaches 0.42
success (0.27 at generation 10; 0.3 first reached after 8.3 generations, 6/6 learners). Sparse (rescue + success
only) reaches 0.03 (1/6 learners reached 0.3, after 14). In 107 of 120 sparse generations every candidate scored
exactly the same, so there was nothing to learn from; with the full reward that happened in 4/120. The shaping earns
its keep. Final success sd across learners: 0.09 (full), 0.07 (sparse).

## Limitations (read before trusting it)
* **No learning agent has been trained on this yet.** The tests and the automated search cover a hand-made policy
  family; a learner may find exploits outside it. The real validation is still training PPO (roadmap Phase 4b), then
  reading the component breakdown of its best episodes.
* **The potential still leaks door jams.** Exit distances use the true jammed doors, so when a door jams the
  progress signal changes before the rescuer has touched it (up to 0.36 in one tick on `door_jam`/`compound`, 20/20
  probe episodes), and afterwards it points along the true detour. That matters only for a learner that sees rewards
  during an episode (a recurrent policy fed its rewards, an LLM agent shown them). It is not fixed: computing distances
  with nominal doors would make the shaping point into the jammed door, and it would change every `door_jam`/`compound`
  return row. If you build such an agent, fix this first.
* **Nearness accumulates; contact does not.** Proximity is charged every tick, contact once. A dependent left one step
  from flames for about 13 ticks costs as much as one who touches them, so over a long wait, letting the fire reach them
  can be cheaper than keeping them near it alive. No tested policy exploits this (dependents never step into fire, and
  the rescuer cannot push them), but a cap on total nearness per dependent would close it.
* **The death charge assumes a static hazard.** It continues the current per-tick cost to the horizon. If fire
  later moves closer, staying alive could cost more than the charge assumes (usually then dependents are lost and
  charged too). It is tested, not proved.
* **Failures are heavy-tailed by design.** A death can cost around -5 to -10 against about +1.5 for a good
  success. If that makes learning too cautious, lower `death_extra` or normalize returns; do not remove the
  principle that dying must not be cheaper than carrying on.
* **Proximity uses walking distance to flames, not smoke.** Smoke hurts visibility in the simulator but is not
  penalized. That is a deliberate scope limit.
* **Weights are reasoned, not tuned.** They come from "a successful rescue should be clearly positive, a
  failure clearly negative, and contact far worse than nearness". `RewardConfig` exposes all of them for sweeps.

## Changing the reward
1. Edit `RewardConfig` or `RewardTracker.update`.
2. Add a degenerate policy to `scripts/audit_reward.py` for any exploit you can think of, and a test.
3. `python -m pytest tests/test_reward.py` and `python scripts/audit_reward.py`: honest effort must still rank
   above every shortcut.
4. `python scripts/find_exploits.py --budget small` (no confirmed loophole, no new leak) and
   `python scripts/reward_sweep.py --part sweep` (both checks hold at the defaults).
