# The reward

**In one line:** get everyone out as fast as possible, and keep the people you are rescuing away from the flames.

The reward lives in `src/emberpath/reward.py`. It is built to be *read*: every step reports each component
separately (`info["reward_components"]` in the Gym env), so you can always see what the agent was paid for.

## What it rewards and penalizes

| Component | Formula (defaults) | What it is for |
|---|---|---|
| `time` | `-0.02` every tick | Pressure to be quick. Every wasted step costs. |
| `progress` | `0.03 x (potential(next) - potential(now))` | A dense "you are getting closer" signal (details below). |
| `rescue` | `+1.0` per dependent that reaches an exit | The point of the exercise. |
| `success` | `+1.0` when everyone is out and the rescuer is at an exit | Finishing the job, not just parts of it. |
| `fire_proximity` | `-0.10 x closeness` per waiting or following dependent, per tick | **Near flames.** Closeness is 0.75 at one step from fire, 0.5 at two, 0.25 at three, and 0 at four or more. Measured along walkable paths, since fire doesn't jump walls. |
| `fire_contact` | `-1.0` per dependent lost to flames | **Touching flames.** About 13x the worst single-tick proximity charge. |
| `agent_death` | see "Failing early" below | The rescuer dying must never be a good option. |

**Efficiency** is reported but not directly rewarded: `fewest possible steps / steps taken`, for successful
episodes only (`info["efficiency"]`, and a column in `emberpath eval`). The "fewest possible" ignores fire and
how slowly children move, so 1.0 is a ceiling that real runs rarely reach.

### A real episode
Ranch, one dog, a door jams, an agent that knows the true rules (seed 1). 54 steps against a geometric minimum of
21, so efficiency 0.39 (the jam forces a detour):

```
time -1.08   progress +0.63   rescue +1.00   success +1.00   fire_proximity 0.00   fire_contact 0.00   TOTAL +1.55
```

## Progress shaping, and why it is safe
Rewarding "being closer" is the classic way to speed up learning and also the classic way to create a loophole.
The potential is

```
potential = - (steps each waiting/following dependent still needs to reach an exit)
            - (steps from the rescuer to the nearest dependent still waiting)
            - 2 x (longest distance in the house) for each dependent already lost
```

and `progress` is the *change* in potential. Because it is a difference of a function of the state, summed over
an episode it collapses to `potential(end) - potential(start)`. So walking in circles earns exactly nothing, and
the shaping cannot change which behavior is best; it only makes good behavior easier to find (Ng, Harada &
Russell, 1999). The lost-dependent term matters: if losing someone simply deleted their distance from the sum,
the agent would be *paid* for letting them burn.

## Exploits we tried (and the defence for each)

| Exploit | Why a naive reward pays for it | Defence | Test |
|---|---|---|---|
| **Suicide.** Walk into the flames. | The old reward charged `-1` for dying but `-0.01 x 150 = -1.5` for timing out, so an agent that thought it would fail preferred dying. | If the rescuer dies the meter keeps running: the current per-tick cost until the horizon, plus contact for everyone left behind, plus a flat `1.0`. | `test_exploit_suicide_is_never_cheaper_than_stalling` |
| **Let them burn.** | Losing a dependent removes their distance from a naive "distance left" signal. | Lost dependents are held at a fixed value worse than any reachable distance. | `test_exploit_losing_a_dependent_never_raises_the_potential` |
| **Farm progress by looping.** | Rewarding "moved closer" pays every time you step closer again. | Potential-based shaping telescopes; loops net to zero. | `test_exploit_progress_reward_cannot_be_farmed_by_looping`, `test_exploit_back_and_forth_pays_nothing` |
| **Abandon them.** Run out alone. | Time cost ends sooner if you stop trying. | Lose rescue and success bonuses, pay contact if they burn, and the potential charges you for walking away from the nearest waiting dependent. | `test_exploit_abandoning_the_dependent_scores_worse_than_rescuing` |
| **Dawdle.** | None, if speed isn't priced. | Per-tick cost, so a slower success scores lower than a faster one. | `test_faster_rescue_scores_higher` |
| **Treat near-fire as free.** | A binary "touched fire" penalty gives no reason to stay clear until it's too late. | Smooth proximity penalty, with contact costing far more than any amount of nearness. | `test_fire_proximity_grows_as_a_dependent_nears_the_flames`, `test_touching_costs_far_more_than_being_near` |

### Audit with real policies
`python scripts/audit_reward.py` runs deliberately degenerate policies under the old and the new reward
(ranch, one dog, jammed door, 20 seeds). Full output: [reward_audit.md](reward_audit.md).

| policy | old reward | new reward (mean) | new reward (median) | rescued |
|---|---|---|---|---|
| oracle (honest, correct rules) | 1.84 | 0.09 | 1.47 | 0.85 |
| baseline (honest, stale rules) | -1.66 | -5.01 | -4.39 | 0.00 |
| stall | -1.82 | -5.34 | -5.55 | 0.00 |
| **suicide** | **-1.24** | **-6.20** | -5.09 | 0.00 |
| abandon | -1.24 | -4.15 | -4.10 | 0.00 |

* **Old reward:** suicide (-1.24) outscored an honest agent that was struggling (-1.66). That is the loophole,
  confirmed with a real policy and not just arithmetic.
* **New reward:** suicide is the worst policy. Honest rescue (oracle) ranks first under both.
* **About `abandon` ranking above `baseline`:** not an exploit. The baseline is stuck at a jammed door and
  earns nothing; abandoning at least gets out. The meaningful comparison is abandon versus a *successful* honest
  agent: about 4 points apart.
* **About the oracle mean of 0.09:** 3 of 20 oracle episodes fail in this hot-fire scenario, and failures are
  charged heavily on purpose. The median (1.47) is the typical successful episode.

My first version of the audit had a bug worth recording: the suicide policy planned with plain wall geometry,
bumped the jammed door forever, and "failed to die" in 18 of 20 episodes. A broken exploit policy proves nothing
about the reward, so the audit now makes it jam-aware and prints outcomes so this kind of mistake is visible.

## Limitations (read before trusting it)
* **No learning agent has been trained on this yet.** The tests cover exploits I thought of. A learning agent
  may find ones I didn't; the real validation is training PPO (roadmap Phase 4b), then reading the component
  breakdown of its best episodes.
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
