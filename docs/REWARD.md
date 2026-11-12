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
When a spouse is in the house the "fewest possible" counts only the rescuer's walk to an exit (she may fetch the
others herself), so efficiency there is very low and only useful for comparing agents with each other. To compare
the adaptive agent against the oracle, use `steps vs oracle` in the results tables (`evaluate.step_ratio`):
adaptive steps / oracle steps, counted only on seeds where *both* got everyone out, so dying early or leaving
someone behind can never look fast.

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
(steps are walked around walls and around the jammed doors someone has already bumped into, not the true jams)

once the episode is over:
potential = - 2 x (longest distance in the house at the start) for each dependent lost to the flames
            (all the distance terms are 0)
```

and `progress` is the *change* in potential. Because it is a difference of a function of the state, summed over
an episode it collapses to `potential(end) - potential(start)`. So walking in circles earns exactly nothing, and
the shaping only makes good behavior easier to find (Ng, Harada & Russell, 1999). The lost-dependent term matters:
if losing someone simply deleted their distance from the sum, the agent would be *paid* for letting them burn.

**The end-of-episode potential.** Until the final verification (2026-11) the potential at the end of an episode was
the same formula as during it, so a failed episode kept its final distances as partial credit. That broke AGENTS.md
invariant 8 once the jam fix below made the distances evidence-only: an unfound jam keeps them optimistic, so in
`make audit` `stall` (-14.36) and `door_shy` (-14.24, never tries the jammed door) both out-scored the honest
`baseline` stuck at that door (-14.66). Now every distance ("work left") term is 0 at a terminal state, so across
episodes from the same start the shaping can only differ by how many people were lost to the flames: no ending
earns credit for where people stood or which jams were never found. Ng et al.'s invariance needs a terminal potential
that is the same for every ending; this one keeps a fixed charge per loss, because a pure zero would refund the
loss term and make letting a dependent burn at once tie a timeout in which they survive
(`test_exploit_letting_them_burn_does_not_stop_the_clock` would fail). So the shaping is invariant except that it
adds `0.03 x 2 x (longest distance)` (about 2.3 on the ranch) to the cost of each death by fire, which only ever
favours saving people. Success episodes are unchanged (everyone out already meant potential 0).

"That it knows about" matters for the `hidden_dependent` novelty: a hidden visitor who is still waiting is left out
of the pickup term and of `fire_proximity` (`reward.perceived`, mirroring `World.observe`). Their exit distance stays
in the potential, constant while they wait, so finding them causes no jump; they count in full in every outcome term
(rescue, success, contact, death charge).

"Already bumped into" matters for `door_jam` and `compound`: distances treat as blocked only `reward.known_jams` =
`world.bumped_jams - world.broken_doors` (doors someone walked into and found jammed, and not since forced open).
Until 2026-11 they used the true `world.rules.jammed_doors`, which leaked the jam (see the exploits table and
"Independent verification" below).

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
| **Read the jammed door out of the reward** (found in the independent WP-C verification). | The potential's distances used the TRUE jammed doors, so the progress payment fell by 0.36-0.42 on the very tick a door jammed (ranch / dog: 100/100 held-out seeds), before anyone had touched it, and afterwards pointed along the true detour. `jam_snoop`, which sees only its own progress payments and observable positions, used that to plan around the door and matched the oracle (0.90 rescued vs the baseline's 0.00; kids 0.78 vs 0.47). | Distances use only jams the rescuer has evidence of (`known_jams`). | `test_exploit_progress_does_not_reveal_a_jammed_door_before_it_is_tried`, `test_jam_snoop_learns_nothing_about_unfound_jams`, `test_leak_probe_sees_the_jammed_door_only_in_the_leaky_reward` |
| **Never find the jam, or just stand still** (opened by the fix above; closed in the final verification). | With an evidence-only potential an unfound jam keeps the potential optimistic, and a failed episode kept its final potential, so a failure that never tried the door kept the progress a stuck agent loses at the bump: `door_shy` -14.24 and `stall` -14.36 against the stuck honest `baseline` -14.66. | The distance terms of the potential are 0 at every terminal state, so failures differ in shaping only by losses to fire. Now `door_shy` ties the baseline exactly and `stall` is below it, on every seed checked (ranch single and kids, door_jam); the audit prints `EXPLOIT` if either beats the baseline. | `test_exploit_no_ending_earns_partial_shaping_credit`, `test_exploit_never_finding_the_jam_does_not_beat_looking` |
| **Treat near-fire as free.** | A binary "touched fire" penalty gives no reason to stay clear until it's too late. | Smooth proximity penalty, with contact costing far more than any amount of nearness. | `test_fire_proximity_grows_as_a_dependent_nears_the_flames`, `test_touching_costs_far_more_than_being_near` |

### Audit with real policies
`python scripts/audit_reward.py` runs deliberately degenerate policies under the old and the new reward
(ranch, one dog, jammed door, 20 seeds). Full output: [reward_audit.md](reward_audit.md).

| policy | old reward | new reward (mean) | new reward (median) | rescued |
|---|---|---|---|---|
| oracle (honest, correct rules) | 2.07 | 3.41 | 5.47 | 0.90 |
| adaptive (honest, learns the jam by bumping it) | 1.52 | 0.37 | 5.18 | 0.75 |
| baseline (honest, stale rules, stuck at the jam) | -1.63 | -14.03 | -14.19 | 0.00 |
| stall | -1.80 | -14.08 | -14.19 | 0.00 |
| **suicide** | **-1.27** | **-14.96** | -13.37 | 0.00 |
| abandon | -1.24 | -14.03 | -14.19 | 0.00 |
| reward_snoop | -1.63 | -14.03 | -14.19 | 0.00 |
| door_shy | -1.63 | -14.03 | -14.19 | 0.00 |
| jam_snoop | -1.63 | -14.03 | -14.19 | 0.00 |

(Current numbers, after the end-of-episode potential fix; `find_exploits.py --budget small` finds 0 loopholes in
960 episodes. Suicide's *median* is above the baseline's: on 8 of 20 seeds dying early is a little cheaper than staying
stuck while the fire closes in, the "death charge assumes a static hazard" limitation below. Its mean stays last.)

`reward_snoop` is the baseline plus any hidden person the reward's potential gives away (it probes the potential at
every blueprint slot). It must play exactly like the baseline; the audit prints `LEAK` if it does not. With
`--novelty hidden_dependent` the old potential let it rescue 0.80 of dependents on 10 seeds (8/10 successes) against the
baseline's 0.40 (0/10); now it is identical to the baseline.

Three rows were added in the 2026-11 verification (current numbers in [reward_audit.md](reward_audit.md)): `adaptive`
(honest, learns the jam by bumping it: +0.37, 0.75 rescued), `jam_snoop` (the baseline plus any jammed door its own
progress payments give away; with the old true-jam potential it scored +3.37 and rescued 0.90, the oracle's numbers;
now identical to the baseline, and the audit prints `LEAK` if it is not) and `door_shy` (never tries the jammed door,
cheating to know which: -14.03, tied with the stuck baseline and far below every honest agent that gets through; the
audit prints `EXPLOIT` if it beats oracle or adaptive, or if it or `stall` beats the baseline).

* **Old reward:** suicide (-1.24) outscored an honest agent that was struggling (-1.66). That is the loophole,
  confirmed with a real policy and not just arithmetic.
* **New reward:** suicide is the worst policy. Honest rescue (oracle) ranks first under both.
* **About `abandon` tying `baseline`:** not an exploit. The baseline is stuck at a jammed door and rescues nobody;
  abandoning rescues nobody either, and since a loss keeps the clock running they now score the same (-14.03). The
  meaningful comparison is abandon versus an honest agent that gets through: about 17 points apart. (With the married
  household `abandon` scores above the stuck baseline, because the spouse walks out by herself, so leaving her is
  not abandoning anyone.)
* **About the oracle mean of 3.41:** 2 of 20 oracle episodes fail in this hot-fire scenario, and failures are
  charged heavily on purpose. The median (5.47) is the typical successful episode.

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
* Legacy reward, small budget (240 policies x 4 seeds, about 13 s): when this was written it **rediscovered the
  suicide loophole** on `door_jam`, confirmed on 12 fresh seeds: `fire=-1.00` (walk into the flames), died 0.92 vs
  0.00, return -1.45 vs the stuck baseline's -1.58. **Rerun in the final verification (current dynamics), it no longer
  confirms it:** the in-sample search still finds fire-seeking policies that out-score the stuck baseline by dying
  (`test_the_small_search_rediscovers_the_legacy_suicide_loophole`, which checks the in-sample archive, passes), but
  the best door_jam candidate did not hold up on 12 fresh seeds (-1.63 vs -1.57), so the script reports 0 confirmed.
  The hand-written `suicide` policy still exposes the legacy loophole in `make audit` (-1.27 vs -1.63) and
  `test_loophole_rule_flags_the_legacy_suicide_and_not_the_new_reward` still passes. The fire, spouse and tile
  changes since then altered every fire; the search budget was not raised to chase the old result.
* New reward: no confirmed loophole. Small budget: 240 policies, 960 episodes, 10-21 s (rerun in the final
  verification, before and after the end-of-episode potential fix: 0 confirmed, leak probe 0/12 on every preset). Medium budget: 800 policies x 6
  seeds, 4800 episodes, 83 s. One medium candidate on `hidden_dependent` (return -0.54 vs oracle -0.84, success 0.50
  vs 0.83 on 6 seeds) did not hold up on 20 fresh seeds. This is a negative result over a small, hand-made family; it is
  not a proof.
* **Leak probe** (same script). The novelty world and a no-novelty control are stepped with the same actions while
  observations and evidence are identical; any reward difference there tells a learner something it could not see.
  `hidden_dependent`: progress leaked up to 0.39 per tick in 12/12 episodes before the fix, none after.
  `door_jam` / `compound`: progress leaked 0.36 per tick in 12/12 episodes until the evidence-only jam fix, none after.
  `carpet_flashover`: none.

**Weight sweep** (`python scripts/reward_sweep.py --part sweep`, 40 seeds, about 2 min). Each `RewardConfig` weight
x0, x0.25, x0.5, x2, x4 (radius 1, 2, 3, 6, 8), one at a time, ranking stall, suicide, abandon, baseline and oracle on
the four presets. At the defaults both checks hold everywhere ("honest first": oracle above stall, suicide and abandon;
"dying never pays": suicide below stall and baseline). Mean Kendall tau against the default ranking is 0.85 or more for
every setting. The checks break only at `progress_weight` x4 (0.12: suicide edges above a struggling agent on
`door_jam`) and, before the evidence-only jam fix, `proximity_weight` x4 (0.4: abandon above oracle on `compound`;
rerun after the fix, only `progress_weight` 0.12 breaks). Before the "partial" fix, the
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
* **Only a weak learner has been trained on it.** A PPO policy (docs/RL.md) was trained for 2.2 M steps and its
  component breakdowns read: 2,000 held-out and 20,528 training episodes. No exploit: the best episodes are genuine
  rescues with exactly the telescoped progress, the best non-success (-9.92) sits far below the worst success
  (+0.73), dying averages below timing out, and it waits less than the planner. Its worst habit, bringing one child
  out and losing the other, is charged heavily, so it is a learning failure, not a loophole. The policy only reaches
  42% / 7% success, so a stronger learner could still find exploits this one could not reach.
* **Not looking for a jam used to be worth a little; fixed for undiscounted returns.** Before the end-of-episode
  potential fix, a failure that never tried the jammed door kept up to 0.42 more progress than one stuck at it, and
  `stall` / `door_shy` out-ranked the stuck baseline (an invariant-8 violation, judged and fixed in the final
  verification; see "The end-of-episode potential" above). Now they tie or rank below it. **For a discounted learner
  (gamma 0.99) a small edge remains**: the gamma mismatch below charges (1 - gamma) x 0.03 x |potential| per tick, and
  a stuck agent that has bumped the jam carries a larger |potential| for the rest of the episode, so in the
  *discounted* return `door_shy` still beats the stuck baseline by 0.2-0.4 (ranch / single / door_jam) and 0.1 (kids),
  and `stall` by 0.2-0.4 with the dog (it is 2.5 below with the kids), on seeds 0-3. Suicide stays far below in the
  discounted return too. Reshaping with the learner's own gamma would close it.
* **The learned checkpoints were trained under older rewards.** `runs/ppo_*`, `runs/bc_*`, `runs/bcppo` and every
  Dreamer run were trained before the end-of-episode potential fix (the original PPO runs also before the jam-leak
  fix). The change only moves the final progress payment of failed episodes, and those checkpoints were not
  retrained. Their held-out scores and success rates (docs/RL.md, docs/DREAMER.md) do not depend on the reward at
  evaluation time, but a retrained policy could differ.
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

## Independent verification (WP-C check, 2026-11)
Commands: `make audit`, `python scripts/find_exploits.py`, `python scripts/reward_sweep.py --part sweep`, plus
throwaway scripts that stepped 1,020 full episodes (ranch and studio; single, kids, married; every novelty; baseline,
oracle, sampled PPO, random actions; seeds 1000..1009) and 233 branch points.

* **Telescoping holds exactly.** Over all 1,020 episodes, sum of `progress` = 0.03 x (potential(end) - potential(start))
  to 2e-16. The shaping uses gamma = 1. It was not zero at terminal states, so the sum depended on the end state: among
  failures, ending with people nearer an exit scored a little higher (partial credit, at most 0.03 x a few dozen cells).
  Policy invariance in the Ng et al. sense therefore held only up to that end-state term. (Fixed in the final
  verification: the distance terms are now 0 at every terminal state; only a fixed charge per loss to fire remains.)
* **PPO's gamma (0.99) does not match the shaping's (1).** For a discounted learner the gamma = 1 shaping equals the
  gamma-consistent one (gamma potential(s') - potential(s)) plus (1 - gamma) x 0.03 x potential(s') per tick, an extra
  cost proportional to the work left (about 0.01 a tick early in an episode, comparable to the 0.02 time cost). In the
  discounted return it averages -0.34 per episode (range -0.01 to -1.64, against a mean |discounted return| of 8.0). It
  pushes toward finishing sooner, which the reward wants anyway, and it is larger after a loss (the lost value is the
  worst potential), so it never pays for a bad ending. Not changed; a learner that needs exact invariance should use
  gamma = 1 or reshape with its own gamma.
* **Terminal charges are as documented.** Final-tick components checked on every one of the 1,020 endings (444
  success, 100 partial, 278 timeout, 198 agent_lost): success pays +5 with one tick of time; partial pays time to the
  horizon; timeout charges -10 for everyone still inside; a death charges at least time to the horizon + 10 per person
  left + 1. No violations.
* **Dying is usually, not always, worse than stalling.** From 233 mid-episode states (baseline play, every 10 ticks
  once fire exists) the episode was branched into "walk into the fire" and "wait to the end" on identical fire. Dying
  paid more in 18 (8%), by 0.04 to 3.3. All 18 are hopeless (nobody rescued in either branch). Two causes, both under
  the existing "death charge assumes a static hazard" limitation: the stalled branch keeps paying proximity as the fire
  closes in (up to 4.8 more on the studio with the kids), and dying freezes the potential while a stalled dependent who
  is later lost drops to the lost value (up to +1.3 progress for dying). Against an honest agent that is still trying
  (the audit's `baseline`) suicide stays last. A charge at the worst-case proximity rate would close the gap and make
  deaths much more expensive; not done here.
* **No double counting at the horizon.** `EmberPathEnv.step` returns `terminated=True, truncated=False` for every
  ending (timeout included), and `rl_baseline.compute_gae` masks the bootstrap with that done flag; the obs after a
  done is the next episode's reset, and its value is never used for the finished one (tests in
  `tests/test_ppo_math.py`). The time-remaining channel keeps the timeout Markov, so treating it as terminal is right.
* **The door-jam leak was real and is fixed.** Measured over 100 held-out seeds with the novelty world and a control
  stepped in lockstep until the rescuer could tell them apart: with the dog, the progress payment differed on the jam
  tick in 100/100 episodes for every policy tried (baseline -0.36, PPO -0.42, random -0.36 on average), and later
  ticks kept differing (mean 3 to 26 ticks per episode, up to 0.42). With the kids it never differed on the jam tick but
  did later in 93-100/100 episodes (up to 0.36), as the rescuer neared the door; with the spouse in 3-4/100. A
  feed-forward PPO policy cannot read it at test time (it never sees rewards), but its training signal and any
  learner fed its rewards (recurrent policies, LLM agents shown rewards, the trajectory dataset's per-step reward) can:
  `jam_snoop` turned it into oracle-level play using only its own payments. After the fix (`known_jams`) the lockstep
  difference is 0 in every one of those 1,800 episodes, `jam_snoop` plays exactly like the baseline, and the leak
  probe reports 0/12 on every preset. `make results` and `make results-house` are byte-identical before and after (the
  planners never read the reward). Audit returns moved only for policies that never bump the door in failing episodes
  (oracle mean 3.37 -> 3.41, stall -14.73 -> -14.36, abandon -14.66 -> -14.46). The efficiency lower bound still uses the
  true jams; it is computed once at the start (before any jam) and reported only when the episode is over.
* **Effect on PPO training (one seed, not a controlled study).** Re-running the first 254k novelty-phase steps from the
  same frozen policy and seed under the fixed reward: door_jam held-out success stayed at 0-2% (old 2% greedy / 1%
  sampled, fixed 0% / 1%); sampled scores moved by at most 0.04 anywhere. Greedy scores moved by -0.21 to +0.12 in
  settings the jam cannot touch (e.g. kids with no novelty), which is the whole training trajectory forking after the
  first changed reward, i.e. seed-level noise. The trained checkpoints in `runs/` were trained on the leaky reward.

## Changing the reward
1. Edit `RewardConfig` or `RewardTracker.update`.
2. Add a degenerate policy to `scripts/audit_reward.py` for any exploit you can think of, and a test.
3. `python -m pytest tests/test_reward.py` and `python scripts/audit_reward.py`: honest effort must still rank
   above every shortcut.
4. `python scripts/find_exploits.py --budget small` (no confirmed loophole, no new leak) and
   `python scripts/reward_sweep.py --part sweep` (both checks hold at the defaults).
