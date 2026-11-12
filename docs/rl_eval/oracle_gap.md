## Why does BC+PPO score above the oracle?

**Question.** On the 8 trained ranch settings (dog / kids × none, door_jam, carpet_flashover, hidden_dependent;
held-out fires 1000..1099) BC+PPO (`runs/bcppo/final.pt`) pools to 0.930, against 0.911 for the oracle and 0.903 for
the adaptive agent. The oracle is the same `HazardAwarePlanner` given the true rules, so "beats the oracle" can only
mean the planner is not reward-optimal, the comparison is chance, or the learner sees something it should not. This
section measures how much each explanation accounts for.

**Short answer.**

* Only **one** per-setting win over the oracle survives multiple-comparison correction: **dog / hidden_dependent**,
  +0.090 [+0.050, +0.135], Holm-adjusted p = 0.003 over the 20 bcppo-vs-planner tests (0.016 over 140 tests covering
  every learned variant). Every other "≥ oracle" row is consistent with chance.
* **(a) The "unknown person first" handicap in `_rank` is the largest single cause.** A test-only fix (treat a hidden,
  not-yet-found person like the dog: no priority over anyone) closes **all** of the kids / hidden_dependent gap
  (+0.040 of +0.040) and 39% of the dog / hidden_dependent gap (+0.035 of +0.090). Pooled, it is +0.009 of the +0.019
  gap to the oracle (50%) and +0.005 of the +0.026 gap to the adaptive agent (19%).
* **(b) `save_margin` and the waiting rules explain almost nothing.** Of 32 pre-declared combinations, chosen on
  training fires 0..99, the best changes only the "don't wait for followers near flames" radius (3 to 1). On the
  held-out fires it adds +0.015 on dog / hidden_dependent and +0.001 pooled. No `save_margin` value helped on the
  training fires.
* **(c) What is left is real, but small and specific to the dog.** With both patches the oracle pools to 0.921;
  BC+PPO is +0.008 [-0.009, +0.025] above it, a tie. Against the patched *adaptive* agent BC+PPO is still +0.022
  [+0.004, +0.038]. The residual is in the dog household (+0.033 [+0.006, +0.060] over its 4 settings vs the patched
  oracle). In the traces it comes from one learned habit: when the fire is about to cut off the route, the policy
  turns back and waits for it to burn out, or takes the other exit, so the unreliable dog is not caught by the
  flames. The planner presses on. BC+PPO pays for this with longer episodes and more rescuer deaths.
* **(d) Leaks: none found.** Lockstep evidence test on both checkpoints: 0 differing decisions in 530 paired episodes
  each. The test still catches the oracle.
* **A retrain under the current reward** (`runs/formal_eval/train/bcppo_s0/ckpt_p100.pt`) shows the same pattern
  (dog / hidden_dependent +0.090 vs the oracle, Holm p = 0.01; tied with the patched oracle when pooled,
  +0.002 [-0.015, +0.021]). Its other single-setting gains are in different settings, which supports reading them as
  noise.

### Method

* **Agents and fires.** The cached held-out episodes of `scripts/ppo_variants_eval.py` (seeds 1000..1099, shared
  harness `evaluate.run_episode`). The study re-runs every episode it uses through a tracing wrapper (same harness,
  plus a per-tick `record` callback). The re-runs reproduce `runs/variants_eval/` exactly: identical score, steps and
  outcome for bcppo, oracle and adaptive in all 10 ranch settings.
* **Statistics.** Score = fraction rescued, 0 if the rescuer dies. 95% paired-bootstrap intervals over the 100 seeds
  (2,000 resamples, seed 0, as in `evaluate.paired_bootstrap`). "Pooled" = each seed's mean over the 8 trained ranch
  settings, then bootstrapped. Shares are ratios of means, with intervals from the same resamples; they are unstable
  wherever the gap is near 0. p-values are two-sided paired bootstraps of the centred differences (20,000 resamples).
  A sign-flip permutation test agrees.
* **Test-only planner patches** (scratch/oracle_gap/og.py; **nothing in `src/` was changed**):
  * `rank=unk2`: monkeypatches `emberpath.planner._rank` in the worker process so that `"unknown"` maps to 2 (as the
    dog) instead of 0 (as a child). Also tried: `unk1` (unknown = adult) and `flat` (every rank 1, no priorities).
  * `sm`, `mw`, `pg`: non-default `save_margin`, `max_wait`, `pace_gap` keyword arguments to `HazardAwarePlanner`.
  * `wd`: a subclass whose `_maybe_wait_for_followers` takes the danger radius (hard-coded 3) as a parameter.
    `wd=3` reproduces the default planner path for path on the episodes checked.
  * The oracle variant swaps the planner inside `OracleAgent`. The adaptive variant swaps it inside `AdaptiveAgent`
    and is never handed the world.
* **Selection on training fires only.** The candidates and the criterion were fixed before any result was seen (see
  the docstring of scratch/oracle_gap/grid.py): the oracle's pooled score over the 8 ranch settings on seeds 0..99.
  First the rank fix, from {unk1, unk2, flat}: unk2 0.908, unk1 0.899, flat 0.896, default 0.886. Then, on top of
  unk2, the 32-cell grid `save_margin` ∈ {0.35, 0.5, 0.7, 1.0} × (`max_wait`, `pace_gap`) ∈ {(6,2), (0,2), (12,2),
  (6,4)} × wait-danger radius ∈ {3, 1}. Winner: `wd=1` with everything else default (0.915 vs 0.908; the top 8 cells
  all have `wd=1` and lie within 0.003 of each other). Each chosen variant was then run **once** on 1000..1099.
* **Order of attribution.** Sequential: G = bcppo − planner; (a) = planner_R − planner (rank fix); (b) = planner_RM −
  planner_R (margin/wait added); residual = bcppo − planner_RM. The two patches interact: on dog / hidden_dependent,
  `wd=1` alone is -0.010 and helps only once the rank is fixed. So the reverse order is reported too.

### Which settings: BC+PPO ≥ the planner, with multiple-comparison correction

bcppo minus the default planners. p = paired bootstrap. Holm F1 = over the 20 tests {oracle, adaptive} × 10 ranch
settings. Holm F2 = over 140 tests: every learned variant in `runs/variants_eval/table.md` (ppo_old, bc, bc_gru,
bc_plain, bcppo_200k, bcppo_400k, bcppo) × {oracle, adaptive} × 10 settings, which is about how many intervals were
looked at before the "beats the oracle" rows were singled out.

| setting | bcppo − oracle | p | Holm F1 | Holm F2 | bcppo − adaptive | p | Holm F1 | Holm F2 |
|---|---|---:|---:|---:|---|---:|---:|---:|
| dog / none | +0.040 [-0.010, +0.100] | 0.21 | 1.00 | 1.00 | (same: planners identical) | | | |
| dog / door_jam | +0.000 [-0.070, +0.070] | 1.00 | 1.00 | 1.00 | +0.030 [-0.030, +0.090] | 0.39 | 1.00 | 1.00 |
| dog / carpet_flashover | +0.050 [+0.000, +0.100] | 0.08 | 1.00 | 1.00 | (same) | | | |
| **dog / hidden_dependent** | **+0.090 [+0.050, +0.135]** | **0.0001** | **0.003** | **0.016** | +0.055 [+0.020, +0.100] | 0.009 | 0.15 | 0.85 |
| kids / none | -0.015 [-0.050, +0.020] | 0.48 | 1.00 | 1.00 | (same) | | | |
| kids / door_jam | -0.045 [-0.095, +0.005] | 0.09 | 1.00 | 1.00 | +0.020 [-0.020, +0.060] | 0.41 | 1.00 | 1.00 |
| kids / carpet_flashover | -0.010 [-0.040, +0.015] | 0.59 | 1.00 | 1.00 | (same) | | | |
| kids / hidden_dependent | +0.040 [+0.000, +0.083] | 0.07 | 1.00 | 1.00 | (same as oracle) | | | |
| dog / compound (untrained) | -0.270 [-0.370, -0.170] | <0.001 | 0.001 | 0.007 | -0.100 [-0.170, -0.040] | 0.003 | 0.06 | 0.33 |
| kids / compound (untrained) | +0.010 [-0.010, +0.030] | 0.44 | 1.00 | 1.00 | +0.005 [-0.010, +0.025] | 0.76 | 1.00 | 1.00 |
| pooled 8 trained | +0.019 [-0.000, +0.038] | 0.057 | | | +0.026 [+0.006, +0.046] | 0.010 | | |

Settings where bcppo ≥ oracle as a point estimate: dog / none, door_jam (tie), carpet_flashover, hidden_dependent;
kids / hidden_dependent, compound. Against the adaptive agent, add kids / door_jam. After correction only dog /
hidden_dependent vs the oracle stands. Only 5 to 21 of the 100 seeds differ in any row, so a single row rests on a
handful of episodes: dog / none +0.04 is 6 seeds won and 2 lost.

### Decomposition (held-out 1000..1099; `R` = rank fix unk2, `M` = wait-danger radius 1, both chosen on 0..99)

**vs the oracle**, sequential order (a) then (b):

| setting | bcppo | oracle | gap G | (a) rank fix | share of G | (b) + wait rule | share | residual bcppo − oracle_RM | share |
|---|---:|---:|---|---|---|---|---|---|---|
| dog / none | 0.930 | 0.890 | +0.040 [-0.010, +0.100] | 0 | 0% | 0 | 0% | +0.040 [-0.010, +0.100] | 100% |
| dog / door_jam | 0.890 | 0.890 | +0.000 [-0.070, +0.070] | 0 | n/a | 0 | n/a | +0.000 [-0.070, +0.070] | n/a |
| dog / carpet_flashover | 0.960 | 0.910 | +0.050 [+0.000, +0.100] | 0 | 0% | 0 | 0% | +0.050 [+0.000, +0.100] | 100% |
| dog / hidden_dependent | 0.980 | 0.890 | +0.090 [+0.050, +0.135] | +0.035 [-0.010, +0.080] | 39% [-15%, 76%] | +0.015 [+0.000, +0.040] | 17% [0%, 50%] | +0.040 [+0.010, +0.075] | 44% [12%, 85%] |
| kids / hidden_dependent | 0.937 | 0.897 | +0.040 [+0.000, +0.083] | +0.040 [+0.007, +0.077] | 100% [5%, 350%] | 0 | 0% | +0.000 [-0.030, +0.030] | 0% |
| kids / none, door_jam, carpet | 0.960 / 0.805 / 0.975 | 0.975 / 0.850 / 0.985 | -0.015 / -0.045 / -0.010 (all CIs cover 0) | 0 | | 0 / -0.005 / 0 | | -0.015 / -0.040 / -0.010 | |
| kids / compound | 0.495 | 0.485 | +0.010 [-0.010, +0.030] | 0 | 0% | +0.005 [+0.000, +0.015] | 50% | +0.005 [-0.010, +0.025] | 50% |
| **pooled 8 trained** | 0.930 | 0.911 | **+0.019 [-0.000, +0.038]** | **+0.009 [+0.003, +0.016]** | **50% [-24%, 244%]** | **+0.001 [-0.004, +0.006]** | **7% [-53%, 55%]** | **+0.008 [-0.009, +0.025]** | **43% [-170%, 126%]** |

Reverse order (wait rule first, then rank fix). Dog / hidden_dependent: `wd=1` alone -0.010 [-0.045, +0.020], then
the rank fix +0.060 [+0.010, +0.110] (67% of G). Pooled: -0.002, then +0.013 (69%). Either way the rank fix carries
nearly all of what the patches explain. The wait rule only helps once the dog is fetched first.

**vs the adaptive agent** (same patches applied to `AdaptiveAgent`):

| setting | G | (a) rank fix | (b) + wait rule | residual bcppo − adaptive_RM |
|---|---|---|---|---|
| dog / hidden_dependent | +0.055 [+0.020, +0.100] | 0 (the adaptive agent hears the visitor only after it has the dog) | +0.015 [+0.000, +0.040] (27%) | +0.040 [+0.010, +0.075] (73%) |
| kids / hidden_dependent | +0.040 [+0.000, +0.083] | +0.040 [+0.007, +0.077] (100%) | 0 | +0.000 [-0.030, +0.030] |
| kids / door_jam | +0.020 [-0.020, +0.060] | 0 | -0.020 [-0.050, +0.015] | +0.040 [-0.005, +0.085] |
| **pooled 8 trained** | **+0.026 [+0.006, +0.046]** | **+0.005 [+0.001, +0.010] (19% [4%, 56%])** | **-0.001 (-2%)** | **+0.022 [+0.004, +0.038] (83% [52%, 112%])** |

Against the adaptive agent most of the pooled edge is left over after the patches. That leftover is the adaptive
agent's *information* gap (it must discover the jam and the visitor; kids / door_jam is 0.785 vs the oracle's 0.850),
plus the dog-handling residual below. The patches fix neither.

**Per household, bcppo − oracle_RM** (pooled over that household's 4 trained settings): dog +0.033 [+0.006, +0.060]
(p = 0.02); kids -0.016 [-0.038, +0.003]. All the residual is with the dog.

**Retrained checkpoint** (`bcppo_s0p100` = `runs/formal_eval/train/bcppo_s0/ckpt_p100.pt`, current reward). Pooled
0.924. vs oracle +0.013 [-0.006, +0.032]; vs oracle_R +0.004; vs oracle_RM +0.002 [-0.015, +0.021]; vs adaptive_RM
+0.016 [-0.003, +0.034]. Dog / hidden_dependent: +0.090 vs the oracle, of which the rank fix is +0.035 and the wait
rule +0.015, leaving +0.040 [+0.005, +0.075], the same as `final.pt`. Kids / hidden_dependent: +0.020 vs the oracle,
-0.020 vs oracle_R (the rank fix more than closes it). Dog / none +0.060 [+0.010, +0.120]; dog / carpet_flashover
now -0.010. These single-setting wins move between checkpoints, which fits chance.

### Traces (c): what the policy does differently

Held-out means over 100 episodes ("lost" = dependents lost, summed over episodes):

| setting | agent | success | died | steps | WAITs | first reached (tick) | lost | pickup order |
|---|---|---:|---:|---:|---:|---|---|---|
| dog / hidden | oracle | 82% | 3% | 73.8 | 28.4 | visitor 9.0, dog 34.9 | dog 17, visitor 4 | visitor→dog 92 |
| | oracle_R | 88% | 2% | 63.1 | 21.5 | dog 11.0, visitor 25.7 | dog 11, visitor 3 | dog→visitor 100 |
| | oracle_RM | 90% | 1% | 63.2 | 21.5 | dog 11.0, visitor 25.7 | dog 10, visitor 2 | dog→visitor 100 |
| | bcppo | 97% | 0% | 69.5 | 25.7 | dog 11.0, visitor 26.1 | dog 3, visitor 0 | dog→visitor 100 |
| kids / hidden | oracle | 81% | 0% (11% timeout) | 116.3 | 55.1 | child 39.9, visitor 52.0 | child 8, visitor 9 | child→visitor→child 76 |
| | oracle_R | 86% | 1% (4%) | 85.0 | 36.5 | child 22.2, visitor 62.2 | child 4, visitor 10 | child→child→visitor 86 |
| | bcppo | 85% | 1% (5%) | 87.4 | 38.2 | child 23.3, visitor 64.0 | child 1, visitor 10 | child→child→visitor 85 |
| dog / none | oracle (= baseline = adaptive) | 89% | 1% | 43.7 | 10.5 | dog 11.0 | dog 11 | |
| | bcppo | 93% | 4% | 47.7 | 12.2 | dog 11.0 | dog 6 | |
| dog / carpet | oracle | 91% | 0% | 45.2 | 11.1 | dog 11.0 | dog 9 | |
| | bcppo | 96% | 1% | 48.2 | 11.9 | dog 11.0 | dog 3 | |

1. **Rescue order (hidden visitor).** With the default `_rank` the oracle goes for the hidden visitor first: tick 9 vs
   the dog's 35 with the dog; with the kids, it skips the second child to fetch the far-away visitor. The visitor
   moves at half pace, so the whole group crawls (oracle on kids / hidden: 116 ticks, 11% timeouts). BC+PPO learned
   the dog first and the children first. Its first-pickup and visitor ticks match the rank-fixed planner to within
   about a tick (dog 11.0 vs 11.0, visitor 26.1 vs 25.7; kids 23.3 vs 22.2, visitor 64.0 vs 62.2). So on the hidden
   visitor it rediscovered the fixed planner's order; it did not find a better one.
2. **Not losing a following dog (the residual).** The dog follows only 70% of the ticks it may move (`follow_prob:dog`
   0.7), and the planner's timing assumes it keeps pace. Every dog-household loss of the planner, and most of the
   residual, is the dog caught by the fire while following. Compare the 20 seeds in dog / none, carpet_flashover and
   hidden_dependent where BC+PPO beats oracle_RM. BC+PPO left by the *other* exit in 16 of them. In 12 it backtracked
   at least 3 cells along the hallway (the planner: 0 of 20). Those episodes ran 1.6 to 2 times longer (e.g. none:
   67 vs 34 ticks). The policy waits for the fire to burn out, or goes around it. Burnt cells cannot catch again, but
   the planner's forecast never expects a burning cell to clear. The cost is risk: in dog / none BC+PPO dies in 4
   episodes, the oracle in 1.

Example episodes (`scratch/oracle_gap/examples.py`):

* **dog / hidden_dependent, seed 1016.** Oracle: t=1 "I heard someone at (2, 10); going to find them". Bonds the
  visitor at t=9, then walks toward the dog at the visitor's half pace, stopping every other tick ("Waiting for Lily
  to catch up (3 steps behind)"). The dog is lost at its hiding place at t=34, 3 steps before the rescuer reaches it.
  Score 0.5. BC+PPO: dog at t=11, visitor at t=27, both out at t=83/87. Score 1.0. Oracle_R behaves like BC+PPO here
  (success, 79 ticks).
* **dog / none, seed 1015** (no novelty, so oracle = baseline = adaptive). Planner: dog at t=11, then "Route clear:
  leading 1 to the exit, 10 steps" through the burning bottom room to the near exit (12, 11). The rescuer is out at
  t=21; the lagging dog dies at (11, 11), one cell from the exit, at t=25. Score 0. BC+PPO: same pickup, then back
  north and east along the hallway to the far exit (6, 23), out at t=41. Score 1.
* **dog / carpet_flashover, seed 1042** (patched oracle_RM). The planner heads east along the hallway toward fire
  coming out of the carpet room, then "No safe route to an exit right now; holding position" at (6, 12) for 6 ticks,
  then pushes through. The dog is lost at (6, 14), t=36. BC+PPO reaches (6, 13), turns back west, returns through the
  now burnt-out bottom room and leaves by (12, 11) at t=57 with the dog.
* **dog / hidden_dependent, seed 1005** (patched oracle_RM, the residual). Both fetch the dog (t=11) then the visitor
  (t=27). The planner leads them east into the fire and the dog is lost at t=50. Score 0.5. BC+PPO waits in the hallway
  around (7, 10) for most of 50 ticks and leaves by the bottom exit at t=108/113. Score 1.0.

### Leak check (e)

`scratch/oracle_gap/leak.py`: the lockstep test of `tests/test_evidence_discipline.py` run on the checkpoints. The
novelty world and its no-novelty control use the same seed and are stepped while the agent's observations and events
are identical; decisions (action and reason) must match. ranch + studio, every household and applicable novelty,
seeds 1000..1004, greedy and sampled (340 pairs), plus ranch hidden_dependent × {dog, kids} on seeds 1005..1099,
greedy (190 pairs). Also a reward-leak probe in the style of `find_exploits.leak_probe`, driven by the policy's own
actions: the largest per-tick reward-component difference while nothing perceivable differs.

| checkpoint | pairs | ticks compared | differing decisions | reward-leak max |
|---|---:|---:|---:|---:|
| runs/bcppo/final.pt | 340 + 190 | 3,411 + 2,090 | **0** | 0.0 |
| bcppo_s0/ckpt_p100.pt | 340 + 190 | 3,337 + 2,090 | **0** | 0.0 |
| oracle (positive control: must fail) | 2 | | caught on door_jam and hidden_dependent | |

The standard part matches `docs/RL.md` (340 episodes, 3,411 ticks, 0 differing). `tests/test_evidence_discipline.py`
and `tests/test_no_leaks.py`: 52 passed. The win is not a leak. It is also not memorised fires: training used seeds
0..999, and every number here is on 1000..1099.

### Does the reward change matter here?

No. `src/emberpath/reward.py` changed on 2026-11-08 at 23:22: terminal potential distance terms are now 0. The
episode **score** compared here is `rescued / total`, or 0 if the rescuer died. It is computed from `world.counts()`
in `evaluate.run_episode` and does not read the reward. The planners never see the reward. `runs/variants_eval/`
(written 21:38 to 21:51) predates the change, and this study's cache (23:43 onward) postdates it. The bcppo, oracle
and adaptive episodes are identical in both, in all 10 settings. `planner.py`, `world.py` and `evaluate.py` last
changed at 17:36, before every cache. The reward change can only affect *training*. That is why the retrained
`ckpt_p100.pt` is reported alongside: same conclusions.

### Proposals (not applied: they would shift every benchmark)

1. **Fix the oracle's `_rank`:** map `"unknown"` to 2, or drop priorities for people not yet found. Measured, held
   out: oracle +0.035 on dog / hidden_dependent and +0.040 on kids / hidden_dependent, +0.009 pooled; no other setting
   changes. It also raises the adaptive agent on kids / hidden_dependent (+0.040). It does not change dog /
   hidden_dependent for the adaptive agent, which hears the visitor only after it has the dog. On training fires 0..99,
   unk2 is the best of the three candidates (+0.022 pooled). Whether "children first" should cover a person not yet
   seen is a policy choice. Under the score metric it costs score, and it makes the oracle a weaker reference than
   the adaptive agent on hidden_dependent (gap closed above 100%). If the priority stays, report the oracle as
   "privileged planner with children-first priorities", not as an upper bound.
2. **Wait for followers closer to the flames** (danger radius 3 to 1). Training fires +0.007 pooled; held out
   +0.001 pooled, +0.015 on dog / hidden_dependent, -0.005 on kids / door_jam. Too small to justify a benchmark shift
   on its own.
3. **Model unreliable followers and fire burn-out** (structural, not tried as a tuned parameter). The residual comes
   from routes the planner's model cannot value: a following dog that lags, and a fire front that clears after
   `burn_steps`. A planner that timed routes with the expected lag (move_period / follow_prob) and could plan to wait
   for burn-out would target exactly the 20 dog episodes above. A crude version (pace divided by follow_prob, tried
   post hoc on training fires only) did **not** help (0.906 vs 0.908), so a real fix needs the burn-out forecast too.
4. **Report the oracle comparison with multiplicity in mind.** One setting survives Holm correction. Pooled, BC+PPO
   vs the oracle is +0.019 [-0.000, +0.038] (p = 0.057) before any fix and +0.008 [-0.009, +0.025] after it.

### Negative results

* `save_margin` ∈ {0.35, 0.7, 1.0}: no gain on training fires over 0.5 at any wait setting (best non-0.5 cell 0.9142
  vs 0.9150).
* `max_wait` 0 (never wait for followers): -0.023 pooled on training fires. `max_wait` 12: no gain. `pace_gap` 4:
  -0.014.
* `rank=flat` (no priorities at all): 0.896 on training fires, worse than unk2 (0.908). The children-first ordering
  among the household is worth keeping. Only the unknown person's priority hurts.
* `rank=unk1` (unknown = adult) does not help with the dog, because an adult still outranks the dog.
* Follower-reliability pace (post hoc, training only): no gain, so not run on held-out fires.
* The patched planners tie BC+PPO but do not beat it anywhere significantly. BC+PPO remains below the oracle on the
  untrained dog / compound (-0.270).

### Limitations

One BC+PPO training seed per reward version (the formal-eval seeds 1 and 2 were not available when this ran). Each
row's interval covers evaluation fires only, not training variance. Shares are ratios of small differences, so their
intervals are wide; read the absolute differences. The patches were chosen on 100 training fires. They were tuned for
the oracle, not for the adaptive agent. The trace analysis of "why" is descriptive (exits, backtracks, ticks) and
does not prove the policy's internal reason.

### Reproduce (from the repo root, ≤ 2 cores)

```bash
export OMP_NUM_THREADS=1
# held-out caches for the default agents and both checkpoints, rank variants on test + train
.venv/bin/python scratch/oracle_gap/run.py --specs bcppo bcppo_s0p100 oracle adaptive oracle:rank=unk1 oracle:rank=unk2 \
    oracle:rank=flat adaptive:rank=unk1 adaptive:rank=unk2 adaptive:rank=flat --settings ranch8 compound \
    --seedsets test train --workers 2
.venv/bin/python scratch/oracle_gap/grid.py --stage rank                # picks unk2 on seeds 0..99
.venv/bin/python scratch/oracle_gap/grid.py --stage grid --workers 2    # 32-cell grid on 0..99, picks wd=1 (~40 min)
.venv/bin/python scratch/oracle_gap/run.py --specs oracle:rank=unk2,fp=1 oracle:rank=unk2,wd=1,fp=1 \
    --settings ranch8 --seedsets train --workers 2                       # post-hoc follower-pace check (negative)
.venv/bin/python scratch/oracle_gap/analyze.py    # runs the remaining held-out jobs; all tables -> runs/oracle_gap/analysis.md
.venv/bin/python scratch/oracle_gap/holm.py       # multiple-comparison table (F1, F2) from runs/variants_eval/
.venv/bin/python scratch/oracle_gap/leak.py --workers 2                 # lockstep + reward-leak probe -> runs/oracle_gap/leak.json
.venv/bin/python scratch/oracle_gap/examples.py ranch/single/hidden_dependent 1016 oracle bcppo oracle:rank=unk2
.venv/bin/python scratch/oracle_gap/examples.py ranch/single/none 1015 oracle bcppo
.venv/bin/python scratch/oracle_gap/examples.py ranch/single/carpet_flashover 1042 oracle:rank=unk2,wd=1 bcppo
.venv/bin/python scratch/oracle_gap/examples.py ranch/single/hidden_dependent 1005 oracle:rank=unk2,wd=1 bcppo
```

Outputs: `runs/oracle_gap/analysis.md` (every decomposition, significance and trace table, both checkpoints),
`runs/oracle_gap/decomposition.json`, `runs/oracle_gap/selection.json` (training-fire scores of every candidate),
`runs/oracle_gap/holm.json`, `runs/oracle_gap/leak.json`, and per-episode traces in `runs/oracle_gap/cache/`.
