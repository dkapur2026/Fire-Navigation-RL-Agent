"""How much do the reward's conclusions depend on its weights? Three studies, all seeded.

    python scripts/reward_sweep.py                 # all three, a few minutes
    python scripts/reward_sweep.py --part sweep    # or: ablation, reliability

sweep        Scale each RewardConfig weight (x0 .. x4; the proximity radius 1 .. 8 steps) one at a time and rank the
             audit policies by mean return (ranch / single, every novelty preset). Reports Kendall's tau against the
             default ranking and where a check breaks: "honest first" (oracle above stall, suicide and abandon) and
             "dying never pays" (suicide below both stall and the struggling baseline). Each episode is played once
             and scored by every setting at the same time, since the policies do not see the reward.
ablation     Sparse (rescue and success only) versus the full reward, as the training signal for a small evolution
             strategy over the scripted policy family of find_exploits.py, on the studio. Measures how fast the true
             success rate improves and how often a generation gets no signal at all (every candidate tied).
reliability  Spread of a fixed good policy's (oracle's) return across seeds, and how many seeds it takes to tell two
             policies apart.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_reward import POLICIES, rollout  # noqa: E402
from find_exploits import HIGH, LOW, PARAMS, blueprint, evaluate  # noqa: E402

from emberpath.dependents import HOUSEHOLDS  # noqa: E402
from emberpath.reward import RewardConfig  # noqa: E402
from emberpath.scenarios import NOVELTIES  # noqa: E402

DEFAULT = RewardConfig()
SCALES = (0.0, 0.25, 0.5, 2.0, 4.0)
RADII = (1, 2, 3, 6, 8)
SETTINGS = [("default", None, DEFAULT)] + [
    (f.name, v, dataclasses.replace(DEFAULT, **{f.name: v}))
    for f in dataclasses.fields(RewardConfig)
    for v in (RADII if f.name == "proximity_radius" else [getattr(DEFAULT, f.name) * s for s in SCALES])
]
SWEPT = [p for p in POLICIES if p != "reward_snoop"]  # identical to baseline under this reward (tests/test_exploits.py)
DEGENERATE = ("stall", "suicide", "abandon")
SPARSE = RewardConfig(step_cost=0.0, progress_weight=0.0, proximity_weight=0.0, contact_penalty=0.0, death_extra=0.0)
Z = 1.96 + 0.84  # two-sided 5% test with 80% power


def kendall_tau(a, b) -> float:
    """Kendall's tau-b between two score vectors over the same items."""
    pairs = list(itertools.combinations(range(len(a)), 2))
    s = sum(np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]) for i, j in pairs)
    ties_a = sum(a[i] == a[j] for i, j in pairs)
    ties_b = sum(b[i] == b[j] for i, j in pairs)
    return float(s / math.sqrt((len(pairs) - ties_a) * (len(pairs) - ties_b)))


def checks(mean: dict[str, float]) -> tuple[bool, bool]:
    honest_first = all(mean["oracle"] > mean[d] for d in DEGENERATE)
    dying_never_pays = mean["suicide"] < min(mean["stall"], mean["baseline"])
    return honest_first, dying_never_pays


def returns_job(job) -> list[float]:
    policy, novelty, seed = job
    _, sums, _ = rollout(POLICIES[policy], blueprint("ranch"), HOUSEHOLDS["single"], novelty, seed, [c for *_, c in SETTINGS])
    return [sum(s.values()) for s in sums]


def play_all(pool, seeds: int) -> dict:
    """{novelty: {policy: array[setting, seed]}} of episode returns."""
    jobs = [(p, n, s) for n in NOVELTIES for p in SWEPT for s in range(seeds)]
    out = iter(pool.map(returns_job, jobs, chunksize=4))
    return {n: {p: np.array([next(out) for _ in range(seeds)]).T for p in SWEPT} for n in NOVELTIES}


def sweep(data: dict, seeds: int) -> None:
    print(f"## Weight sweep: ranch / single, {len(SWEPT)} audit policies x {seeds} seeds per novelty, one weight at a time")
    base = {n: [data[n][p][0].mean() for p in SWEPT] for n in NOVELTIES}
    print("Default ranking: " + "; ".join(
        f"{n}: " + " > ".join(sorted(SWEPT, key=lambda p: -data[n][p][0].mean())) for n in NOVELTIES))
    print("\n| weight | value | min tau | mean tau | honest first | dying never pays | broken in |")
    print("|---|---|---|---|---|---|---|")
    breaks: dict[str, list] = {}
    for i, (name, value, _) in enumerate(SETTINGS):
        taus, hf, dn, broken = [], 0, 0, []
        for n in NOVELTIES:
            mean = {p: data[n][p][i].mean() for p in SWEPT}
            taus.append(kendall_tau([mean[p] for p in SWEPT], base[n]))
            ok_h, ok_d = checks(mean)
            hf, dn = hf + ok_h, dn + ok_d
            tags = [t for t, ok in (("honest", ok_h), ("dying", ok_d)) if not ok]
            if tags:
                broken.append(f"{n} ({'+'.join(tags)})")
        if broken:
            breaks.setdefault(name, []).append(value)
        k = len(NOVELTIES)
        shown = "-" if value is None else f"{value:g}"
        print(f"| {name} | {shown} | {min(taus):.2f} | {np.mean(taus):.2f} | {hf}/{k} | {dn}/{k} | {', '.join(broken) or '-'} |")
    print("\nWhere a check breaks: " + ("; ".join(f"{w} at {', '.join('its default' if v is None else f'{v:g}' for v in vs)}" for w, vs in breaks.items()) or "nowhere"))


def reliability(data: dict, seeds: int) -> None:
    print(f"\n## Signal reliability: default reward, ranch / single, {seeds} seeds")
    print("n = seeds per policy to detect a return difference of 0.5 (or 0.25) at 5% significance with 80% power.")
    print("Unpaired uses oracle's own spread; paired uses oracle - baseline on the same seeds (identical fire).\n")
    print("| novelty | oracle mean | oracle sd | median | n unpaired (0.5 / 0.25) | paired sd | n paired (0.5 / 0.25) |")
    print("|---|---|---|---|---|---|---|")
    for n in NOVELTIES:
        o, b = data[n]["oracle"][0], data[n]["baseline"][0]
        sd, sd_d = o.std(ddof=1), (o - b).std(ddof=1)
        nu = [math.ceil(2 * (Z * sd / d) ** 2) for d in (0.5, 0.25)]
        npd = " / ".join(str(math.ceil((Z * sd_d / d) ** 2)) for d in (0.5, 0.25)) if sd_d else "identical episodes"
        print(f"| {n} | {o.mean():+.2f} | {sd:.2f} | {np.median(o):+.2f} | {nu[0]} / {nu[1]} | {sd_d:.2f} | {npd} |")


# ------------------------------------------------------------------ sparse vs shaped
ABL_SCENARIO = ("studio", "married", None)


def learn(job) -> dict:
    """OpenAI-style ES with antithetic pairs and rank-normalized fitness (ties share a rank, so a generation where
    every candidate scores the same gets no update). True success is tracked on fixed held-out seeds."""
    cfg, learner_seed, gens, pairs, k = job
    rng = np.random.default_rng([learner_seed, 11])
    span, theta = HIGH - LOW, rng.uniform(LOW, HIGH)
    held_out = range(5000, 5010)
    curve, flat = [evaluate(None, [theta], ABL_SCENARIO, held_out, DEFAULT)[0]["success"]], 0
    for g in range(gens):
        eps = rng.standard_normal((pairs, len(PARAMS)))
        cands = [np.clip(theta + sign * 0.1 * span * e, LOW, HIGH) for e in eps for sign in (1, -1)]
        train = range(1000 * learner_seed + g * k, 1000 * learner_seed + (g + 1) * k)
        fit = np.array([r["return"] for r in evaluate(None, cands, ABL_SCENARIO, train, cfg)])
        ranks = np.array([(fit < f).sum() + ((fit == f).sum() - 1) / 2 for f in fit]) / max(1, len(fit) - 1) - 0.5
        flat += bool(np.all(fit == fit[0]))
        grad = sum((ranks[2 * i] - ranks[2 * i + 1]) * eps[i] for i in range(pairs)) / pairs
        theta = np.clip(theta + span * grad, LOW, HIGH)
        curve.append(evaluate(None, [theta], ABL_SCENARIO, held_out, DEFAULT)[0]["success"])
    final = evaluate(None, [theta], ABL_SCENARIO, held_out, cfg)[0]
    return {"curve": curve, "flat": flat, "final_return": final["return"]}


def ablation(pool, learners: int, gens: int) -> None:
    pairs, k = 4, 3
    print(f"\n## Sparse vs shaped: ES on {'/'.join(str(x) for x in ABL_SCENARIO[:2])}, {learners} learner seeds x {gens}"
          f" generations x {2 * pairs} candidates x {k} episodes; success on 10 held-out seeds")
    print("| reward | success at gen 0 / 5 / 10 / end | gens to reach 0.3 | final success sd | flat generations |")
    print("|---|---|---|---|---|")
    for name, cfg in (("full", DEFAULT), ("sparse", SPARSE)):
        runs = list(pool.map(learn, [(cfg, s, gens, pairs, k) for s in range(learners)]))
        curves = np.array([r["curve"] for r in runs])
        mean = curves.mean(0)
        reach = [next((g for g, v in enumerate(c) if v >= 0.3), None) for c in curves]
        hit = [r for r in reach if r is not None]
        reach_txt = f"{np.mean(hit):.1f} ({len(hit)}/{learners} learners)" if hit else f"never (0/{learners})"
        pts = " / ".join(f"{mean[g]:.2f}" for g in (0, 5, 10, gens))
        flat = sum(r["flat"] for r in runs)
        print(f"| {name} | {pts} | {reach_txt} | {curves[:, -1].std():.2f} | {flat}/{learners * gens} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("all", "sweep", "ablation", "reliability"), default="all")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--learners", type=int, default=6)
    ap.add_argument("--gens", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        if a.part in ("all", "sweep", "reliability"):
            data = play_all(pool, a.seeds)
            if a.part != "reliability":
                sweep(data, a.seeds)
            if a.part != "sweep":
                reliability(data, a.seeds)
        if a.part in ("all", "ablation"):
            ablation(pool, a.learners, a.gens)
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
