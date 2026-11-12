"""Regenerate docs/results.md: baseline vs oracle vs adaptive, every household x every novelty.

    python scripts/make_results.py                    # ranch, 60 seeds, all cores -> docs/results.md
    python scripts/make_results.py --blueprint house  # another house -> docs/results_house.md
    python scripts/make_results.py --seeds 20         # quicker

Run this after ANY change to world dynamics, the planner, or default rules: those numbers are the
project's calibration. The 'oracle gap' column is what the adaptive agent is trying to close; with no
novelty it must be +0.00. The fuller study (ablations, delays, recovery curves) is scripts/adaptive_study.py.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from emberpath.blueprint import Blueprint
from emberpath.dependents import HOUSEHOLDS
from emberpath.evaluate import evaluate, gap_ratio, paired_bootstrap, step_ratio
from emberpath.scenarios import applicable

ROOT = Path(__file__).resolve().parents[1]
NOVELTIES = [None, "door_jam", "carpet_flashover", "compound", "hidden_dependent"]
PROFILES = list(HOUSEHOLDS)
AGENTS = ["baseline", "oracle", "adaptive"]
MIN_GAP = 0.05  # below this oracle gap, gap_closed is noise over a near-zero denominator


def job(args):
    blueprint, profile, novelty, seeds = args
    bp = Blueprint.from_file(ROOT / "blueprints" / f"{blueprint}.txt")
    if not applicable(bp, HOUSEHOLDS[profile], novelty):
        return profile, novelty, None
    return profile, novelty, evaluate(bp, HOUSEHOLDS[profile], AGENTS, novelty, range(seeds))


def ci(point: float, lo: float, hi: float, pct: bool = False) -> str:
    f = (lambda x: f"{x:.0%}") if pct else (lambda x: f"{x:.2f}")
    return f"{f(point)} [{f(lo)}, {f(hi)}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--blueprint", default="ranch")
    a = ap.parse_args()
    jobs = [(a.blueprint, p, n, a.seeds) for p in PROFILES for n in NOVELTIES]
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        out = list(pool.map(job, jobs))

    lines = [
        f"# Results: {a.blueprint} blueprint, {a.seeds} seeds per cell",
        "",
        "Mean score (fraction of dependents rescued, 0 if the rescuer dies). Same seeds and same fires for every agent.",
        "`oracle gap` = oracle - baseline: how much a correct rulebook is worth. With no novelty it must be +0.00.",
        "`gap closed` = (adaptive - baseline) / (oracle - baseline); `n/a` when the oracle gap is under "
        f"{MIN_GAP:.2f}. Brackets: 95% paired-bootstrap intervals over seeds. Regenerate with `make results`;",
        "ablations, detection delays and recovery curves: `python scripts/adaptive_study.py` (docs/adaptive_study.md).",
        "`efficiency (o / a)` = fewest possible steps (ignoring fire and child pace) / steps taken, mean over each "
        "agent's own successes.",
        "`steps vs oracle` = adaptive steps / oracle steps over the n seeds where BOTH got everyone out (1.00 = as "
        "fast as the oracle,",
        "higher = slower). The oracle is the same planner given the true rules, not a proven shortest rescue.",
        "",
        "| household | novelty | baseline | oracle | oracle gap | adaptive | adaptive - baseline | gap closed | all out (b / o / a) "
        "| efficiency (o / a) | steps vs oracle |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for profile, novelty, res in out:
        if res is None:
            lines.append(f"| {profile} | {novelty or 'none'} | n/a: the house cannot host this (too few slots) |||||||||")
            continue
        ok = {ag: sum(r.outcome == "success" for r in res if r.agent == ag) / a.seeds for ag in AGENTS}
        mean = {ag: paired_bootstrap(res, lambda m, ag=ag: m[ag]) for ag in AGENTS}
        gap = mean["oracle"][0] - mean["baseline"][0]
        lift = paired_bootstrap(res, lambda m: m["adaptive"] - m["baseline"])
        closed = ci(*paired_bootstrap(res, gap_ratio()), pct=True) if gap >= MIN_GAP else "n/a"
        eff = {ag: [r.efficiency for r in res if r.agent == ag and r.efficiency is not None]
               for ag in ("oracle", "adaptive")}
        eff = {ag: f"{sum(v) / len(v):.2f}" if v else "n/a" for ag, v in eff.items()}
        sr, slo, shi, sn = step_ratio(res)
        steps = f"{sr:.2f} [{slo:.2f}, {shi:.2f}] (n={sn})" if sn else "n/a"
        lines.append(
            f"| {profile} | {novelty or 'none'} | {mean['baseline'][0]:.2f} | {mean['oracle'][0]:.2f} | {gap:+.2f} "
            f"| {ci(*mean['adaptive'])} | {lift[0]:+.2f} [{lift[1]:+.2f}, {lift[2]:+.2f}] | {closed} "
            f"| {ok['baseline']:.0%} / {ok['oracle']:.0%} / {ok['adaptive']:.0%} "
            f"| {eff['oracle']} / {eff['adaptive']} | {steps} |"
        )

    path = ROOT / "docs" / ("results.md" if a.blueprint == "ranch" else f"results_{a.blueprint}.md")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
