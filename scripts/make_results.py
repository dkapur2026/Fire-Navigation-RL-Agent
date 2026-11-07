"""Regenerate docs/results.md: baseline vs oracle (and adaptive, once it exists) on the ranch.

    python scripts/make_results.py            # 60 seeds, all cores
    python scripts/make_results.py --seeds 20 # quicker

Run this after ANY change to world dynamics, the planner, or default rules: those numbers are the
project's calibration, and the 'oracle gap' column is what the adaptive agent is trying to close.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from emberpath.blueprint import Blueprint
from emberpath.dependents import HOUSEHOLDS
from emberpath.evaluate import evaluate, gap_closed, summarize

ROOT = Path(__file__).resolve().parents[1]
NOVELTIES = [None, "door_jam", "carpet_flashover", "compound"]
PROFILES = ["single", "married", "kids"]


def job(args):
    profile, novelty, seeds, agents = args
    bp = Blueprint.from_file(ROOT / "blueprints" / "ranch.txt")
    try:
        res = evaluate(bp, HOUSEHOLDS[profile], agents, novelty, range(seeds))
    except NotImplementedError:
        res = evaluate(bp, HOUSEHOLDS[profile], [a for a in agents if a != "adaptive"], novelty, range(seeds))
    return profile, novelty, summarize(res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60)
    a = ap.parse_args()
    agents = ["baseline", "oracle", "adaptive"]
    jobs = [(p, n, a.seeds, agents) for p in PROFILES for n in NOVELTIES]
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        out = list(pool.map(job, jobs))

    have_adaptive = any("adaptive" in s for _, _, s in out)
    head = "| household | novelty | baseline | oracle |" + (" adaptive | gap closed |" if have_adaptive else " oracle gap |")
    sep = "|---|---|---|---|" + ("---|---|" if have_adaptive else "---|")
    lines = [
        f"# Results: ranch blueprint, {a.seeds} seeds per cell",
        "",
        "Mean score (fraction of dependents rescued, 0 if the rescuer dies). Same seeds and same fires for every agent.",
        "`oracle gap` = oracle - baseline: how much a correct rulebook is worth. With no novelty it must be 0.",
        "",
        head,
        sep,
    ]
    for profile, novelty, s in out:
        b, o = s["baseline"]["mean_score"], s["oracle"]["mean_score"]
        row = f"| {profile} | {novelty or 'none'} | {b:.2f} | {o:.2f} |"
        if have_adaptive:
            g = gap_closed(s)
            row += f" {s['adaptive']['mean_score']:.2f} | {'n/a' if g is None else f'{g:.0%}'} |"
        else:
            row += f" {o - b:+.2f} |"
        lines.append(row)

    path = ROOT / "docs" / "results.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
