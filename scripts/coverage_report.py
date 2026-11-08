"""How diverse are the procedural houses, and does oracle success really fall with tier?

    python scripts/coverage_report.py              # 20 houses per tier, 30 fire seeds each (~1-2 min, all cores)
    python scripts/coverage_report.py --houses 5   # quicker

Tiers are assigned from cheap measurements (no agent is simulated); the oracle-success table is the
independent check that those measurements mean what they claim. Exits non-zero if success is not
monotone non-increasing in tier (within --tol).
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from emberpath.blueprint import Blueprint
from emberpath.grid import neighbors4
from emberpath.procgen import TIERS, generate, measure, oracle_success

METRICS = ("route_len", "doors", "exits", "fire_exit", "margin", "p_lost")


def topology(bp: Blueprint) -> tuple:
    """Rooms are floor regions split by doors; the signature is the room graph's shape (degrees, exits, links)."""
    room: dict = {}
    n_rooms = 0
    for cell in bp.materials:
        if cell in room or cell in bp.doors or cell in bp.exits:
            continue
        for c in bp.reachable(cell, frozenset(bp.doors) | frozenset(bp.exits)):
            room[c] = n_rooms
        n_rooms += 1
    links = Counter()
    for d in bp.doors + bp.exits:
        near = {room[n] for n in neighbors4(d, bp.height, bp.width) if n in room}
        for r in near:
            links[(r, "E" if d in bp.exits else "D")] += 1
    per_room = sorted((links[(r, "D")], links[(r, "E")]) for r in range(n_rooms))
    return (len(per_room), tuple(per_room))


def job(args: tuple[int, int, int]) -> dict:
    tier, seed, seeds = args
    t = time.perf_counter()
    bp = generate(seed, tier)
    gen_ms = 1000 * (time.perf_counter() - t)
    return {"tier": tier, "gen_ms": gen_ms, "topo": topology(bp), **measure(bp), "oracle": oracle_success(bp, seeds)}


def hist(values: list[float], bins: int = 8) -> str:
    counts, edges = np.histogram(values, bins=bins)
    bar = "".join(" .:-=+*#%@"[min(9, math.ceil(9 * c / max(counts)))] for c in counts)
    return f"[{edges[0]:>6.1f} {bar} {edges[-1]:<6.1f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", type=int, default=20, help="houses per tier")
    ap.add_argument("--seeds", type=int, default=30, help="fire seeds per house for oracle success")
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()

    jobs = [(t, s, a.seeds) for t in TIERS for s in range(a.houses)]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        rows = list(ex.map(job, jobs))

    print(f"{len(rows)} houses ({a.houses} per tier), oracle over {a.seeds} fire seeds each, kids household\n")
    print("metric histograms (8 bins, min .. max):")
    for m in METRICS:
        print(f"  {m:<10}{hist([r[m] for r in rows])}")

    x = np.array([[r[m] for m in METRICS] for r in rows], float)
    z = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1)
    d = [np.abs(z[i] - z[j]).mean() for i, j in itertools.combinations(range(len(z)), 2)]
    topos = Counter(r["topo"] for r in rows)
    print(f"\npairwise distance (mean |z| over metrics): mean {np.mean(d):.2f}  p5 {np.percentile(d, 5):.2f}  "
          f"min {min(d):.2f}")
    print(f"distinct topologies: {len(topos)} / {len(rows)}  (most common appears {topos.most_common(1)[0][1]}x)")
    print(f"generation: {np.mean([r['gen_ms'] for r in rows]):.1f} ms per house (mean, tier-targeted)\n")

    print(f"{'tier':<6}{'houses':>7}{'oracle':>8}{'  95% CI':<14}{'route':>7}{'doors':>7}{'fire->exit':>11}"
          f"{'margin':>8}{'p_lost':>8}")
    means = []
    for t in TIERS:
        rs = [r for r in rows if r["tier"] == t]
        s = np.array([r["oracle"] for r in rs])
        half = 1.96 * s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")
        means.append(s.mean())
        avg = {m: np.mean([r[m] for r in rs]) for m in METRICS}
        print(f"{t:<6}{len(rs):>7}{s.mean():>8.2f}  ±{half:<11.2f}{avg['route_len']:>7.1f}{avg['doors']:>7.1f}"
              f"{avg['fire_exit']:>11.1f}{avg['margin']:>8.1f}{avg['p_lost']:>8.2f}")
    monotone = all(b <= a_ + a.tol for a_, b in zip(means, means[1:]))
    print(f"\noracle success monotone non-increasing in tier (tol {a.tol}): {'yes' if monotone else 'NO'}")
    return 0 if monotone else 1


if __name__ == "__main__":
    raise SystemExit(main())
