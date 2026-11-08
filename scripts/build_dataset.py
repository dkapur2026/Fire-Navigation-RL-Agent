"""Build the trajectory dataset (WP-K): episodes from several agents across houses, households, novelties and seeds.

    python scripts/build_dataset.py --out /tmp/ds --episodes 200 [--agents baseline oracle adaptive random] [--workers 4] [--gzip]

Same arguments -> byte-identical files. Splits are by (blueprint, seed) group. See docs/DATA.md.
"""

from __future__ import annotations

import argparse
import time

from emberpath.dataset import DEFAULT_AGENTS, DEFAULT_BLUEPRINTS, build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--agents", nargs="+", default=list(DEFAULT_AGENTS), help="any of agents.AGENTS, plus 'random'")
    ap.add_argument("--blueprints", nargs="+", default=list(DEFAULT_BLUEPRINTS),
                    help="ranch, studio, a path, gen:<tier> (fresh house per pass) or gen:<tier>:<seed>")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gzip", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    m = build(a.out, a.episodes, tuple(a.agents), tuple(a.blueprints), a.max_steps, a.workers, a.gzip)
    dt = time.perf_counter() - t0
    print(f"{m['records']} records {m['by_split']} from {m['jobs']} jobs in {dt:.1f}s ({m['records'] / dt:.1f} records/s)")
    print(f"skipped {len(m['skipped_jobs'])} jobs (preset not applicable), dropped {len(m['dropped_duplicates'])} duplicate "
          f"trajectories; both listed in {a.out}/manifest.json")


if __name__ == "__main__":
    main()
