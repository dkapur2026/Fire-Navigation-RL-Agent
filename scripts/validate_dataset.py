"""Validate a dataset built by scripts/build_dataset.py; exit 1 if any check fails. See docs/DATA.md.

    python scripts/validate_dataset.py /tmp/ds [--no-replay] [--report stats.md]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from emberpath.dataset import read_dataset, stats, validate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--no-replay", action="store_true", help="skip the (slower) replay-determinism check")
    ap.add_argument("--report", help="also write the statistics report (markdown) here")
    a = ap.parse_args()
    t0 = time.perf_counter()
    manifest, splits = read_dataset(a.root)
    issues = validate(manifest, splits, replay=not a.no_replay)
    report = stats(splits, issues)
    print(report)
    if a.report:
        Path(a.report).write_text(report)
    n = sum(map(len, splits.values()))
    print(f"validated {n} records in {time.perf_counter() - t0:.1f}s; skipped jobs {len(manifest['skipped_jobs'])}, "
          f"dropped duplicates {len(manifest['dropped_duplicates'])} (see manifest.json)")
    for i in issues[:50]:
        print(f"FAIL [{i.check}] {i.record}: {i.detail}")
    print("issues by check:", dict(Counter(i.check for i in issues)) or "none")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
