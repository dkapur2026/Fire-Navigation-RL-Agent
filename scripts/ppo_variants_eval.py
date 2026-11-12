"""Held-out comparison of PPO variants (docs/RL.md, "Improvements tried") with the planners and the original PPO.

    python scripts/ppo_variants_eval.py --ckpt bc=runs/bc_adaptive/bc.pt bcppo=runs/bcppo/final.pt --workers 4

Same protocol as docs/RL.md "Independent verification": ranch dog / kids x none, door_jam, carpet_flashover,
hidden_dependent, compound, plus studio / dog / none; seeds 1000..1099; the shared harness (`evaluate.run_episode`);
95% paired-bootstrap intervals over seeds (`evaluate.paired_bootstrap`, 2,000 resamples). Episode results are cached
per (agent, setting) in runs/variants_eval/ so the planners run once. Writes runs/variants_eval/table.md.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "variants_eval"
SEEDS = list(range(1000, 1100))
SETTINGS = [("ranch", p, n) for p in ("single", "kids")
            for n in ("none", "door_jam", "carpet_flashover", "hidden_dependent", "compound")] + [("studio", "single", "none")]
PLANNERS = ("baseline", "oracle", "adaptive")


def _key(label: str, setting) -> Path:
    return CACHE / f"{label}__{'_'.join(setting)}.json"


def _job(args):
    label, ckpt, setting, greedy = args
    import torch

    torch.set_num_threads(1)
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.evaluate import run_episode

    bp_name, prof, nov = setting
    bp, hh = load_blueprint(bp_name), HOUSEHOLDS[prof]
    novelty = None if nov == "none" else nov
    if ckpt is None:
        res = [run_episode(bp, hh, label, novelty, s) for s in SEEDS]
    else:
        from emberpath.rl_baseline import evaluate_policy, load_policy

        net, _ = load_policy(ckpt)
        res = evaluate_policy(net, bp, hh, novelty, SEEDS, greedy=greedy)
    rows = [{**asdict(r), "agent": label} for r in res]
    _key(label, setting).write_text(json.dumps(rows))
    return label, setting


def load(label, setting):
    from emberpath.evaluate import EpisodeResult

    rows = json.loads(_key(label, setting).read_text())
    return [EpisodeResult(**{**r, "patches": tuple(tuple(p) for p in r["patches"])}) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", nargs="*", default=[], help="label=path (greedy); label_sampled=path for sampled")
    ap.add_argument("--refs", nargs="+", default=["ppo_old", "baseline", "oracle", "adaptive"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-run cached results")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    agents = {p: None for p in PLANNERS}
    agents["ppo_old"] = str(ROOT / "runs/ppo_novelty/final.pt")
    for spec in a.ckpt:
        label, path = spec.split("=", 1)
        agents[label] = path
    jobs = [(lab, ck, st, not lab.endswith("_sampled")) for lab, ck in agents.items() for st in SETTINGS
            if a.force and ck is not None and lab in dict(s.split("=", 1) for s in a.ckpt) or not _key(lab, st).exists()]
    if jobs:
        with ProcessPoolExecutor(a.workers) as ex:
            for lab, st in ex.map(_job, jobs):
                print("done", lab, "/".join(st), flush=True)

    from emberpath.evaluate import paired_bootstrap

    learned = [lab for lab in agents if lab not in PLANNERS]
    lines = ["| setting | " + " | ".join(agents) + " |", "|---|" + "---|" * len(agents)]
    for st in SETTINGS:
        cells = []
        for lab in agents:
            r = load(lab, st)
            cells.append(f"{np.mean([x.outcome == 'success' for x in r]):.0%} / {np.mean([x.score for x in r]):.2f}")
        lines.append(f"| {'/'.join(st)} | " + " | ".join(cells) + " |")
    lines += ["", "Score differences (variant - reference), 95% paired bootstrap over the 100 seeds:", ""]
    for lab in learned:
        refs = [r for r in a.refs if r != lab]
        lines += [f"**{lab}**", "", "| setting | " + " | ".join(f"- {r}" for r in refs) + " | success - adaptive |",
                  "|---|" + "---|" * (len(refs) + 1)]
        for st in SETTINGS:
            cells = []
            for ref in refs:
                res = load(lab, st) + load(ref, st)
                pt, lo, hi = paired_bootstrap(res, lambda m, x=lab, y=ref: m[x] - m[y])
                cells.append(f"{pt:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            res = load(lab, st) + load("adaptive", st)
            pt, lo, hi = paired_bootstrap(res, lambda m, x=lab: m[x] - m["adaptive"],
                                          value=lambda r: float(r.outcome == "success"))
            cells.append(f"{pt:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            lines.append(f"| {'/'.join(st)} | " + " | ".join(cells) + " |")
        lines.append("")
    out = CACHE / "table.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
