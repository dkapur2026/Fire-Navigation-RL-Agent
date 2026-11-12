"""Formal, pre-registered evaluation of BC+PPO ("Imitation + PPO") and DreamerV3 + imitation.

The protocol (training seeds, evaluation sets, metrics, the comparison family, the bootstrap and the Holm correction)
is fixed in runs/formal_eval/PREREGISTRATION.md and read from runs/formal_eval/protocol.json; this script only
executes it. Subcommands:

    python scripts/formal_eval.py manifest --method bcppo --seed 0     # checkpoint manifest of one training run
    python scripts/formal_eval.py eval --workers 4                     # episodes (cached per job)
    python scripts/formal_eval.py lockstep --workers 4                 # evidence lockstep on every evaluated ckpt
    python scripts/formal_eval.py stats                                # results.json + results.md
    python scripts/formal_eval.py all --workers 4                      # eval + lockstep + stats

Episodes go through the shared harness (`evaluate.run_episode`, via `rl_baseline.evaluate_policy` /
`dreamer.evaluate_dreamer`, which refuse training seeds). Results are cached in runs/formal_eval/cache/ keyed by the
job and the checkpoint's sha256, so a rerun only plays what changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "formal_eval"
TRAIN = OUT / "train"
CACHE = OUT / "cache"
PROTOCOL = OUT / "protocol.json"
TRAINING_FIRE_SEEDS = range(0, 1000)
PLANNERS = ("baseline", "oracle", "adaptive")


# --------------------------------------------------------------------------- statistics (pure, tested)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (same order as the input), capped at 1 and made monotone."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def boot_p(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value for 'difference = 0': twice the smaller tail mass at 0 (ties count in both
    tails, so a difference that is exactly 0 in every resample gets p = 1), floored at 1 / (B + 1)."""
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return math.nan
    lo, hi = np.mean(d <= 0), np.mean(d >= 0)
    return float(min(1.0, max(2 * min(lo, hi), 1.0 / (len(d) + 1))))


def _resampled_means(x: np.ndarray, s_idx, u_idx, f_idx) -> np.ndarray:
    """Means of x[s, u, f] over resampled index sets; one value per resample. x: (S, U, F)."""
    B = len(f_idx)
    out = np.empty(B)
    for b in range(B):
        out[b] = x[np.ix_(s_idx[b], u_idx[b], f_idx[b])].mean()
    return out


def paired_diff_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0,
                          resample_seeds: bool = True, resample_units: bool = True,
                          pair_seeds: bool = False) -> dict:
    """Difference of means A - B with a paired bootstrap. ``a``/``b``: (S, U, F) per-episode-cell values (training
    seed x unit x fire), already averaged over the settings within a unit. Fires (and units, if ``resample_units``)
    are resampled with the SAME indices for A and B (paired: the same fires and houses). With ``resample_seeds`` the
    training seeds of each method are resampled too (two-level / hierarchical bootstrap; a method with one "seed",
    e.g. a planner, contributes no seed variance); ``pair_seeds`` uses the same seed indices for A and B (e.g. a
    fine-tuned policy and its own clone). Returns point, 95% percentile interval, two-sided p, bootstrap SE."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(f"A and B need the same units and fires, got {a.shape} and {b.shape}")
    rng = np.random.default_rng(seed)
    U, F = a.shape[1], a.shape[2]
    f_idx = rng.integers(0, F, (n_boot, F))
    u_idx = rng.integers(0, U, (n_boot, U)) if resample_units and U > 1 else np.tile(np.arange(U), (n_boot, 1))

    def seed_idx(S):
        if resample_seeds and S > 1:
            return rng.integers(0, S, (n_boot, S))
        return np.tile(np.arange(S), (n_boot, 1))

    sa = seed_idx(a.shape[0])
    if pair_seeds:
        if a.shape[0] != b.shape[0]:
            raise ValueError("paired seeds need the same number of training seeds")
        sb = sa
    else:
        sb = seed_idx(b.shape[0])
    d = _resampled_means(a, sa, u_idx, f_idx) - _resampled_means(b, sb, u_idx, f_idx)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(a.mean() - b.mean()), "lo": float(lo), "hi": float(hi), "p": boot_p(d),
            "se": float(d.std(ddof=1)), "n_boot": n_boot}


def seed_split(x: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Training-seed vs fire variance for one method's (S, U, F) values: per-seed means, the between-seed SD, the
    fire-only bootstrap SE of the seed-averaged mean (seeds fixed) and the two-level SE (seeds, then fires)."""
    x = np.asarray(x, float)
    per_seed = x.mean(axis=(1, 2))
    rng = np.random.default_rng(seed)
    S, U, F = x.shape
    f_idx = rng.integers(0, F, (n_boot, F))
    u_idx = rng.integers(0, U, (n_boot, U)) if U > 1 else np.zeros((n_boot, 1), int)
    fixed = np.tile(np.arange(S), (n_boot, 1))
    fire_only = _resampled_means(x, fixed, u_idx, f_idx)
    s_idx = rng.integers(0, S, (n_boot, S)) if S > 1 else fixed
    hier = _resampled_means(x, s_idx, u_idx, f_idx)
    return {"per_seed": per_seed.tolist(), "mean": float(x.mean()),
            "between_seed_sd": float(per_seed.std(ddof=1)) if S > 1 else None,
            "fire_se": float(fire_only.std(ddof=1)), "hier_se": float(hier.std(ddof=1))}


# --------------------------------------------------------------------------- protocol helpers


def load_protocol(path: Path = PROTOCOL) -> dict:
    return json.loads(Path(path).read_text())


def fire_seeds(spec) -> list[int]:
    return list(range(spec[0], spec[1]))


def held_out_houses(tiers: Sequence[int], per_tier: int, start: int, canvas: tuple[int, int]) -> list[str]:
    """The pre-registered house rule: for each tier, the first ``per_tier`` generator seeds >= ``start`` whose
    procedural house fits the policies' canvas (pad_to refuses larger houses)."""
    from emberpath.procgen import generate

    out = []
    for t in tiers:
        s, found = start, 0
        while found < per_tier:
            try:
                bp = generate(s, t)
            except RuntimeError:
                s += 1
                continue
            if bp.height <= canvas[0] and bp.width <= canvas[1]:
                out.append(f"gen:{t}:{s}")
                found += 1
            s += 1
    return out


def training_blueprints(proto: dict) -> set[str]:
    """Every house any evaluated checkpoint was trained on (from the protocol's record of the recipes)."""
    return set(proto["training_blueprints"])


def set_settings(proto: dict, name: str) -> list[tuple[str, str, str]]:
    """(blueprint, profile, novelty) cells of one evaluation set, in a fixed order."""
    es = proto["eval_sets"][name]
    return [tuple(x) for x in es["settings"]]


def units_of(settings: Sequence[tuple[str, str, str]]) -> list[str]:
    """The bootstrap units of a set: one per house (the ranch sets have one unit)."""
    return list(dict.fromkeys(s[0] for s in settings))


# --------------------------------------------------------------------------- checkpoints and manifests


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def _step_of(p: Path) -> int:
    return int(p.stem.split("_")[1])


def build_manifest(method: str, seed: int, run: Path | None = None) -> dict:
    """Write runs/formal_eval/train/<method>_s<seed>/manifest.json (the shared checkpoint convention): the 0% points
    and one checkpoint per 10% of training, hard-linked as ckpt_p<pct>.pt next to it. Paths are repo-relative."""
    run = run or TRAIN / f"{method}_s{seed}"
    rel = lambda p: str(Path(p).resolve().relative_to(ROOT))  # noqa: E731
    points = [{"pct": 0, "label": "untrained", "path": rel(run / "ckpt_p000_untrained.pt"), "step": 0}]
    if method == "bcppo":
        _link(run / "bc" / "bc.pt", run / "ckpt_p000_bc.pt")
        points.append({"pct": 0, "label": "bc_clone", "path": rel(run / "ckpt_p000_bc.pt"), "step": 0})
        sub = run / "ppo"
    elif method.startswith("dreamer_il"):
        sub = run / "dreamer"
    else:
        raise ValueError(method)
    hist = json.loads((sub / "history.json").read_text())
    total = sum(p["steps"] for p in hist["phases"])
    steps = sorted((_step_of(p) for p in sub.glob("step_*.pt")))
    final_step = int(hist.get("steps") or (hist["history"][-1]["step"] if hist["history"] else 0))
    stopped_early = final_step < total
    marks = []
    for k in range(1, 11):
        target = total * k / 10
        hits = [s for s in steps if s >= target - 1e-9]
        if k == 10:
            marks.append((100, sub / "final.pt", final_step))
        elif hits and hits[0] <= final_step:
            # the first checkpoint at or after k*10% (they snap to rollout / env-tick boundaries)
            marks.append((10 * k, sub / f"step_{hits[0]}.pt", hits[0]))
    for pct, src, st in marks:
        dst = run / f"ckpt_p{pct:03d}.pt"
        _link(src, dst)
        points.append({"pct": pct, "label": f"{pct}%", "path": rel(dst), "step": int(st)})
    man = {"method": method, "seed": seed, "total_steps": int(total), "final_step": final_step,
           "stopped_early": stopped_early, "points": points,
           "wall_clock_s": round(float(hist.get("wall_clock", 0.0)), 1),
           "sha256": {p["path"]: sha256(ROOT / p["path"]) for p in points},
           # the reward the run trained under (it changed on 2026-11-08 23:22, before every formal_eval run)
           "reward_py_sha256": sha256(ROOT / "src" / "emberpath" / "reward.py")}
    if method == "bcppo":
        bc_log = run / "bc" / "bc.pt"
        import torch

        meta = torch.load(bc_log, map_location="cpu", weights_only=False)
        man["bc"] = {k: meta["bc"][k] for k in ("teacher", "samples", "episodes_per_task", "collect_seconds")}
        man["bc"]["train_seeds_min_max"] = [min(meta["bc"]["train_seeds_used"]), max(meta["bc"]["train_seeds_used"])]
        man["bc"]["final_val_acc"] = meta["bc"]["history"][-1]["val_acc"]
        man["bc"]["seconds"] = meta["bc"]["history"][-1]["time"]
    (run / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def validate_manifest(man: dict, root: Path = ROOT) -> list[str]:
    """Problems with a manifest against the shared convention (empty list = fine)."""
    errs = []
    for k in ("method", "seed", "total_steps", "points"):
        if k not in man:
            errs.append(f"missing {k}")
    if errs:
        return errs
    pts = man["points"]
    labels = [p["label"] for p in pts]
    if not pts or pts[0]["label"] != "untrained" or pts[0]["pct"] != 0:
        errs.append("the first point must be the 0% 'untrained' checkpoint")
    if man["method"] == "bcppo" and "bc_clone" not in labels:
        errs.append("bcppo needs the 0% 'bc_clone' point")
    pcts = [p["pct"] for p in pts]
    if pcts != sorted(pcts):
        errs.append("points must be in increasing pct order")
    if pcts[-1] != 100:
        errs.append("the last point must be 100%")
    for p in pts:
        if Path(p["path"]).is_absolute():
            errs.append(f"{p['path']}: paths must be repo-relative")
        elif not (root / p["path"]).exists():
            errs.append(f"{p['path']}: missing file")
        if p["pct"] > 0 and p["label"] != f"{p['pct']}%":
            errs.append(f"label of {p['pct']}% point should be '{p['pct']}%'")
    return errs


# --------------------------------------------------------------------------- agents and episodes


def agent_specs(proto: dict) -> dict[str, dict]:
    """label -> {"kind": planner|ppo|dreamer, "method", "seed", "ckpt"} for every agent in the protocol whose
    checkpoint exists (missing training runs are reported, not silently skipped)."""
    specs = {p: {"kind": "planner", "method": p, "seed": None, "ckpt": None} for p in PLANNERS}
    for method, m in proto["methods"].items():
        for s in m["seeds"]:
            ck = ROOT / m["checkpoint"].format(seed=s)
            if ck.exists():
                specs[f"{method}_s{s}"] = {"kind": m["kind"], "method": method, "seed": s, "ckpt": str(ck)}
    for label, ref in proto.get("fixed_checkpoints", {}).items():
        ck = ROOT / ref["path"]
        if ck.exists():
            specs[label] = {"kind": ref["kind"], "method": label, "seed": None, "ckpt": str(ck)}
    return specs


def _cache_file(label: str, spec: dict, setting, seeds: Sequence[int]) -> Path:
    stamp = sha256(Path(spec["ckpt"]))[:16] if spec["ckpt"] else "planner"
    key = hashlib.sha256(repr((label, stamp, tuple(setting), seeds[0], seeds[-1], len(seeds))).encode()).hexdigest()
    return CACHE / f"{label}__{'_'.join(setting).replace(':', '-')}__{key[:12]}.json"


def _episode_job(args) -> tuple[str, tuple, list]:
    label, spec, setting, seeds, cache = args
    import torch

    torch.set_num_threads(1)
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.evaluate import run_episode

    bp_name, prof, nov = setting
    bp, hh = load_blueprint(bp_name), HOUSEHOLDS[prof]
    novelty = None if nov == "none" else nov
    if set(seeds) & set(TRAINING_FIRE_SEEDS):
        raise ValueError("evaluation must use held-out fire seeds only")
    if spec["kind"] == "planner":
        res = [run_episode(bp, hh, label, novelty, s) for s in seeds]
    elif spec["kind"] == "ppo":
        from emberpath.rl_baseline import evaluate_policy, load_policy

        net, _ = load_policy(spec["ckpt"])
        res = evaluate_policy(net, bp, hh, novelty, seeds, greedy=True)
    elif spec["kind"] == "dreamer":
        from emberpath.dreamer import evaluate_dreamer, load_dreamer

        res = evaluate_dreamer(load_dreamer(spec["ckpt"]), bp, hh, novelty, seeds, greedy=True)
    else:
        raise ValueError(spec["kind"])
    rows = [{"seed": r.seed, "score": r.score, "success": float(r.outcome == "success"),
             "died": float(r.outcome == "agent_lost"),
             "outcome": r.outcome, "steps": r.steps, "reward": r.reward, "rescued": r.rescued, "total": r.total}
            for r in res]
    Path(cache).write_text(json.dumps(rows))
    return label, tuple(setting), rows


def plan_jobs(proto: dict, specs: dict, labels: Sequence[str] | None = None, sets: Sequence[str] | None = None):
    jobs = []
    for set_name in sets or proto["eval_sets"]:
        seeds = fire_seeds(proto["eval_sets"][set_name]["fire_seeds"])
        for st in set_settings(proto, set_name):
            for label, spec in specs.items():
                if labels and label not in labels:
                    continue
                f = _cache_file(label, spec, st, seeds)
                if not f.exists():
                    jobs.append((label, spec, st, seeds, str(f)))
    return jobs


def run_jobs(jobs, workers: int, fn=_episode_job) -> None:
    if not jobs:
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    # the slowest agents first so the pool drains evenly
    order = {"dreamer": 0, "ppo": 1, "planner": 2}
    jobs = sorted(jobs, key=lambda j: order.get(j[1]["kind"], 3))
    t0 = time.time()
    if workers <= 1:
        for j in jobs:
            fn(j)
        return
    with ProcessPoolExecutor(workers) as ex:
        for i, (label, st, _) in enumerate(ex.map(fn, jobs, chunksize=1)):
            print(f"[{i + 1}/{len(jobs)}] {label} {'/'.join(st)} ({time.time() - t0:.0f}s)", flush=True)


def load_rows(label: str, spec: dict, setting, seeds) -> list[dict]:
    return json.loads(_cache_file(label, spec, setting, seeds).read_text())


def cube(proto: dict, specs: dict, labels: Sequence[str], set_name: str, metric: str = "score") -> np.ndarray:
    """(S, U, F) array for a method made of the given seed labels: per training seed, per unit (house), per fire,
    the mean of ``metric`` over that unit's settings."""
    seeds = fire_seeds(proto["eval_sets"][set_name]["fire_seeds"])
    settings = set_settings(proto, set_name)
    units = units_of(settings)
    out = np.zeros((len(labels), len(units), len(seeds)))
    for i, lab in enumerate(labels):
        for j, u in enumerate(units):
            cells = [st for st in settings if st[0] == u]
            vals = []
            for st in cells:
                rows = sorted(load_rows(lab, specs[lab], st, seeds), key=lambda r: r["seed"])
                assert [r["seed"] for r in rows] == seeds
                vals.append([r[metric] for r in rows])
            out[i, j] = np.mean(vals, axis=0)
    return out


# --------------------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="write the checkpoint manifest of one training run")
    m.add_argument("--method", required=True)
    m.add_argument("--seed", type=int, required=True)
    for name in ("eval", "lockstep", "stats", "all", "curve"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--labels", nargs="*", default=None, help="only these agents (default: all in the protocol)")
        p.add_argument("--sets", nargs="*", default=None, help="only these evaluation sets")
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        man = build_manifest(a.method, a.seed)
        errs = validate_manifest(man)
        print(json.dumps({k: man[k] for k in ("method", "seed", "total_steps", "final_step")}),
              f"{len(man['points'])} points", "OK" if not errs else errs)
        return 1 if errs else 0
    return run_cli(a)


# --------------------------------------------------------------------------- lockstep evidence test


def _make_agent(spec: dict, model, bp, world, greedy: bool, seed: int):
    if spec["kind"] == "ppo":
        from emberpath.rl_baseline import ppo_factory

        return ppo_factory(model, greedy=greedy, seed=seed)(bp, world)
    if spec["kind"] == "dreamer":
        from emberpath.dreamer import dreamer_factory

        return dreamer_factory(model, greedy=greedy, seed=seed)(bp, world)
    from emberpath.agents import make_agent

    return make_agent(spec["method"], bp, world)  # planners (and the oracle control) get what make_agent gives


def lockstep_pair(spec: dict, model, bp, hh, novelty: str, seed: int, greedy: bool) -> tuple[int, bool]:
    """tests/test_evidence_discipline.py's lockstep on any agent: the novelty world and its no-novelty control with
    the same seed, stepped together while the agent's inputs are identical. Returns (ticks compared, leaked)."""
    from emberpath.scenarios import NOVELTIES
    from emberpath.world import World

    worlds = [World(bp, hh, seed=seed, novelties=NOVELTIES[novelty](bp, hh)), World(bp, hh, seed=seed)]
    agents = [_make_agent(spec, model, bp, w, greedy, seed) for w in worlds]
    obs, events, t = [w.observe() for w in worlds], [[], []], 0
    while not any(w.done for w in worlds) and obs[0] == obs[1] and events[0] == events[1]:
        ds = [ag.act(o, e) for ag, o, e in zip(agents, obs, events)]
        if ds[0] != ds[1]:
            return t, True
        rs = [w.step(d.action) for w, d in zip(worlds, ds)]
        obs, events = [r.obs for r in rs], [r.events for r in rs]
        t += 1
    return t, False


def _lockstep_job(args) -> dict:
    label, spec, proto_lock, cache = args
    import torch

    torch.set_num_threads(1)
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.scenarios import NOVELTIES, applicable

    model = None
    if spec["kind"] == "ppo":
        from emberpath.rl_baseline import load_policy

        model = load_policy(spec["ckpt"])[0]
    elif spec["kind"] == "dreamer":
        from emberpath.dreamer import load_dreamer

        model = load_dreamer(spec["ckpt"])
    pairs = ticks = 0
    leaks = []
    modes = (True, False) if spec["kind"] in ("ppo", "dreamer") else (True,)
    for bp_name in proto_lock["blueprints"]:
        bp = load_blueprint(bp_name)
        for prof, hh in HOUSEHOLDS.items():
            for nov in sorted(NOVELTIES):
                if not applicable(bp, hh, nov):
                    continue
                for s in fire_seeds(proto_lock["fire_seeds"]):
                    for greedy in modes:
                        t, leaked = lockstep_pair(spec, model, bp, hh, nov, s, greedy)
                        pairs += 1
                        ticks += t
                        if leaked:
                            leaks.append([bp_name, prof, nov, s, greedy, t])
    res = {"label": label, "ckpt": str(Path(spec["ckpt"]).relative_to(ROOT)) if spec["ckpt"] else None,
           "pairs": pairs, "ticks": ticks, "leaks": leaks}
    Path(cache).write_text(json.dumps(res))
    return res


def _lock_cache(label: str, spec: dict) -> Path:
    stamp = sha256(Path(spec["ckpt"]))[:16] if spec["ckpt"] else spec["method"]
    return CACHE / f"lockstep__{label}__{stamp}.json"


def run_lockstep(proto: dict, specs: dict, workers: int) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(lab, sp, proto["lockstep"], str(_lock_cache(lab, sp))) for lab, sp in specs.items()]
    todo = [j for j in jobs if not Path(j[3]).exists()]
    if todo:
        with ProcessPoolExecutor(max(1, workers)) as ex:
            for r in ex.map(_lockstep_job, todo):
                print(f"lockstep {r['label']}: {r['pairs']} pairs, {r['ticks']} ticks, {len(r['leaks'])} leaks",
                      flush=True)
    return [json.loads(Path(j[3]).read_text()) for j in jobs]


# --------------------------------------------------------------------------- learning curves (fires subset)


def curve_specs(proto: dict) -> dict[str, dict]:
    out = {}
    for method, m in proto["methods"].items():
        if method == "bc_clone":
            continue
        for s in m["seeds"]:
            mf = TRAIN / f"{method}_s{s}" / "manifest.json"
            if not mf.exists():
                continue
            for pt in json.loads(mf.read_text())["points"]:
                tag = "p000_" + pt["label"] if pt["pct"] == 0 else f"p{pt['pct']:03d}"
                out[f"curve_{method}_s{s}_{tag}"] = {"kind": m["kind"], "method": method, "seed": s,
                                                     "ckpt": str(ROOT / pt["path"]), "pct": pt["pct"],
                                                     "label": pt["label"], "step": pt["step"]}
    return out


CURVE_SEEDS = list(range(1000, 1020))


def run_curve(proto: dict, workers: int, methods: Sequence[str] = ("bcppo",)) -> dict:
    specs = {k: v for k, v in curve_specs(proto).items() if v["method"] in methods}
    settings = set_settings(proto, "fires")
    jobs = [(lab, sp, st, CURVE_SEEDS, str(_cache_file(lab, sp, st, CURVE_SEEDS)))
            for lab, sp in specs.items() for st in settings]
    run_jobs([j for j in jobs if not Path(j[4]).exists()], workers)
    lock = run_lockstep(proto, specs, workers)
    rows = []
    for lab, sp in specs.items():
        sc = [np.mean([r["score"] for r in load_rows(lab, sp, st, CURVE_SEEDS)]) for st in settings]
        su = [np.mean([r["success"] for r in load_rows(lab, sp, st, CURVE_SEEDS)]) for st in settings]
        rows.append({"label": lab, "method": sp["method"], "seed": sp["seed"], "pct": sp["pct"],
                     "point": sp["label"], "step": sp["step"], "score": float(np.mean(sc)),
                     "success": float(np.mean(su))})
    return {"rows": rows, "lockstep": lock, "fire_seeds": [CURVE_SEEDS[0], CURVE_SEEDS[-1] + 1]}


# --------------------------------------------------------------------------- statistics over the cached episodes


def method_labels(proto: dict, specs: dict) -> dict[str, list[str]]:
    out = {p: [p] for p in proto["planners"]}
    for method, m in proto["methods"].items():
        labs = [f"{method}_s{s}" for s in m["seeds"] if f"{method}_s{s}" in specs]
        if labs:
            out[method] = labs
    for label in proto.get("fixed_checkpoints", {}):
        if label in specs:
            out[label] = [label]
    return out


def compute_stats(proto: dict, specs: dict) -> dict:
    bs = proto["bootstrap"]
    B, rs = bs["n_boot"], bs["rng_seed"]
    labels = method_labels(proto, specs)
    res: dict = {"protocol_created_utc": proto["created_utc"], "methods": {}, "comparisons": [], "missing": []}
    for set_name in proto["eval_sets"]:
        res["methods"][set_name] = {}
        for meth, labs in labels.items():
            row = {"labels": labs, "score": seed_split(cube(proto, specs, labs, set_name), B, rs)}
            for metric in ("success", "died", "steps", "reward"):
                x = cube(proto, specs, labs, set_name, metric)
                row[metric] = {"mean": float(x.mean()), "per_seed": x.mean(axis=(1, 2)).tolist()}
            res["methods"][set_name][meth] = row
    fam = []
    for meth, ref, set_name in proto["comparisons"]:
        if meth not in labels or ref not in labels:
            res["missing"].append([meth, ref, set_name])
            continue
        a, b = cube(proto, specs, labels[meth], set_name), cube(proto, specs, labels[ref], set_name)
        pair = meth == "bcppo" and ref == "bc_clone" and a.shape[0] == b.shape[0]
        prim = paired_diff_bootstrap(a, b, B, rs, resample_seeds=True, pair_seeds=pair)
        sec = paired_diff_bootstrap(a.mean(0, keepdims=True), b.mean(0, keepdims=True), B, rs, resample_seeds=False)
        succ = paired_diff_bootstrap(cube(proto, specs, labels[meth], set_name, "success"),
                                     cube(proto, specs, labels[ref], set_name, "success"), B, rs,
                                     resample_seeds=True, pair_seeds=pair)
        per_seed = (a.mean(axis=(1, 2))[:, None] - b.mean(axis=(1, 2))[None, :]).mean(1).tolist() \
            if not pair else (a.mean(axis=(1, 2)) - b.mean(axis=(1, 2))).tolist()
        fam.append({"method": meth, "ref": ref, "set": set_name, "n_seeds": [a.shape[0], b.shape[0]],
                    "means": [float(a.mean()), float(b.mean())], "primary": prim, "fire_only": sec,
                    "success_diff": succ, "per_seed_diff": per_seed})
    adj = holm([c["primary"]["p"] for c in fam])
    alpha = proto["multiple_testing"]["alpha"]
    for c, p in zip(fam, adj):
        c["p_holm"] = p
        c["verdict"] = ("no significant difference" if p >= alpha else
                        f"{c['method']} > {c['ref']}" if c["primary"]["diff"] > 0 else f"{c['method']} < {c['ref']}")
    res["comparisons"] = fam
    res["family_size"] = len(fam)
    # reproducibility: bcppo_s0 vs the original run
    if "bcppo_s0" in specs and "bcppo_original" in specs:
        import torch

        sa = torch.load(specs["bcppo_s0"]["ckpt"], map_location="cpu", weights_only=False)["state_dict"]
        sb = torch.load(specs["bcppo_original"]["ckpt"], map_location="cpu", weights_only=False)["state_dict"]
        maxdiff = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)
        rep = {"identical_weights": maxdiff == 0.0, "max_abs_weight_diff": maxdiff}
        for set_name in proto["eval_sets"]:
            a = cube(proto, specs, ["bcppo_s0"], set_name)
            b = cube(proto, specs, ["bcppo_original"], set_name)
            rep[set_name] = paired_diff_bootstrap(a, b, B, rs, resample_seeds=False)
            rep[set_name]["means"] = [float(a.mean()), float(b.mean())]
            same = 0
            n = 0
            for st in set_settings(proto, set_name):
                seeds = fire_seeds(proto["eval_sets"][set_name]["fire_seeds"])
                ra = load_rows("bcppo_s0", specs["bcppo_s0"], st, seeds)
                rb = load_rows("bcppo_original", specs["bcppo_original"], st, seeds)
                same += sum((x["outcome"], x["steps"], x["rescued"]) == (y["outcome"], y["steps"], y["rescued"])
                            for x, y in zip(ra, rb))
                n += len(ra)
            rep[set_name]["identical_episodes"] = [same, n]
        res["reproducibility"] = rep
    return res


def _f(x, nd=3, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def render_md(res: dict, proto: dict) -> str:
    L = ["# Formal evaluation: BC+PPO and DreamerV3 + imitation", "",
         f"Pre-registered {proto['created_utc']} (runs/formal_eval/PREREGISTRATION.md). Primary metric: pooled score "
         "(fraction rescued, 0 if the rescuer died). Greedy policies. Intervals: 95% percentile, "
         f"{proto['bootstrap']['n_boot']} paired resamples; primary = two-level bootstrap (training seeds, then houses, "
         f"then fires); Holm over the {res['family_size']} pre-registered comparisons at alpha "
         f"{proto['multiple_testing']['alpha']}.", ""]
    names = {"fires": "Held-out fires (ranch, 8 trained settings, seeds 1000..1099)",
             "houses": "Held-out houses (9 procedural houses x 8 settings, seeds 1000..1029)",
             "compound": "Untrained novelty: compound (ranch, dog + kids, seeds 1000..1099)"}
    for set_name in proto["eval_sets"]:
        L += [f"## {names.get(set_name, set_name)}", "",
              "| method | pooled score | per training seed | between-seed SD | fire SE | two-level SE | success | "
              "died | steps |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
        for meth, row in res["methods"][set_name].items():
            sc = row["score"]
            ps = ", ".join(f"{v:.3f}" for v in sc["per_seed"]) if len(sc["per_seed"]) > 1 else ""
            L.append(f"| {meth} | {sc['mean']:.3f} | {ps} | {_f(sc['between_seed_sd'], 3, False)} | "
                     f"{sc['fire_se']:.3f} | {sc['hier_se']:.3f} | {row['success']['mean']:.1%} | "
                     f"{row['died']['mean']:.1%} | {row['steps']['mean']:.0f} |")
        L += ["", "| comparison | difference | 95% CI (two-level) | raw p | Holm p | verdict | fire-only CI | "
                  "success diff [CI] | per-seed diffs |", "|---|---:|---|---:|---:|---|---|---|---|"]
        for c in res["comparisons"]:
            if c["set"] != set_name:
                continue
            p, s, sd = c["primary"], c["fire_only"], c["success_diff"]
            L.append(f"| {c['method']} - {c['ref']} | {_f(p['diff'])} | [{_f(p['lo'])}, {_f(p['hi'])}] | "
                     f"{p['p']:.4f} | {c['p_holm']:.4f} | {c['verdict']} | [{_f(s['lo'])}, {_f(s['hi'])}] | "
                     f"{_f(sd['diff'], 2)} [{_f(sd['lo'], 2)}, {_f(sd['hi'], 2)}] | "
                     f"{', '.join(_f(x, 3) for x in c['per_seed_diff'])} |")
        L.append("")
    if res.get("missing"):
        L += ["**Comparisons not run (method not trained or evaluated):** "
              + "; ".join(f"{m} - {r} ({s})" for m, r, s in res["missing"]), ""]
    rep = res.get("reproducibility")
    if rep:
        L += ["## Reproducibility: bcppo_s0 vs runs/bcppo/final.pt", "",
              f"Identical weights: {rep['identical_weights']} (max |weight difference| {rep['max_abs_weight_diff']:.3g}).",
              "", "| set | bcppo_s0 | original | difference [fire-only 95% CI] | p | identical episodes |",
              "|---|---:|---:|---|---:|---:|"]
        for set_name in proto["eval_sets"]:
            r = rep[set_name]
            L.append(f"| {set_name} | {r['means'][0]:.3f} | {r['means'][1]:.3f} | {_f(r['diff'])} [{_f(r['lo'])}, "
                     f"{_f(r['hi'])}] | {r['p']:.4f} | {r['identical_episodes'][0]}/{r['identical_episodes'][1]} |")
        L.append("")
    lock = res.get("lockstep", [])
    if lock:
        L += ["## Lockstep evidence test (every evaluated checkpoint)", "",
              "ranch + studio, every household x applicable novelty, fire seeds 1000..1004, greedy and sampled.", "",
              "| agent | checkpoint | pairs | ticks compared | differing decisions |", "|---|---|---:|---:|---:|"]
        for r in lock:
            L.append(f"| {r['label']} | {r['ckpt'] or '-'} | {r['pairs']} | {r['ticks']} | {len(r['leaks'])} |")
        L.append("")
    cur = res.get("curve")
    if cur:
        L += [f"## Learning curves (fires set, seeds {cur['fire_seeds'][0]}..{cur['fire_seeds'][1] - 1})", "",
              "| method | seed | point | env step | score | success | lockstep leaks |", "|---|---:|---|---:|---:|---:|---:|"]
        lk = {r["label"]: len(r["leaks"]) for r in cur["lockstep"]}
        for r in sorted(cur["rows"], key=lambda r: (r["method"], r["seed"], r["pct"], r["point"] != "untrained")):
            L.append(f"| {r['method']} | {r['seed']} | {r['point']} | {r['step']} | {r['score']:.3f} | "
                     f"{r['success']:.1%} | {lk.get(r['label'], 'n/a')} |")
        L.append("")
    return "\n".join(L) + "\n"


def run_cli(a) -> int:
    proto = load_protocol()
    specs = agent_specs(proto)
    if a.labels:
        specs = {k: v for k, v in specs.items() if k in a.labels}
    if a.cmd in ("eval", "all"):
        jobs = plan_jobs(proto, specs, None, a.sets)
        print(f"{len(jobs)} episode jobs to run ({len(specs)} agents)", flush=True)
        run_jobs(jobs, a.workers)
    if a.cmd == "curve":
        run_curve(proto, a.workers, tuple(a.sets or ("bcppo",)))
        return 0
    if a.cmd in ("lockstep", "all"):
        run_lockstep(proto, specs, a.workers)
    if a.cmd in ("stats", "all"):
        specs = agent_specs(proto)
        res = compute_stats(proto, specs)
        lock_specs = dict(specs)
        lock_specs["oracle"] = specs["oracle"]
        res["lockstep"] = [json.loads(_lock_cache(k, v).read_text()) for k, v in specs.items()
                           if _lock_cache(k, v).exists()]
        cs = curve_specs(proto)
        if any(_cache_file(k, v, set_settings(proto, "fires")[0], CURVE_SEEDS).exists() for k, v in cs.items()):
            done = {k: v for k, v in cs.items() if all(_cache_file(k, v, st, CURVE_SEEDS).exists()
                                                         for st in set_settings(proto, "fires"))}
            res["curve"] = {"fire_seeds": [CURVE_SEEDS[0], CURVE_SEEDS[-1] + 1], "rows": [], "lockstep": []}
            for lab, sp in done.items():
                sts = set_settings(proto, "fires")
                res["curve"]["rows"].append({
                    "label": lab, "method": sp["method"], "seed": sp["seed"], "pct": sp["pct"], "point": sp["label"],
                    "step": sp["step"],
                    "score": float(np.mean([np.mean([r["score"] for r in load_rows(lab, sp, st, CURVE_SEEDS)])
                                            for st in sts])),
                    "success": float(np.mean([np.mean([r["success"] for r in load_rows(lab, sp, st, CURVE_SEEDS)])
                                              for st in sts]))})
                if _lock_cache(lab, sp).exists():
                    res["curve"]["lockstep"].append(json.loads(_lock_cache(lab, sp).read_text()))
        (OUT / "results.json").write_text(json.dumps(res, indent=1))
        (OUT / "results.md").write_text(render_md(res, proto))
        print(render_md(res, proto))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
