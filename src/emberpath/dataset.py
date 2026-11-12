"""Trajectory dataset (WP-K, SPEC D1-D3): versioned episode records, deterministic builds, validators.

One JSON object per episode (JSONL, optionally gzip). Stdlib + the simulator only: no pandas, no pyarrow.
The schema, every validator and what each one catches are documented in docs/DATA.md.

Evidence discipline: the per-step side of a record (``steps[*].obs`` / ``steps[*].events``) holds only what
the rescuer could perceive (``World.observe()`` and the ``Event`` log). The hidden truth (which novelty, which
rules changed and when) lives in the episode-level ``labels`` and nowhere else. ``validate`` replays every
episode (any mismatch fails) and ``check_leaks`` names evidence-discipline violations as ``leak``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

from emberpath.agents import make_agent
from emberpath.blueprint import Blueprint
from emberpath.dependents import HOUSEHOLDS
from emberpath.planner import Decision
from emberpath.procgen import measure, tier_of
from emberpath.reward import COMPONENTS, RewardConfig, RewardTracker
from emberpath.rules import key_cell, parse_key
from emberpath.scenarios import NOVELTIES, applicable
from emberpath.grid import MOVES, WAIT, manhattan
from emberpath.world import HEARING_RANGE, REVEAL_STEP, WAITING, Event, Observation, World

SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")
NOVELTY_NAMES = ("none", *NOVELTIES)
HOUSEHOLD_NAMES = tuple(HOUSEHOLDS)
DEFAULT_BLUEPRINTS = ("ranch", "studio", "gen:1", "gen:2", "gen:3", "gen:4", "gen:5")  # 7: coprime with 3 x 5
DEFAULT_AGENTS = ("baseline", "oracle", "adaptive", "random")
PRIVILEGED = {"oracle"}  # agents whose actions/reasons are computed from the true rules
OBS_FIELDS = {"pos": list, "alive": bool, "fire_on": list, "fire_off": list, "deps": list, "smoke": float}
STEP_FIELDS = {"obs": dict, "action": int, "reason": str, "reward": list, "events": dict}
RECORD_FIELDS = {
    "schema_version": int, "id": str, "content_hash": str, "split": str, "blueprint": str, "blueprint_hash": str,
    "tier": int, "household": str, "seed": int, "agent": str, "privileged": bool, "max_steps": int,
    "labels": dict, "steps": list, "final": dict, "outcome": str, "n_steps": int, "return": float,
    "score": float, "efficiency": (float, type(None)), "quality": float,
}
OUTCOMES = ("success", "partial", "agent_lost", "timeout")


# ------------------------------------------------------------------ houses
@lru_cache(maxsize=256)
def load_blueprint(name: str) -> Blueprint:
    """`ranch` / `studio` / a path, or a procedural house `gen:<tier>:<seed>` (cached: houses are reused per job)."""
    from emberpath.cli import load_blueprint as load  # lazy: the CLI pulls in rendering

    return load(name)


def blueprint_for(spec: str, job: int, n_specs: int) -> Blueprint:
    """`gen:<tier>` draws a fresh house per pass over the blueprint list; anything else is a fixed house."""
    if spec.startswith("gen:") and spec.count(":") == 1:
        spec = f"{spec}:{job // n_specs}"
    return load_blueprint(spec)


def blueprint_hash(bp: Blueprint) -> str:
    return hashlib.sha256("\n".join(bp.rows).encode()).hexdigest()[:16]


@lru_cache(maxsize=256)
def tier(name: str) -> int:
    """Procedural houses carry their tier; shipped houses are measured with the same yardstick (procgen.tier_of)."""
    return int(name.split(":")[1]) if name.startswith("gen:") else tier_of(measure(load_blueprint(name)))


# ------------------------------------------------------------------ encoding
def _flat(cells: Iterable[tuple[int, int]], width: int) -> list[int]:
    return sorted(r * width + c for r, c in cells)


def compact_obs(obs: Observation, prev_fire: frozenset, width: int) -> dict:
    """Observation -> small dict. Fire is delta-coded (cells r*W+c that started / stopped burning since the previous
    observation); `burnt` is then implied (every cell that stopped burning). Full observability is assumed."""
    return {
        "pos": list(obs.agent_pos),
        "alive": obs.alive,
        "fire_on": _flat(obs.burning - prev_fire, width),
        "fire_off": _flat(prev_fire - obs.burning, width),
        "deps": [[d.id, d.kind, d.pos[0], d.pos[1], d.status] for d in obs.dependents],
        "smoke": round(obs.smoke_at_agent, 4),
    }


def tally(events: list[Event]) -> dict[str, list[int]]:
    """Events of one tick -> {key: [trials, successes]}. Lossless for exchangeable (Beta-Bernoulli) evidence."""
    out: dict[str, list[int]] = {}
    for e in events:
        n = out.setdefault(e.key, [0, 0])
        n[0] += 1
        n[1] += e.outcome
    return dict(sorted(out.items()))


# ------------------------------------------------------------------ policies
@dataclass
class RandomPolicy:
    """Scripted, seeded random walk: a deliberately poor policy so the data holds failures, not just experts."""

    seed: int

    def __post_init__(self) -> None:
        self.rng = random.Random(f"random:{self.seed}")

    def act(self, obs: Observation, events: list[Event]) -> Decision:
        return Decision(self.rng.randrange(5), "Scripted random walk (a deliberately poor policy, for data diversity).")


@dataclass
class ReplayPolicy:
    """Plays back recorded actions (and their reasons); waits if the record runs out before the world ends."""

    steps: list

    def act(self, obs: Observation, events: list[Event]) -> Decision:
        i = obs.step
        if i < len(self.steps):
            return Decision(int(self.steps[i]["action"]), str(self.steps[i]["reason"]))
        return Decision(WAIT, "Replay ran out of recorded actions.")


# ------------------------------------------------------------------ episodes
def run_record(bp: Blueprint, household: str, novelty: str, agent: str, seed: int, max_steps: int = 150,
               replay: list | None = None) -> dict:
    """Run one episode and return its record (split, id and quality are filled in by `finish`)."""
    specs = NOVELTIES[novelty](bp, HOUSEHOLDS[household]) if novelty != "none" else []
    world = World(bp, HOUSEHOLDS[household], seed=seed, novelties=specs, max_steps=max_steps)
    if replay is not None:
        policy = ReplayPolicy(replay)
    elif agent == "random":
        policy = RandomPolicy(seed)
    else:
        policy = make_agent(agent, bp, world)
    tracker = RewardTracker(world)
    steps: list[dict] = []
    obs, events, fire = world.observe(), [], frozenset()
    while not world.done:
        o, fire = compact_obs(obs, fire, bp.width), obs.burning
        d = policy.act(obs, events)
        res = world.step(d.action)
        r = tracker.update(world)
        steps.append({"obs": o, "action": int(d.action), "reason": d.reason,
                      "reward": [r.components[k] for k in COMPONENTS], "events": tally(res.events)})
        obs, events = res.obs, res.events
    c = world.counts()
    return {
        "schema_version": SCHEMA_VERSION, "blueprint": bp.name, "blueprint_hash": blueprint_hash(bp),
        "tier": tier(bp.name), "household": household, "seed": seed, "agent": agent,
        "privileged": agent in PRIVILEGED, "max_steps": max_steps,
        "labels": {"novelty": novelty, "true_patches": [[s.step, s.patch.key, s.patch.value] for s in world.novelties]},
        "steps": steps,
        "final": {"obs": compact_obs(obs, fire, bp.width), **c},
        "outcome": world.outcome, "n_steps": world.t, "return": tracker.total,
        "score": 0.0 if not world.agent_alive else c["rescued"] / max(1, c["total"]),
        "efficiency": tracker.efficiency(world),
    }


def _sha(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:16]


def split_of(bp_hash: str, seed: int) -> str:
    """Split by (blueprint, seed) group: every agent and novelty on the same house and fire stays together."""
    k = int(_sha(bp_hash, seed), 16) % 10
    return "train" if k < 8 else "val" if k == 8 else "test"


def quality(rec: dict) -> float:
    """0..1, threshold to keep good demonstrations: score x (0.5 + 0.5 x efficiency); 0 for failures."""
    return round(rec["score"] * (0.5 + 0.5 * (rec["efficiency"] or 0.0)), 6)


def finish(rec: dict) -> dict:
    rec["content_hash"] = _sha(rec["blueprint_hash"], rec["household"], rec["labels"]["novelty"], rec["seed"],
                               [s["action"] for s in rec["steps"]])
    rec["id"] = _sha(rec["blueprint_hash"], rec["household"], rec["labels"]["novelty"], rec["seed"], rec["agent"])
    rec["split"] = split_of(rec["blueprint_hash"], rec["seed"])
    rec["quality"] = quality(rec)
    return rec


# ------------------------------------------------------------------ build
def job_cell(j: int, blueprints: tuple[str, ...]) -> tuple[str, str, str]:
    """Job j -> (blueprint spec, household, novelty) by residues. With len(blueprints) coprime to 15 every triple
    appears once per lcm-period, and every pair far sooner (e.g. all blueprint x novelty pairs in 35 jobs)."""
    return blueprints[j % len(blueprints)], HOUSEHOLD_NAMES[j % len(HOUSEHOLD_NAMES)], NOVELTY_NAMES[j % len(NOVELTY_NAMES)]


def run_job(args: tuple) -> tuple[int, list[dict], str | None]:
    """All agents on one (house, household, novelty, seed). Returns (job, records, skip reason)."""
    j, blueprints, agents, max_steps = args
    spec, household, novelty = job_cell(j, blueprints)
    n = len(blueprints)
    for k in range(n):  # a house too small for this household / novelty hands the job to the next house in the list
        bp = blueprint_for(blueprints[(blueprints.index(spec) + k) % n], j, n)
        if applicable(bp, HOUSEHOLDS[household], None if novelty == "none" else novelty):
            return j, [finish(run_record(bp, household, novelty, a, j, max_steps)) for a in agents], None
    return j, [], f"{household}/{novelty}: no listed house can host it"


def dumps(rec: dict) -> str:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"))


def build(out: str | Path, episodes: int, agents: tuple[str, ...] = DEFAULT_AGENTS,
          blueprints: tuple[str, ...] = DEFAULT_BLUEPRINTS, max_steps: int = 150, workers: int = 1,
          compress: bool = False) -> dict:
    """Stream `episodes` records into out/<split>.jsonl[.gz] and write out/manifest.json. Byte-identical for the same
    arguments (any `workers`). Skipped jobs and dropped duplicates are listed in the manifest, never silently lost."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ext = ".jsonl.gz" if compress else ".jsonl"
    files = {s: (gzip.GzipFile(out / f"{s}{ext}", "wb", mtime=0) if compress else open(out / f"{s}{ext}", "wb"))
             for s in SPLITS}
    counts, seen, skipped, dropped = Counter(), {}, [], []
    pool = None
    if workers > 1:
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(workers)
    j = 0
    try:
        while sum(counts.values()) < episodes:
            need = -(-(episodes - sum(counts.values())) // len(agents))
            batch = [(k, blueprints, agents, max_steps) for k in range(j, j + need)]
            j += need
            for _, recs, skip in (pool.imap(run_job, batch) if pool else map(run_job, batch)):
                if skip:
                    skipped.append(skip)
                for rec in recs:
                    if sum(counts.values()) >= episodes:
                        break
                    if rec["content_hash"] in seen:
                        dropped.append({"id": rec["id"], "agent": rec["agent"], "duplicate_of": seen[rec["content_hash"]]})
                        continue
                    seen[rec["content_hash"]] = rec["id"]
                    files[rec["split"]].write((dumps(rec) + "\n").encode())
                    counts[rec["split"]] += 1
    finally:
        if pool:
            pool.close()
        for f in files.values():
            f.close()
    manifest = {
        "schema_version": SCHEMA_VERSION, "records": sum(counts.values()), "by_split": {s: counts[s] for s in SPLITS},
        "files": {s: f"{s}{ext}" for s in SPLITS}, "jobs": j, "skipped_jobs": skipped, "dropped_duplicates": dropped,
        "args": {"episodes": episodes, "agents": list(agents), "blueprints": list(blueprints), "max_steps": max_steps},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


# ------------------------------------------------------------------ reading
def read_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    with (gzip.open(p, "rt") if p.suffix == ".gz" else open(p)) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_dataset(root: str | Path) -> tuple[dict, dict[str, list[dict]]]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    return manifest, {s: list(read_jsonl(root / manifest["files"][s])) for s in SPLITS}


# ------------------------------------------------------------------ validators
@dataclass(frozen=True)
class Issue:
    check: str  # schema | replay | leak | return | outcome | duplicate | split_leakage | outlier | manifest
    record: str
    detail: str


def _types_ok(obj: dict, spec: dict) -> list[str]:
    bad = [f"missing {k}" for k in spec if k not in obj] + [f"unexpected field {k}" for k in obj if k not in spec]
    for k, t in spec.items():
        v = obj.get(k)
        ok = isinstance(v, t) and not (t is int and isinstance(v, bool)) or (t is float and type(v) is int)
        if k in obj and not ok:
            bad.append(f"{k} has type {type(v).__name__}")
    return bad


def check_schema(rec: dict) -> list[str]:
    bad = _types_ok(rec, RECORD_FIELDS)
    if bad:
        return bad
    if rec["schema_version"] != SCHEMA_VERSION:
        bad.append(f"schema_version {rec['schema_version']} != {SCHEMA_VERSION}")
    for k, allowed in (("split", SPLITS), ("household", HOUSEHOLD_NAMES), ("outcome", OUTCOMES)):
        if rec[k] not in allowed:
            bad.append(f"{k}={rec[k]!r}")
    if rec["labels"].get("novelty") not in NOVELTY_NAMES or set(rec["labels"]) != {"novelty", "true_patches"}:
        bad.append(f"labels={rec['labels']!r}")
    for i, st in enumerate([*rec["steps"], {"obs": rec["final"]["obs"]}]):
        errs = _types_ok(st["obs"], OBS_FIELDS) if isinstance(st.get("obs"), dict) else ["obs missing"]
        if i < len(rec["steps"]):
            errs += _types_ok(st, STEP_FIELDS)
            errs += [] if len(st.get("reward", ())) == len(COMPONENTS) else ["reward length"]
        bad += [f"step {i}: {e}" for e in errs]
    return bad


def check_static(rec: dict, cfg: RewardConfig = RewardConfig()) -> list[Issue]:
    """Checks that need no simulation: return = sum of components, outcome vs final state, impossible values."""
    out, rid = [], rec["id"]
    total = sum(sum(s["reward"]) for s in rec["steps"])
    if not math.isclose(total, rec["return"], abs_tol=1e-6):
        out.append(Issue("return", rid, f"return {rec['return']:.6f} != sum of components {total:.6f}"))
    f, n = rec["final"], len(rec["steps"])
    alive, full = f["obs"]["alive"], f["rescued"] + f["lost"] == f["total"]
    expect = {"success": alive and full and f["lost"] == 0, "partial": alive and full and f["lost"] > 0,
              "agent_lost": not alive, "timeout": alive and n == rec["max_steps"]}
    if not expect[rec["outcome"]]:
        out.append(Issue("outcome", rid, f"outcome {rec['outcome']} contradicts final state {f}"))
    score = f["rescued"] / max(1, f["total"]) if alive else 0.0
    if not math.isclose(score, rec["score"]) or not math.isclose(quality(rec), rec["quality"]):
        out.append(Issue("outcome", rid, f"score/quality {rec['score']}/{rec['quality']} do not match final state"))
    if not 1 <= n <= rec["max_steps"] or rec["n_steps"] != n:
        out.append(Issue("outlier", rid, f"{n} steps recorded, n_steps={rec['n_steps']}, max_steps={rec['max_steps']}"))
    k = f["total"]
    horizon_left = rec["max_steps"] - n if rec["outcome"] == "partial" else 0  # a loss charges time to the horizon
    for i, s in enumerate(rec["steps"]):
        c = dict(zip(COMPONENTS, s["reward"]))
        last = i == n - 1
        ok = (
            s["action"] in range(5)
            and math.isclose(c["time"], -cfg.step_cost * (1 + horizon_left * last))
            and _multiple(c["rescue"], cfg.rescue_bonus, 0, k)
            and _multiple(c["fire_contact"], -cfg.contact_penalty, 0, k)
            and (c["success"] == 0 or (last and math.isclose(c["success"], cfg.success_bonus)))
            and -cfg.proximity_weight * k - 1e-9 <= c["fire_proximity"] <= 0
            and (c["agent_death"] == 0 or (last and c["agent_death"] < 0))
            and abs(c["progress"]) <= cfg.progress_weight * 4 * (k + 1) * _area(rec["blueprint"])
        )
        if not ok:
            out.append(Issue("outlier", rid, f"step {i}: impossible action/reward {s['action']} {c}"))
            break
    return out


def _area(name: str) -> int:
    bp = load_blueprint(name)
    return bp.height * bp.width


def check_leaks(rec: dict) -> list[Issue]:
    """Evidence discipline, from the record alone. A hidden dependent may appear in an observation only once it has
    bonded (never as `waiting`); a door event needs the rescuer to have just walked into that door; a hearing event
    needs the slot within HEARING_RANGE of where the rescuer ended the tick, or (from REVEAL_STEP) a call that was heard. Extra obs fields are caught by the schema."""
    out, rid = [], rec["id"]
    n_declared = len(HOUSEHOLDS[rec["household"]].dependents)
    adults = {i for i, p in enumerate(HOUSEHOLDS[rec["household"]].dependents) if p.autonomous}
    hidden = {key_cell(k) for _, k, v in rec["labels"]["true_patches"] if parse_key(k)[0] == "occupied" and v}
    obs = [s["obs"] for s in rec["steps"]] + [rec["final"]["obs"]]
    for i, s in enumerate(rec["steps"]):
        for d in s["obs"]["deps"]:
            if d[0] >= n_declared and d[4] == WAITING and (d[2], d[3]) in hidden:
                out.append(Issue("leak", rid, f"step {i}: hidden dependent shown at {(d[2], d[3])} while still waiting"))
        (r, c), nxt = obs[i]["pos"], tuple(obs[i + 1]["pos"])
        for key in s["events"]:
            fam = parse_key(key)[0]
            if fam == "door_jammed" and not (
                s["action"] != WAIT and key_cell(key) == (r + MOVES[s["action"]][0], c + MOVES[s["action"]][1])
            ) and not any(  # or an adult acting alone walked into it
                manhattan(key_cell(key), (d[2], d[3])) <= 1 for d in s["obs"]["deps"] if d[0] in adults
            ):
                out.append(Issue("leak", rid, f"step {i}: {key} reported without bumping that door"))
            elif fam == "occupied" and manhattan(key_cell(key), nxt) > HEARING_RANGE and not (
                i >= REVEAL_STEP and s["events"][key][0] == s["events"][key][1]
            ):
                out.append(Issue("leak", rid, f"step {i}: {key} reported out of hearing range"))
    return out


def _multiple(x: float, unit: float, lo: int, hi: int) -> bool:
    return any(math.isclose(x, m * unit, abs_tol=1e-9) for m in range(lo, hi + 1))


def check_replay(rec: dict) -> list[Issue]:
    """Re-run the recorded actions with the recorded seed: everything recorded must come out again. Per-step content the
    world never showed is also caught here, as a mismatch; `check_leaks` names the evidence-discipline cases."""
    rid = rec["id"]
    bp = load_blueprint(rec["blueprint"])
    if blueprint_hash(bp) != rec["blueprint_hash"]:
        return [Issue("replay", rid, f"blueprint {rec['blueprint']} hash differs from the record")]
    re = run_record(bp, rec["household"], rec["labels"]["novelty"], rec["agent"], rec["seed"], rec["max_steps"],
                    replay=rec["steps"])
    out: list[Issue] = []
    for i, (a, b) in enumerate(zip(rec["steps"], re["steps"])):
        if a != b:
            diff = sorted(k for k in a if a[k] != b.get(k))
            return [Issue("replay", rid, f"step {i} differs on replay: {diff}")]
    for k in ("outcome", "n_steps", "final", "labels", "efficiency"):
        if rec[k] != re[k]:
            out.append(Issue("replay", rid, f"{k} differs on replay: {rec[k]!r} vs {re[k]!r}"))
    if not math.isclose(rec["return"], re["return"], abs_tol=1e-9):
        out.append(Issue("replay", rid, f"return differs on replay: {rec['return']} vs {re['return']}"))
    return out


def validate(manifest: dict, splits: dict[str, list[dict]], replay: bool = True) -> list[Issue]:
    issues: list[Issue] = []
    hashes: dict[str, str] = {}
    groups: dict[tuple, set[str]] = {}
    n = 0
    for split, recs in splits.items():
        for rec in recs:
            n += 1
            rid = str(rec.get("id", f"{split}#{n}"))
            bad = check_schema(rec)
            if bad:
                issues += [Issue("leak" if "unexpected field" in b and "step" in b else "schema", rid, b) for b in bad]
                continue
            if rec["split"] != split:
                issues.append(Issue("split_leakage", rid, f"labelled {rec['split']} but stored in {split}"))
            ref = finish(dict(rec))
            if rec["split"] != ref["split"]:  # the split is a pure function of the group, so builds never disagree
                issues.append(Issue("split_leakage", rid, f"labelled {rec['split']}, but its (blueprint, seed) group belongs to {ref['split']}"))
            if rec["id"] != ref["id"] or rec["content_hash"] != ref["content_hash"]:
                issues.append(Issue("schema", rid, "id/content_hash do not match the record"))
            if rec["content_hash"] in hashes:
                issues.append(Issue("duplicate", rid, f"same house+household+novelty+seed+actions as {hashes[rec['content_hash']]}"))
            hashes.setdefault(rec["content_hash"], rid)
            groups.setdefault((rec["blueprint_hash"], rec["seed"]), set()).add(split)
            issues += check_static(rec) + check_leaks(rec)
            if replay:
                issues += check_replay(rec)
    issues += [Issue("split_leakage", f"{g[0]}/seed {g[1]}", f"group in {sorted(s)}") for g, s in groups.items() if len(s) > 1]
    if n != manifest.get("records") or any(len(splits[s]) != manifest["by_split"].get(s) for s in SPLITS):
        issues.append(Issue("manifest", "-", f"{n} records on disk, manifest says {manifest.get('records')} {manifest.get('by_split')}"))
    return issues


# ------------------------------------------------------------------ statistics
def stats(splits: dict[str, list[dict]], issues: list[Issue] = ()) -> str:
    """Markdown balance report: counts by tier/novelty/household/agent/outcome/split, returns, quality."""
    recs = [r for rs in splits.values() for r in rs]
    flagged = {i.record for i in issues}
    lines = [f"# Dataset statistics\n\n{len(recs)} episodes, {sum(r['n_steps'] for r in recs)} steps, "
             f"{sum(r['outcome'] != 'success' for r in recs)} not fully successful, {len(flagged)} flagged by validators.\n"]
    for name, key in (("split", lambda r: r["split"]), ("tier", lambda r: r["tier"]), ("novelty", lambda r: r["labels"]["novelty"]),
                      ("household", lambda r: r["household"]), ("agent", lambda r: r["agent"]), ("blueprint kind", lambda r: r["blueprint"].split(":")[0])):
        c = Counter(key(r) for r in recs)
        lines.append(f"**{name}:** " + ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=str)) + "  ")
    oc = Counter((r["agent"], r["outcome"]) for r in recs)
    lines.append("\n| agent | n | " + " | ".join(OUTCOMES) + " | return p10 / p50 / p90 | mean quality |\n|---|---|"
                 + "---|" * (len(OUTCOMES) + 2))
    for a in sorted({r["agent"] for r in recs}):
        rs = [r for r in recs if r["agent"] == a]
        q = statistics.quantiles([r["return"] for r in rs], n=10) if len(rs) > 1 else [rs[0]["return"]] * 9
        lines.append(f"| {a} | {len(rs)} | " + " | ".join(str(oc[a, o]) for o in OUTCOMES)
                     + f" | {q[0]:.2f} / {q[4]:.2f} / {q[8]:.2f} | {statistics.fmean(r['quality'] for r in rs):.2f} |")
    kinds = sorted({r["blueprint"].split(":")[0] for r in recs})
    have = {(r["blueprint"].split(":")[0], r["household"], r["labels"]["novelty"]) for r in recs}
    missing = [f"{k}/{h}/{v}" for k in kinds for h in HOUSEHOLD_NAMES for v in NOVELTY_NAMES if (k, h, v) not in have]
    lines.append(f"\n**coverage gaps** (house kind/household/novelty; inapplicable presets are listed in the manifest): "
                 f"{', '.join(missing) or 'none'}")
    qs = Counter(min(int(r["quality"] * 5), 4) for r in recs)
    lines.append("\n**quality histogram** (0-0.2 ... 0.8-1): " + ", ".join(str(qs[b]) for b in range(5)))
    return "\n".join(lines) + "\n"
