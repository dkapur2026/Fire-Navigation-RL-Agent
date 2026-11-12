"""Training-progress GIFs: one whole held-out episode per checkpoint (0%, 10%, ..., 100% of training).

    python scripts/make_training_gifs.py                                  # bcppo + dreamer_il, seed-0 manifests
    python scripts/make_training_gifs.py --methods bcppo
    python scripts/make_training_gifs.py --manifest path/to/manifest.json --out /tmp/gifs
    python scripts/make_training_gifs.py --method bcppo --checkpoints 0:untrained=a.pt 0:bc_clone=b.pt 100=c.pt@602112
    python scripts/make_gifs.py --training ...                            # the same, through make_gifs.py
    make training-gifs

Input: a manifest (runs/formal_eval/train/<method>_s<seed>/manifest.json by default) listing
{"method", "seed", "total_steps", "points": [{"pct", "label", "path", "step"}, ...]}, or an explicit checkpoint list
(PCT[:LABEL]=PATH[@STEP]). Output: <out>/<method>/<blueprint>_<profile>_<novelty>_seed<N>/<method>_p<pct>_<label>.gif
with an index.md there (one row per checkpoint), and <out>/<method>/index.md listing the sets.

Every checkpoint plays the SAME fire (one held-out seed, 1000..1099) in the same setting, so the GIFs of one set
differ only in the policy. Each row also carries an aggregate: greedy success rate over the fixed held-out seeds
1000..1000+K-1 (``--aggregate-seeds K``, default 20) in that setting, so the trend is not read off one episode.

Two ways to fix the fire, both decided before looking at any episode:
* ``--seed N`` (default 1000): ranch / single (the dog) / hidden_dependent, the setting where BC+PPO's held-out score
  rose at every checkpoint (docs/RL.md); seed 1000 is simply the first held-out seed.
* ``--seed first-clone-failure``: the first held-out seed >= 1000 in the first of ``--select-settings`` (default
  kids/door_jam, then kids/hidden_dependent) where the after-imitation ("bc_clone") checkpoint does NOT get everyone
  out. This asks whether PPO fine-tuning fixes a case the imitation clone fails; whatever happens is shown. Learned agents are loaded only through the
evidence-disciplined wrappers (``PPOAgent`` / ``DreamerAgent`` via their factories): they see Observations and
Events, never the world.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = ("bcppo", "dreamer_il")
TITLES = {"bcppo": "PPO+imitation", "dreamer_il": "DreamerV3+imitation"}
HELD_OUT = range(1000, 1100)


@dataclass
class Point:
    pct: int
    label: str
    path: str
    step: int | None = None


def manifest_path(method: str, train_seed: int) -> Path:
    return ROOT / "runs" / "formal_eval" / "train" / f"{method}_s{train_seed}" / "manifest.json"


def read_manifest(path: Path) -> tuple[str, list[Point], dict]:
    m = json.loads(Path(path).read_text())
    pts = [Point(int(p["pct"]), str(p.get("label") or f"{int(p['pct'])}%"), str(p["path"]), p.get("step"))
           for p in m["points"]]
    return m["method"], pts, m


def parse_checkpoint(spec: str) -> Point:
    """PCT[:LABEL]=PATH[@STEP]."""
    head, sep, tail = spec.partition("=")
    if not sep:
        raise SystemExit(f"--checkpoints entry {spec!r}: expected PCT[:LABEL]=PATH[@STEP]")
    pct, _, label = head.partition(":")
    path, _, step = tail.rpartition("@") if "@" in tail else (tail, "", "")
    return Point(int(pct), label or f"{int(pct)}%", path, int(step) if step else None)


def resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return ROOT / p


def progress_label(method: str, pt: Point) -> str:
    """What the banner and index say about where in training this checkpoint is."""
    if pt.label == "untrained":
        return "0% of training (untrained: random init)"
    if pt.label == "bc_clone":
        after = "before PPO" if method.startswith(("bcppo", "ppo")) else "before RL"
        return f"0% of training (after imitation, {after})"
    return f"{pt.pct}% of training"


def slug(pt: Point) -> str:
    tail = re.sub(r"[^A-Za-z0-9]+", "_", pt.label.replace("%", "pct")).strip("_")
    return f"p{pt.pct:03d}_{tail}" if tail else f"p{pt.pct:03d}"


_CACHE: dict = {}


def _load(path: Path):
    """("dreamer" | "ppo", model), loaded once per path."""
    import torch

    key = str(path)
    if key not in _CACHE:
        meta = torch.load(path, map_location="cpu", weights_only=False)
        if "deter" in meta.get("config", {}):  # a Dreamer checkpoint
            from emberpath.dreamer import load_dreamer

            _CACHE[key] = ("dreamer", load_dreamer(path))
        else:
            from emberpath.rl_baseline import load_policy

            _CACHE[key] = ("ppo", load_policy(path)[0])
    return _CACHE[key]


def factory(path: Path, greedy: bool, max_steps: int, seed: int):
    """An agent factory over the checkpoint, through the evidence-disciplined wrappers (never handed the world)."""
    kind, model = _load(path)
    if kind == "dreamer":
        from emberpath.dreamer import dreamer_factory

        return dreamer_factory(model, greedy=greedy, max_steps=max_steps, seed=seed)
    from emberpath.rl_baseline import ppo_factory

    return ppo_factory(model, greedy=greedy, max_steps=max_steps, seed=seed)


@dataclass
class Setting:
    blueprint: str
    profile: str
    novelty: str
    seed: int
    rule: str = ""  # how the seed was chosen, written into the index

    @property
    def name(self) -> str:
        return f"{self.blueprint}_{self.profile}_{self.novelty}_seed{self.seed}"


def episode(ckpt: Path, st: Setting, seed: int, a: argparse.Namespace, greedy: bool, record=None):
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.evaluate import run_episode

    agent_seed = a.agent_seed if a.agent_seed is not None else seed
    return run_episode(load_blueprint(st.blueprint), HOUSEHOLDS[st.profile], factory(ckpt, greedy, a.max_steps,
                       agent_seed), None if st.novelty == "none" else st.novelty, seed, a.max_steps, record=record)


def aggregate(ckpt: Path, st: Setting, a: argparse.Namespace) -> dict | None:
    """Greedy success rate and mean score over the fixed held-out seeds 1000..1000+K-1 in this setting."""
    if a.aggregate_seeds <= 0:
        return None
    res = [episode(ckpt, st, s, a, greedy=True) for s in range(1000, 1000 + a.aggregate_seeds)]
    return {"n": len(res), "success": sum(r.outcome == "success" for r in res),
            "score": sum(r.score for r in res) / len(res)}


def play(method: str, pt: Point, st: Setting, a: argparse.Namespace, out_dir: Path, keep: list | None = None) -> dict:
    """Play the episode once to learn its outcome, then again (deterministically, same fire and agent seed) while
    recording frames, so every frame's banner can carry the final result."""
    from emberpath.cli import _banner
    from emberpath.render import render_frame, save_gif

    ckpt = resolve(pt.path)
    greedy = a.policy == "greedy"
    res = episode(ckpt, st, st.seed, a, greedy)
    prog = progress_label(method, pt)
    head = f"{TITLES.get(method, method)} - {prog} - {res.rescued}/{res.total} rescued ({res.outcome})"
    frames: list = []

    def record(world, decision) -> None:
        extra = _banner(world)
        frames.append(render_frame(world, caption=decision.reason if decision else "Start",
                                   banner=f"{head}  {extra}" if extra else head))

    again = episode(ckpt, st, st.seed, a, greedy, record=record)
    if (again.outcome, again.rescued, again.steps) != (res.outcome, res.rescued, res.steps):
        raise RuntimeError(f"{ckpt}: replay differed from the first run ({res} vs {again}); not deterministic")
    name = f"{method}_{slug(pt)}.gif"
    save_gif(frames, out_dir / name)
    if keep is not None:  # for the combined start-to-finish GIF: shrunk and palettised now, to bound memory
        keep.append((head, [_small(f, a.combined_scale) for f in frames[::max(1, a.combined_stride)]]))
    return {"pct": pt.pct, "label": pt.label, "progress": prog, "step": pt.step, "path": pt.path,
            "outcome": res.outcome, "rescued": res.rescued, "total": res.total, "success": res.outcome == "success",
            "score": res.score, "steps": res.steps, "gif": name, "agg": aggregate(ckpt, st, a)}


def _small(frame, scale: float):
    from PIL import Image

    if scale != 1.0:
        frame = frame.resize((max(1, round(frame.width * scale)), max(1, round(frame.height * scale))), Image.LANCZOS)
    return frame.convert("P", palette=Image.ADAPTIVE)


def title_card(text: str, size: tuple[int, int]):
    """A plain card announcing the next checkpoint, the same size as the episode frames."""
    import textwrap

    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, (24, 24, 28))
    d = ImageDraw.Draw(img)
    lines = textwrap.wrap(text, width=max(20, size[0] // 8))
    y = size[1] // 2 - 8 * len(lines)
    for line in lines:
        d.text((size[0] // 2, y), line, fill=(235, 235, 235), anchor="ma")
        y += 16
    return img.convert("P", palette=Image.ADAPTIVE)


def save_combined(parts: list, path: Path, title: str, ms_per_frame: int = 120, card_ms: int = 1600) -> None:
    """Every checkpoint's episode back to back, untrained -> fully trained, each preceded by a title card."""
    size = parts[0][1][0].size
    frames, durations = [title_card(title, size)], [card_ms + 400]
    for head, eps in parts:
        frames.append(title_card(head, size))
        durations.append(card_ms)
        frames += eps
        durations += [ms_per_frame] * (len(eps) - 1) + [900]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def _rel(path) -> str:
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


FIXED_RULE = ("Seed fixed in advance (the first held-out seed), not chosen by how the episodes look.")
SELECT_RULE = ("Seed chosen by a rule stated before any episode was looked at: the first held-out seed >= 1000, in "
               "setting {settings}, where the after-imitation 0% checkpoint (bc_clone) does NOT get everyone out. "
               "{detail}")


def write_index(method: str, rows: list[dict], st: Setting, a: argparse.Namespace, out_dir: Path, source: str,
                total_steps) -> None:
    title = TITLES.get(method, method)
    k = a.aggregate_seeds
    lines = [f"# Training progress: {title} (`{method}`), {st.blueprint} / {st.profile} / {st.novelty}, "
             f"seed {st.seed}", "",
             f"Setting: blueprint `{st.blueprint}`, household `{st.profile}`, novelty `{st.novelty}`, held-out fire "
             f"seed {st.seed}, max {a.max_steps} steps, {a.policy} actions. The same fire at every checkpoint.", "",
             f"**How the seed was chosen:** {st.rule}", "",
             "One episode per checkpoint is an illustration, not a measurement."
             + (f" The aggregate column is greedy success over the {k} fixed held-out seeds 1000..{999 + k} in this "
                "setting (and the mean score), so the trend is not read off one episode; formal-eval's held-out "
                "evaluation is the authoritative number." if k > 0 else ""), "",
             f"Checkpoints from: `{source}`" + (f" (total training steps: {total_steps:,})"
                                                 if isinstance(total_steps, int) else ""), "",
             "| pct | label | step | checkpoint | outcome | rescued | success | score | episode length | "
             + (f"success, seeds 1000..{999 + k} (mean score) | " if k > 0 else "") + "GIF |",
             "|---|---|---|---|---|---|---|---|---|" + ("---|" if k > 0 else "") + "---|"]
    for r in rows:
        step = f"{r['step']:,}" if isinstance(r["step"], int) else "-"
        agg = r["agg"]
        aggc = f"{agg['success']}/{agg['n']} = {agg['success'] / agg['n']:.0%} ({agg['score']:.2f}) | " if agg else ""
        lines.append(f"| {r['pct']}% | {r['progress']} | {step} | `{r['path']}` | {r['outcome']} | "
                     f"{r['rescued']}/{r['total']} | {'yes' if r['success'] else 'no'} | {r['score']:.2f} | "
                     f"{r['steps']} | {aggc}[{r['gif']}]({r['gif']}) |")
    regen = ["python", "scripts/make_training_gifs.py", "--blueprint", st.blueprint, "--max-steps", str(a.max_steps),
             "--policy", a.policy, "--aggregate-seeds", str(k), "--out", _rel(a.out)]
    if a.seed == AUTO:
        regen += ["--seed", AUTO, "--select-settings", *a.select_settings]
    else:
        regen += ["--profile", st.profile, "--novelty", st.novelty, "--seed", str(st.seed)]
    regen += ["--manifest", source] if source.endswith(".json") else ["--method", method, "--checkpoints", "..."]
    lines += ["", "Regenerate: `" + " ".join(shlex.quote(x) for x in regen) + "`"]
    (out_dir / "index.md").write_text("\n".join(lines) + "\n")


def write_method_index(method: str, out: Path) -> None:
    """<out>/<method>/index.md: one line per set (subfolder), with its seed rule."""
    lines = [f"# Training progress: {TITLES.get(method, method)} (`{method}`)", "",
             "Each folder plays one fixed held-out fire at every checkpoint (0%, 10%, ..., 100% of training).", ""]
    for d in sorted(p for p in (out / method).iterdir() if (p / "index.md").exists()):
        rule = next((ln.removeprefix("**How the seed was chosen:** ") for ln in (d / "index.md").read_text()
                     .splitlines() if ln.startswith("**How the seed was chosen:**")), "")
        lines.append(f"* [{d.name}]({d.name}/index.md): {rule}")
    (out / method / "index.md").write_text("\n".join(lines) + "\n")


def select_setting(method: str, points: list[Point], a: argparse.Namespace) -> Setting:
    """The first held-out seed, in the first listed setting that has one, where bc_clone does not get everyone out."""
    clone = next((p for p in points if p.label == "bc_clone"), None)
    if clone is None:
        raise SystemExit(f"{method}: --seed {AUTO} needs a bc_clone checkpoint")
    tried = []
    for spec in a.select_settings:
        profile, novelty = spec.split("/")
        for seed in HELD_OUT:
            st = Setting(a.blueprint, profile, novelty, seed)
            r = episode(resolve(clone.path), st, seed, a, a.policy == "greedy")
            if r.rescued < r.total:
                detail = (f"Result: {spec}, seed {seed} (the clone: {r.rescued}/{r.total} rescued, {r.outcome}); "
                          + (f"no seed in 1000..1099 qualified in {', '.join(tried)}; " if tried else "")
                          + f"{seed - 999} seed(s) checked in {spec}.")
                st.rule = SELECT_RULE.format(settings=" then ".join(a.select_settings), detail=detail)
                return st
        tried.append(spec)
    raise SystemExit(f"{method}: the clone gets everyone out on every held-out seed in {a.select_settings}")


def build(method: str, points: list[Point], a: argparse.Namespace, source: str, total_steps=None) -> list[dict]:
    if a.seed == AUTO:
        st = select_setting(method, points, a)
    else:
        st = Setting(a.blueprint, a.profile, a.novelty, int(a.seed), FIXED_RULE)
    out_dir = Path(a.out) / method / st.name
    out_dir.mkdir(parents=True, exist_ok=True)
    points = sorted(points, key=lambda p: (p.pct, p.label != "untrained"))  # untrained before bc_clone at 0%
    rows, parts = [], ([] if a.combined else None)
    for pt in points:
        r = play(method, pt, st, a, out_dir, keep=parts)
        agg = f"  agg {r['agg']['success']}/{r['agg']['n']}" if r["agg"] else ""
        print(f"{method:<12} {st.name:<38} {r['progress']:<45} {r['outcome']:<10} rescued {r['rescued']}/"
              f"{r['total']} score {r['score']:.2f} steps {r['steps']:>3}{agg}", flush=True)
        rows.append(r)
    if parts:
        name = f"{method}_training_start_to_finish.gif"
        save_combined(parts, out_dir / name, f"{TITLES.get(method, method)} on {st.blueprint} / {st.profile} / "
                      f"{st.novelty}, held-out fire {st.seed}: the same fire at every stage of training")
        print(f"combined: {out_dir / name}", flush=True)
    write_index(method, rows, st, a, out_dir, source, total_steps)
    if parts:
        with open(out_dir / "index.md", "a") as f:
            f.write(f"\nWhole training in one GIF (every checkpoint above, back to back): "
                    f"[{name}]({name})\n")
    write_method_index(method, Path(a.out))
    return rows


AUTO = "first-clone-failure"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                    help="read runs/formal_eval/train/<method>_s<train-seed>/manifest.json for each")
    ap.add_argument("--train-seed", type=int, default=0, help="training seed of the manifests to read")
    ap.add_argument("--manifest", nargs="+", default=None, help="explicit manifest file(s); overrides --methods")
    ap.add_argument("--method", default=None, help="method name for --checkpoints (bcppo, dreamer_il, ...)")
    ap.add_argument("--checkpoints", nargs="+", default=None, metavar="PCT[:LABEL]=PATH[@STEP]",
                    help="explicit checkpoint list instead of a manifest (needs --method)")
    ap.add_argument("--blueprint", default="ranch")
    ap.add_argument("--profile", default="single")
    ap.add_argument("--novelty", default="hidden_dependent")
    ap.add_argument("--seed", default="1000",
                    help=f"held-out fire seed (1000..1099), the same for every GIF; or {AUTO!r} (see above)")
    ap.add_argument("--select-settings", nargs="+", default=["kids/door_jam", "kids/hidden_dependent"],
                    metavar="PROFILE/NOVELTY", help=f"settings tried in order by --seed {AUTO}")
    ap.add_argument("--aggregate-seeds", type=int, default=20,
                    help="per checkpoint, greedy success over held-out seeds 1000..1000+K-1 (0 = off)")
    ap.add_argument("--agent-seed", type=int, default=None, help="seed of the agent's own sampling (default: --seed)")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--policy", choices=["greedy", "sampled"], default="greedy")
    ap.add_argument("--threads", type=int, default=1, help="torch threads")
    ap.add_argument("--out", default=str(ROOT / "gifs" / "training"))
    ap.add_argument("--combined", action="store_true",
                    help="also write <method>_training_start_to_finish.gif: every checkpoint's episode back to back")
    ap.add_argument("--combined-scale", type=float, default=0.6, help="frame scale in the combined GIF")
    ap.add_argument("--combined-stride", type=int, default=1, help="keep every Nth frame in the combined GIF")
    return ap


def _check(blueprint: str, profile: str, novelty: str) -> None:
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.scenarios import NOVELTIES, applicable

    if profile not in HOUSEHOLDS or (novelty != "none" and novelty not in NOVELTIES):
        raise SystemExit(f"unknown household {profile!r} or novelty {novelty!r}")
    if not applicable(load_blueprint(blueprint), HOUSEHOLDS[profile], None if novelty == "none" else novelty):
        raise SystemExit(f"novelty {novelty!r} does not apply to {blueprint} / {profile}")


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    if a.seed == AUTO:
        for spec in a.select_settings:
            if spec.count("/") != 1:
                raise SystemExit(f"--select-settings entry {spec!r}: expected PROFILE/NOVELTY")
            _check(a.blueprint, *spec.split("/"))
    else:
        if not a.seed.isdigit() or int(a.seed) not in HELD_OUT:
            raise SystemExit(f"--seed must be a held-out fire seed in 1000..1099 or {AUTO!r} (got {a.seed}); "
                             "0..999 are training")
        _check(a.blueprint, a.profile, a.novelty)
    if not 0 <= a.aggregate_seeds <= len(HELD_OUT):
        raise SystemExit("--aggregate-seeds must be in 0..100")
    import torch

    torch.set_num_threads(a.threads)

    jobs: list[tuple[str, list[Point], str, object]] = []
    missing: list[str] = []
    if a.checkpoints:
        if not a.method:
            raise SystemExit("--checkpoints needs --method")
        jobs.append((a.method, [parse_checkpoint(s) for s in a.checkpoints], "explicit checkpoint list", None))
    else:
        paths = [Path(p) for p in a.manifest] if a.manifest else [manifest_path(m, a.train_seed) for m in a.methods]
        for p in paths:
            if not p.exists():
                missing.append(str(p))
                continue
            method, pts, meta = read_manifest(p)
            jobs.append((method, pts, _rel(p), meta.get("total_steps")))
    for method, pts, src, total in jobs:
        build(method, pts, a, src, total)
    for p in missing:
        print(f"PENDING: manifest not found: {p}", file=sys.stderr)
    return 0 if jobs else 1


if __name__ == "__main__":
    sys.exit(main())
