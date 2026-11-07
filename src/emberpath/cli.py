"""Command line: `emberpath run | eval | show | list`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emberpath.blueprint import Blueprint
from emberpath.dependents import HOUSEHOLDS
from emberpath.evaluate import evaluate, gap_closed, run_episode, summarize, write_csv
from emberpath.render import render_ascii, render_frame, save_gif
from emberpath.scenarios import NOVELTIES
from emberpath.world import World

BLUEPRINT_DIRS = [Path.cwd() / "blueprints", Path(__file__).resolve().parents[2] / "blueprints"]


def load_blueprint(name_or_path: str) -> Blueprint:
    p = Path(name_or_path)
    if p.exists():
        return Blueprint.from_file(p)
    for d in BLUEPRINT_DIRS:
        cand = d / f"{name_or_path}.txt"
        if cand.exists():
            return Blueprint.from_file(cand)
    raise SystemExit(f"blueprint {name_or_path!r} not found (looked in {[str(d) for d in BLUEPRINT_DIRS]})")


def cmd_list(_: argparse.Namespace) -> int:
    print("households:")
    for h in HOUSEHOLDS.values():
        print(f"  {h.label:<8} {h.description}")
    print("novelties:  " + ", ".join(NOVELTIES))
    for d in BLUEPRINT_DIRS:
        if d.exists():
            print("blueprints: " + ", ".join(sorted(p.stem for p in d.glob("*.txt"))))
            break
    return 0


def cmd_show(a: argparse.Namespace) -> int:
    bp = load_blueprint(a.blueprint)
    w = World(bp, HOUSEHOLDS[a.profile])
    print(render_ascii(w))
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    bp, hh = load_blueprint(a.blueprint), HOUSEHOLDS[a.profile]
    frames: list = []
    lines: list[str] = []

    def record(world: World, decision) -> None:
        if a.gif:
            cap = decision.reason if decision else "Start"
            frames.append(render_frame(world, caption=cap, banner=_banner(world)))
        if decision and not a.quiet:
            lines.append(f"t={world.t:>3}  {decision.reason}")
        if a.ascii and decision:
            print(f"--- t={world.t} ---\n{render_ascii(world)}")

    try:
        res = run_episode(bp, hh, a.agent, a.novelty, a.seed, a.max_steps, record=record)
    except NotImplementedError as exc:
        print(f"\nThe '{a.agent}' agent depends on a feature that isn't built yet:\n  {exc}\nSee docs/ROADMAP.md.")
        return 2

    last = None
    for line in lines:  # collapse repeated reasons for readability
        body = line.split("  ", 1)[1]
        if body != last:
            print(line)
        last = body
    print(f"\n{hh.label} / {a.agent} / novelty={a.novelty or 'none'} / seed={a.seed}")
    print(f"outcome: {res.outcome.upper()}   rescued {res.rescued}/{res.total}   steps {res.steps}   score {res.score:.2f}")
    if a.gif:
        save_gif(frames, a.gif)
        print(f"wrote {a.gif}")
    return 0


def _banner(world: World) -> str:
    if world.novelty_log:
        step, patch = world.novelty_log[-1]
        from emberpath.explain import describe_patch

        return f"[truth changed at t={step}: {describe_patch(patch)}]"
    return ""


def cmd_eval(a: argparse.Namespace) -> int:
    bp, hh = load_blueprint(a.blueprint), HOUSEHOLDS[a.profile]
    try:
        results = evaluate(bp, hh, a.agents, a.novelty, range(a.seeds), a.max_steps)
    except NotImplementedError as exc:
        print(f"An agent depends on a feature that isn't built yet: {exc}")
        return 2
    summ = summarize(results)
    print(f"{a.blueprint} / {a.profile} / novelty={a.novelty or 'none'} / {a.seeds} seeds\n")
    print(f"{'agent':<10}{'success':>9}{'score':>8}{'return':>9}{'effic.':>8}{'steps':>8}{'died':>7}{'timeout':>9}")
    for agent, s in summ.items():
        eff = "n/a" if s["mean_efficiency"] != s["mean_efficiency"] else f"{s['mean_efficiency']:.2f}"
        print(
            f"{agent:<10}{s['success_rate']:>9.0%}{s['mean_score']:>8.2f}{s['mean_return']:>9.2f}{eff:>8}"
            f"{s['mean_steps']:>8.1f}{s['agent_lost_rate']:>7.0%}{s['timeout_rate']:>9.0%}"
        )
    g = gap_closed(summ)
    if g is not None:
        print(f"\ngap closed by adaptive agent: {g:.0%}")
    if a.csv:
        write_csv(results, a.csv)
        print(f"wrote {a.csv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="emberpath", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--blueprint", default="ranch")
        sp.add_argument("--profile", default="single", choices=list(HOUSEHOLDS))
        sp.add_argument("--novelty", default=None, choices=list(NOVELTIES))
        sp.add_argument("--max-steps", type=int, default=150)

    r = sub.add_parser("run", help="run one episode")
    common(r)
    r.add_argument("--agent", default="baseline", choices=["baseline", "oracle", "adaptive"])
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--gif", default=None, help="write an animated GIF here")
    r.add_argument("--ascii", action="store_true", help="print every frame as text")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("eval", help="compare agents over many seeds")
    common(e)
    e.add_argument("--agents", nargs="+", default=["baseline", "oracle"])
    e.add_argument("--seeds", type=int, default=30)
    e.add_argument("--csv", default=None)
    e.set_defaults(fn=cmd_eval)

    s = sub.add_parser("show", help="print a blueprint")
    s.add_argument("--blueprint", default="ranch")
    s.add_argument("--profile", default="kids", choices=list(HOUSEHOLDS))
    s.set_defaults(fn=cmd_show)

    l = sub.add_parser("list", help="list households, novelties, blueprints")
    l.set_defaults(fn=cmd_list)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
