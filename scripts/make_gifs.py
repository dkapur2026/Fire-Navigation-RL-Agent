"""Write a GIF for every household x novelty x agent on one house (skipping combinations the house cannot host).

    python scripts/make_gifs.py                                   # house, all households/novelties, 3 agents
    python scripts/make_gifs.py --blueprint ranch --seed 4
    python scripts/make_gifs.py --profiles kids family --novelties door_jam --agents baseline adaptive
    python scripts/make_gifs.py --training [options]               # training-progress GIFs: make_training_gifs.py

Files: <out>/<blueprint>/<household>_<novelty>_<agent>_seed<seed>.gif, plus index.md listing each outcome.
Every GIF of one (household, novelty, seed) faces the same fire, so they can be compared side by side.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from emberpath.agents import AGENTS
from emberpath.cli import load_blueprint, main as cli
from emberpath.dependents import HOUSEHOLDS
from emberpath.scenarios import NOVELTIES, applicable


def one(job: tuple[str, str, str, str, int, Path]) -> tuple[str, str]:
    blueprint, profile, novelty, agent, seed, path = job
    args = ["run", "--blueprint", blueprint, "--profile", profile, "--agent", agent, "--seed", str(seed),
            "--gif", str(path), "--quiet"]
    if novelty != "none":
        args += ["--novelty", novelty]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli(args)
    outcome = next(line for line in out.getvalue().splitlines() if line.startswith("outcome:"))
    return path.name, outcome.removeprefix("outcome:").strip()


def _training(argv: list[str]) -> int:
    """Hand everything after --training to scripts/make_training_gifs.py (one GIF per training checkpoint)."""
    import importlib.util

    path = Path(__file__).with_name("make_training_gifs.py")
    spec = importlib.util.spec_from_file_location("make_training_gifs", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod.main(argv)


def main() -> None:
    if "--training" in sys.argv[1:]:
        sys.exit(_training([x for x in sys.argv[1:] if x != "--training"]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blueprint", default="house")
    ap.add_argument("--profiles", nargs="+", default=list(HOUSEHOLDS), choices=list(HOUSEHOLDS))
    ap.add_argument("--novelties", nargs="+", default=["none", *NOVELTIES], choices=["none", *NOVELTIES])
    ap.add_argument("--agents", nargs="+", default=["baseline", "oracle", "adaptive"], choices=list(AGENTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gifs")
    ap.add_argument("--training", action="store_true",
                    help="instead: training-progress GIFs; the other options are make_training_gifs.py's (see its --help)")
    a = ap.parse_args()

    bp = load_blueprint(a.blueprint)
    out = Path(a.out) / a.blueprint
    out.mkdir(parents=True, exist_ok=True)
    jobs, skipped = [], []
    for p in a.profiles:
        for n in a.novelties:
            if not applicable(bp, HOUSEHOLDS[p], None if n == "none" else n):
                skipped.append(f"{p} / {n}")
                continue
            jobs += [(a.blueprint, p, n, ag, a.seed, out / f"{p}_{n}_{ag}_seed{a.seed}.gif") for ag in a.agents]
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        results = list(pool.map(one, jobs))

    lines = [f"# GIFs: {a.blueprint}, seed {a.seed}", "", "| file | outcome |", "|---|---|"]
    lines += [f"| [{name}]({name}) | {outcome} |" for name, outcome in results]
    if skipped:
        lines += ["", "Not applicable to this house (too few slots): " + ", ".join(skipped)]
    (out / "index.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {len(results)} GIFs to {out}/")


if __name__ == "__main__":
    main()
