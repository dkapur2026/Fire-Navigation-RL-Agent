"""Detector false-alarm rate and detection delay against threshold, on real ranch rollouts.

    python scripts/false_alarms.py             # 30 seeds x every household
    python scripts/false_alarms.py --seeds 10  # quicker

Each episode is driven by BaselineAgent (it never changes its belief, so the event stream does not depend on
the detector) and recorded once; the per-tick event batches are then replayed into a fresh detector for each
(threshold, decay) setting, peeking at ``findings()`` after every tick exactly like an agent does.

* quiet worlds (no novelty): false alarm = any finding at any tick of the episode.
* novel worlds: delay = first flagged tick of the changed rule minus the tick it changed; "missed" if the
  rule is never flagged; "other" = episodes where some unchanged rule was flagged too.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from emberpath.agents import BaselineAgent
from emberpath.blueprint import Blueprint
from emberpath.dependents import HOUSEHOLDS
from emberpath.novelty import BetaBernoulliDetector
from emberpath.rules import nominal_rules, parse_key
from emberpath.scenarios import NOVELTIES
from emberpath.world import Event, World

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
DECAYS = (1.0, 0.9)
DEFAULT = BetaBernoulliDetector(nominal_rules())


def rollout(bp: Blueprint, household: str, novelty: str | None, seed: int) -> tuple[list[list[Event]], dict]:
    """Per-tick event batches of one baseline episode, and {changed key: tick it changed}."""
    hh = HOUSEHOLDS[household]
    specs = NOVELTIES[novelty](bp, hh) if novelty else []
    world = World(bp, hh, seed=seed, novelties=specs)
    agent = BaselineAgent(bp, world)
    obs, events, batches = world.observe(), [], []
    while not world.done:
        res = world.step(agent.act(obs, events).action)
        obs, events = res.obs, res.events
        batches.append(events)
    return batches, {s.patch.key: s.step for s in specs}


def first_flags(batches: list[list[Event]], threshold: float, decay: float) -> dict[str, int]:
    det = BetaBernoulliDetector(nominal_rules(), threshold=threshold, decay=decay)
    first: dict[str, int] = {}
    for batch in batches:
        det.observe(batch)
        for f in det.findings():
            first.setdefault(f.key, f.first_flagged_step)
    return first


def _median(xs: list[int]) -> str:
    return f"{statistics.median(xs):.0f}" if xs else "-"


def _first_trial(batches: list[list[Event]], key: str, since: int) -> int | None:
    return next((e.step - since for b in batches for e in b if e.key == key and e.step >= since), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    bp = Blueprint.from_file(ROOT / "blueprints" / "ranch.txt")
    t0 = time.perf_counter()
    runs = {
        nov: [rollout(bp, hh, nov, s) for hh in HOUSEHOLDS for s in range(args.seeds)]
        for nov in (None, *NOVELTIES)
    }
    sim = time.perf_counter() - t0
    n_ep = len(runs[None])
    print(f"ranch, BaselineAgent, {n_ep} episodes per scenario ({len(HOUSEHOLDS)} households x {args.seeds} seeds)")
    print(f"default detector: threshold={DEFAULT.threshold}, min_trials={DEFAULT.min_trials}, decay={DEFAULT.decay}\n")

    print("False alarms on quiet worlds (any finding at any tick, peeking every tick)")
    print(f"{'threshold':>9} " + " ".join(f"{'decay ' + str(d):>11}" for d in DECAYS))
    for th in THRESHOLDS:
        rates = [sum(bool(first_flags(b, th, d)) for b, _ in runs[None]) / n_ep for d in DECAYS]
        print(f"{th:>9.1f} " + " ".join(f"{r:>10.1%} " for r in rates))

    print("\nDetection on novel worlds: median ticks from the change to the flag (missed %)")
    print("'1st trial' = median ticks until the changed rule is tested at all (a floor no detector can beat);")
    print(f"'other' = episodes where an unchanged rule was also flagged, at threshold {DEFAULT.threshold}")
    for d in DECAYS:
        print(f"decay {d}\n  {'novelty / rule':<30}{'1st trial':>9} " + "".join(f"{'th=' + str(th):>12}" for th in THRESHOLDS) + f"{'other':>7}")
        for nov in NOVELTIES:
            flags = {th: [first_flags(b, th, d) for b, _ in runs[nov]] for th in THRESHOLDS}
            for family in sorted({parse_key(k)[0] for _, ch in runs[nov] for k in ch}):
                rows = [(i, k, step) for i, (_, ch) in enumerate(runs[nov]) for k, step in ch.items() if parse_key(k)[0] == family]
                tested = [t for i, k, step in rows if (t := _first_trial(runs[nov][i][0], k, step)) is not None]
                line = f"  {nov + ' / ' + family:<30}{_median(tested):>9} "
                for th in THRESHOLDS:
                    delays = [flags[th][i][k] - step for i, k, step in rows if k in flags[th][i]]
                    line += f"{_median(delays):>5} ({1 - len(delays) / len(rows):>4.0%})"
                other = sum(any(k not in ch for k in f) for f, (_, ch) in zip(flags[DEFAULT.threshold], runs[nov]))
                print(line + f"{other / len(runs[nov]):>7.0%}")

    ticks, t1 = 0, time.perf_counter()
    for batches, _ in runs["compound"]:
        det = BetaBernoulliDetector(nominal_rules())
        for batch in batches:
            det.observe(batch)
            det.findings()
            ticks += 1
    cost = (time.perf_counter() - t1) / ticks * 1e6
    print(f"\ndetector cost: {cost:.0f} us per tick (observe + findings, compound episodes); simulation {sim:.0f} s")


if __name__ == "__main__":
    main()
