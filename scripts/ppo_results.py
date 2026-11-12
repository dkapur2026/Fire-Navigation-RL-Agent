"""PPO results for docs/RL.md: held-out eval vs the planner agents, learning curve, recovery, exploit inspection.

    python scripts/ppo_results.py --nominal runs/ppo_nominal --novelty runs/ppo_novelty

`--nominal` is the run trained without novelties (the "frozen" policy); `--novelty` continues it with novelties
sampled per episode (the "keeps learning" policy, and the one compared in the main table). Writes the plot to
docs/media/ppo_learning_curve.png and the tables to runs/ppo_results.md (copied into docs/RL.md by hand).
Every PPO and planner number is on the held-out seeds 1000..1099.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ["single", "kids"]
NOVELTIES = [None, "door_jam", "carpet_flashover", "hidden_dependent", "compound"]  # compound: never trained on
PLANNERS = ["baseline", "oracle", "adaptive"]


def _bp():
    from emberpath.cli import load_blueprint

    return load_blueprint("ranch")


def _planner_job(args):
    agent, profile, novelty, seeds = args
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.evaluate import evaluate

    return evaluate(_bp(), HOUSEHOLDS[profile], [agent], novelty, seeds)


def _ppo_job(args):
    ckpt, label, profile, novelty, seeds, greedy = args
    import torch

    torch.set_num_threads(1)
    from dataclasses import replace

    from emberpath.dependents import HOUSEHOLDS
    from emberpath.rl_baseline import evaluate_policy, load_policy

    net, _ = load_policy(ckpt)
    res = evaluate_policy(net, _bp(), HOUSEHOLDS[profile], novelty, seeds, greedy=greedy)
    return [replace(r, agent=label + ("" if greedy else "_sampled")) for r in res]


def _components_job(args):
    ckpt, profile, novelty, seeds, greedy = args
    import torch

    torch.set_num_threads(1)
    from emberpath.agents import BaselineAgent
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.rl_baseline import PPOAgent, episode_components, load_policy

    net = load_policy(ckpt)[0] if ckpt else None
    bp, hh = _bp(), HOUSEHOLDS[profile]
    out = []
    for s in seeds:
        ag = PPOAgent(net, bp, greedy=greedy, seed=s) if ckpt else BaselineAgent(bp)
        e = episode_components(ag, bp, hh, novelty, s)
        e.update(profile=profile, novelty=novelty or "none", greedy=greedy, policy="ppo" if ckpt else "baseline")
        out.append(e)
    return out


def summarize(rows):
    n = len(rows)
    return {
        "success": sum(r.outcome == "success" for r in rows) / n,
        "score": sum(r.score for r in rows) / n,
        "died": sum(r.outcome == "agent_lost" for r in rows) / n,
        "timeout": sum(r.outcome == "timeout" for r in rows) / n,
        "steps": sum(r.steps for r in rows) / n,
        "ret": sum(r.reward for r in rows) / n,
    }


def rolling(xs, k=100):
    xs = np.asarray(xs, float)
    c = np.cumsum(np.insert(xs, 0, 0))
    out = np.empty(len(xs))
    for i in range(len(xs)):
        lo = max(0, i + 1 - k)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", default="runs/ppo_nominal")
    ap.add_argument("--novelty", default="runs/ppo_novelty")
    ap.add_argument("--seeds", type=int, default=100, help="held-out seeds from 1000")
    ap.add_argument("--curve-seeds", type=int, default=20, help="held-out seeds per setting for checkpoint evals")
    ap.add_argument("--plot", default=str(ROOT / "docs" / "media" / "ppo_learning_curve.png"))
    ap.add_argument("--md", default="runs/ppo_results.md")
    a = ap.parse_args()
    from emberpath.rl_baseline import EVAL_SEEDS

    seeds = list(EVAL_SEEDS)[: a.seeds]
    nom, nov = Path(a.nominal), Path(a.novelty)
    settings = [(p, n) for p in PROFILES for n in NOVELTIES]
    lines: list[str] = []
    out = lines.append

    # ------------------------------------------------ held-out evaluation
    jobs_p = [(ag, p, n, seeds) for ag in PLANNERS for p, n in settings]
    jobs_r = [(str(c / "final.pt"), lab, p, n, seeds, g) for c, lab in ((nov, "ppo"), (nom, "ppo_frozen"))
              for p, n in settings for g in (True, False)]
    # checkpoint curve: greedy + sampled held-out success at every saved checkpoint, on the trained settings
    ckpts = []
    offset = 0
    for run in (nom, nov):
        hist = json.loads((run / "history.json").read_text())
        for f in sorted(run.glob("step_*.pt"), key=lambda q: int(q.stem.split("_")[1])):
            ckpts.append((offset + int(f.stem.split("_")[1]), str(f)))
        ckpts.append((offset + hist["history"][-1]["step"], str(run / "final.pt")))
        offset += hist["history"][-1]["step"]
    curve_settings = [(p, n) for p in PROFILES for n in NOVELTIES[:4]]
    jobs_c = [(ck, f"c{st}", p, n, seeds[: a.curve_seeds], g) for st, ck in ckpts for p, n in curve_settings
              for g in (True, False)]
    jobs_x = [(str(nov / "final.pt"), p, n, seeds, g) for p, n in settings for g in (True, False)]
    jobs_x += [(None, p, n, seeds, True) for p, n in settings]  # baseline, for the door comparison
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        res_p = list(pool.map(_planner_job, jobs_p))
        res_r = list(pool.map(_ppo_job, jobs_r))
        res_c = list(pool.map(_ppo_job, jobs_c))
        res_x = list(pool.map(_components_job, jobs_x))
    table: dict[tuple, dict] = {}
    for job, rows in zip(jobs_p, res_p):
        table[(job[0], job[1], job[2])] = summarize(rows)
    for job, rows in zip(jobs_r, res_r):
        table[(rows[0].agent, job[2], job[3])] = summarize(rows)
    agents = ["ppo", "ppo_sampled", "ppo_frozen", "ppo_frozen_sampled", *PLANNERS]

    out(f"## Held-out evaluation: ranch, seeds {seeds[0]}..{seeds[-1]}\n")
    out("success = everyone out; score = fraction rescued (0 if the rescuer dies); died = rescuer lost; steps = "
        "mean episode length. `ppo` = greedy (argmax), `_sampled` = actions drawn from the policy. "
        "`compound` was never trained on.\n")
    out("| household | novelty | agent | success | score | died | timeout | steps | return |")
    out("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for p, n in settings:
        for ag in agents:
            s = table[(ag, p, n)]
            out(f"| {p} | {n or 'none'} | {ag} | {s['success']:.0%} | {s['score']:.2f} | {s['died']:.0%} | "
                f"{s['timeout']:.0%} | {s['steps']:.0f} | {s['ret']:.2f} |")
    out("")
    out("### Degradation when the novelty arrives (score with novelty minus score without, same household)\n")
    out("| household | novelty | " + " | ".join(agents) + " |")
    out("|---|---|" + "---:|" * len(agents))
    for p in PROFILES:
        for n in NOVELTIES[1:]:
            out(f"| {p} | {n} | " + " | ".join(
                f"{table[(ag, p, n)]['score'] - table[(ag, p, None)]['score']:+.2f}" for ag in agents) + " |")
    out("")

    # ------------------------------------------------ recovery (keeps learning)
    hn = json.loads((nom / "history.json").read_text())
    hv = json.loads((nov / "history.json").read_text())
    last_phase = max(e["phase"] for e in hn["episodes"])
    pre = [e for e in hn["episodes"] if e["phase"] == last_phase][-400:]
    pre_succ = float(np.mean([e["outcome"] == "success" for e in pre]))
    out("### Keeps learning: steps until training success on each novelty recovers to 90% of the pre-novelty level\n")
    out(f"Pre-novelty level: training success over the last {len(pre)} nominal ranch episodes of the nominal run "
        f"= {pre_succ:.0%}; target = {0.9 * pre_succ:.0%} (rolling mean over that novelty's last 100 episodes, both "
        "households).\n")
    out("| novelty | success, first 100 eps after the switch | steps to recover | episodes (of that novelty) |")
    out("|---|---:|---:|---:|")
    by_task: dict[str, list] = {}
    for e in hv["episodes"]:
        by_task.setdefault(e["task"].split("/")[2], []).append(e)
    for n, eps in by_task.items():
        r = rolling([e["outcome"] == "success" for e in eps])
        hit = next((i for i in range(99, len(eps)) if r[i] >= 0.9 * pre_succ), None)
        first = float(np.mean([e["outcome"] == "success" for e in eps[:100]]))
        out(f"| {n} | {first:.0%} | {eps[hit]['step'] if hit is not None else 'not reached'} | "
            f"{hit + 1 if hit is not None else len(eps)} |")
    out("")

    # ------------------------------------------------ exploit inspection
    ppo_eps = [e for rows in res_x for e in rows if e["policy"] == "ppo"]
    base_eps = [e for rows in res_x for e in rows if e["policy"] == "baseline"]
    comps = list(ppo_eps[0]["components"])

    def row(e):
        c = " | ".join(f"{e['components'][k]:+.2f}" for k in comps)
        return (f"| {e['profile']}/{e['novelty']} | {e['seed']} | {'greedy' if e['greedy'] else 'sampled'} | "
                f"{e['outcome']} | {e['steps']} | {e['return']:.2f} | {c} |")

    head = "| setting | seed | mode | outcome | steps | return | " + " | ".join(comps) + " |"
    sep = "|---|---:|---|---|---:|---:|" + "---:|" * len(comps)
    out("## Exploit inspection (trained policy, held-out seeds)\n")
    out("Top 8 episodes by return:\n")
    out(head)
    out(sep)
    for e in sorted(ppo_eps, key=lambda e: -e["return"])[:8]:
        out(row(e))
    out("\nTop 8 episodes by return that did NOT end in success:\n")
    out(head)
    out(sep)
    for e in sorted([e for e in ppo_eps if e["outcome"] != "success"], key=lambda e: -e["return"])[:8]:
        out(row(e))
    out("\nMean components by outcome (all PPO eval episodes, greedy and sampled):\n")
    out("| outcome | n | return | " + " | ".join(comps) + " |")
    out("|---|---:|---:|" + "---:|" * len(comps))
    for oc, k in Counter(e["outcome"] for e in ppo_eps).most_common():
        g = [e for e in ppo_eps if e["outcome"] == oc]
        out(f"| {oc} | {k} | {np.mean([e['return'] for e in g]):.2f} | " +
            " | ".join(f"{np.mean([e['components'][c] for e in g]):+.2f}" for c in comps) + " |")
    # behavioural probes
    def waits(e):
        return sum(a == 4 for a in e["actions"]) / max(1, len(e["actions"]))

    def revisits(e):
        return 1 - len(set(e["path"])) / len(e["path"])

    out("\nBehaviour probes (mean over episodes; dawdling and loops would show as high wait / revisit):\n")
    out("| policy | mode | outcome | n | wait fraction | revisited-cell fraction | steps |")
    out("|---|---|---|---:|---:|---:|---:|")
    for label, eps in (("ppo", ppo_eps), ("baseline", base_eps)):
        for g in sorted({e["greedy"] for e in eps}, reverse=True):
            for oc in ("success", "other"):
                sel = [e for e in eps if e["greedy"] == g and ((e["outcome"] == "success") == (oc == "success"))]
                if sel:
                    out(f"| {label} | {'greedy' if g else 'sampled'} | {oc} | {len(sel)} | "
                        f"{np.mean([waits(e) for e in sel]):.2f} | {np.mean([revisits(e) for e in sel]):.2f} | "
                        f"{np.mean([e['steps'] for e in sel]):.0f} |")
    # the true-jam progress leak: does the policy avoid the door that jams before it has bumped it?
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.scenarios import door_jam_single as door_jam  # this probe was written for the legacy single jam

    bp = _bp()
    out("\nThe jammed door (reward-leak probe): share of episodes whose path ever steps onto / bumps that door:\n")
    out("| policy | mode | household | door used, no novelty | door bumped, door_jam |")
    out("|---|---|---|---:|---:|")
    for label, eps in (("ppo", ppo_eps), ("baseline", base_eps)):
        for g in sorted({e["greedy"] for e in eps}, reverse=True):
            for p in PROFILES:
                door = door_jam(bp, HOUSEHOLDS[p])[0].patch.key.split(":")[1]
                cell = tuple(int(x) for x in door.split(","))
                none = [e for e in eps if e["greedy"] == g and e["profile"] == p and e["novelty"] == "none"]
                jam = [e for e in eps if e["greedy"] == g and e["profile"] == p and e["novelty"] == "door_jam"]
                used = np.mean([cell in map(tuple, e["path"]) for e in none])
                bumped = np.mean([[*cell] in [list(b) for b in e["bumped"]] for e in jam])
                out(f"| {label} | {'greedy' if g else 'sampled'} | {p} | {used:.0%} | {bumped:.0%} |")
    out("")

    # ------------------------------------------------ learning curve
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cur: dict[tuple, list] = {}
    for job, rows in zip(jobs_c, res_c):
        step = int(job[1][1:])
        cur.setdefault((step, job[5]), []).extend(rows)
    pts = sorted(cur)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=110)
    xs, ys = [], []
    for run, off in ((hn, 0), (hv, hn["history"][-1]["step"])):
        for e in run["episodes"]:
            if (off == 0 and e["phase"] == last_phase) or off:
                xs.append(off + e["step"])
                ys.append(e["outcome"] == "success")
    ax.plot(xs, rolling(ys, 300), color="#b8b7b0", lw=1.5, label="training episodes (rolling 300; trained settings only)")
    for g, col, lab in ((True, "#2a78d6", "PPO held-out, greedy"), (False, "#eb6834", "PPO held-out, sampled")):
        sx = [s for s, gg in pts if gg == g]
        sy = [np.mean([r.outcome == "success" for r in cur[(s, g)]]) for s in sx]
        ax.plot(sx, sy, color=col, lw=2, marker="o", ms=5, label=lab)
    for ag, ls in (("adaptive", "-"), ("baseline", ":")):
        ref = np.mean([table[(ag, p, n)]["success"] for p, n in curve_settings])
        ax.axhline(ref, color="#52514e", lw=1, ls=ls)
        ax.text(ax.get_xlim()[0], ref + 0.01, f" {ag} agent, same 8 settings: {ref:.0%}", fontsize=8,
                color="#52514e", va="bottom")
    split = hn["history"][-1]["step"]
    warm = sum(p["steps"] for p in hn["phases"][:-1])
    ax.axvline(split, color="#52514e", lw=1, ls="--")
    ax.text(split, 1.02, " novelties switched on", fontsize=8, color="#52514e", va="bottom")
    ax.axvline(warm, color="#52514e", lw=1, ls=":")
    ax.text(warm, 1.02, " ranch", fontsize=8, color="#52514e", va="bottom")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("success rate, ranch (8 settings)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout()
    fig.savefig(a.plot)
    out("## Learning curve (held-out success at each checkpoint, "
        f"{a.curve_seeds} seeds x {len(curve_settings)} trained settings)\n")
    out("| env steps | greedy | sampled |")
    out("|---:|---:|---:|")
    for s in sorted({s for s, _ in pts}):
        out(f"| {s} | " + " | ".join(f"{np.mean([r.outcome == 'success' for r in cur[(s, g)]]):.0%}"
                                     for g in (True, False)) + " |")
    Path(a.md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {a.plot} and {a.md}")


if __name__ == "__main__":
    main()
