"""Dreamer results for docs/DREAMER.md: held-out eval against PPO and the planners (paired bootstrap CIs),
degradation when the novelty arrives, the learning curve, the reward-component inspection, and world-model
diagnostics (reward head on successes, imagined vs real returns).

    python scripts/dreamer_results.py --studio runs/dreamer_studio --ranch runs/dreamer_ranch \
        --ppo runs/ppo_novelty/final.pt

Demo-seeded (planner-guided) runs, docs/DREAMER.md "Planner-guided Dreamer":

    python scripts/dreamer_results.py --guided-studio runs/dreamer_demo_studio --guided-ranch runs/dreamer_demo_ranch \
        --studio runs/dreamer_studio_v2 --ranch runs/dreamer_ranch --ppo runs/ppo_novelty/final.pt \
        --workers 4 --md runs/dreamer_guided_results.md

Writes runs/dreamer_results.md and runs/dreamer_results.json (gitignored) and docs/media/dreamer_learning_curve.png.
Every number is on the held-out seeds 1000..1099, the same seeds (so the same fires) for every agent.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ["single", "kids"]
NOVELTIES = [None, "door_jam", "carpet_flashover", "hidden_dependent", "compound"]  # compound: never trained on
PLANNERS = ["baseline", "oracle", "adaptive"]
LABEL = {"single": "dog", "kids": "kids"}


def _bp(name="ranch"):
    from emberpath.cli import load_blueprint

    return load_blueprint(name)


def _planner_job(args):
    agent, profile, novelty, seeds = args
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.evaluate import evaluate

    return evaluate(_bp(), HOUSEHOLDS[profile], [agent], novelty, seeds)


def _ppo_job(args):
    ckpt, profile, novelty, seeds, greedy = args
    import torch

    torch.set_num_threads(1)
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.rl_baseline import evaluate_policy, load_policy

    net, _ = load_policy(ckpt)
    return evaluate_policy(net, _bp(), HOUSEHOLDS[profile], novelty, seeds, greedy=greedy)


def _dreamer_job(args):
    ckpt, label, bpname, profile, novelty, seeds, greedy = args
    import torch

    torch.set_num_threads(1)
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.dreamer import evaluate_dreamer, load_dreamer

    model = load_dreamer(ckpt)
    res = evaluate_dreamer(model, _bp(bpname), HOUSEHOLDS[profile], novelty, seeds, greedy=greedy)
    return [replace(r, agent=label + ("" if greedy else "_sampled")) for r in res]


def _components_job(args):
    ckpt, profile, novelty, seeds, greedy = args
    import torch

    torch.set_num_threads(1)
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.dreamer import DreamerAgent, load_dreamer
    from emberpath.rl_baseline import episode_components

    model = load_dreamer(ckpt)
    bp, hh = _bp(), HOUSEHOLDS[profile]
    out = []
    for s in seeds:
        e = episode_components(DreamerAgent(model, bp, greedy=greedy, seed=s), bp, hh, novelty, s)
        e.update(profile=profile, novelty=novelty or "none", greedy=greedy)
        e["distinct"] = len(set(e.pop("path")))
        out.append(e)
    return out


def _diagnose_job(args):
    """World-model checks on real held-out episodes played by the agent: per-step predicted vs real reward (and on
    the success step), predicted value at t=0 vs the real discounted return, reconstruction of the rescuer cell,
    and open-loop (imagined, 15 steps) reward prediction along the real action sequence."""
    ckpt, profile, novelty, seeds, *rest = args
    driver_name = rest[0] if rest else None  # "adaptive": the planner plays, the model only watches
    import torch

    torch.set_num_threads(1)
    from emberpath.agents import AdaptiveAgent
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.dreamer import DreamerAgent, _onehot, load_dreamer, twohot_mean
    from emberpath.reward import RewardTracker
    from emberpath.scenarios import NOVELTIES as NOV
    from emberpath.world import World

    m = load_dreamer(ckpt)
    bp, hh = _bp(), HOUSEHOLDS[profile]
    g = m.cfg.gamma
    rows = []
    for s in seeds:
        world = World(bp, hh, seed=s, novelties=NOV[novelty](bp, hh) if novelty else [])
        tracker = RewardTracker(world)
        agent = DreamerAgent(m, bp, greedy=True, seed=s)
        driver = AdaptiveAgent(bp) if driver_name == "adaptive" else None
        obs, events, xs, acts, rews = world.observe(), [], [], [], [0.0]
        while not world.done:
            xs.append(agent.features(obs, events))
            d = driver.act(obs, events) if driver else agent.act(obs, events)
            acts.append(d.action)
            res = world.step(d.action)
            rews.append(tracker.update(world).total)
            obs, events = res.obs, res.events
        xs.append(agent.features(obs, events))
        with torch.no_grad():
            x = torch.as_tensor(np.stack(xs))[None]
            T = x.shape[1]
            emb = m.wm.embed(x)
            h, z = m.wm.rssm.initial(1)
            gen = torch.Generator().manual_seed(s)
            prev = [0] + acts
            feats = []
            for t in range(T):
                h, z, _, _ = m.wm.rssm.obs_step(h, z, _onehot(torch.tensor([prev[t]])), emb[:, t],
                                                torch.tensor([float(t == 0)]), gen=gen)
                feats.append(torch.cat([h, z], -1))
            f = torch.cat(feats)
            rp = twohot_mean(torch.softmax(m.wm.reward(f), -1), m.bins).numpy()
            cp = torch.sigmoid(m.wm.cont(f)).squeeze(-1).numpy()
            v = twohot_mean(torch.softmax(m.ac.critic(f), -1), m.bins).numpy()
            rec = m.wm.decode(f)
            agent_ok = float((rec[..., 3].flatten(1).argmax(1) == x[0, ..., 3].flatten(1).argmax(1)).float().mean())
            # open loop: from the state at t=0, imagine along the real actions for up to 15 steps
            D = m.cfg.deter
            hh_, zz = f[:1, :D], f[:1, D:]
            ol = []
            for t in range(min(15, len(acts))):
                hh_, prior = m.wm.rssm.img_step(hh_, zz, _onehot(torch.tensor([acts[t]])))
                zz = m.wm.rssm.sample(prior, gen)
                ol.append(float(twohot_mean(torch.softmax(m.wm.reward(torch.cat([hh_, zz], -1)), -1), m.bins)))
        real = np.array(rews)
        disc = sum(r * g**i for i, r in enumerate(real[1:]))
        rows.append({"seed": s, "profile": profile, "novelty": novelty or "none", "outcome": world.outcome,
                     "steps": world.t, "return": float(real.sum()), "disc_return": float(disc), "value0": float(v[0]),
                     "reward_abs_err": float(np.abs(rp[1:] - real[1:]).mean()),
                     "final_reward": float(real[-1]), "final_reward_pred": float(rp[-1]),
                     "final_cont_pred": float(cp[-1]), "mid_cont_pred": float(cp[1:-1].mean()) if T > 2 else None,
                     "agent_cell_acc": agent_ok,
                     "open_loop_reward_err": float(np.abs(np.array(ol) - real[1:1 + len(ol)]).mean()) if ol else None})
    return rows


def summarize(rows):
    n = len(rows)
    return {"success": sum(r.outcome == "success" for r in rows) / n, "score": sum(r.score for r in rows) / n,
            "died": sum(r.outcome == "agent_lost" for r in rows) / n,
            "timeout": sum(r.outcome == "timeout" for r in rows) / n,
            "steps": sum(r.steps for r in rows) / n, "ret": sum(r.reward for r in rows) / n}


def ci(results, a, b, value):
    from emberpath.evaluate import paired_bootstrap

    rows = [r for r in results if r.agent in (a, b)]
    return paired_bootstrap(rows, lambda m: m[a] - m[b], value=value)


def fmt_ci(t):
    p, lo, hi = t
    return f"{p:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def ckpts_of(run: Path, offset: int):
    hist = json.loads((run / "history.json").read_text())
    out = [(offset + int(f.stem.split("_")[1]), str(f)) for f in run.glob("step_*.pt")]
    out.append((offset + hist["steps"], str(run / "final.pt")))
    return sorted(out), offset + hist["steps"], hist


def _lockstep_job(args):
    """The evidence-discipline lockstep check (tests/test_evidence_discipline.py) on a trained checkpoint: one seed
    with and without each novelty; while the agent was handed identical input its decisions must be identical."""
    ckpt, seeds = args
    import torch

    torch.set_num_threads(1)
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.dreamer import dreamer_factory, load_dreamer
    from emberpath.scenarios import NOVELTIES as NOV, applicable
    from emberpath.world import World

    model, bp = load_dreamer(ckpt), _bp()
    checked, leaks = 0, []
    for novelty in sorted(NOV):
        for profile in PROFILES:
            hh = HOUSEHOLDS[profile]
            if not applicable(bp, hh, novelty):
                continue
            for seed in seeds:
                for greedy in (True, False):
                    worlds = [World(bp, hh, seed=seed, novelties=NOV[novelty](bp, hh)), World(bp, hh, seed=seed)]
                    agents = [dreamer_factory(model, greedy=greedy, seed=seed)(bp, w) for w in worlds]
                    obs, events, t = [w.observe() for w in worlds], [[], []], 0
                    while not any(w.done for w in worlds) and obs[0] == obs[1] and events[0] == events[1]:
                        ds = [a.act(o, e) for a, o, e in zip(agents, obs, events)]
                        if ds[0] != ds[1]:
                            leaks.append((novelty, profile, seed, greedy, t))
                            break
                        rs = [w.step(d.action) for w, d in zip(worlds, ds)]
                        obs, events = [r.obs for r in rs], [r.events for r in rs]
                        t += 1
                    checked += 1
    return {"ckpt": ckpt, "checked": checked, "leaks": leaks}


CACHE = ROOT / "runs" / "dreamer_guided_cache"


def _cached(args):
    """Run ``fn(job)`` once per (function, job, checkpoint file's mtime) and keep the result on disk."""
    import hashlib
    import pickle

    name, job = args
    stamp = [os.path.getmtime(x) for x in job if isinstance(x, str) and x.endswith(".pt") and os.path.exists(x)]
    key = hashlib.sha256(repr((name, job, stamp)).encode()).hexdigest()[:20]
    f = CACHE / f"{name}_{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    res = globals()[name](job)
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(res))
    return res


def guided_main(a) -> None:
    """Demo-seeded Dreamer (docs/DREAMER.md, "Planner-guided Dreamer") against unguided Dreamer, PPO and the
    planners, with the demo-seeded run evaluated at the checkpoint where the demo share first hit 0 and at the end."""
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.evaluate import evaluate
    from emberpath.rl_baseline import EVAL_SEEDS, TRAIN_SEEDS

    seeds = list(EVAL_SEEDS)[: a.seeds]
    assert not set(seeds) & set(TRAIN_SEEDS)
    gs, gr = Path(a.guided_studio), Path(a.guided_ranch)
    settings = [(p, n) for p in PROFILES for n in NOVELTIES]
    lines: list[str] = []
    out = lines.append
    hist_gs = json.loads((gs / "history.json").read_text())
    hist_gr = json.loads((gr / "history.json").read_text())
    ck_gr = sorted((int(f.stem.split("_")[1]), str(f)) for f in gr.glob("step_*.pt"))
    ck_gr.append((hist_gr["steps"], str(gr / "final.pt")))
    dz = hist_gr["demo"]["demo_zero"]
    dreamers = {"demo_final": (str(gr / "final.pt"), (True, False)),
                "demo_zero": (str(gr / "demo_zero.pt"), (True, False)),
                "dreamer": (str(Path(a.ranch) / "final.pt"), (True,))}
    nh = Path(a.guided_ranch_noheard) if a.guided_ranch_noheard else None
    if nh:
        dreamers.update({"nh_final": (str(nh / "final.pt"), (True, False)), "nh_zero": (str(nh / "demo_zero.pt"), (True,))})
        hist_nh = json.loads((nh / "history.json").read_text())
    jobs_d = [(ck, lab, "ranch", p, n, seeds, g) for lab, (ck, gg) in dreamers.items() for p, n in settings
              for g in gg]
    ppos = {"ppo": a.ppo, **({"ppo_best": a.ppo_best} if a.ppo_best else {})}
    jobs_ppo = [(ck, lab, p, n, seeds, g) for lab, ck in ppos.items() for p, n in settings for g in (True, False)]
    jobs_p = [(ag, p, n, seeds) for ag in PLANNERS for p, n in settings]
    studio_ck = {"demo_final": str(gs / "final.pt"), "demo_zero": str(gs / "demo_zero.pt"),
                 "dreamer": str(Path(a.studio) / "final.pt")}
    jobs_s = [(ck, lab, "studio", "single", None, seeds, g) for lab, ck in studio_ck.items() for g in (True, False)]
    cseeds = seeds[: a.curve_seeds]
    jobs_c = [(ck, f"c{st}", "ranch", p, n, cseeds, True) for st, ck in ck_gr for p, n in settings if n != "compound"]
    jobs_x = [(str(gr / "final.pt"), p, n, seeds, True) for p, n in settings]
    jobs_diag = [(str(gr / "final.pt"), p, n, seeds[:30]) for p, n in settings]
    diag_ck = {"demo_final": str(gr / "final.pt"), "demo_zero": str(gr / "demo_zero.pt"),
               "dreamer": str(Path(a.ranch) / "final.pt")}
    jobs_watch = [(ck, p, n, seeds[:30], "adaptive") for ck in diag_ck.values() for p, n in settings
                  if n in (None, "door_jam")]
    jobs_lock = [(str(gr / "final.pt"), [1000, 1001]), (str(gr / "demo_zero.pt"), [1000, 1001]),
                 (str(gs / "final.pt"), [1000, 1001])]
    if a.only_static:  # precompute what does not depend on the demo-seeded runs
        static = ([("_dreamer_job", j) for j in jobs_d if "demo" not in j[0]]
                  + [("_dreamer_job", j) for j in jobs_s if "demo" not in j[0]]
                  + [("_ppo_label_job", j) for j in jobs_ppo] + [("_planner_job", j) for j in jobs_p])
        with ProcessPoolExecutor(a.workers) as pool:
            list(pool.map(_cached, static))
        print(f"cached {len(static)} jobs")
        return
    with ProcessPoolExecutor(a.workers) as pool:
        cm = lambda name, jobs: pool.map(_cached, [(name, j) for j in jobs])  # noqa: E731
        f_d = cm("_dreamer_job", jobs_d)
        f_s = cm("_dreamer_job", jobs_s)
        f_ppo = cm("_ppo_label_job", jobs_ppo)
        f_p = cm("_planner_job", jobs_p)
        f_c = pool.map(_dreamer_job, jobs_c)
        f_x = pool.map(_components_job, jobs_x)
        f_diag = pool.map(_diagnose_job, jobs_diag)
        f_lock = pool.map(_lockstep_job, jobs_lock)
        f_watch = pool.map(_diagnose_job, jobs_watch)
        res_d, res_s, res_ppo, res_p = list(f_d), list(f_s), list(f_ppo), list(f_p)
        res_c, res_x, res_diag, res_lock = list(f_c), list(f_x), list(f_diag), list(f_lock)
        res_watch = list(f_watch)
    by: dict[tuple, list] = {}
    for job, rows in zip(jobs_d, res_d):
        by[(rows[0].agent, job[3], job[4] or "none")] = rows
    for job, rows in zip(jobs_ppo, res_ppo):
        by[(rows[0].agent, job[2], job[3] or "none")] = rows
    for job, rows in zip(jobs_p, res_p):
        by[(job[0], job[1], job[2] or "none")] = rows
    succ = lambda r: float(r.outcome == "success")  # noqa: E731
    score = lambda r: r.score  # noqa: E731

    # ---------------- training record
    def demo_line(h, name):
        d = h["demo"]
        return (f"| {name} | {h['steps']} | {h['updates']} | {h['wall_clock']:.0f}s | {d['demo_episodes']} | "
                f"{d['demo_transitions']} | {d['demo_steps_replayed']} | {d['schedule']['start']:.2f} -> 0 by step "
                f"{d['schedule']['until_step']} (first all-own batch: step {d['demo_zero']['step']}, update "
                f"{d['demo_zero']['updates']}) |")
    out("## Training record\n")
    out("| run | env steps | updates | wall clock | demo episodes (adaptive, training seeds) | demo transitions | "
        "demo steps replayed | demo share schedule |\n|---|---:|---:|---:|---:|---:|---:|---|")
    out(demo_line(hist_gs, gs.name))
    out(demo_line(hist_gr, gr.name))
    if nh:
        out(demo_line(hist_nh, nh.name))
    out("")

    # ---------------- studio
    out(f"## Studio sanity check (studio / dog / no novelty, seeds {seeds[0]}..{seeds[-1]})\n")
    out("| agent | success | score | died | timeout | steps |\n|---|---:|---:|---:|---:|---:|")
    st_rows = {}
    for rows in res_s:
        st_rows[rows[0].agent] = rows
    rnd = [replace(r, agent="random") for r in _random_studio(seeds)]
    st_rows["random"] = rnd
    for ag in ("baseline", "adaptive"):
        st_rows[ag] = evaluate(_bp("studio"), HOUSEHOLDS["single"], [ag], None, seeds)
    for ag, rows in st_rows.items():
        sm = summarize(rows)
        out(f"| {ag} | {sm['success']:.0%} | {sm['score']:.2f} | {sm['died']:.0%} | {sm['timeout']:.0%} | "
            f"{sm['steps']:.0f} |")
    allst = [r for rows in st_rows.values() for r in rows]
    out("\nPaired (success): " + "; ".join(
        f"{x} - {y} {fmt_ci(ci(allst, x, y, succ))}" for x, y in
        (("demo_final_sampled", "random"), ("demo_final", "random"), ("demo_final_sampled", "dreamer_sampled"),
         ("demo_zero_sampled", "random"), ("demo_final_sampled", "adaptive"))) + "\n")

    # ---------------- ranch table
    agents = ["demo_final", "demo_final_sampled", "demo_zero", "demo_zero_sampled",
              *(["nh_final", "nh_final_sampled", "nh_zero"] if nh else []), "dreamer", *ppos,
              *[f"{k}_sampled" for k in ppos], *PLANNERS]
    out(f"## Held-out evaluation: ranch, seeds {seeds[0]}..{seeds[-1]}\n")
    out("Each cell: success / score / died / mean steps.\n")
    out("| household | novelty | " + " | ".join(agents) + " |")
    out("|---|---|" + "---|" * len(agents))
    summary = {}
    for p, n in settings:
        cells = []
        for ag in agents:
            sm = summarize(by[(ag, p, n or "none")])
            summary[f"{ag}|{p}|{n or 'none'}"] = sm
            cells.append(f"{sm['success']:.0%} / {sm['score']:.2f} / {sm['died']:.0%} / {sm['steps']:.0f}")
        out(f"| {LABEL[p]} | {n or 'none'} | " + " | ".join(cells) + " |")
    trained = [(p, n) for p, n in settings if n != "compound"]
    def mean8(ag, key):
        return np.mean([summary[f"{ag}|{p}|{n or 'none'}"][key] for p, n in trained])

    out("\nMean over the 8 trained settings (success / score): " + "; ".join(
        f"{ag} {mean8(ag, 'success'):.0%} / {mean8(ag, 'score'):.2f}" for ag in agents) + "\n")
    pairs = [("demo_final", "dreamer"), ("demo_final", "ppo"), ("demo_final_sampled", "ppo_sampled"),
             ("demo_final", "adaptive"), ("demo_final", "baseline"), ("demo_final", "demo_zero"),
             ("demo_final_sampled", "demo_zero_sampled")]
    if "ppo_best" in ppos:
        pairs.insert(3, ("demo_final", "ppo_best"))
    if nh:
        pairs += [("nh_final", "dreamer"), ("nh_zero", "dreamer"), ("demo_final", "nh_final")]
    for value, vname in ((succ, "success"), (score, "score")):
        out(f"## Paired differences in {vname} (95% paired bootstrap over the {len(seeds)} seeds)\n")
        out("| household | novelty | " + " | ".join(f"{x} - {y}" for x, y in pairs) + " |")
        out("|---|---|" + "---|" * len(pairs))
        for p, n in settings:
            rows = [r for ag in agents for r in by[(ag, p, n or "none")]]
            out(f"| {LABEL[p]} | {n or 'none'} | " + " | ".join(fmt_ci(ci(rows, x, y, value)) for x, y in pairs)
                + " |")
        out("")
    out("## Degradation when the novelty arrives (score with novelty minus without; 95% paired bootstrap)\n")
    deg_agents = ["demo_final", "demo_zero", *(["nh_final"] if nh else []), "dreamer", "ppo", *(["ppo_best"] if "ppo_best" in ppos else []),
                  *PLANNERS]
    out("| household | novelty | " + " | ".join(deg_agents) + " |")
    out("|---|---|" + "---|" * len(deg_agents))
    for p in PROFILES:
        for n in NOVELTIES[1:]:
            cells = []
            for ag in deg_agents:
                rows = [replace(r, agent="nov") for r in by[(ag, p, n)]] + \
                       [replace(r, agent="none") for r in by[(ag, p, "none")]]
                cells.append(fmt_ci(ci(rows, "nov", "none", score)))
            out(f"| {LABEL[p]} | {n} | " + " | ".join(cells) + " |")
    out("")

    # ---------------- curve
    pts: dict[int, list] = {}
    for job, rows in zip(jobs_c, res_c):
        pts.setdefault(int(job[1][1:]), []).extend(r.outcome == "success" for r in rows)
    sched = {r["step"]: r["demo_frac"] for r in hist_gr["history"]}
    out(f"## Learning curve of the demo-seeded ranch run (greedy, {len(cseeds)} seeds x 8 trained settings)\n")
    out("| env steps (ranch run) | demo share near here | held-out success |\n|---:|---:|---:|")
    for st in sorted(pts):
        near = min(sched, key=lambda k: abs(k - st)) if sched else None
        out(f"| {st / 1000:.0f}k | {sched.get(near, float('nan')):.2f} | {np.mean(pts[st]):.0%} |")
    out(f"\n(Demo share first 0 at ranch step {dz['step']}.)\n")

    # ---------------- components
    eps = [e for rows in res_x for e in rows]
    comp_names = list(eps[0]["components"])
    out("## Reward components (greedy demo_final, all held-out ranch episodes)\n")
    for title, sel in (("Top 8 by return", eps), ("Top 8 by return that did NOT end in success",
                                                  [e for e in eps if e["outcome"] != "success"])):
        out(f"{title}:\n")
        out("| household | novelty | seed | outcome | steps | return | " + " | ".join(comp_names) + " |")
        out("|---|---|---:|---|---:|---:|" + "---:|" * len(comp_names))
        for e in sorted(sel, key=lambda e: -e["return"])[:8]:
            out(f"| {LABEL[e['profile']]} | {e['novelty']} | {e['seed']} | {e['outcome']} | {e['steps']} | "
                f"{e['return']:.2f} | " + " | ".join(f"{e['components'][c]:+.2f}" for c in comp_names) + " |")
        out("")
    out("| outcome | episodes | return | " + " | ".join(comp_names) + " |")
    out("|---|---:|---:|" + "---:|" * len(comp_names))
    for oc, k in Counter(e["outcome"] for e in eps).most_common():
        es = [e for e in eps if e["outcome"] == oc]
        out(f"| {oc} | {k} | {np.mean([e['return'] for e in es]):.2f} | " +
            " | ".join(f"{np.mean([e['components'][c] for e in es]):+.2f}" for c in comp_names) + " |")
    waits = np.mean([np.mean(np.array(e["actions"]) == 4) for e in eps])
    distinct = np.mean([e["distinct"] for e in eps])
    out(f"\nWaits: {waits:.0%} of ticks. Largest progress total: "
        f"{max(e['components']['progress'] for e in eps):.2f}. Distinct cells per episode: {distinct:.1f}.\n")

    # ---------------- diagnostics
    diag = [r for rows in res_diag for r in rows]
    out("## World-model diagnostics (greedy demo_final, first 30 held-out seeds per setting)\n")
    out("| household | novelty | reward err / step | final reward: real / predicted | continue end / mid | "
        "rescuer cell decoded | value(t=0) / real discounted return |\n|---|---|---:|---|---|---:|---|")
    for p, n in settings:
        rs = [r for r in diag if r["profile"] == p and r["novelty"] == (n or "none")]
        out(f"| {LABEL[p]} | {n or 'none'} | {np.mean([r['reward_abs_err'] for r in rs]):.3f} | "
            f"{np.mean([r['final_reward'] for r in rs]):+.2f} / {np.mean([r['final_reward_pred'] for r in rs]):+.2f} | "
            f"{np.mean([r['final_cont_pred'] for r in rs]):.2f} / "
            f"{np.nanmean([r['mid_cont_pred'] if r['mid_cont_pred'] is not None else np.nan for r in rs]):.2f} | "
            f"{np.mean([r['agent_cell_acc'] for r in rs]):.0%} | "
            f"{np.mean([r['value0'] for r in rs]):+.2f} / {np.mean([r['disc_return'] for r in rs]):+.2f} |")
    sd = [r for r in diag if r["outcome"] == "success"]
    if sd:
        out(f"\nOn the {len(sd)} successful diagnosed episodes the final step pays "
            f"{np.mean([r['final_reward'] for r in sd]):+.2f} and the reward head predicts "
            f"{np.mean([r['final_reward_pred'] for r in sd]):+.2f}.\n")
    out("## Does the world model know what a rescue looks like? (it watches the ADAPTIVE agent's held-out "
        "episodes, first 30 seeds of dog/kids x none/door_jam)\n")
    out("| model | episodes | successes | final step on successes: real / predicted reward | on all: real / "
        "predicted | continue prob. at the end / mid | value(t=0) / real discounted return |")
    out("|---|---:|---:|---|---|---|---|")
    for (lab, ck), i in zip(diag_ck.items(), range(len(diag_ck))):
        k = len(jobs_watch) // len(diag_ck)
        rs = [r for rows in res_watch[i * k:(i + 1) * k] for r in rows]
        sd = [r for r in rs if r["outcome"] == "success"]
        out(f"| {lab} | {len(rs)} | {len(sd)} | {np.mean([r['final_reward'] for r in sd]):+.2f} / "
            f"{np.mean([r['final_reward_pred'] for r in sd]):+.2f} | {np.mean([r['final_reward'] for r in rs]):+.2f} / "
            f"{np.mean([r['final_reward_pred'] for r in rs]):+.2f} | {np.mean([r['final_cont_pred'] for r in rs]):.2f} / "
            f"{np.nanmean([r['mid_cont_pred'] if r['mid_cont_pred'] is not None else np.nan for r in rs]):.2f} | "
            f"{np.mean([r['value0'] for r in rs]):+.2f} / {np.mean([r['disc_return'] for r in rs]):+.2f} |")
    out("")
    out("## Evidence discipline: lockstep on the trained checkpoints\n")
    for r in res_lock:
        out(f"* `{r['ckpt']}`: {r['checked']} lockstep pairs (every novelty x household x seeds 1000, 1001 x "
            f"greedy and sampled), leaks: {r['leaks'] or 'none'}")
    out("")
    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.md).write_text("\n".join(lines) + "\n")
    Path(a.md).with_suffix(".json").write_text(json.dumps(
        {"summary": summary, "curve": {k: float(np.mean(v)) for k, v in pts.items()}, "diag": diag,
         "lockstep": res_lock}, default=float))
    print("\n".join(lines))


def reward_probe(demos_path: str, updates: int = 400, threads: int = 2) -> None:
    """Where does the reward signal get lost? Two learners on the SAME adaptive demos (training seeds; the last 50
    episodes held back for checking), same encoder, same rate (3e-4), batches of 1024 steps:
    (1) the full world model (RSSM) trained on demos only, reward read from its posterior features;
    (2) no RSSM: the encoder applied to the previous and the current frame plus the action, straight into a
    reward head of the same two-hot kind. Prints the predicted reward on success / loss / ordinary steps."""
    import pickle
    import time

    import torch
    from torch import nn

    from emberpath import dreamer as DR

    torch.set_num_threads(threads)
    demos = pickle.loads(Path(demos_path).read_bytes())
    canvas = tuple(demos["canvas"])
    tr, va = dict(demos, episodes=demos["episodes"][:-50]), dict(demos, episodes=demos["episodes"][-50:])

    def report(tag, u, rp, r, t0):
        big, neg = r > 0.5, r < -0.5
        sm = ~big & ~neg
        print(f"{tag} update {u}: held-back predicted reward on success/rescue steps {float(rp[big].mean()):+.2f} "
              f"(real {float(r[big].mean()):+.2f}), on loss steps {float(rp[neg].mean()):+.2f} "
              f"(real {float(r[neg].mean()):+.2f}), on other steps {float(rp[sm].mean()):+.3f} "
              f"(real {float(r[sm].mean()):+.3f})  ({time.time() - t0:.0f}s)", flush=True)

    # (1) the world model
    torch.manual_seed(0)
    m = DR.make_dreamer(canvas, DR.DreamerConfig(model_lr=3e-4))
    opt = torch.optim.Adam(m.wm.parameters(), lr=3e-4)
    rt, rv = DR.demo_replay(tr, 16, canvas, 1), DR.demo_replay(va, 4, canvas, 2)
    vb = {k: torch.as_tensor(v) for k, v in rv.sample(16, 64).items()}
    t0 = time.time()
    for u in range(1, updates + 1):
        b = {k: torch.as_tensor(v) for k, v in rt.sample(16, 64).items()}
        loss, _, _ = DR.world_model_loss(m, b["obs"], b["act"], b["reward"], b["first"], b["term"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.wm.parameters(), 1000)
        opt.step()
        if u % 100 == 0:
            with torch.no_grad():
                _, _, f = DR.world_model_loss(m, vb["obs"], vb["act"], vb["reward"], vb["first"], vb["term"])
                rp = DR.twohot_mean(torch.softmax(m.wm.reward(f.reshape(-1, f.shape[-1])), -1), m.bins)
            report("RSSM world model", u, rp, vb["reward"].reshape(-1), t0)

    # (2) the same encoder and head kind, no RSSM
    def pairs(d):
        rep = DR.demo_replay(d, 1, canvas, 0)
        T = int(rep.count[0])
        b = rep.sample(1, T)
        i = np.nonzero(b["first"][0] == 0)[0]
        return [torch.as_tensor(x) for x in (b["obs"][0, i - 1], b["obs"][0, i], b["act"][0, i], b["reward"][0, i])]

    torch.manual_seed(0)
    m = DR.make_dreamer(canvas, DR.DreamerConfig())
    hidden = m.cfg.hidden
    head = nn.Sequential(nn.Linear(2 * hidden + DR.N_ACTIONS, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                         nn.Linear(hidden, m.cfg.bins))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    opt = torch.optim.Adam(list(m.wm.encoder.parameters()) + list(head.parameters()), lr=3e-4)
    P, V = pairs(tr), pairs(va)
    pred = lambda x0, x1, a: head(torch.cat([m.wm.embed(x0), m.wm.embed(x1), DR._onehot(a)], -1))  # noqa: E731
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for u in range(1, updates + 1):
        i = torch.randint(len(P[3]), (1024,), generator=g)
        loss = DR.twohot_nll(pred(P[0][i], P[1][i], P[2][i]), P[3][i], m.bins).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if u % 100 == 0:
            with torch.no_grad():
                rp = DR.twohot_mean(torch.softmax(pred(*V[:3]), -1), m.bins)
            report("encoder + reward head, no RSSM", u, rp, V[3], t0)


def _ppo_label_job(args):
    ckpt, label, profile, novelty, seeds, greedy = args
    rows = _ppo_job((ckpt, profile, novelty, seeds, greedy))
    return [replace(r, agent=label + ("" if greedy else "_sampled")) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--studio", default="runs/dreamer_studio_v2")
    ap.add_argument("--ranch", default="runs/dreamer_ranch")
    ap.add_argument("--ppo", default="runs/ppo_novelty/final.pt")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--curve-seeds", type=int, default=20)
    ap.add_argument("--curve-every", type=int, default=1, help="evaluate every k-th checkpoint")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--plot", default=str(ROOT / "docs" / "media" / "dreamer_learning_curve.png"))
    ap.add_argument("--md", default="runs/dreamer_results.md")
    ap.add_argument("--guided-studio", default=None, help="demo-seeded studio run (switches to the guided report)")
    ap.add_argument("--guided-ranch", default=None, help="demo-seeded ranch run")
    ap.add_argument("--guided-ranch-noheard", default=None, help="demo-seeded ranch run without the hearing plane")
    ap.add_argument("--ppo-best", default=None, help="the best improved PPO checkpoint, if any (docs/RL.md)")
    ap.add_argument("--only-static", action="store_true", help="guided report: only precompute the cached jobs "
                    "that do not depend on the demo-seeded runs")
    ap.add_argument("--probe", default=None, help="a demos.pkl: run the reward-signal probe (reward_probe) only")
    a = ap.parse_args()
    if a.probe:
        return reward_probe(a.probe)
    if a.guided_ranch:
        return guided_main(a)
    from emberpath.rl_baseline import EVAL_SEEDS, TRAIN_SEEDS

    seeds = list(EVAL_SEEDS)[: a.seeds]
    assert not set(seeds) & set(TRAIN_SEEDS)
    studio, ranch = Path(a.studio), Path(a.ranch)
    final = str(ranch / "final.pt")
    settings = [(p, n) for p in PROFILES for n in NOVELTIES]
    lines: list[str] = []
    out = lines.append

    ck_s, off, hist_s = ckpts_of(studio, 0)
    ck_r, total, hist_r = ckpts_of(ranch, off)
    curve_settings = [(p, n) for p in PROFILES for n in NOVELTIES[:4]]
    cseeds = seeds[: a.curve_seeds]
    jobs_p = [(ag, p, n, seeds) for ag in PLANNERS for p, n in settings]
    jobs_ppo = [(a.ppo, p, n, seeds, g) for p, n in settings for g in (True, False)]
    jobs_d = [(final, "dreamer", "ranch", p, n, seeds, g) for p, n in settings for g in (True, False)]
    jobs_studio = [(str(studio / "final.pt"), "dreamer", "studio", "single", None, seeds, g) for g in (True, False)]
    curve = [ck_s[-1]] + ck_r[:: a.curve_every]  # the studio warm-up's end, then every ranch checkpoint
    if ck_r[-1] not in curve:
        curve.append(ck_r[-1])
    jobs_c = [(ck, f"c{st}", "ranch", p, n, cseeds, g) for st, ck in curve for p, n in curve_settings
              for g in (True, False)]
    jobs_x = [(final, p, n, seeds, True) for p, n in settings]
    jobs_diag = [(final, p, n, seeds[:30]) for p, n in settings]
    with ProcessPoolExecutor(a.workers) as pool:
        res_d = list(pool.map(_dreamer_job, jobs_d))
        res_studio = list(pool.map(_dreamer_job, jobs_studio))
        res_ppo = list(pool.map(_ppo_job, jobs_ppo))
        res_p = list(pool.map(_planner_job, jobs_p))
        res_c = list(pool.map(_dreamer_job, jobs_c))
        res_x = list(pool.map(_components_job, jobs_x))
        res_diag = list(pool.map(_diagnose_job, jobs_diag))

    by: dict[tuple, list] = {}
    for job, rows in zip(jobs_d, res_d):
        by[(rows[0].agent, job[3], job[4] or "none")] = rows
    for job, rows in zip(jobs_ppo, res_ppo):
        by[(rows[0].agent, job[1], job[2] or "none")] = rows
    for job, rows in zip(jobs_p, res_p):
        by[(job[0], job[1], job[2] or "none")] = rows

    # ------------------------------------------------ studio sanity check
    out("## Studio sanity check (studio / dog / no novelty, seeds 1000..1099)\n")
    out("| agent | success | score | died | steps |\n|---|---:|---:|---:|---:|")
    for rows in res_studio:
        s = summarize(rows)
        out(f"| {rows[0].agent} (studio checkpoint, {off // 1000}k steps) | {s['success']:.0%} | {s['score']:.2f} | "
            f"{s['died']:.0%} | {s['steps']:.0f} |")
    from emberpath.evaluate import evaluate
    from emberpath.dependents import HOUSEHOLDS

    for ag in ("baseline",):
        rows = evaluate(_bp("studio"), HOUSEHOLDS["single"], [ag], None, seeds)
        s = summarize(rows)
        out(f"| {ag} | {s['success']:.0%} | {s['score']:.2f} | {s['died']:.0%} | {s['steps']:.0f} |")
    rnd = _random_studio(seeds)
    s = summarize(rnd)
    out(f"| random (uniform actions) | {s['success']:.0%} | {s['score']:.2f} | {s['died']:.0%} | {s['steps']:.0f} |\n")

    # ------------------------------------------------ main table
    agents = ["dreamer", "dreamer_sampled", "ppo", "ppo_sampled", *PLANNERS]
    out(f"## Held-out evaluation: ranch, seeds {seeds[0]}..{seeds[-1]}\n")
    out("| household | novelty | agent | success | score | died | timeout | steps | return |")
    out("|---|---|---|---:|---:|---:|---:|---:|---:|")
    summary = {}
    for p, n in settings:
        for ag in agents:
            s = summarize(by[(ag, p, n or "none")])
            summary[f"{ag}|{p}|{n or 'none'}"] = s
            out(f"| {LABEL[p]} | {n or 'none'} | {ag} | {s['success']:.0%} | {s['score']:.2f} | {s['died']:.0%} | "
                f"{s['timeout']:.0%} | {s['steps']:.0f} | {s['ret']:.2f} |")
    out("")
    # paired CIs
    out("## Paired differences (95% paired bootstrap over the 100 seeds)\n")
    out("| household | novelty | dreamer - ppo (success) | dreamer - ppo (score) | dreamer - adaptive (success) | "
        "dreamer - adaptive (score) | dreamer_sampled - ppo_sampled (success) |")
    out("|---|---|---|---|---|---|---|")
    succ = lambda r: float(r.outcome == "success")  # noqa: E731
    score = lambda r: r.score  # noqa: E731
    cis = {}
    for p, n in settings:
        rows = [r for ag in agents for r in by[(ag, p, n or "none")]]
        c = [ci(rows, "dreamer", "ppo", succ), ci(rows, "dreamer", "ppo", score),
             ci(rows, "dreamer", "adaptive", succ), ci(rows, "dreamer", "adaptive", score),
             ci(rows, "dreamer_sampled", "ppo_sampled", succ)]
        cis[f"{p}|{n or 'none'}"] = c
        out(f"| {LABEL[p]} | {n or 'none'} | " + " | ".join(fmt_ci(x) for x in c) + " |")
    out("")
    # degradation
    out("## Degradation when the novelty arrives (score with novelty minus without; 95% paired bootstrap)\n")
    out("| household | novelty | " + " | ".join(agents) + " |")
    out("|---|---|" + "---|" * len(agents))
    for p in PROFILES:
        for n in NOVELTIES[1:]:
            cells = []
            for ag in agents:
                rows = [replace(r, agent="nov") for r in by[(ag, p, n)]] + \
                       [replace(r, agent="none") for r in by[(ag, p, "none")]]
                cells.append(fmt_ci(ci(rows, "nov", "none", score)))
            out(f"| {LABEL[p]} | {n} | " + " | ".join(cells) + " |")
    out("")

    # ------------------------------------------------ learning curve
    pts: dict[int, dict] = {}
    for job, rows in zip(jobs_c, res_c):
        st = int(job[1][1:])
        key = "greedy" if job[6] else "sampled"
        d = pts.setdefault(st, {"greedy": [], "sampled": []})
        d[key] += [r.outcome == "success" for r in rows]
    out(f"## Learning curve (held-out success, {len(cseeds)} seeds x 8 trained ranch settings per point)\n")
    out("| env steps | greedy | sampled |\n|---:|---:|---:|")
    curve_rows = []
    for st in sorted(pts):
        g, sm = np.mean(pts[st]["greedy"]), np.mean(pts[st]["sampled"])
        curve_rows.append((st, g, sm))
        out(f"| {st / 1000:.0f}k | {g:.0%} | {sm:.0%} |")
    out("")
    out(f"Wall clock: studio {hist_s['wall_clock']:.0f}s ({off} env steps, {hist_s['updates']} updates); ranch "
        f"{hist_r['wall_clock']:.0f}s ({total - off} env steps, {hist_r['updates']} updates).\n")
    _plot(a.plot, curve_rows, hist_s, hist_r, off, summary)

    # ------------------------------------------------ reward components
    eps = [e for rows in res_x for e in rows]
    out("## Reward components (greedy dreamer, all 1000 held-out ranch episodes)\n")
    comp_names = list(eps[0]["components"])
    out("Top 8 by return:\n")
    out("| household | novelty | seed | outcome | steps | return | " + " | ".join(comp_names) + " |")
    out("|---|---|---:|---|---:|---:|" + "---:|" * len(comp_names))
    for e in sorted(eps, key=lambda e: -e["return"])[:8]:
        out(f"| {LABEL[e['profile']]} | {e['novelty']} | {e['seed']} | {e['outcome']} | {e['steps']} | "
            f"{e['return']:.2f} | " + " | ".join(f"{e['components'][c]:+.2f}" for c in comp_names) + " |")
    out("\nTop 8 by return that did NOT end in success:\n")
    out("| household | novelty | seed | outcome | steps | return | " + " | ".join(comp_names) + " |")
    out("|---|---|---:|---|---:|---:|" + "---:|" * len(comp_names))
    for e in sorted([e for e in eps if e["outcome"] != "success"], key=lambda e: -e["return"])[:8]:
        out(f"| {LABEL[e['profile']]} | {e['novelty']} | {e['seed']} | {e['outcome']} | {e['steps']} | "
            f"{e['return']:.2f} | " + " | ".join(f"{e['components'][c]:+.2f}" for c in comp_names) + " |")
    out("\nMean return and components by outcome:\n")
    out("| outcome | episodes | return | " + " | ".join(comp_names) + " |")
    out("|---|---:|---:|" + "---:|" * len(comp_names))
    for oc, k in Counter(e["outcome"] for e in eps).most_common():
        es = [e for e in eps if e["outcome"] == oc]
        out(f"| {oc} | {k} | {np.mean([e['return'] for e in es]):.2f} | " +
            " | ".join(f"{np.mean([e['components'][c] for e in es]):+.2f}" for c in comp_names) + " |")
    waits = np.mean([np.mean(np.array(e["actions"]) == 4) for e in eps])
    max_prog = max(e["components"]["progress"] for e in eps)
    succ_eps = [e for e in eps if e["outcome"] == "success"]
    min_succ = min((e["return"] for e in succ_eps), default=float("nan"))
    max_fail = max((e["return"] for e in eps if e["outcome"] != "success"), default=float("nan"))
    out(f"\nWaits: {waits:.0%} of ticks. Largest progress total: {max_prog:.2f}. Worst success return: {min_succ:.2f}; "
        f"best non-success return: {max_fail:.2f}.\n")
    bumped = {}
    for p in PROFILES:
        es = [e for e in eps if e["profile"] == p and e["novelty"] in ("door_jam", "compound")]
        bumped[p] = float(np.mean([bool(e["bumped"]) for e in es]))
    out(f"Door_jam / compound episodes that bumped the jammed door: dog {bumped['single']:.0%}, kids "
        f"{bumped['kids']:.0%}.\n")

    # ------------------------------------------------ diagnostics
    diag = [r for rows in res_diag for r in rows]
    out("## World-model diagnostics (greedy dreamer, first 30 held-out seeds per setting)\n")
    out("| household | novelty | reward abs err / step | open-loop (15 step) reward err | final reward: real / "
        "predicted | continue at the end / mid | rescuer cell decoded | value(t=0) / real discounted return |")
    out("|---|---|---:|---:|---|---|---:|---|")
    for p, n in settings:
        rs = [r for r in diag if r["profile"] == p and r["novelty"] == (n or "none")]
        out(f"| {LABEL[p]} | {n or 'none'} | {np.mean([r['reward_abs_err'] for r in rs]):.3f} | "
            f"{np.nanmean([r['open_loop_reward_err'] or np.nan for r in rs]):.3f} | "
            f"{np.mean([r['final_reward'] for r in rs]):+.2f} / {np.mean([r['final_reward_pred'] for r in rs]):+.2f} | "
            f"{np.mean([r['final_cont_pred'] for r in rs]):.2f} / "
            f"{np.nanmean([r['mid_cont_pred'] if r['mid_cont_pred'] is not None else np.nan for r in rs]):.2f} | "
            f"{np.mean([r['agent_cell_acc'] for r in rs]):.0%} | "
            f"{np.mean([r['value0'] for r in rs]):+.2f} / {np.mean([r['disc_return'] for r in rs]):+.2f} |")
    sd = [r for r in diag if r["outcome"] == "success"]
    if sd:
        out(f"\nOn the {len(sd)} successful diagnosed episodes the success step pays {np.mean([r['final_reward'] for r in sd]):+.2f}"
            f" on average and the reward head predicts {np.mean([r['final_reward_pred'] for r in sd]):+.2f} there "
            f"(from the posterior, after seeing the final frame).\n")
    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.md).write_text("\n".join(lines) + "\n")
    Path(a.md).with_suffix(".json").write_text(json.dumps({"summary": summary, "cis": cis, "curve": curve_rows,
                                                           "diag": diag}, default=float))
    print("\n".join(lines))


def _random_studio(seeds):
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.evaluate import run_episode
    from emberpath.planner import Decision

    class Random:
        name = "random"

        def __init__(self, s):
            self.rng = np.random.default_rng(s)

        def act(self, obs, events):
            return Decision(int(self.rng.integers(5)), "Random move.")

    return [run_episode(_bp("studio"), HOUSEHOLDS["single"], lambda bp, w, s=s: Random(s), None, s) for s in seeds]


def _plot(path, curve_rows, hist_s, hist_r, off, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # PPO's held-out curve, same protocol (20 seeds x the same 8 settings), from docs/RL.md (scripts/ppo_results.py)
    ppo = [(254e3, 0.00, 0.00), (504e3, 0.01, 0.04), (754e3, 0.01, 0.09), (1.00e6, 0.08, 0.09),
           (1.16e6, 0.02, 0.10), (1.41e6, 0.12, 0.10), (1.66e6, 0.17, 0.11), (1.91e6, 0.12, 0.11),
           (2.16e6, 0.14, 0.13)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=110, gridspec_kw={"width_ratios": [1.4, 1]})
    xs = [r[0] for r in curve_rows]
    ax.plot(xs, [r[1] for r in curve_rows], "o-", color="#c2410c", label="Dreamer, greedy (held out)")
    ax.plot(xs, [r[2] for r in curve_rows], "o--", color="#c2410c", alpha=0.55, label="Dreamer, sampled (held out)")
    ax.plot([p[0] for p in ppo], [p[1] for p in ppo], "s-", color="#2563eb", label="PPO, greedy (held out)")
    ax.plot([p[0] for p in ppo], [p[2] for p in ppo], "s--", color="#2563eb", alpha=0.55, label="PPO, sampled")
    adaptive = np.mean([summary[f"adaptive|{p}|{n or 'none'}"]["success"] for p in PROFILES for n in NOVELTIES[:4]])
    baseline = np.mean([summary[f"baseline|{p}|{n or 'none'}"]["success"] for p in PROFILES for n in NOVELTIES[:4]])
    ax.axhline(adaptive, color="#15803d", lw=1, ls=":", label=f"adaptive planner ({adaptive:.0%})")
    ax.axhline(baseline, color="#6b7280", lw=1, ls=":", label=f"textbook-rules planner ({baseline:.0%})")
    ax.axvline(off, color="#9ca3af", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("environment steps (log scale; Dreamer's include the studio warm-up left of the dashed line)")
    ax.set_ylabel("held-out success, 8 ranch settings")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Dreamer vs PPO on the ranch (seeds 1000..1019 per setting)", fontsize=10)
    # right: world-model training diagnostics against env steps (studio then ranch)
    rows = [(r["step"], r) for r in hist_s["history"]] + [(off + r["step"], r) for r in hist_r["history"]]
    st = [x for x, _ in rows]
    ax2.plot(st, [r["recon"] for _, r in rows], color="#7c3aed", label="reconstruction loss (nats / frame)")
    ax2.plot(st, [r["reward_abs_err_big"] for _, r in rows], color="#dc2626",
             label="reward error on |reward| > 0.5 steps")
    ax2.plot(st, [-r["value"] for _, r in rows], color="#0891b2", label="- critic value (imagination start)")
    ax2.set_yscale("log")
    ax2.axvline(off, color="#9ca3af", lw=0.8, ls="--")
    ax2.set_xlabel("environment steps (studio, then ranch)")
    ax2.set_title("World-model training diagnostics", fontsize=10)
    ax3 = ax2.twinx()
    ax3.plot(st, [r["train_success"] or 0 for _, r in rows], color="#111827", lw=0.8, alpha=0.6,
             label="training success (rolling 100 episodes)")
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("training success")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


if __name__ == "__main__":
    main()
