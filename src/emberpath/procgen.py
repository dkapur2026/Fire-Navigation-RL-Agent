"""Procedural houses with a solvability guarantee and measured, graded difficulty (WP-J / W8).

A house is a grid of rectangular rooms (rows and columns of varying size). One room row may be a
hallway (its rooms merged). Rooms are joined by a random spanning tree of doorways plus extra links
that create loops (alternative routes); most doorways are doors (`D`), some are open archways.
Every house is emitted as ASCII and parsed by `Blueprint.parse`, so the format has one source of truth.

Difficulty is *measured*, not assumed. Generation knobs (room count, carpet share, exits, where the fire
starts) only steer the sampler; `measure` then computes route length, doors, fire-to-exit distance, the
chance the fire reaches a dependent before the earliest possible pickup (`p_lost`, fire-only Monte Carlo)
and the safety margin of the best escape plan against the planner's own fire forecast (`margin`).
`tier_of` thresholds those, and `generate(seed, tier)` resamples until the measured tier matches. The
thresholds were chosen by looking at oracle success over 400 houses; `scripts/coverage_report.py` checks
on fresh houses that oracle success falls with tier (docs/ARCHITECTURE.md, "Procedural houses").
"""

from __future__ import annotations

import itertools
import math
import random
from collections import deque

import numpy as np

from emberpath.blueprint import Blueprint
from emberpath.dependents import HOUSEHOLDS, Household
from emberpath.grid import Cell, bfs_distances, manhattan, neighbors4
from emberpath.planner import HazardAwarePlanner
from emberpath.rules import nominal_rules
from emberpath.scenarios import door_jam_single  # the legacy single jam: changing the filter would change every house
from emberpath.world import is_fire_tick

TIERS = (1, 2, 3, 4, 5)
N_SLOTS = 3  # households of up to two people keep a spare slot for `hidden_dependent`; the family (3) fills them
PROFILE = "kids"  # difficulty is measured on the hardest household: two slow children
P_LOST_CUTS = (0.0, 0.25)  # p_lost above these -> tier 4, tier 5
MARGIN_CUTS = (0, 45)  # with p_lost == 0: margin at or below these -> tier 3, tier 2; above -> tier 1
# Fire start distance to the nearest of dependents 1-2 (tier 3: to the nearest exit), by tier. Steers the sampler only.
_FIRE_GAP = {1: (8, 99), 2: (6, 14), 3: (2, 6), 4: (2, 6), 5: (2, 4)}
_PROBE = Household("probe", "three dependents", HOUSEHOLDS["kids"].dependents[:1] * N_SLOTS)


def _layout(rng: random.Random, tier: int) -> str:
    """Draw one candidate house (may be degenerate; `generate` filters)."""
    nr, nc = rng.randint(2, 3), rng.randint(2, 3 + (tier >= 3))
    hs = [rng.randint(2, 4) for _ in range(nr)]
    ws = [rng.randint(3, 7) for _ in range(nc)]
    r0 = [1 + sum(h + 1 for h in hs[:i]) for i in range(nr)]
    c0 = [1 + sum(w + 1 for w in ws[:j]) for j in range(nc)]
    H, W = r0[-1] + hs[-1] + 1, c0[-1] + ws[-1] + 1
    g = [["#"] * W for _ in range(H)]
    hall = rng.randrange(nr) if rng.random() < 0.6 else -1
    node = {(i, j): (i, 0) if i == hall else (i, j) for i in range(nr) for j in range(nc)}
    carpet = 0.15 + 0.12 * tier
    floor = {n: rng.choices(".ct", (1.0, carpet, 0.35))[0] for n in set(node.values())}
    for (i, j), n in node.items():
        for r in range(r0[i], r0[i] + hs[i]):
            for c in range(c0[j] - (i == hall and j > 0), c0[j] + ws[j]):
                g[r][c] = floor[n]

    # Doorways: spanning tree over room nodes, plus 1-3 extra links that make loops (alternative routes).
    pairs = [((i, j), (i, j + 1)) for i in range(nr) for j in range(nc - 1) if i != hall]
    pairs += [((i, j), (i + 1, j)) for i in range(nr - 1) for j in range(nc)]
    rng.shuffle(pairs)
    root = {n: n for n in set(node.values())}

    def find(n):
        while root[n] != n:
            n = root[n]
        return n

    tree, extra = [], []
    for a, b in pairs:
        ra, rb = find(node[a]), find(node[b])
        (extra if ra == rb else tree).append((a, b))
        root[ra] = rb
    for (i, j), (k, l) in tree + extra[: rng.randint(1, 3)]:
        if k == i:  # vertical wall between (i, j) and (i, j + 1)
            cell = (rng.randrange(r0[i], r0[i] + hs[i]), c0[l] - 1)
        else:
            cell = (r0[k] - 1, rng.randrange(c0[j], c0[j] + ws[j]))
        g[cell[0]][cell[1]] = "D" if rng.random() < 0.8 else floor[node[(i, j)]]

    # Exits on the outer wall (never in a corner).
    sides = [(0, c) for c in range(1, W - 1)] + [(H - 1, c) for c in range(1, W - 1)]
    sides += [(r, 0) for r in range(1, H - 1)] + [(r, W - 1) for r in range(1, H - 1)]
    exits = rng.sample(sides, rng.randint(1, 3 - (tier >= 4)))
    for r, c in exits:
        g[r][c] = "E"

    # Rescuer and three dependent slots anywhere. The fire goes nearer the first two dependents in harder
    # tiers, and near an exit in tier 3 (dependents safe, escape contested).
    cells = [(r, c) for r in range(H) for c in range(W) if g[r][c] in ".ct"]
    picks = rng.sample(cells, 4)
    lo, hi = _FIRE_GAP[tier]
    targets = exits if tier == 3 else picks[1:3]
    free = [x for x in cells if x not in picks]
    near = [x for x in free if lo <= min(manhattan(x, s) for s in targets) <= hi]
    for (r, c), ch in zip(picks + [rng.choice(near or free)], "A123F"):
        g[r][c] = ch
    return "\n".join("".join(row) for row in g)


def _escape_time(bp: Blueprint, adj: dict, eta: dict[Cell, float], order: tuple[Cell, ...], margin: float) -> int | None:
    """Ticks to collect `order` and reach an exit while staying `margin` ticks ahead of the forecast fire.

    The forecast is fixed, so arriving earlier is never worse and a BFS that prunes late cells is exact.
    After the first pickup the children set the pace: one cell every two ticks. None = infeasible."""
    here, now, pace = bp.agent_start, 0, 1
    for goals in [{s} for s in order] + [set(bp.exits)]:
        seen, frontier = {here}, deque([(here, now)])
        while frontier:
            cell, t = frontier.popleft()
            if cell in goals:
                break
            for nxt in adj[cell]:
                if nxt not in seen and eta.get(nxt, math.inf) - (t + pace) >= margin:
                    seen.add(nxt)
                    frontier.append((nxt, t + pace))
        else:
            return None
        here, now, pace = cell, t, 2
    return now


def _first_fire(bp: Blueprint, horizon: int, samples: int = 24) -> np.ndarray:
    """Monte Carlo of the fire alone (it ignores the agent): tick each cell first ignites, per sample.

    Same dynamics as `World._spread_fire` under nominal rules (one trial per burning neighbour, cells burn
    `burn_steps` fire turns then go out), on the world's fire clock (a turn only on fire ticks: slow at first),
    vectorised over samples. Fixed seed, so `measure` is deterministic."""
    rules, rng = nominal_rules(), np.random.default_rng(0)
    q = np.ones((bp.height, bp.width))  # chance one burning neighbour fails to ignite the cell
    for cell, mat in bp.materials.items():
        if mat in rules.spread_prob and cell not in bp.exits:
            q[cell] = 1.0 - rules.spread_prob[mat]
    first = np.full((samples, bp.height, bp.width), np.inf)
    for cell in bp.ignitions:
        first[:, cell[0], cell[1]] = 0
    turn = np.where(first == 0, 0.0, np.inf)  # fire turn each cell ignited on (burn-out counts turns, not ticks)
    k = 0
    for t in range(1, horizon + 1):
        if not is_fire_tick(t - 1):
            continue
        k += 1
        b = ((turn <= k - 1) & (turn > k - 1 - rules.burn_steps)).astype(np.int8)
        if not b.any():
            break
        n = np.zeros_like(b)
        n[:, 1:] += b[:, :-1]
        n[:, :-1] += b[:, 1:]
        n[:, :, 1:] += b[:, :, :-1]
        n[:, :, :-1] += b[:, :, 1:]
        new = (first == np.inf) & (rng.random(first.shape) >= q**n)
        first[new], turn[new] = t, k
    return first


def measure(bp: Blueprint, n: int = 2) -> dict[str, float]:
    """Difficulty measurements for rescuing the first `n` dependents. No agent is simulated.

    * `p_lost`: chance (fire Monte Carlo) that the fire reaches a dependent before the earliest possible
      pickup, for the best pickup order. This is what mostly makes houses fail.
    * `margin`: the largest safety margin (ticks ahead of the planner's own fire forecast) with which some
      order and route still gets everyone out; negative means the forecast fire wins every race."""
    eta = HazardAwarePlanner(bp, nominal_rules()).fire_eta(frozenset(bp.ignitions), step=0)  # from the first tick
    orders = list(itertools.permutations(bp.slots[:n]))
    adj = {c: [x for x in neighbors4(c, bp.height, bp.width) if x in bp.materials] for c in bp.materials}
    lo, hi = -64, 64  # largest feasible margin, by bisection (capped: beyond 64 the fire is irrelevant)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if any(_escape_time(bp, adj, eta, o, mid) is not None for o in orders) else (lo, mid - 1)
    dist = {c: bfs_distances(c, bp.materials.__contains__, bp.height, bp.width) for c in (bp.agent_start, *bp.ignitions, *bp.slots[:n])}
    pickups = []  # earliest pickup tick per dependent; children halve the pace after the first pickup
    for order in orders:
        t, here, times = 0, bp.agent_start, {}
        for k, slot in enumerate(order):
            t += (1 + (k > 0)) * dist[here][slot]
            here, times[slot] = slot, t
        pickups.append(times)
    fire_dist = dist[bp.ignitions[0]]  # fire spreads at most one cell per tick: if it cannot win, skip the sampling
    if any(all(fire_dist.get(c, math.inf) > t for c, t in p.items()) for p in pickups):
        p_lost = 0.0
    else:
        first = _first_fire(bp, max(max(p.values()) for p in pickups))
        p_lost = min(1 - np.mean(np.all([first[:, r, c] > t for (r, c), t in p.items()], axis=0)) for p in pickups)
    return {
        "route_len": min(_escape_time(bp, adj, {}, o, 0) for o in orders),
        "doors": len(bp.doors),
        "exits": len(bp.exits),
        "fire_exit": min(fire_dist.get(e, 99) for e in bp.exits),
        "margin": lo,
        "p_lost": round(float(p_lost), 3),
    }


def tier_of(m: dict[str, float]) -> int:
    """Tier 1 (easy) .. 5 (hard) from the measurements; thresholds chosen from data (docs/ARCHITECTURE.md)."""
    if m["p_lost"] > P_LOST_CUTS[1]:
        return 5
    if m["p_lost"] > P_LOST_CUTS[0]:
        return 4
    return 3 if m["margin"] <= MARGIN_CUTS[0] else 2 if m["margin"] <= MARGIN_CUTS[1] else 1


def _acceptable(bp: Blueprint) -> bool:
    """Solvable for 3 dependents and not degenerate (door_jam is checked last in `generate`: it is slower)."""
    try:
        bp.validate(N_SLOTS)
    except ValueError:
        return False
    dist = bp.reachable(bp.agent_start)
    fire = bp.ignitions[0]
    return (
        min(dist.get(e, 0) for e in bp.exits) > 3  # rescuer not beside an exit
        and manhattan(bp.agent_start, fire) > 2  # fire not beside the rescuer
        and all(manhattan(s, fire) > 1 for s in bp.slots)
        and fire in dist
    )


def generate(seed: int, tier: int = 3, max_tries: int = 2000) -> Blueprint:
    """A solvable house whose *measured* tier is `tier`. Deterministic in (seed, tier)."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    rng = random.Random(f"emberpath:{tier}:{seed}")
    for _ in range(max_tries):
        bp = Blueprint.parse(_layout(rng, tier), name=f"gen:{tier}:{seed}")
        if _acceptable(bp) and tier_of(measure(bp)) == tier:
            try:
                door_jam_single(bp, _PROBE)  # some door on the mission route can jam and leave the house solvable
                return bp
            except ValueError:
                pass
    raise RuntimeError(f"no tier-{tier} house found for seed {seed} in {max_tries} tries")


def from_spec(spec: str) -> Blueprint:
    """`gen:<tier>:<seed>` -> house (the CLI's `--blueprint gen:3:7`)."""
    _, tier, seed = spec.split(":")
    return generate(int(seed), int(tier))


def oracle_success(bp: Blueprint, seeds: int = 30, profile: str = PROFILE) -> float:
    """Empirical oracle success rate over `seeds` fire seeds, no novelty."""
    from emberpath.evaluate import evaluate  # lazy: evaluate imports agents, which is heavier

    res = evaluate(bp, HOUSEHOLDS[profile], ["oracle"], None, range(seeds))
    return sum(r.outcome == "success" for r in res) / len(res)
