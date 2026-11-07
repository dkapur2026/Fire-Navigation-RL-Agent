"""Phase 3: a symbolic, hazard-aware planner. No learning.

It plans with a *belief* (a ``Rules`` object). If the belief is wrong, the plan is wrong. That
is the whole point of the project: this planner is the "brain" whose beliefs get repaired.

How it decides, every tick:
  1. Predict when fire will reach each cell (expected ticks = 1 / believed spread probability per
     cell crossed, from every flame). Compare with when the rescuer could get there. Cells the
     fire is likely to reach first cost extra: 1 + risk_weight * risk. Burning cells and
     believed-jammed doors are walls. Because the forecast uses the believed spread rates, a
     wrong belief (fire is faster than I think) produces wrong routes.
  2. Try every order of collecting the people still waiting (brute force, fine for <= 3), and pick
     the order with the cheapest route that ends at an exit.
  3. Take the first step of that route, and explain why (``Decision.reason``).
  4. If followers have fallen behind and there is no fire nearby, wait for them (up to a limit).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from emberpath.blueprint import Blueprint
from emberpath.grid import (
    WAIT,
    Cell,
    action_toward,
    bfs_distances,
    dijkstra,
    dijkstra_multi,
    manhattan,
    path_to,
)
from emberpath.rules import Rules
from emberpath.world import FOLLOWING, WAITING, DependentView, Observation


@dataclass(frozen=True)
class Decision:
    action: int
    reason: str  # plain-English explanation, shown to humans
    target: Cell | None = None


class HazardAwarePlanner:
    def __init__(
        self,
        blueprint: Blueprint,
        belief: Rules,
        risk_weight: float = 8.0,
        risk_margin: float = 8.0,
        pace_gap: int = 2,
        max_wait: int = 6,
    ) -> None:
        self.bp = blueprint
        self.belief = belief
        self.risk_weight = risk_weight
        self.risk_margin = risk_margin  # ticks of safety slack we want between us and the fire front
        self.pace_gap = pace_gap
        self.max_wait = max_wait
        self._wait_streak = 0

    # --------------------------------------------------------------- pricing
    def fire_eta(self, burning: frozenset[Cell]) -> dict[Cell, float]:
        """Expected ticks until fire reaches each cell, under the BELIEVED spread rates."""
        bp = self.bp

        def crossing_time(cell: Cell) -> float:
            material = bp.material(cell)
            if not bp.passable(cell) or material in (None, "exit"):
                return math.inf
            return min(60.0, 1.0 / max(self.belief.spread_prob[material], 1e-3))

        if not burning:
            return {}
        return dijkstra_multi(sorted(burning), crossing_time, bp.height, bp.width)[0]

    def _enter_cost(self, obs: Observation):
        bp = self.bp
        burning = obs.burning
        eta_fire = self.fire_eta(burning)
        eta_me = bfs_distances(
            obs.agent_pos,
            lambda c: bp.passable(c) and c not in burning and c not in self.belief.jammed_doors,
            bp.height,
            bp.width,
        )
        cache: dict[Cell, float] = {}

        def cost(cell: Cell) -> float:
            if cell in cache:
                return cache[cell]
            if not bp.passable(cell) or cell in burning or cell in self.belief.jammed_doors:
                out = math.inf
            else:
                slack = eta_fire.get(cell, math.inf) - eta_me.get(cell, 0)
                risk = min(1.0, max(0.0, (self.risk_margin - slack) / self.risk_margin))
                out = 1.0 + self.risk_weight * risk
            cache[cell] = out
            return out

        return cost

    # -------------------------------------------------------------- deciding
    def act(self, obs: Observation) -> Decision:
        bp = self.bp
        cost = self._enter_cost(obs)
        me = obs.agent_pos
        waiting = [d for d in obs.dependents if d.status == WAITING]
        following = [d for d in obs.dependents if d.status == FOLLOWING]

        dist_me, parent_me = dijkstra(me, cost, bp.height, bp.width)

        # Let followers catch up if it is safe to.
        decision = self._maybe_wait_for_followers(obs, following)
        if decision:
            return decision
        self._wait_streak = 0

        # Which waiting people can we actually reach?
        reachable = [d for d in waiting if d.pos in dist_me]
        order = self._best_order(me, reachable, dist_me, cost)

        if order:
            first = order[0]
            path = path_to(parent_me, me, first.pos)
            if len(path) >= 2:
                eta = len(path) - 1
                return Decision(
                    action_toward(me, path[1]),
                    f"Heading to {first.name} ({first.kind}), {eta} steps away.",
                    first.pos,
                )

        exit_cell = self._nearest_exit(dist_me)
        if exit_cell is None:
            return Decision(WAIT, "No safe route to an exit right now; holding position.")
        if exit_cell == me:
            return Decision(WAIT, "At the exit; waiting for everyone to arrive.", me)
        path = path_to(parent_me, me, exit_cell)
        carrying = len(following)
        who = f"leading {carrying} to" if carrying else "heading to"
        return Decision(
            action_toward(me, path[1]),
            f"Route clear: {who} the exit, {len(path) - 1} steps.",
            exit_cell,
        )

    # --------------------------------------------------------------- helpers
    def _nearest_exit(self, dist: dict[Cell, float]) -> Cell | None:
        options = [e for e in self.bp.exits if e in dist]
        return min(options, key=lambda e: dist[e]) if options else None

    def _best_order(self, me, people: list[DependentView], dist_me, cost) -> list[DependentView]:
        if not people:
            return []
        bp = self.bp
        from_person = {p.id: dijkstra(p.pos, cost, bp.height, bp.width)[0] for p in people}
        best: tuple[float, list[DependentView]] | None = None
        for perm in itertools.permutations(people):
            total, here_dist = 0.0, dist_me
            feasible = True
            for p in perm:
                leg = here_dist.get(p.pos)
                if leg is None:
                    feasible = False
                    break
                total += leg
                here_dist = from_person[p.id]
            if not feasible:
                continue
            exits = [here_dist[e] for e in bp.exits if e in here_dist]
            if not exits:
                continue
            total += min(exits)
            if best is None or total < best[0]:
                best = (total, list(perm))
        return best[1] if best else []

    def _maybe_wait_for_followers(self, obs: Observation, following: list[DependentView]):
        if not following or self._wait_streak >= self.max_wait:
            return None
        bp = self.bp
        dist = bfs_distances(
            obs.agent_pos,
            lambda c: bp.passable(c) and c not in self.belief.jammed_doors,
            bp.height,
            bp.width,
        )
        laggard = max(following, key=lambda f: dist.get(f.pos, 0))
        gap = dist.get(laggard.pos, 0)
        danger = any(manhattan(obs.agent_pos, b) <= 3 for b in obs.burning)
        if gap > self.pace_gap and not danger:
            self._wait_streak += 1
            return Decision(WAIT, f"Waiting for {laggard.name} to catch up ({gap} steps behind).")
        return None
