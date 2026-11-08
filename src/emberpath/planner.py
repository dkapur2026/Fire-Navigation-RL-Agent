"""Phase 3: a symbolic, hazard-aware planner. No learning.

It plans with a *belief* (a ``Rules`` object). If the belief is wrong, the plan is wrong. That
is the whole point of the project: this planner is the "brain" whose beliefs get repaired.

How it decides, every tick:
  1. Predict when fire will reach each cell (expected ticks = 1 / believed spread probability per
     cell crossed, from every flame). Compare with when the rescuer could get there. Cells the
     fire is likely to reach first cost extra: 1 + risk_weight * risk. Burning cells and
     believed-jammed doors are walls. Because the forecast uses the believed spread rates, a
     wrong belief (fire is faster than I think) produces wrong routes.
  2. Try every order of collecting the people still waiting (brute force, fine for <= 3). For each
     order, estimate when the rescuer reaches each person (slowing to the slowest follower's pace once
     someone follows) and whether that is before the forecast fire gets there. Pick the order that
     reaches the most people ahead of the fire, then the cheapest route that ends at an exit. So when
     fire is closing in on one person, they are fetched first if that saves more people in total. Cells the belief marks ``occupied``
     (someone unexpected is hiding there) count as people waiting, until the rescuer has stood next
     to the cell (whoever hid there is then following) or it has burnt.
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
    neighbors4,
    path_to,
)
from emberpath.rules import Rules
from emberpath.world import FOLLOWING, PANIC_RADIUS, WAITING, DependentView, Observation


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
        save_margin: float = 0.5,
    ) -> None:
        self.bp = blueprint
        self.belief = belief
        self.risk_weight = risk_weight
        self.risk_margin = risk_margin  # ticks of safety slack we want between us and the fire front
        self.pace_gap = pace_gap
        self.max_wait = max_wait
        # `fire_eta` follows one chain of mean crossing times, but real fire tries every burning neighbour each
        # tick: on ranch the fire's 10th-percentile arrival is 0.44-0.59x the forecast (200 fires, 5 cells). To
        # judge "can I reach them before the fire?" we plan against that bad case, not the average.
        self.save_margin = save_margin
        self._wait_streak = 0
        self._searched: set[Cell] = set()  # cells the rescuer has stood on or next to (from step 1 on)
        self._crossing: tuple[dict[str, float], dict[Cell, float]] | None = None  # (rates, table) cache

    # --------------------------------------------------------------- pricing
    def fire_eta(self, burning: frozenset[Cell]) -> dict[Cell, float]:
        """Expected ticks until fire reaches each cell, under the BELIEVED spread rates."""
        if not burning:
            return {}
        bp = self.bp
        return dijkstra_multi(sorted(burning), self._crossing_times().__getitem__, bp.height, bp.width)[0]

    def _crossing_times(self) -> dict[Cell, float]:
        """Ticks for fire to cross each cell under the believed spread rates; rebuilt only when those change."""
        rates = self.belief.spread_prob
        if self._crossing is None or self._crossing[0] != rates:
            bp = self.bp
            table = {}
            for cell in itertools.product(range(bp.height), range(bp.width)):
                material = bp.material(cell)
                if not bp.passable(cell) or material in (None, "exit"):
                    table[cell] = math.inf
                else:
                    table[cell] = min(60.0, 1.0 / max(rates[material], 1e-3))
            self._crossing = (dict(rates), table)
        return self._crossing[1]

    def _steps_from(self, cell: Cell, burning: frozenset[Cell], broken: frozenset[Cell] = frozenset()) -> dict[Cell, int]:
        """Walking distance in ticks, around flames and believed-jammed (unbroken) doors."""
        bp = self.bp
        jammed = self.belief.jammed_doors - broken
        return bfs_distances(cell, lambda c: bp.passable(c) and c not in burning and c not in jammed, bp.height, bp.width)

    def _enter_cost(self, obs: Observation, eta_fire: dict[Cell, float], eta_me: dict[Cell, int]):
        bp = self.bp
        burning = obs.burning
        cache: dict[Cell, float] = {}

        def cost(cell: Cell) -> float:
            if cell in cache:
                return cache[cell]
            if not bp.passable(cell) or cell in burning:
                out = math.inf
            elif cell in self.belief.jammed_doors and cell not in obs.broken_doors:
                # forcible only from the hallway side: expected ticks to kick it in; from inside a room, a wall
                outside = obs.agent_pos in bp.door_outside.get(cell, ())
                out = 1.0 / max(self.belief.force_prob["door"], 1e-3) if outside else math.inf
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
        me = obs.agent_pos
        waiting = [d for d in obs.dependents if d.status == WAITING]
        following = [d for d in obs.dependents if d.status == FOLLOWING and d.leader is None]
        if obs.step > 0:  # a dependents update has run with the rescuer here: anyone adjacent has bonded
            self._searched.update(neighbors4(me, bp.height, bp.width), (me,))
        if self.belief.occupied:
            waiting += self._hidden_targets(obs, waiting)
        adults = [d for d in obs.dependents if d.status == FOLLOWING and d.leader == d.id]
        needy = {a.id: why for a in adults if (why := self._needs_help(a, obs))}
        waiting += [a for a in adults if a.id in needy]

        # Let followers catch up if it is safe to.
        decision = self._maybe_wait_for_followers(obs, following)
        if decision:
            return decision
        self._wait_streak = 0

        eta_fire = self.fire_eta(obs.burning)
        steps_me = self._steps_from(me, obs.burning, obs.broken_doors)
        cost = self._enter_cost(obs, eta_fire, steps_me)
        dist_me, parent_me = dijkstra(me, cost, bp.height, bp.width)

        # Which waiting people can we actually reach, and which will an adult in the household fetch first?
        helped = self._fetched_by_adults(obs, [d for d in waiting if d.id not in needy], steps_me, set(needy))
        reachable = [d for d in waiting if d.pos in dist_me and d.id not in helped]
        order, urgent = self._best_order(obs, reachable, dist_me, cost, eta_fire, steps_me, following)

        if order:
            first = order[0]
            path = path_to(parent_me, me, first.pos)
            if len(path) >= 2:
                eta = _steps(len(path) - 1)
                if first.id < 0:
                    why = f"I heard someone at {first.pos}; going to find them, {eta} away."
                elif first.id in needy:
                    why = f"Going to help {first.name}: {needy[first.id]}, {eta} away."
                else:
                    why = f"Heading to {first.name} ({first.kind}), {eta} away."
                if urgent:
                    why += " The fire will reach them first, so they go before the closer pickup."
                return Decision(action_toward(me, path[1]), why, first.pos)

        exit_cell = self._nearest_exit(dist_me)
        if exit_cell is None:
            return Decision(WAIT, "No safe route to an exit right now; holding position.")
        if exit_cell == me:
            return Decision(WAIT, "At the exit; waiting for everyone to arrive.", me)
        path = path_to(parent_me, me, exit_cell)
        carrying = len(following)
        who = f"leading {carrying} to" if carrying else "heading to"
        also = "".join(f" {helper} is getting {name}." for name, helper in helped.values())
        return Decision(
            action_toward(me, path[1]),
            f"Route clear: {who} the exit, {_steps(len(path) - 1)}.{also}",
            exit_cell,
        )

    # --------------------------------------------------------------- helpers
    def _hidden_targets(self, obs: Observation, waiting: list[DependentView]) -> list[DependentView]:
        """Believed-occupied cells not yet searched or burnt, as stand-in people (negative ids) to collect."""
        seen = {d.pos for d in waiting}
        cells = sorted(self.belief.occupied - self._searched - obs.burnt - seen)
        return [DependentView(-1 - i, "unknown", "someone", c, WAITING) for i, c in enumerate(cells)]

    def _needs_help(self, a: DependentView, obs: Observation) -> str:
        """Why an adult acting alone needs the rescuer, or "" if they can manage. Only what the rescuer can see:
        panic is visible, and the fire and the floor plan say whether they are about to panic or are cut off."""
        if a.panicked:
            return "they are panicking"
        if any(manhattan(a.pos, b) <= PANIC_RADIUS + 1 for b in obs.burning):
            return "the fire is close to them"
        theirs = self._steps_from(a.pos, obs.burning, obs.broken_doors)
        if not any(e in theirs for e in self.bp.exits):
            return "they have no way out"
        return ""

    def _fetched_by_adults(self, obs: Observation, waiting, steps_me, needy=frozenset()) -> dict[int, tuple[str, str]]:
        """Waiting people an adult acting alone will reach before the rescuer could: {id: (name, adult's name)}.
        Mirrors the adult's known habit (nearest waiting child first), using only what the rescuer can see."""
        out: dict[int, tuple[str, str]] = {}
        known = [d for d in waiting if d.id >= 0 and d.status == WAITING]
        for a in (d for d in obs.dependents if d.status == FOLLOWING and d.leader == d.id and d.id not in needy):
            theirs = self._steps_from(a.pos, obs.burning)
            options = [d for d in known if d.pos in theirs and d.id not in out]
            if options:
                child = min(options, key=lambda d: theirs[d.pos])
                if theirs[child.pos] <= steps_me.get(child.pos, math.inf):
                    out[child.id] = (child.name, a.name)
        return out

    def _nearest_exit(self, dist: dict[Cell, float]) -> Cell | None:
        options = [e for e in self.bp.exits if e in dist]
        return min(options, key=lambda e: dist[e]) if options else None

    def _best_order(self, obs, people, dist_me, cost, eta_fire, steps_me, following):
        """Best pickup order: most people reached before the forecast fire, then cheapest route to an exit.
        Returns (order, urgent), where `urgent` says the chosen first pickup is not the one the cheapest
        route would have started with (so the reason can say why)."""
        if not people:
            return [], False
        bp = self.bp
        from_person = {p.id: dijkstra(p.pos, cost, bp.height, bp.width)[0] for p in people}
        steps_from = {p.id: self._steps_from(p.pos, obs.burning) for p in people}
        pace0 = max((f.pace for f in following), default=1)
        best = cheapest = None
        for perm in itertools.permutations(people):
            total, here_dist, here_steps = 0.0, dist_me, steps_me
            t, pace, saved = 0.0, pace0, 0
            for p in perm:
                leg = here_dist.get(p.pos)
                if leg is None or p.pos not in here_steps:
                    break
                total += leg
                t += here_steps[p.pos] * pace
                saved += t < self.save_margin * eta_fire.get(p.pos, math.inf)
                pace = max(pace, p.pace)
                here_dist, here_steps = from_person[p.id], steps_from[p.id]
            else:
                exits = [here_dist[e] for e in bp.exits if e in here_dist]
                if exits:
                    total += min(exits)
                    if best is None or (-saved, total) < best[0]:
                        best = ((-saved, total), list(perm))
                    if cheapest is None or total < cheapest[0]:
                        cheapest = (total, perm[0].id)
        if best is None:
            return [], False
        return best[1], best[1][0].id != cheapest[1]

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


def _steps(n: int) -> str:
    return f"{n} step" if n == 1 else f"{n} steps"
