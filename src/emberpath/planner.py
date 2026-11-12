"""Phase 3: a symbolic, hazard-aware planner. No learning.

It plans with a *belief* (a ``Rules`` object). If the belief is wrong, the plan is wrong. That
is the whole point of the project: this planner is the "brain" whose beliefs get repaired.

How it decides, every tick:
  1. Predict when fire will reach each cell (expected fire turns = 1 / believed spread probability per
     cell crossed, from every flame), then turn fire turns into ticks with the public fire clock (the
     fire starts slowly: ``world.FireClock``). Compare with when the rescuer could get there. Cells the
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
    bfs_multi,
    dijkstra,
    dijkstra_multi,
    manhattan,
    neighbors4,
    path_to,
)
from emberpath.rules import Rules
from emberpath.world import (
    CARRY_EVERY,
    CARRY_RADIUS,
    FOLLOWING,
    PANIC_RADIUS,
    WAITING,
    DependentView,
    Observation,
    fire_ticks_for_rounds,
)


@dataclass(frozen=True)
class Decision:
    action: int
    reason: str  # plain-English explanation, shown to humans
    target: Cell | None = None
    explanation: str = ""  # set only on a tick where an adaptive agent changed its mind: what changed and why (explain.py)


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
        go_back_for_children: bool = True,
        fire_escape: bool = True,
        out_when_burned: bool = True,
    ) -> None:
        self.bp = blueprint
        self.belief = belief
        self.risk_weight = risk_weight
        self.risk_margin = risk_margin  # ticks of safety slack we want between us and the fire front
        self.pace_gap = pace_gap
        self.max_wait = max_wait
        # `fire_eta` follows one chain of mean crossing times, but real fire tries every burning neighbour each
        # tick: on ranch the fire's 10th-percentile arrival is 0.44-0.59x the forecast (200 fires, 5 cells). To
        # judge "can I reach them before the fire?" we plan against that bad case, not the average. The margin
        # scales fire TURNS (that is where the spread is random), and the result is then timed by the fire clock.
        self.save_margin = save_margin
        # Turn back to carry a following child with flames near them. Without burn tolerance this roughly broke even and
        # doubled rescuer deaths in carpet fires; with adults surviving 2-5 ticks in flames it is a net gain (summed
        # baseline + oracle score over 24 child cells 36.29 -> 36.77), though house family_dog carpet still loses.
        self.go_back_for_children = go_back_for_children
        # Standing in flames: get clear of them first (stepping through burning cells if that is the way out).
        # Off = the old behaviour (flames are walls, even underfoot, so a rescuer caught by the fire stood still).
        self.fire_escape = fire_escape
        self.out_when_burned = out_when_burned  # once burned, no new pickups: straight out with whoever is with me
        self._wait_streak = 0
        self._searched: set[Cell] = set()  # cells the rescuer has stood on or next to (from step 1 on)
        self._heading_to: int | None = None  # whom the last decision went for: kept unless switching saves more
        self._crossing: tuple[dict[str, float], dict[Cell, float]] | None = None  # (rates, table) cache

    # --------------------------------------------------------------- pricing
    def fire_rounds(self, burning: frozenset[Cell], burnt: frozenset[Cell] = frozenset()) -> dict[Cell, float]:
        """Expected fire TURNS until fire reaches each cell, under the BELIEVED spread rates. Like the real fire, it
        moves only between 4-neighbouring floor cells, never into an exit, and never through a cell that has burnt
        out (a burnt cell cannot catch again, so it is a firebreak)."""
        if not burning:
            return {}
        bp = self.bp
        table = self._crossing_times()
        cost = table.__getitem__ if not burnt else (lambda c: math.inf if c in burnt else table[c])
        return dijkstra_multi(sorted(burning), cost, bp.height, bp.width)[0]

    def fire_eta(
        self, burning: frozenset[Cell], burnt: frozenset[Cell] = frozenset(), *, step: int, margin: float = 1.0
    ) -> dict[Cell, float]:
        """Expected TICKS until fire reaches each cell, seen from an observation at tick `step`: `fire_rounds` (scaled
        by `margin`, for a pessimistic forecast) timed by the public fire clock, which is slow early on. `step` is
        required: the same number of fire turns takes more ticks at the start of an episode than later."""
        return {c: fire_ticks_for_rounds(step, margin * r) for c, r in self.fire_rounds(burning, burnt).items()}

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
        """Cost of stepping into each cell; flames and believed-jammed doors seen from inside are walls."""
        bp = self.bp
        burning = obs.burning
        cache: dict[Cell, float] = {}

        def cost(cell: Cell) -> float:
            if cell in cache:
                return cache[cell]
            if not bp.passable(cell):
                out = math.inf
            elif cell in burning:
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
        mine = [d for d in obs.dependents if d.status == FOLLOWING and d.leader is None]
        following = [d for d in mine if not d.carried]  # walking behind me; the carried are with me already
        if obs.step > 0:  # a dependents update has run with the rescuer here: anyone adjacent has bonded
            self._searched.update(neighbors4(me, bp.height, bp.width), (me,))
        if self.belief.occupied:
            waiting += self._hidden_targets(obs, waiting)
        adults = [d for d in obs.dependents if d.status == FOLLOWING and d.leader == d.id]
        needy = {a.id: why for a in adults if (why := self._needs_help(a, obs))}
        waiting += [a for a in adults if a.id in needy]
        # A child following me who has fallen behind with flames close to them: go back and carry them.
        near = bfs_multi(obs.burning, self.bp.passable, self.bp.height, self.bp.width, max_depth=CARRY_RADIUS)
        for d in following if self.go_back_for_children else ():
            if d.kind == "child" and d.pos in near and manhattan(d.pos, me) > 1:
                needy[d.id] = "the fire is close and they are behind me; I will carry them"
                waiting.append(d)

        # Standing in flames: before anything else, get to the nearest cell that is not burning.
        if self.fire_escape and me in obs.burning:
            return self._escape_fire(obs)

        # Let followers catch up if it is safe to.
        decision = self._maybe_wait_for_followers(obs, following)
        if decision:
            return decision
        self._wait_streak = 0

        rounds = self.fire_rounds(obs.burning, obs.burnt)
        eta_fire = {c: fire_ticks_for_rounds(obs.step, r) for c, r in rounds.items()}
        # the pessimistic forecast ("can I get there before the fire?"): save_margin x the fire turns, then timed
        eta_safe = {c: fire_ticks_for_rounds(obs.step, self.save_margin * r) for c, r in rounds.items()}
        steps_me = self._steps_from(me, obs.burning, obs.broken_doors)
        cost = self._enter_cost(obs, eta_fire, steps_me)
        dist_me, parent_me = dijkstra(me, cost, bp.height, bp.width)

        # Which waiting people can we actually reach, and which will an adult in the household fetch first?
        helped = self._fetched_by_adults(obs, [d for d in waiting if d.id not in needy], steps_me, set(needy))
        reachable = [d for d in waiting if d.pos in dist_me and d.id not in helped]
        if self.out_when_burned and obs.burns:  # burned already: no more pickups, out with whoever is with me
            reachable = []
        order, urgent = self._best_order(obs, reachable, dist_me, cost, eta_fire, steps_me, following, eta_safe)

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
                    why += (" Children first, and the fire will reach this one soonest." if _rank(first) == 0
                            else " They go before the closer pickup: the others are safer for now.")
                return Decision(action_toward(me, path[1]), why, first.pos)

        pace = self._pace(mine)
        exit_cell, detour = self._exit_in_time(me, dist_me, parent_me, eta_safe, pace)
        if exit_cell is None and not following:  # nobody walking behind me: crossing a little fire is survivable
            crossing = self._cross_to_exit(obs)
            if crossing:
                return crossing
        if exit_cell is None:  # flames burn out after a while; crossing them would leave walking followers behind
            return Decision(WAIT, "No safe route to an exit right now; holding position.")
        if exit_cell == me:
            return Decision(WAIT, "At the exit; waiting for everyone to arrive.", me)
        path = path_to(parent_me, me, exit_cell)
        n_carried = sum(d.carried for d in mine)
        who = f"leading {len(mine)} to" if mine else "heading to"
        if self.out_when_burned and obs.burns:
            who = f"burned, so straight out: {who}"
        held = f" (carrying {n_carried})" if n_carried else ""
        why = " The nearer exit is one we might not reach before the fire." if detour else ""
        also = "".join(f" {helper} is getting {name}." for name, helper in helped.values())
        return Decision(
            action_toward(me, path[1]),
            f"Route clear: {who} the exit{held}, {_steps(len(path) - 1)}.{why}{also}",
            exit_cell,
        )

    # --------------------------------------------------------------- helpers
    def _hidden_targets(self, obs: Observation, waiting: list[DependentView]) -> list[DependentView]:
        """Believed-occupied cells not yet searched or burnt, as stand-in people (negative ids) to collect."""
        seen = {d.pos for d in waiting}
        cells = sorted(self.belief.occupied - self._searched - obs.burnt - seen)
        return [DependentView(-1 - i, "unknown", "someone", c, WAITING) for i, c in enumerate(cells)]

    def _cross_to_exit(self, obs: Observation) -> Decision | None:
        """The exit reached through the fewest burning cells (then fewest steps), if that is fewer burning cells than
        the burns I have left. Only used when everyone with me is carried: walking followers cannot cross flames."""
        bp, me = self.bp, obs.agent_pos
        jammed = self.belief.jammed_doors - obs.broken_doors
        cost = lambda c: (math.inf if not bp.passable(c) or c in jammed else 1.0 + 100.0 * (c in obs.burning))  # noqa: E731
        dist, parent = dijkstra(me, cost, bp.height, bp.width)
        exits = [e for e in bp.exits if e in dist]
        if not exits:
            return None
        exit_cell = min(exits, key=dist.get)
        path = path_to(parent, me, exit_cell)
        fire = sum(c in obs.burning for c in path[1:])
        if fire >= obs.burns_left:
            return None
        return Decision(action_toward(me, path[1]), f"Every way out is burning: crossing {fire} burning "
                        f"{'cell' if fire == 1 else 'cells'} to the exit, {_steps(len(path) - 1)}.", exit_cell)

    def _escape_fire(self, obs: Observation) -> Decision:
        """Step toward the nearest safe place: an exit (fire never enters one, and it is the way out) or a cell at least
        2 steps from any flame; else any cell that is not burning. Ties: an exit, then where the fire is slowest."""
        bp, me = self.bp, obs.agent_pos
        jammed = self.belief.jammed_doors - obs.broken_doors
        dist = bfs_distances(me, lambda c: bp.passable(c) and c not in jammed, bp.height, bp.width)
        near_fire = bfs_multi(obs.burning, bp.passable, bp.height, bp.width, max_depth=1)
        exits = set(bp.exits)
        safe = [c for c in dist if c not in near_fire or c in exits] or [c for c in dist if c not in obs.burning]
        if not safe:
            return Decision(WAIT, "I am in the flames with nowhere to go.")
        eta = self.fire_eta(obs.burning, obs.burnt, step=obs.step)
        target = min(safe, key=lambda c: (dist[c], c not in exits, -eta.get(c, math.inf)))
        parent = dijkstra(me, lambda c: 1.0 if bp.passable(c) and c not in jammed else math.inf, bp.height, bp.width)[1]
        path = path_to(parent, me, target)
        return Decision(action_toward(me, path[1]), f"I am burning ({obs.burns_left} more and I fall): getting clear of "
                        f"the flames first, {_steps(len(path) - 1)}, then out.", target)

    def _needs_help(self, a: DependentView, obs: Observation) -> str:
        """Why an adult acting alone needs the rescuer, or "" if they can manage. Only what the rescuer can see:
        panic is visible, and the fire and the floor plan say whether they are about to panic or are cut off."""
        if a.panicked:
            return "they are panicking"
        bp = self.bp  # walking distance, as the world measures panic range: flames behind a wall do not count
        if a.pos in bfs_multi(obs.burning, bp.passable, bp.height, bp.width, max_depth=PANIC_RADIUS + 1):
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

    @staticmethod
    def _pace(mine: list[DependentView]) -> float:
        """Ticks per step for the group I lead: the slowest walker, and 4/3 while I carry someone."""
        walk = max((d.pace for d in mine if not d.carried), default=1)
        return walk * (CARRY_EVERY / (CARRY_EVERY - 1) if any(d.carried for d in mine) else 1)

    def _exit_in_time(self, me, dist, parent, eta_safe, pace) -> tuple[Cell | None, bool]:
        """The cheapest exit whose route everyone I lead covers before the pessimistic fire forecast (`eta_safe`, in
        ticks, see `fire_eta(margin=save_margin)`) at every cell; if none is safe, the cheapest exit. Returns (exit,
        whether a nearer exit was passed over as unsafe)."""
        options = sorted((e for e in self.bp.exits if e in dist), key=dist.get)
        for e in options:
            path = path_to(parent, me, e)
            if all(i * pace < eta_safe.get(c, math.inf) for i, c in enumerate(path)):
                return e, e != options[0]
        return (options[0] if options else None), False

    def _best_order(self, obs, people, dist_me, cost, eta_fire, steps_me, following, eta_safe):
        """Best pickup order, by priority:
          1. CHILDREN reached before the (pessimistic) fire forecast, the most endangered child first: saving the
             child the fire reaches soonest outranks saving a safer one;
          2. nobody of lower priority before someone of higher (children, then adults, then the dog), so the dog
             never delays a child;
          3. most people reached in time;
          4. keep going for whoever I was already heading to (no flip-flopping between equal plans);
          5. first the person the fire reaches soonest (the forecast follows walls and doors);
          6. the cheapest route that ends at an exit.
        Returns (order, urgent), where `urgent` says the chosen first pickup is not the one the cheapest
        route would have started with (so the reason can say why)."""
        if not people:
            return [], False
        bp = self.bp
        from_person = {p.id: dijkstra(p.pos, cost, bp.height, bp.width)[0] for p in people}
        steps_from = {p.id: self._steps_from(p.pos, obs.burning) for p in people}
        fetching = {p.id for p in people}  # going back for a follower is not slowed by that follower
        pace0 = self._pace([d for d in obs.dependents if d.status == FOLLOWING and d.leader is None and d.id not in fetching])
        kids_by_danger = sorted((p.id for p in people if _rank(p) == 0), key=lambda i: eta_fire.get(
            next(p.pos for p in people if p.id == i), math.inf))
        best = cheapest = None
        for perm in itertools.permutations(people):
            total, here_dist, here_steps = 0.0, dist_me, steps_me
            t, pace, saved, reached = 0.0, pace0, 0, set()
            for p in perm:
                leg = here_dist.get(p.pos)
                if leg is None or p.pos not in here_steps:
                    break
                total += leg
                t += here_steps[p.pos] * pace
                in_time = t < eta_safe.get(p.pos, math.inf)
                saved += in_time
                if in_time:
                    reached.add(p.id)
                pace = max(pace, p.pace)
                here_dist, here_steps = from_person[p.id], steps_from[p.id]
            else:
                exits = [here_dist[e] for e in bp.exits if e in here_dist]
                if exits:
                    total += min(exits)
                    first = perm[0]
                    kids = tuple(i not in reached for i in kids_by_danger)  # False (= saved) sorts first
                    inversions = sum(_rank(a) > _rank(b) for a, b in itertools.combinations(perm, 2))
                    key = (kids, inversions, -saved, first.id != self._heading_to, eta_fire.get(first.pos, math.inf), total)
                    if best is None or key < best[0]:
                        best = (key, list(perm))
                    if cheapest is None or total < cheapest[0]:
                        cheapest = (total, perm[0].id)
        if best is None:
            return [], False
        self._heading_to = best[1][0].id
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
        near = bfs_multi(obs.burning, bp.passable, bp.height, bp.width, max_depth=3)  # walking: walls stop fire
        danger = obs.agent_pos in near
        if gap > self.pace_gap and not danger:
            self._wait_streak += 1
            return Decision(WAIT, f"Waiting for {laggard.name} to catch up ({gap} steps behind).")
        return None


def _rank(d: DependentView) -> int:
    """Rescue priority: children (and a hidden person, most likely a child) 0, adults 1, the dog 2."""
    return {"child": 0, "unknown": 0, "dog": 2}.get(d.kind, 1)


def _steps(n: int) -> str:
    return f"{n} step" if n == 1 else f"{n} steps"
