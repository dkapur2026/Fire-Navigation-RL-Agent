"""The reward: get everyone out fast, and keep the people you are rescuing away from flames.

Two goals, in the words of the brief:
  * EFFICIENT NAVIGATION: bring the dependents to the exit in as little time as possible.
  * FIRE SAFETY: penalize a dependent being NEAR flames, and heavily penalize it TOUCHING them.

Components (every step reports each one separately, see ``RewardBreakdown``):

  time            -step_cost every tick. The pressure to be quick.
  progress        progress_weight * (potential(s') - potential(s)). Dense "you are getting closer" signal.
  rescue          +rescue_bonus per dependent that reaches an exit.
  success         +success_bonus when everyone is out and the rescuer is at an exit.
  fire_proximity  -proximity_weight * sum over waiting/following dependents of closeness-to-fire in [0, 1).
  fire_contact    -contact_penalty per dependent lost to flames (it touched fire).
  agent_death     if the rescuer dies: charged everything it would have paid by carrying on to the horizon.

Reward-hacking defences (each has a test in tests/test_reward.py and a row in docs/REWARD.md):

  1. Progress is POTENTIAL-BASED with gamma = 1. Its sum over any episode telescopes to
     potential(end) - potential(start), so no loop or oscillation can farm it, and it cannot change which
     policy is optimal (Ng, Harada & Russell 1999). It only makes learning easier.
  2. A lost dependent can never RAISE the potential. Without this, losing someone would delete their distance
     term and pay the agent for it. Lost dependents are held at a fixed worse-than-anything value instead.
  3. Failing early never costs less than carrying on. If the rescuer dies, the meter keeps running: it is
     charged the per-tick rate it was paying (time + proximity) until the horizon, plus contact for everyone
     left behind, plus ``death_extra``. The naive "-1 on death" was cheaper than a 150-tick timeout at -0.01 a
     tick, so dying could beat trying. (Tested on static hazards, not proved: if fire later gets nearer, staying
     alive could cost more than the charge assumes, though dependents are then usually lost and charged anyway.)
  4. Abandoning a dependent loses the rescue and success bonuses AND moves the agent away from the nearest
     waiting dependent, which the potential charges for.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from emberpath.grid import Cell, bfs_distances, bfs_multi
from emberpath.world import FOLLOWING, LOST, WAITING, World

COMPONENTS = (
    "time",
    "progress",
    "rescue",
    "success",
    "fire_proximity",
    "fire_contact",
    "agent_death",
)


@dataclass(frozen=True)
class RewardConfig:
    step_cost: float = 0.02
    progress_weight: float = 0.03  # per cell of potential
    rescue_bonus: float = 1.0
    success_bonus: float = 1.0
    proximity_weight: float = 0.10  # per dependent per tick at the closest non-touching distance (scaled by closeness)
    proximity_radius: int = 4  # fire this many steps away or farther costs nothing
    contact_penalty: float = 1.0
    death_extra: float = 1.0


@dataclass(frozen=True)
class RewardBreakdown:
    components: dict[str, float]

    @property
    def total(self) -> float:
        return sum(self.components.values())


def proximity(distance: int, radius: int) -> float:
    """Closeness to fire in [0, 1): 0 at `radius` or farther, rising linearly as fire gets nearer.
    Distance 0 means touching; a touching dependent is already lost, handled by `fire_contact`."""
    if distance < 1:
        return 1.0
    return max(0.0, (radius - distance) / radius)


def mission_lower_bound(world: World) -> float:
    """Fewest steps the rescuer needs, ignoring fire and how fast anyone else moves:
    best order of visiting every dependent, then the nearest exit. inf if impossible."""
    bp = world.blueprint
    jammed = world.rules.jammed_doors

    def passable(c: Cell) -> bool:
        return bp.passable(c) and c not in jammed

    people = [d.pos for d in world.dependents if d.status in (WAITING, FOLLOWING)]
    from_agent = bfs_distances(world.agent_pos, passable, bp.height, bp.width)
    from_person = {p: bfs_distances(p, passable, bp.height, bp.width) for p in set(people)}

    def to_exit(dist: dict[Cell, int]) -> float:
        return min((dist[e] for e in bp.exits if e in dist), default=math.inf)

    if not people:
        return to_exit(from_agent)
    best = math.inf
    for order in itertools.permutations(people):
        total, dist = 0.0, from_agent
        for p in order:
            total += dist.get(p, math.inf)
            dist = from_person[p]
        best = min(best, total + to_exit(dist))
    return best


class RewardTracker:
    """Call ``update(world)`` after every ``world.step``; it returns that tick's ``RewardBreakdown``."""

    def __init__(self, world: World, config: RewardConfig | None = None) -> None:
        self.cfg = config or RewardConfig()
        self.bp = world.blueprint
        self._exit_cache: dict[frozenset, tuple[dict[Cell, int], int]] = {}
        self._prev_phi = self.potential(world)
        self._prev_counts = world.counts()
        self._was_alive = world.agent_alive
        self.total = 0.0
        self.lower_bound = mission_lower_bound(world)

    # ----------------------------------------------------------- geometry
    def _exit_distances(self, world: World) -> tuple[dict[Cell, int], int]:
        """Steps from every walkable cell to the nearest exit (fire ignored; doors that are really jammed block)."""
        key = frozenset(world.rules.jammed_doors)
        if key not in self._exit_cache:
            bp = self.bp
            dist = bfs_multi(bp.exits, lambda c: bp.passable(c) and c not in key, bp.height, bp.width)
            self._exit_cache[key] = (dist, max(dist.values(), default=0))
        return self._exit_cache[key]

    def potential(self, world: World) -> float:
        """Higher = closer to done. Minus the steps the dependents still need to reach an exit, minus the steps
        from the rescuer to the nearest dependent still waiting, minus a fixed worst value per lost dependent."""
        exit_d, d_max = self._exit_distances(world)
        bp = self.bp
        jammed = world.rules.jammed_doors
        lost_value = 2 * d_max  # >= exit distance + pickup distance of anyone, so losing someone never helps
        phi = 0.0
        waiting: list[Cell] = []
        for d in world.dependents:
            if d.status in (WAITING, FOLLOWING):
                phi -= exit_d.get(d.pos, lost_value)
                if d.status == WAITING:
                    waiting.append(d.pos)
            elif d.status == LOST:
                phi -= lost_value
        if waiting:
            dist = bfs_distances(
                world.agent_pos, lambda c: bp.passable(c) and c not in jammed, bp.height, bp.width
            )
            phi -= min(dist.get(p, lost_value) for p in waiting)
        return phi

    def fire_proximity(self, world: World) -> float:
        """Sum over waiting/following dependents of closeness-to-fire (unweighted), measured along walkable paths
        (fire does not leap through walls)."""
        burning = world.burning_cells()
        if not burning:
            return 0.0
        r = self.cfg.proximity_radius
        bp = self.bp
        dist = bfs_multi(burning, bp.passable, bp.height, bp.width, max_depth=r)
        return sum(
            proximity(dist[d.pos], r)
            for d in world.dependents
            if d.status in (WAITING, FOLLOWING) and d.pos in dist
        )

    # -------------------------------------------------------------- reward
    def update(self, world: World) -> RewardBreakdown:
        cfg = self.cfg
        counts = world.counts()
        phi = self.potential(world)
        comp = dict.fromkeys(COMPONENTS, 0.0)
        comp["time"] = -cfg.step_cost
        comp["progress"] = cfg.progress_weight * (phi - self._prev_phi)
        comp["rescue"] = cfg.rescue_bonus * (counts["rescued"] - self._prev_counts["rescued"])
        comp["success"] = cfg.success_bonus if world.outcome == "success" else 0.0
        prox = self.fire_proximity(world)
        comp["fire_proximity"] = -cfg.proximity_weight * prox
        comp["fire_contact"] = -cfg.contact_penalty * (counts["lost"] - self._prev_counts["lost"])
        if self._was_alive and not world.agent_alive:
            # Dying does not stop the meter: keep charging the per-tick rate being paid right now until the
            # horizon, plus contact for everyone left behind, plus a flat extra.
            unresolved = sum(d.status in (WAITING, FOLLOWING) for d in world.dependents)
            remaining = max(0, world.max_steps - world.t)
            rate = cfg.step_cost + cfg.proximity_weight * prox
            comp["agent_death"] = -(rate * remaining + cfg.contact_penalty * unresolved + cfg.death_extra)
        self._prev_phi, self._prev_counts, self._was_alive = phi, counts, world.agent_alive
        out = RewardBreakdown(comp)
        self.total += out.total
        return out

    def efficiency(self, world: World) -> float | None:
        """Fewest-possible steps / steps actually taken, in (0, 1]. Only defined for fully successful episodes.
        1.0 means as fast as geometry allows (slower dependents mean even a perfect run is often below 1)."""
        if world.outcome != "success" or not math.isfinite(self.lower_bound) or world.t == 0:
            return None
        return min(1.0, max(self.lower_bound, 1.0) / world.t)
