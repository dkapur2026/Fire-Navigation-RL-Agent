"""Small grid utilities shared by the world, the planner and the tests."""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Callable, Iterable, Iterator

Cell = tuple[int, int]

# Action indices. The first four are moves, in this order.
MOVES: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES: tuple[str, ...] = ("up", "down", "left", "right", "wait")
WAIT = 4


def neighbors4(cell: Cell, height: int, width: int) -> Iterator[Cell]:
    r, c = cell
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield (nr, nc)


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def action_toward(src: Cell, dst: Cell) -> int:
    """Action index that moves from src to the adjacent cell dst."""
    delta = (dst[0] - src[0], dst[1] - src[1])
    if delta not in MOVES:
        raise ValueError(f"{dst} is not adjacent to {src}")
    return MOVES.index(delta)


def bfs_distances(
    source: Cell, passable: Callable[[Cell], bool], height: int, width: int
) -> dict[Cell, int]:
    """Unweighted shortest-path distances from `source` to every reachable cell."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        cur = queue.popleft()
        for nxt in neighbors4(cur, height, width):
            if nxt not in dist and passable(nxt):
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


def bfs_multi(
    sources: Iterable[Cell],
    passable: Callable[[Cell], bool],
    height: int,
    width: int,
    max_depth: int | None = None,
) -> dict[Cell, int]:
    """Distance from the nearest of several sources, optionally stopping at `max_depth`."""
    srcs = list(sources)
    dist = {s: 0 for s in srcs}
    queue = deque(srcs)
    while queue:
        cur = queue.popleft()
        if max_depth is not None and dist[cur] >= max_depth:
            continue
        for nxt in neighbors4(cur, height, width):
            if nxt not in dist and passable(nxt):
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


def dijkstra(
    source: Cell,
    enter_cost: Callable[[Cell], float],
    height: int,
    width: int,
) -> tuple[dict[Cell, float], dict[Cell, Cell]]:
    """Weighted shortest paths. `enter_cost(cell)` is the price of stepping onto a cell;
    return math.inf for impassable cells. Returns (distance, parent) maps."""
    return dijkstra_multi([source], enter_cost, height, width)


def dijkstra_multi(
    sources: list[Cell],
    enter_cost: Callable[[Cell], float],
    height: int,
    width: int,
) -> tuple[dict[Cell, float], dict[Cell, Cell]]:
    """Like `dijkstra`, but starts from several sources at once (distance = to the nearest one)."""
    dist: dict[Cell, float] = {s: 0.0 for s in sources}
    parent: dict[Cell, Cell] = {}
    heap: list[tuple[float, Cell]] = [(0.0, s) for s in sources]
    heapq.heapify(heap)
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist.get(cur, math.inf):
            continue
        for nxt in neighbors4(cur, height, width):
            cost = enter_cost(nxt)
            if math.isinf(cost):
                continue
            nd = d + cost
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                parent[nxt] = cur
                heapq.heappush(heap, (nd, nxt))
    return dist, parent


def path_to(parent: dict[Cell, Cell], source: Cell, target: Cell) -> list[Cell]:
    """Reconstruct source -> target from a parent map. Empty list if unreachable."""
    if target == source:
        return [source]
    if target not in parent:
        return []
    path = [target]
    while path[-1] != source:
        path.append(parent[path[-1]])
    path.reverse()
    return path
