"""Blueprints: houses drawn as ASCII floor plans.

Legend
    #  wall                      .  wood floor
    c  carpet (burns fast)       t  tile (burns slowly)
    D  door (a floor cell that can jam)
    E  exit (fireproof)          A  where the rescuer starts
    1 2 3  where dependent #1, #2, #3 start (a household uses the first n slots)
    (A, F and 1-3 sit on floor; their material is inferred from the neighbouring cells)
    F  fire starts here (floor material is inferred from the neighbouring cells)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from emberpath.grid import Cell, bfs_distances

WALL, WOOD, CARPET, TILE, DOOR, EXIT = "#", ".", "c", "t", "D", "E"
AGENT, FIRE = "A", "F"
SLOT_CHARS = "123"

_MATERIAL_OF = {WOOD: "wood", CARPET: "carpet", TILE: "tile", DOOR: "door", EXIT: "exit"}


@dataclass(frozen=True)
class Blueprint:
    name: str
    rows: tuple[str, ...]
    agent_start: Cell
    slots: tuple[Cell, ...]
    exits: tuple[Cell, ...]
    doors: tuple[Cell, ...]
    ignitions: tuple[Cell, ...]
    walls: frozenset[Cell]
    materials: dict[Cell, str]

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0])

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.height and 0 <= cell[1] < self.width

    def passable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.walls

    def material(self, cell: Cell) -> str | None:
        return self.materials.get(cell)

    # ------------------------------------------------------------------ parsing
    @classmethod
    def parse(cls, text: str, name: str = "blueprint") -> "Blueprint":
        rows = tuple(line.rstrip("\n") for line in text.strip("\n").splitlines() if line.strip())
        if not rows:
            raise ValueError("empty blueprint")
        width = len(rows[0])
        for i, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(f"row {i} has length {len(row)}, expected {width}")

        walls: set[Cell] = set()
        materials: dict[Cell, str] = {}
        agent: list[Cell] = []
        slots: dict[int, Cell] = {}
        exits: list[Cell] = []
        doors: list[Cell] = []
        fires: list[Cell] = []

        markers: list[Cell] = []  # A / F / digit cells: floor whose material we infer below
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                cell = (r, c)
                if ch == WALL:
                    walls.add(cell)
                elif ch in _MATERIAL_OF:
                    materials[cell] = _MATERIAL_OF[ch]
                    if ch == EXIT:
                        exits.append(cell)
                    if ch == DOOR:
                        doors.append(cell)
                elif ch == AGENT:
                    markers.append(cell)
                    agent.append(cell)
                elif ch == FIRE:
                    markers.append(cell)
                    fires.append(cell)
                elif ch in SLOT_CHARS:
                    markers.append(cell)
                    slots[int(ch)] = cell
                else:
                    raise ValueError(f"unknown character {ch!r} at {cell}")

        # Marker cells take the floor material of their neighbours (a dependent standing in a
        # carpeted bedroom is standing on carpet). Falls back to wood.
        for cell in markers:
            around = [
                materials[n]
                for n in ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1))
                if materials.get(n) in ("wood", "carpet", "tile")
            ]
            materials[cell] = max(("wood", "carpet", "tile"), key=around.count) if around else "wood"

        if len(agent) != 1:
            raise ValueError(f"need exactly one 'A', found {len(agent)}")
        if not exits:
            raise ValueError("need at least one exit 'E'")
        if sorted(slots) != list(range(1, len(slots) + 1)):
            raise ValueError("dependent slots must be numbered 1..n without gaps")

        return cls(
            name=name,
            rows=rows,
            agent_start=agent[0],
            slots=tuple(slots[i] for i in sorted(slots)),
            exits=tuple(exits),
            doors=tuple(doors),
            ignitions=tuple(fires),
            walls=frozenset(walls),
            materials=materials,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Blueprint":
        p = Path(path)
        return cls.parse(p.read_text(), name=p.stem)

    # --------------------------------------------------------------- validation
    def reachable(self, source: Cell, jammed: frozenset[Cell] = frozenset()) -> dict[Cell, int]:
        return bfs_distances(
            source, lambda c: self.passable(c) and c not in jammed, self.height, self.width
        )

    def validate(self, n_dependents: int = 0, jammed: frozenset[Cell] = frozenset()) -> None:
        """Raise if the rescuer cannot reach an exit and the first n dependents (ignoring fire)."""
        if n_dependents > len(self.slots):
            raise ValueError(f"{self.name} has {len(self.slots)} dependent slots, need {n_dependents}")
        dist = self.reachable(self.agent_start, jammed)
        if not any(e in dist for e in self.exits):
            raise ValueError(f"{self.name}: no exit reachable from the start")
        for i in range(n_dependents):
            if self.slots[i] not in dist:
                raise ValueError(f"{self.name}: dependent slot {i + 1} is unreachable")
