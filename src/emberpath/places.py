"""Names people can understand: rooms, numbered doors and exits.

Explanations must say "door 3, between the hallway and the kitchen", not "door_jammed:5,12". This module gives
every part of a blueprint a stable, human name, using only the floor plan (never the rules or the world):

* **Rooms** are the areas of floor that doors separate (connected floor cells, not crossing a door or an exit).
  A room is named by the blueprint's optional ``# room r,c: name`` metadata; otherwise the room with clearly the
  most doors (at least 4) is "the hallway" and the rest are named by where they sit in the house: "the north-west
  room", "the south room", with a number added if two would share a name.
* **Doors** are numbered 1, 2, 3, ... in reading order (top row first, left to right), the order the blueprint
  lists them in. The number depends only on the floor plan, so "door 3" means the same door in every run, in
  every explanation and on the GIF, where the number is drawn on the door.
* **Exits** are named by the room they open from, plus the side of the house when a room has more than one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from emberpath.grid import Cell, neighbors4

if TYPE_CHECKING:
    from emberpath.blueprint import Blueprint


@dataclass(frozen=True)
class Places:
    room_of_cell: dict[Cell, int]  # every floor cell that is not a door or an exit -> room index
    room_names: tuple[str, ...]  # room index -> name, e.g. "kitchen" or "north-west room"
    door_numbers: dict[Cell, int]  # door cell -> 1-based number in reading order
    door_rooms: dict[Cell, tuple[int, ...]]  # door cell -> the rooms on its sides (sorted, distinct)
    exit_names: dict[Cell, str]  # exit cell -> "the kitchen exit" / "the east exit of the family room"

    @classmethod
    def of(cls, bp: "Blueprint") -> "Places":
        doors, exits = set(bp.doors), set(bp.exits)
        room_of: dict[Cell, int] = {}
        areas: list[list[Cell]] = []
        for r in range(bp.height):
            for c in range(bp.width):
                start = (r, c)
                if start in room_of or not bp.passable(start) or start in doors or start in exits:
                    continue
                idx, todo = len(areas), [start]
                room_of[start] = idx
                cells = []
                while todo:
                    cur = todo.pop()
                    cells.append(cur)
                    for n in neighbors4(cur, bp.height, bp.width):
                        if n not in room_of and bp.passable(n) and n not in doors and n not in exits:
                            room_of[n] = idx
                            todo.append(n)
                areas.append(cells)

        given = {room_of[cell]: name for cell, name in bp.room_names if cell in room_of}
        door_rooms = {
            d: tuple(sorted({room_of[n] for n in neighbors4(d, bp.height, bp.width) if n in room_of}))
            for d in bp.doors
        }
        auto = _auto_names(bp, areas, set(given), door_rooms)
        names = tuple(given.get(i, auto.get(i, f"room {i + 1}")) for i in range(len(areas)))

        door_numbers = {d: i + 1 for i, d in enumerate(sorted(bp.doors))}

        exit_rooms = {e: next((room_of[n] for n in neighbors4(e, bp.height, bp.width) if n in room_of), None)
                      for e in bp.exits}
        exit_names = {}
        for e, room in exit_rooms.items():
            side = _side(bp, e)
            if room is None:
                exit_names[e] = f"the {side} exit"
            elif sum(r == room for r in exit_rooms.values()) > 1:
                exit_names[e] = f"the {side} exit of {_the(names[room])}"
            else:
                exit_names[e] = f"the {_bare(names[room])} exit"
        return cls(room_of, names, door_numbers, door_rooms, exit_names)

    # ----------------------------------------------------------------- names
    def room(self, cell: Cell) -> str | None:
        """'the kitchen' for a cell in a named room; a door or exit cell gets the room on one side of it."""
        idx = self.room_of_cell.get(cell)
        if idx is None and cell in self.door_rooms and self.door_rooms[cell]:
            idx = self.door_rooms[cell][0]
        return None if idx is None else _the(self.room_names[idx])

    def door(self, cell: Cell, capital: bool = False) -> str:
        """'door 3' (or 'Door 3')."""
        word = "Door" if capital else "door"
        return f"{word} {self.door_numbers[cell]}" if cell in self.door_numbers else f"{word.lower()} at {cell}"

    def door_between(self, cell: Cell) -> str:
        """'between the hallway and the kitchen' (or 'into the kitchen' when one side is not a room)."""
        rooms = [_the(self.room_names[i]) for i in self.door_rooms.get(cell, ())]
        if len(rooms) >= 2:
            return f"between {rooms[0]} and {rooms[1]}"
        if rooms:
            return f"into {rooms[0]}"
        return ""

    def describe_door(self, cell: Cell, capital: bool = True) -> str:
        """'Door 3 (between the hallway and the kitchen)'."""
        where = self.door_between(cell)
        return f"{self.door(cell, capital)} ({where})" if where else self.door(cell, capital)

    def exit(self, cell: Cell) -> str:
        return self.exit_names.get(cell, "the exit")

    def place(self, cell: Cell) -> str:
        """The most natural name for any cell: a door, an exit, or the room it is in."""
        if cell in self.door_numbers:
            return self.door(cell)
        if cell in self.exit_names:
            return self.exit(cell)
        return self.room(cell) or f"the spot at {cell}"

    def rooms_along(self, path: list[Cell]) -> list[str]:
        """The rooms a path passes through, in order, each once."""
        out: list[str] = []
        for cell in path:
            idx = self.room_of_cell.get(cell)
            if idx is not None and _the(self.room_names[idx]) not in out:
                out.append(_the(self.room_names[idx]))
        return out


def _the(name: str) -> str:
    """'the kitchen'; names that already start with 'the' or a capital (a person's name) are left alone."""
    return name if name.lower().startswith("the ") or name[:1].isupper() else f"the {name}"


def _bare(name: str) -> str:
    return name[4:] if name.lower().startswith("the ") else name


def _side(bp: "Blueprint", cell: Cell) -> str:
    r, c = cell
    if r == 0:
        return "north"
    if r == bp.height - 1:
        return "south"
    if c == 0:
        return "west"
    if c == bp.width - 1:
        return "east"
    return "inner"


def _auto_names(bp: "Blueprint", areas: list[list[Cell]], named: set[int], door_rooms) -> dict[int, str]:
    """Names for rooms the blueprint does not name. The room with the most doors (at least 4, and strictly the
    most) is "the hallway"; the rest are named by their centre, by thirds of the house ("north-west room"),
    numbered when two collide ("north room 1", "north room 2", left to right / top to bottom)."""
    out: dict[int, str] = {}
    n_doors = [sum(i in rooms for rooms in door_rooms.values()) for i in range(len(areas))]
    if n_doors and "hallway" not in {_bare(bp_name) for _, bp_name in bp.room_names}:
        most = max(n_doors)
        if most >= 4 and n_doors.count(most) == 1 and n_doors.index(most) not in named:
            out[n_doors.index(most)] = "hallway"
    for i, cells in enumerate(areas):
        if i in named or i in out:
            continue
        r = sum(c[0] for c in cells) / len(cells)
        c = sum(c[1] for c in cells) / len(cells)
        ns = "north" if r < bp.height / 3 else "south" if r >= 2 * bp.height / 3 else ""
        ew = "west" if c < bp.width / 3 else "east" if c >= 2 * bp.width / 3 else ""
        where = "-".join(x for x in (ns, ew) if x) or "central"
        out[i] = f"{where} room"
    counts: dict[str, int] = {}
    for name in out.values():
        counts[name] = counts.get(name, 0) + 1
    counts["hallway"] = 1
    seen: dict[str, int] = {}
    for i in sorted(out):
        name = out[i]
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            out[i] = f"{name} {seen[name]}"
    return out
