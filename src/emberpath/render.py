"""Rendering: ASCII for the terminal, GIF for demos.

The GIF is the main artifact for non-technical audiences: you can *watch* the agent decide.
Captions under each frame carry the agent's plain-English reason for its move.
A hidden dependent (the ``hidden_dependent`` novelty) shows as ``?`` / a hollow ringed "?" until found.
Doors carry their number (``places.py``: reading order), so "door 3" in an explanation can be found on the picture.
The banner under the caption wraps onto up to ``banner_lines`` lines; the CLI uses it to show, from the tick the
agent noticed a change, what it noticed and what it is doing about it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from emberpath.world import FOLLOWING, LOST, RESCUED, World

_FLOOR = {
    "wood": (222, 196, 160),
    "carpet": (190, 160, 120),
    "tile": (205, 220, 228),
    "door": (150, 100, 60),
    "exit": (110, 200, 120),
}
_WALL = (50, 50, 58)
_BURNT = (60, 52, 48)
_DEP_COLOR = {"dog": (140, 60, 40), "spouse": (150, 90, 200), "child": (230, 200, 50)}
_AGENT = (40, 110, 230)
_HIDDEN = (230, 60, 160)


def render_ascii(world: World) -> str:
    bp = world.blueprint
    glyph: dict = {}
    for d in world.dependents:
        if d.status == LOST:  # the cell shows the fire that took them
            continue
        if d.hiding:
            glyph[d.pos] = "?"
        else:
            glyph[d.pos] = d.profile.glyph.lower() if d.status in (FOLLOWING, RESCUED) else d.profile.glyph
    lines = []
    for r in range(bp.height):
        row = []
        for c in range(bp.width):
            cell = (r, c)
            if cell == world.agent_pos:
                row.append("@" if world.agent_alive else "X")
            elif cell in glyph:
                row.append(glyph[cell])
            elif cell in bp.walls:
                row.append("#")
            elif world.burn[cell] > 0:
                row.append("*")
            elif world.burn[cell] < 0:
                row.append("x")
            elif cell in bp.exits:
                row.append("E")
            elif cell in bp.doors:
                row.append("D" if cell not in world.rules.jammed_doors else "J")
            else:
                row.append(".")
        lines.append("".join(row))
    return "\n".join(lines)


def wrap_banner(text: str, width_chars: int, lines: int) -> list[str]:
    """Wrap `text` to at most `lines` lines of `width_chars`; an overflow ends in '...'."""
    out = textwrap.wrap(text, max(10, width_chars)) or [""]
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1][: max(0, width_chars - 3)].rstrip() + "..."
    return out


def render_frame(world: World, caption: str = "", cell_px: int = 28, banner: str = "", banner_lines: int = 1) -> Image.Image:
    """One frame. The frame height depends on `banner_lines` only, so every frame of a GIF made with one value has
    the same size."""
    bp = world.blueprint
    w, h = bp.width * cell_px, bp.height * cell_px
    foot = 54 + 14 * (max(1, banner_lines) - 1)
    img = Image.new("RGB", (w, h + foot), (24, 24, 28))
    d = ImageDraw.Draw(img, "RGBA")
    numbers = bp.places.door_numbers

    for r in range(bp.height):
        for c in range(bp.width):
            cell = (r, c)
            x0, y0 = c * cell_px, r * cell_px
            box = (x0, y0, x0 + cell_px - 1, y0 + cell_px - 1)
            if cell in bp.walls:
                d.rectangle(box, fill=_WALL)
                continue
            d.rectangle(box, fill=_FLOOR[bp.material(cell)])
            if cell in world.rules.jammed_doors:
                d.rectangle(box, fill=(120, 30, 30))
                d.line((x0 + 4, y0 + 4, x0 + cell_px - 5, y0 + cell_px - 5), fill=(255, 255, 255), width=3)
                d.line((x0 + cell_px - 5, y0 + 4, x0 + 4, y0 + cell_px - 5), fill=(255, 255, 255), width=3)
            if cell in numbers and world.burn[cell] == 0:
                d.text((x0 + 2, y0 + 1), str(numbers[cell]), fill=(255, 255, 255))
            if world.burn[cell] < 0:
                d.rectangle(box, fill=_BURNT)
            elif world.burn[cell] > 0:
                heat = world.burn[cell] / max(1, world.rules.burn_steps)
                d.rectangle(box, fill=(255, int(70 + 130 * (1 - heat)), 20))
            s = float(world.smoke[cell])
            if s > 0.05:
                d.rectangle(box, fill=(70, 70, 80, min(170, int(s * 90))))

    for dep in world.dependents:
        if dep.status == LOST:
            continue
        cx, cy = dep.pos[1] * cell_px + cell_px // 2, dep.pos[0] * cell_px + cell_px // 2
        rad = cell_px // 3
        if dep.hiding:  # unseen by the rescuer: hollow ring with a question mark
            d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=_HIDDEN + (110,), outline=_HIDDEN, width=3)
            d.text((cx - 3, cy - 5), "?", fill=(255, 255, 255))
            continue
        col = _DEP_COLOR[dep.profile.kind]
        d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=col, outline=(20, 20, 20))
        d.text((cx - 3, cy - 5), dep.profile.glyph, fill=(255, 255, 255))
        if dep.status == FOLLOWING:
            d.ellipse((cx - rad - 2, cy - rad - 2, cx + rad + 2, cy + rad + 2), outline=(40, 110, 230), width=2)

    ax, ay = world.agent_pos[1] * cell_px + cell_px // 2, world.agent_pos[0] * cell_px + cell_px // 2
    rad = cell_px // 2 - 3
    d.ellipse((ax - rad, ay - rad, ax + rad, ay + rad), fill=_AGENT if world.agent_alive else (0, 0, 0), outline=(255, 255, 255), width=2)

    c = world.counts()
    status = f"t={world.t}   rescued {c['rescued']}/{c['total']}   lost {c['lost']}"
    if world.outcome:
        status += f"   -> {world.outcome.upper()}"
    d.text((8, h + 6), status, fill=(235, 235, 235))
    d.text((8, h + 22), caption[: w // 6], fill=(170, 210, 255))
    if banner:
        lines = [banner[: w // 6]] if banner_lines <= 1 else wrap_banner(banner, w // 6, banner_lines)
        for i, line in enumerate(lines):
            d.text((8, h + 38 + 14 * i), line, fill=(255, 190, 90))
    return img


def save_gif(frames: list[Image.Image], path: str | Path, ms_per_frame: int = 160, hold_last_ms: int = 1500) -> None:
    if not frames:
        raise ValueError("no frames to save")
    durations = [ms_per_frame] * (len(frames) - 1) + [hold_last_ms]
    # optimize=True drops unused palette entries: lossless (same pixels), files ~40% smaller.
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
