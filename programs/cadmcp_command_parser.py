"""cadmcp_command_parser -- deterministic parameter extractor for CAD commands.

Transferred from the ``nlp_processor.py`` of CAD-MCP. That module turns a free
text draughting instruction into a structured drawing command by scanning it for
coordinates, numbers and keyword-tagged parameters (radius, angles, text height,
rotation, hatch scale) with fixed fallback defaults. The keyword vocabulary in
the original is bilingual (Chinese + English); the *extraction machinery* is a
set of small, exact regular-expression routines and is fully deterministic.

This differs from the harness's other natural-language parsers:

  * ``spec/nlcad_case_frame`` is a Fillmore case-frame semantic parser (verb
    frames + a noun lexicon) -- it does not read coordinate *tuples* ``(x,y,z)``
    out of the surface string or apply per-shape numeric fallbacks;
  * ``surfaces/mcp`` consumes already-structured tool arguments, never text.

Here the pieces mined are:

  * :func:`extract_coordinates` -- pull every ``(x,y[,z])`` / ``x,y[,z]`` tuple;
  * :func:`extract_numbers` -- every signed decimal;
  * :func:`extract_keyword_value` -- the ``<keyword> ... <number>`` pattern used
    for radius / angles / height / rotation / scale;
  * :func:`identify_command` -- action x shape -> a canonical command type;
  * :func:`parse_command` -- assemble a structured command dict (with the
    controller's fallback geometry) that ``drawings.cadmcp_drawing_commands``
    can consume directly.

Stdlib-only, deterministic, no wall clock. English keyword vocabulary.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float, float]

# ``(x,y,z)`` / ``(x,y)`` / ``x,y,z`` / ``x,y`` -- optional parens, optional z.
_COORD_RE = re.compile(
    r"\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)(?:\s*,\s*(-?\d+\.?\d*))?\s*\)?")
_NUMBER_RE = re.compile(r"-?\d+\.?\d*")

# Shape word -> canonical shape.
SHAPE_KEYWORDS: Dict[str, str] = {
    "line": "line", "segment": "line",
    "circle": "circle",
    "arc": "arc",
    "rectangle": "rectangle", "rect": "rectangle", "square": "rectangle",
    "polyline": "polyline", "polygon": "polyline",
    "text": "text", "label": "text",
    "ellipse": "ellipse", "oval": "ellipse",
    "hatch": "hatch", "fill": "hatch",
}

# Action word -> canonical action.
ACTION_KEYWORDS: Dict[str, str] = {
    "draw": "draw", "create": "draw", "add": "draw", "make": "draw",
    "sketch": "draw", "fill": "draw", "hatch": "draw",
    "save": "save",
}


def extract_coordinates(text: str) -> List[Point]:
    """Every ``(x, y[, z])`` tuple in ``text``, z defaulting to 0."""
    out: List[Point] = []
    for m in _COORD_RE.finditer(text):
        x = float(m.group(1))
        y = float(m.group(2))
        z = float(m.group(3)) if m.group(3) else 0.0
        out.append((x, y, z))
    return out


def extract_numbers(text: str) -> List[float]:
    """Every signed decimal number in ``text`` (in order)."""
    return [float(m) for m in _NUMBER_RE.findall(text)]


def extract_keyword_value(text: str, keywords) -> Optional[float]:
    """First ``<keyword> ...<number>`` value, else ``None``.

    Matches the ``(?:kw)[^\\d]*?(-?\\d+\\.?\\d*)`` shape the original uses for
    ``radius`` / ``angle`` / ``height`` / ``rotation`` / ``scale``.
    """
    if isinstance(keywords, str):
        keywords = [keywords]
    # Longest keywords first so "start angle" wins over "start"; word-boundary
    # anchored so single-letter aliases ("r"/"w"/"h") never match inside a word.
    alt = "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))
    m = re.search(rf"\b(?:{alt})\b[^\d-]*?(-?\d+\.?\d*)", text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def identify_command(text: str) -> str:
    """Classify ``text`` into a canonical command type (else ``unknown``)."""
    low = text.lower()
    action = None
    for word, canon in ACTION_KEYWORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            action = canon
            break
    if action == "save" or ("save" in low and action is None):
        return "save"
    if action == "draw":
        for word, shape in SHAPE_KEYWORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", low):
                return f"draw_{shape}"
    return "unknown"


def _first_coord(coords: List[Point], default: Point) -> Point:
    return coords[0] if coords else default


def parse_command(text: str) -> Dict[str, object]:
    """Parse ``text`` into a structured drawing-command dict.

    Applies the controller's fallback geometry (default centre/endpoints/radius)
    when the instruction omits them, so the result is always executable.
    """
    ctype = identify_command(text)
    coords = extract_coordinates(text)
    numbers = extract_numbers(text)

    if ctype == "draw_line":
        if len(coords) >= 2:
            return {"type": "draw_line", "start": coords[0], "end": coords[1]}
        return {"type": "draw_line", "start": (0.0, 0.0, 0.0),
                "end": (100.0, 100.0, 0.0), "defaulted": True}

    if ctype == "draw_circle":
        radius = extract_keyword_value(text, ["radius", "r"])
        if radius is None:
            radius = numbers[0] if numbers else 50.0
        return {"type": "draw_circle",
                "center": _first_coord(coords, (0.0, 0.0, 0.0)),
                "radius": radius}

    if ctype == "draw_arc":
        radius = extract_keyword_value(text, ["radius", "r"])
        if radius is None:
            radius = numbers[0] if numbers else 50.0
        sa = extract_keyword_value(text, ["start angle", "start"])
        ea = extract_keyword_value(text, ["end angle", "end"])
        return {"type": "draw_arc",
                "center": _first_coord(coords, (0.0, 0.0, 0.0)),
                "radius": radius,
                "start_angle": 0.0 if sa is None else sa,
                "end_angle": 90.0 if ea is None else ea}

    if ctype == "draw_rectangle":
        if len(coords) >= 2:
            return {"type": "draw_rectangle",
                    "corner1": coords[0], "corner2": coords[1]}
        width = extract_keyword_value(text, ["width", "w"]) or 100.0
        height = extract_keyword_value(text, ["height", "h"]) or 100.0
        if coords:
            c1 = coords[0]
            c2 = (c1[0] + width, c1[1] + height, c1[2])
        else:
            c1, c2 = (0.0, 0.0, 0.0), (width, height, 0.0)
        return {"type": "draw_rectangle", "corner1": c1, "corner2": c2}

    if ctype == "draw_polyline":
        if len(coords) >= 2:
            closed = "closed" in text.lower()
            return {"type": "draw_polyline", "points": coords, "closed": closed}
        return {"type": "error",
                "message": "polyline needs at least 2 coordinate points"}

    if ctype == "draw_ellipse":
        major = extract_keyword_value(text, ["major"])
        minor = extract_keyword_value(text, ["minor"])
        if major is None or minor is None:
            nums = [n for n in numbers]
            if major is None:
                major = nums[0] if nums else 100.0
            if minor is None:
                minor = nums[1] if len(nums) > 1 else major / 2.0
        rot = extract_keyword_value(text, ["rotation", "rotate", "angle"]) or 0.0
        return {"type": "draw_ellipse",
                "center": _first_coord(coords, (0.0, 0.0, 0.0)),
                "major_axis": major, "minor_axis": minor, "rotation": rot}

    if ctype == "draw_text":
        m = re.search(r"[\"'](.*?)[\"']", text)
        content = m.group(1) if m else "text"
        height = extract_keyword_value(text, ["height", "h"]) or 2.5
        rotation = extract_keyword_value(text, ["rotation", "rotate"]) or 0.0
        return {"type": "draw_text",
                "position": _first_coord(coords, (0.0, 0.0, 0.0)),
                "text": content, "height": height, "rotation": rotation}

    if ctype == "draw_hatch":
        if len(coords) >= 3:
            scale = extract_keyword_value(text, ["scale"]) or 1.0
            m = re.search(r"pattern\s+([A-Za-z0-9_]+)", text, re.IGNORECASE)
            pattern = m.group(1).upper() if m else "SOLID"
            return {"type": "draw_hatch", "points": coords,
                    "pattern_name": pattern, "scale": scale}
        return {"type": "error",
                "message": "hatch needs at least 3 boundary points"}

    if ctype == "save":
        m = re.search(r"[\"'](.*?)[\"']", text)
        return {"type": "save", "file_path": m.group(1) if m else None}

    return {"type": "unknown", "original": text}
