"""Token stream -> sketch-and-extrude geometry (the CAD parser).

It turns a flat merged token stream (see ``reconstruction/skexgen_token_format``)
back into per-SE geometry, and doubles as the *validity oracle* --- the
"invalid %" figure is literally "the parser raised an exception".

Decoding rules that make this representation distinctive:

* a curve is identified purely by its **token count**: 1 = line, 2 = arc,
  4 = circle;
* a curve stores only its *leading* vertices; its end vertex is the first
  vertex of the *next* curve in the loop (wrapping around), so a loop is
  closed by construction;
* a circle is 4 rim points: centre = ``(mid_x(p1, p2), mid_y(p3, p4))`` and
  radius = ``(|p1 - p2| + |p3 - p4|) / 4`` (an average of the two diameters);
* an arc's centre is the circumcentre of (start, mid, end);
* every decoded sketch vertex is mapped back to model space with the extrude
  block's ``scale`` and ``offset`` (which is why those live in the extrude
  branch).

The parser also reproduces the vertex de-duplication table, which is what
gives the ``.obj``-style output its ``l / a / c`` index-based curve records.

Deterministic, stdlib only.
"""
from __future__ import annotations

from math import hypot
from typing import Dict, List, Sequence, Tuple

from harnesscad.domain.reconstruction.tokens.skexgen_extrude import EXT_SEQ_LEN, decode_extrude
from harnesscad.domain.reconstruction.tokens.skexgen_quantize import (
    BIT, CURVE_END, FACE_END, LOOP_END, PIX_OFFSET, SE_END, dequantize,
    split_on, strip_padding, xy_from_pixel,
)

Vec2 = Tuple[float, float]


class SkexGenParseError(ValueError):
    """Raised when a token stream cannot be decoded (the "invalid" case)."""


def circumcenter(a: Vec2, b: Vec2, c: Vec2) -> Vec2:
    """Centre of the circle through three points (arc geometry fit)."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    A = bx - ax
    B = by - ay
    C = cx - ax
    D = cy - ay
    E = A * (ax + bx) + B * (ay + by)
    F = C * (ax + cx) + D * (ay + cy)
    G = 2.0 * (A * (cy - by) - B * (cx - bx))
    if G == 0.0:
        raise SkexGenParseError("degenerate arc (collinear points)")
    return ((D * E - B * F) / G, (A * F - C * E) / G)


def arc_radius(center: Vec2, point: Vec2) -> float:
    return hypot(point[0] - center[0], point[1] - center[1])


def circle_from_rim(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> Tuple[Vec2, float]:
    """The 4-point circle: centre from two diameters, radius averaged."""
    center = (0.5 * (p1[0] + p2[0]), 0.5 * (p3[1] + p4[1]))
    radius = (hypot(p1[0] - p2[0], p1[1] - p2[1]) +
              hypot(p3[0] - p4[0], p3[1] - p4[1])) / 4.0
    return center, radius


def _pixel_point(token: int, bit: int) -> Vec2:
    if token < PIX_OFFSET:
        raise SkexGenParseError("structural token %d where a pixel was expected" % token)
    x, y = xy_from_pixel(token - PIX_OFFSET, bit)
    return (dequantize(x, bit), dequantize(y, bit))


def _apply(point: Vec2, scale: float, offset: Sequence[float]) -> Vec2:
    return (point[0] * scale + offset[0], point[1] * scale + offset[1])


class VertexTable:
    """Insertion-ordered de-duplicating vertex table."""

    def __init__(self) -> None:
        self._keys: List[str] = []
        self._values: List[Tuple[float, float]] = []

    def save(self, x: float, y: float, kind: str = "p") -> int:
        key = "%s:x%sy%s" % (kind, x, y)
        try:
            return self._keys.index(key)
        except ValueError:
            self._keys.append(key)
            self._values.append((x, y))
            return len(self._keys) - 1

    def vertices(self) -> List[Tuple[float, float]]:
        return list(self._values)

    def obj_lines(self) -> List[str]:
        return ["v %s %s" % (x, y) for x, y in self._values]


def split_se(tokens: Sequence[int]) -> List[Tuple[List[int], List[int]]]:
    """Split a merged stream into ``(sketch_tokens, extrude_tokens)`` pairs."""
    body = strip_padding(tokens)
    if not body:
        raise SkexGenParseError("empty token stream")
    if body[-1] != SE_END:
        raise SkexGenParseError("stream must end with the SE end token")
    groups = split_on(body, SE_END)
    if len(groups) % 2 != 0:
        raise SkexGenParseError("odd number of sketch/extrude groups")
    return [(groups[i], groups[i + 1]) for i in range(0, len(groups), 2)]


def decode_sketch(sketch_tokens: Sequence[int], scale: float,
                  offset: Sequence[float], bit: int = BIT) -> List[List[List[Dict]]]:
    """Decode one sketch group into faces -> loops -> curve dicts."""
    if not sketch_tokens or sketch_tokens[-1] != SE_END:
        raise SkexGenParseError("sketch group must end with the SE end token")
    faces_tokens = split_on(list(sketch_tokens[:-1]), FACE_END)
    if not faces_tokens:
        raise SkexGenParseError("sketch has no faces")

    faces: List[List[List[Dict]]] = []
    for face_tokens in faces_tokens:
        loops_tokens = split_on(face_tokens[:-1], LOOP_END)
        if not loops_tokens:
            raise SkexGenParseError("face has no loops")
        loops: List[List[Dict]] = []
        for loop_idx, loop_tokens in enumerate(loops_tokens):
            curves_tokens = split_on(loop_tokens[:-1], CURVE_END)
            if not curves_tokens:
                raise SkexGenParseError("loop has no curves")
            stripped = [c[:-1] for c in curves_tokens]
            loops.append(_decode_loop(stripped, scale, offset, bit,
                                      is_outer=(loop_idx == 0)))
        faces.append(loops)
    return faces


def _decode_loop(curves: Sequence[Sequence[int]], scale: float,
                 offset: Sequence[float], bit: int,
                 is_outer: bool) -> List[Dict]:
    out: List[Dict] = []
    for i, curve in enumerate(curves):
        nxt = curves[(i + 1) % len(curves)]
        if not nxt:
            raise SkexGenParseError("empty curve token group")
        if len(curve) == 1:
            if curve[0] == nxt[0]:
                raise SkexGenParseError("zero-length line")
            start = _apply(_pixel_point(curve[0], bit), scale, offset)
            end = _apply(_pixel_point(nxt[0], bit), scale, offset)
            out.append({"type": "line", "start": start, "end": end,
                        "is_outer": is_outer})
        elif len(curve) == 2:
            if curve[0] == curve[1] or curve[0] == nxt[0] or curve[1] == nxt[0]:
                raise SkexGenParseError("degenerate arc (repeated vertex)")
            start = _pixel_point(curve[0], bit)
            mid = _pixel_point(curve[1], bit)
            end = _pixel_point(nxt[0], bit)
            center = circumcenter(start, mid, end)
            start = _apply(start, scale, offset)
            mid = _apply(mid, scale, offset)
            end = _apply(end, scale, offset)
            center = _apply(center, scale, offset)
            out.append({"type": "arc", "start": start, "mid": mid, "end": end,
                        "center": center, "radius": arc_radius(center, start),
                        "is_outer": is_outer})
        elif len(curve) == 4:
            if len(set(curve)) != 4:
                raise SkexGenParseError("degenerate circle (repeated rim point)")
            pts = [_pixel_point(t, bit) for t in curve]
            center, radius = circle_from_rim(*pts)
            center = _apply(center, scale, offset)
            radius = radius * scale
            if radius <= 0.0:
                raise SkexGenParseError("zero-radius circle")
            out.append({"type": "circle", "center": center, "radius": radius,
                        "is_outer": is_outer})
        else:
            raise SkexGenParseError("curve with %d tokens (expected 1, 2 or 4)"
                                    % len(curve))
    return out


def obj_records(faces: Sequence[Sequence[Sequence[Dict]]]) -> Tuple[List[str], List[str]]:
    """Emit the ``.obj``-style vertex + curve records for one SE."""
    table = VertexTable()
    lines: List[str] = []
    for face in faces:
        lines.append("face")
        for loop_idx, loop in enumerate(face):
            lines.append("out" if loop_idx == 0 else "in")
            for curve in loop:
                if curve["type"] == "line":
                    s = table.save(curve["start"][0], curve["start"][1])
                    e = table.save(curve["end"][0], curve["end"][1])
                    lines.append("l %d %d" % (s, e))
                elif curve["type"] == "arc":
                    c = table.save(curve["center"][0], curve["center"][1])
                    s = table.save(curve["start"][0], curve["start"][1])
                    m = table.save(curve["mid"][0], curve["mid"][1])
                    e = table.save(curve["end"][0], curve["end"][1])
                    lines.append("a %d %d %d %d" % (s, m, c, e))
                else:
                    c = table.save(curve["center"][0], curve["center"][1])
                    r = table.save(curve["radius"], 0.0, "r")
                    lines.append("c %d %d" % (c, r))
    return table.obj_lines(), lines


def parse_tokens(tokens: Sequence[int], bit: int = BIT) -> List[Dict]:
    """Full decode: merged token stream -> list of SE dicts."""
    out: List[Dict] = []
    for sketch_tokens, ext_tokens in split_se(tokens):
        if len(ext_tokens) != EXT_SEQ_LEN:
            raise SkexGenParseError("extrude block has %d tokens (expected %d)"
                                    % (len(ext_tokens), EXT_SEQ_LEN))
        try:
            extrude = decode_extrude(ext_tokens, bit)
        except ValueError as exc:
            raise SkexGenParseError(str(exc))
        faces = decode_sketch(sketch_tokens, extrude["scale"], extrude["offset"], bit)
        vertices, curves = obj_records(faces)
        out.append({"faces": faces, "extrude": extrude,
                    "vertices": vertices, "curves": curves})
    return out


def is_valid(tokens: Sequence[int], bit: int = BIT) -> bool:
    """Validity check: does the parser survive the stream?"""
    try:
        parse_tokens(tokens, bit)
    except (SkexGenParseError, ValueError, IndexError):
        return False
    return True


def invalid_percent(streams: Sequence[Sequence[int]], bit: int = BIT) -> float:
    """Percentage of token streams the parser rejects."""
    if not streams:
        return 0.0
    bad = sum(0 if is_valid(s, bit) else 1 for s in streams)
    return 100.0 * bad / len(streams)
