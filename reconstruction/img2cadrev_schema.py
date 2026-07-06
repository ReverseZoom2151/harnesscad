"""Img2CAD (VLM-Assisted Conditional Factorization) sketch-extrude schema.

This is the deterministic per-stage schema for the Img2CAD reverse-engineering
paper (You et al.). The paper defines a general sketch-extrude CAD language whose
distinctive feature -- relative to the DeepCAD / CADParser vocabularies already in
this codebase -- is that it treats an *extrusion join* and an *extrusion cut* as
two DISTINCT command types (the ``b`` join/cut argument is promoted to the command
type), because the two-stage conditional factorization predicts the discrete
command-type sequence first and the continuous attributes second.

Sketch commands and their continuous attributes (paper Sec. 3.1)::

    L (Line)    (x, y)         endpoint of a line segment
    A (Arc)     (x, y, alpha)  endpoint of an arc with sweep angle alpha
    R (Circle)  (x, y, r)      circle with centre (x, y) and radius r

Extrusion commands and their continuous attributes (paper Sec. 3.2)::

    Ej (Extrude-Join)  (alpha, theta, gamma, x, y, z, e)
    Ec (Extrude-Cut)   (alpha, theta, gamma, x, y, z, e)

where (alpha, theta, gamma) are the extrusion-frame Euler angles, (x, y, z) is the
frame origin and e is the extrusion extent. Join vs cut are separate command types.

The learned components (finetuned Llama3.2 structure prediction, the TrAssembler
flow-matching attribute network, GMFlow) are out of scope. Everything here is pure
and deterministic: attribute dimensioning, sketch-profile validity (closed,
non-self-intersecting, counter-clockwise per the paper's convention) and the
polygon reconstruction used by the metrics and factorization modules.
"""

from __future__ import annotations

import math

# --- command vocabulary -----------------------------------------------------
LINE = "L"
ARC = "A"
CIRCLE = "R"
EXTRUDE_JOIN = "Ej"
EXTRUDE_CUT = "Ec"

SKETCH_COMMANDS: tuple[str, ...] = (LINE, ARC, CIRCLE)
EXTRUDE_COMMANDS: tuple[str, ...] = (EXTRUDE_JOIN, EXTRUDE_CUT)
COMMAND_TYPES: tuple[str, ...] = SKETCH_COMMANDS + EXTRUDE_COMMANDS
COMMAND_INDEX: dict[str, int] = {c: i for i, c in enumerate(COMMAND_TYPES)}

# Continuous-attribute dimensionality per command type (paper Sec. 3).
ATTRIBUTE_DIM: dict[str, int] = {
    LINE: 2,
    ARC: 3,
    CIRCLE: 3,
    EXTRUDE_JOIN: 7,
    EXTRUDE_CUT: 7,
}


def is_command_type(cmd_type: str) -> bool:
    return cmd_type in COMMAND_INDEX


def is_sketch_command(cmd_type: str) -> bool:
    return cmd_type in SKETCH_COMMANDS


def is_extrude_command(cmd_type: str) -> bool:
    return cmd_type in EXTRUDE_COMMANDS


def attribute_dim(cmd_type: str) -> int:
    """Number of continuous attributes carried by ``cmd_type``."""
    if cmd_type not in ATTRIBUTE_DIM:
        raise KeyError(f"unknown command type: {cmd_type!r}")
    return ATTRIBUTE_DIM[cmd_type]


def validate_attribute_vector(cmd_type: str, attrs) -> None:
    """Raise ValueError if ``attrs`` does not match ``cmd_type``'s arity."""
    expected = attribute_dim(cmd_type)
    if len(attrs) != expected:
        raise ValueError(
            f"{cmd_type} expects {expected} attributes, got {len(attrs)}"
        )
    for a in attrs:
        if not isinstance(a, (int, float)):
            raise ValueError(f"{cmd_type} attribute not numeric: {a!r}")


def extrusion_is_cut(cmd_type: str) -> bool:
    """True for Ec, False for Ej. Errors on non-extrude commands."""
    if cmd_type == EXTRUDE_CUT:
        return True
    if cmd_type == EXTRUDE_JOIN:
        return False
    raise ValueError(f"not an extrusion command: {cmd_type!r}")


# --- sketch-profile geometry ------------------------------------------------
def sample_arc(start, end, alpha: float, n: int = 8):
    """Sample ``n`` points along a circular arc from ``start`` to ``end``.

    The arc has signed sweep angle ``alpha`` (radians). Degenerate arcs
    (alpha ~ 0 or coincident endpoints) fall back to the straight chord. Purely
    deterministic; used for area / self-intersection checks.
    """
    sx, sy = start
    ex, ey = end
    if abs(alpha) < 1e-9:
        return [(sx, sy), (ex, ey)]
    cx, cy = (ex - sx), (ey - sy)
    chord = math.hypot(cx, cy)
    if chord < 1e-12:
        return [(sx, sy), (ex, ey)]
    # Radius from chord and sweep: chord = 2 r sin(alpha/2).
    radius = chord / (2.0 * math.sin(abs(alpha) / 2.0))
    # Midpoint of chord.
    mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
    # Distance from chord midpoint to centre.
    h = math.sqrt(max(radius * radius - (chord / 2.0) ** 2, 0.0))
    # Unit normal to the chord; sign follows sweep direction.
    nx, ny = -cy / chord, cx / chord
    sign = 1.0 if alpha > 0 else -1.0
    ctr = (mx - sign * h * nx, my - sign * h * ny)
    a0 = math.atan2(sy - ctr[1], sx - ctr[0])
    pts = []
    for i in range(n + 1):
        ang = a0 + alpha * (i / n)
        pts.append((ctr[0] + radius * math.cos(ang),
                    ctr[1] + radius * math.sin(ang)))
    return pts


def profile_polygon(commands, origin=(0.0, 0.0), arc_samples: int = 8):
    """Reconstruct the 2D polygon traced by a sketch-command sequence.

    ``commands`` is a list of ``(cmd_type, attrs)`` pairs. Line/arc commands are
    chained from ``origin``; a lone circle command yields a sampled circle. Arcs
    are expanded via :func:`sample_arc`. Returns a list of ``(x, y)`` vertices
    (the closing vertex is not duplicated).
    """
    verts = [tuple(map(float, origin))]
    for cmd_type, attrs in commands:
        if cmd_type == LINE:
            verts.append((float(attrs[0]), float(attrs[1])))
        elif cmd_type == ARC:
            end = (float(attrs[0]), float(attrs[1]))
            pts = sample_arc(verts[-1], end, float(attrs[2]), arc_samples)
            verts.extend(pts[1:])
        elif cmd_type == CIRCLE:
            cx, cy, r = float(attrs[0]), float(attrs[1]), float(attrs[2])
            ring = []
            for i in range(max(arc_samples * 2, 8)):
                ang = 2.0 * math.pi * (i / max(arc_samples * 2, 8))
                ring.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
            return ring
        else:
            raise ValueError(f"not a sketch command: {cmd_type!r}")
    # Drop a duplicated closing vertex if present.
    if len(verts) > 1 and _close(verts[0], verts[-1]):
        verts = verts[:-1]
    return verts


def _close(a, b, eps: float = 1e-6) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def signed_area(points) -> float:
    """Shoelace signed area; positive == counter-clockwise."""
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return s / 2.0


def is_counter_clockwise(points) -> bool:
    return signed_area(points) > 0.0


def profile_is_closed(commands, origin=(0.0, 0.0), eps: float = 1e-6) -> bool:
    """True if the line/arc chain returns to ``origin`` (circles are closed)."""
    if len(commands) == 1 and commands[0][0] == CIRCLE:
        return True
    if any(c[0] == CIRCLE for c in commands):
        # A circle mixed with other commands does not form a single chain.
        return False
    pt = tuple(map(float, origin))
    for cmd_type, attrs in commands:
        pt = (float(attrs[0]), float(attrs[1]))
    return _close(pt, tuple(map(float, origin)), eps)


def _seg_intersect(p1, p2, p3, p4) -> bool:
    """Proper intersection test for open segments p1p2 and p3p4."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        if abs(d1) > 1e-12 and abs(d2) > 1e-12 and abs(d3) > 1e-12 and abs(d4) > 1e-12:
            return True
    return False


def is_simple_polygon(points) -> bool:
    """True if no pair of non-adjacent edges properly intersects."""
    n = len(points)
    if n < 4:
        return True
    for i in range(n):
        a1, a2 = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            # Skip adjacent / shared-vertex edges.
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            b1, b2 = points[j], points[(j + 1) % n]
            if _seg_intersect(a1, a2, b1, b2):
                return False
    return True


def profile_is_valid(commands, origin=(0.0, 0.0)) -> bool:
    """Closed, non-self-intersecting, counter-clockwise (paper convention)."""
    if not profile_is_closed(commands, origin):
        return False
    poly = profile_polygon(commands, origin)
    if len(poly) < 3:
        return False
    return is_simple_polygon(poly) and is_counter_clockwise(poly)
