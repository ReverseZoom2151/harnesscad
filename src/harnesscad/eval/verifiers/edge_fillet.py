"""Per-edge fillet ceiling -- the SOUND replacement for the whole-body radius rule.

The problem
-----------
``preflight-RADIUS_TOO_LARGE`` compares a fillet radius against half the smallest
extent of the WHOLE-BODY bounding box. A fillet does not act on the body; it acts
on an EDGE, and an edge need not span the body's smallest extent. A 50x30x6 plate
filleted at r=3.1 on its four VERTICAL edges is valid -- those edges are adjacent
to the 50 mm and 30 mm faces, nowhere near the 6 mm thickness -- yet the
whole-body rule rejects it. That false-positive channel cost the pressure
experiment 8 briefs, so the rule is quarantined HEURISTIC and reaches no model.

The theorem
-----------
Take a filleted edge ``e`` with the two solid faces ``f1, f2`` meeting along it.
A fillet of radius ``r`` is the arc of a circle of radius ``r`` tangent to both
faces; its tangent line on each face lies a perpendicular distance ``r`` from the
edge, *measured in the face, perpendicular to the edge*. So the fillet consumes a
strip of width ``r`` from each adjacent face.

Let ``w_i`` be the extent of face ``f_i`` perpendicular to the edge. Two cases:

* if the PARALLEL edge on the far side of ``f_i`` is ALSO filleted (radius ``r``),
  the two strips together consume ``2r`` of ``w_i``; they would have to OVERLAP --
  which is impossible -- exactly when ``2r > w_i``. (At ``2r == w_i`` the arcs are
  tangent at the mid-plane and the result is a valid BULLNOSE: round the four long
  edges of a ``w x w`` bar at ``r = w/2`` and you get a cylinder. The boundary is
  therefore STRICT.)
* if that opposite edge is NOT filleted, the single strip runs off the far side of
  the face -- the tangent point has nowhere to land -- exactly when ``r > w_i``.

The fillet is degenerate iff either adjacent face fails its condition. ``2r > w``
(resp. ``r > w``) is therefore a THEOREM of tangent-arc geometry, not a guess --
the same kind of offset-surface argument that makes the shell ``2t >= extent``
rule PROVEN. It is proved only inside the scope where ``w_i`` and the edge
topology are *exactly* known (see :func:`box_extents`); outside that scope this
module ABSTAINS rather than guess, so it can never fire on a correct part.

Scope
-----
Exactly one axis-aligned rectangular prism on the XY plane: one ``NewSketch("XY")``,
one ``AddRectangle``, one ``Extrude`` (non-zero), and any number of ``Fillet`` /
``Chamfer`` ops -- nothing else. Under those ops the solid IS the box the sketch +
extrude describe, its twelve edges are enumerable exactly, and CadQuery string
selectors resolve against them deterministically (:mod:`selector_dsl`). Any other
op (boolean, hole, revolve, mirror, pattern, a second sketch, a non-XY plane, a
SetParam that could move a dimension) puts the topology beyond exact knowledge and
the module abstains.

Deterministic, stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.topology import selector_dsl

__all__ = [
    "CODE",
    "Box",
    "box_extents",
    "degenerate_fillets",
    "Finding",
]

#: The diagnostic code this rule emits. Declared PROVEN in verifiers.soundness.
CODE = "edge-fillet-degenerate"

# Exactly the ops under which the solid is provably the box the sketch describes.
_ALLOWED = ("NewSketch", "AddRectangle", "Extrude", "Fillet", "Chamfer")


@dataclass(frozen=True)
class Box:
    """An axis-aligned rectangular prism, world frame."""

    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float

    @property
    def lx(self) -> float:
        return self.x1 - self.x0

    @property
    def ly(self) -> float:
        return self.y1 - self.y0

    @property
    def lz(self) -> float:
        return self.z1 - self.z0


@dataclass(frozen=True)
class Finding:
    """One degenerate filleted edge: which op, which edge, and the arithmetic."""

    op_index: int
    radius: float
    edge: str            # a human key, e.g. "edge along X at (y=30, z=6)"
    extent: float        # the adjacent-face perpendicular extent that is exceeded
    opposite_filleted: bool
    consumed: float      # r, or 2r when the opposite edge is filleted too


def box_extents(ops: Sequence[object]) -> Optional[Box]:
    """The box the ops build, or ``None`` if the ops leave the scope of the proof.

    Returns a :class:`Box` only for a single XY-plane rectangle extruded once, with
    no other geometry-bearing op present. Everything else -> ``None`` (abstain).
    """
    sketches = [o for o in ops if type(o).__name__ == "NewSketch"]
    rects = [o for o in ops if type(o).__name__ == "AddRectangle"]
    extrudes = [o for o in ops if type(o).__name__ == "Extrude"]
    # Every op must be one of the allowed kinds.
    for o in ops:
        if type(o).__name__ not in _ALLOWED:
            return None
    if len(sketches) != 1 or len(rects) != 1 or len(extrudes) != 1:
        return None
    plane = getattr(sketches[0], "plane", "XY")
    if plane != "XY":
        # A non-XY plane would require transforming the world-frame selectors; we
        # do not, so we abstain rather than risk an unsound selector resolution.
        return None
    rect = rects[0]
    extrude = extrudes[0]
    try:
        rx, ry = float(rect.x), float(rect.y)
        rw, rh = float(rect.w), float(rect.h)
        dist = float(extrude.distance)
    except (AttributeError, TypeError, ValueError):
        return None
    if getattr(rect, "sketch", None) != getattr(extrude, "sketch", None):
        return None
    if rw == 0.0 or rh == 0.0 or dist == 0.0:
        return None
    x0, x1 = sorted((rx, rx + rw))
    y0, y1 = sorted((ry, ry + rh))
    z0, z1 = sorted((0.0, dist))
    return Box(x0, y0, z0, x1, y1, z1)


def _edges(box: Box) -> Tuple[List[selector_dsl.Entity], dict]:
    """The twelve box edges as selector entities, plus a topology table.

    The table maps each edge key to its two adjacency constraints. Each constraint
    is ``(perpendicular_extent, opposite_edge_key)``: the extent of an adjacent
    face measured perpendicular to the edge, and the key of the parallel edge on
    the far side of that face (whose fillet, if any, eats the extent from the
    other end).
    """
    xm = (box.x0 + box.x1) / 2.0
    ym = (box.y0 + box.y1) / 2.0
    zm = (box.z0 + box.z1) / 2.0
    xs = (box.x0, box.x1)
    ys = (box.y0, box.y1)
    zs = (box.z0, box.z1)

    entities: List[selector_dsl.Entity] = []
    table: dict = {}

    def other(vals, v):
        return vals[1] if v == vals[0] else vals[0]

    def key(axis, a, b):
        return "%s|%r|%r" % (axis, a, b)

    # Edges along X: vary (y, z). Adjacent faces: z-face (extent Ly, opposite in y)
    # and y-face (extent Lz, opposite in z).
    for yv in ys:
        for zv in zs:
            k = key("X", yv, zv)
            entities.append(selector_dsl.Entity(
                center=(xm, yv, zv), axis=(1.0, 0.0, 0.0),
                geom_type="LINE", name=k))
            table[k] = [
                (box.ly, key("X", other(ys, yv), zv)),
                (box.lz, key("X", yv, other(zs, zv))),
            ]
    # Edges along Y: vary (x, z). Faces: z-face (extent Lx) and x-face (extent Lz).
    for xv in xs:
        for zv in zs:
            k = key("Y", xv, zv)
            entities.append(selector_dsl.Entity(
                center=(xv, ym, zv), axis=(0.0, 1.0, 0.0),
                geom_type="LINE", name=k))
            table[k] = [
                (box.lx, key("Y", other(xs, xv), zv)),
                (box.lz, key("Y", xv, other(zs, zv))),
            ]
    # Edges along Z: vary (x, y). Faces: x-face (extent Ly) and y-face (extent Lx).
    for xv in xs:
        for yv in ys:
            k = key("Z", xv, yv)
            entities.append(selector_dsl.Entity(
                center=(xv, yv, zm), axis=(0.0, 0.0, 1.0),
                geom_type="LINE", name=k))
            table[k] = [
                (box.ly, key("Z", xv, other(ys, yv))),
                (box.lx, key("Z", other(xs, xv), yv)),
            ]
    return entities, table


def _selected_keys(edges: Sequence[str], entities: Sequence[selector_dsl.Entity]) -> set:
    """Resolve a Fillet.edges selector tuple to the set of matched edge keys.

    An empty selector means every edge (the historical Fillet default). Any other
    selector is the union of its resolved strings. A malformed selector string
    resolves to nothing (it selects no edge), which can only make the rule fire
    LESS, never more -- so a parser failure can never manufacture a false positive.
    """
    if not edges:
        return {e.name for e in entities}
    out: set = set()
    for text in edges:
        try:
            matched = selector_dsl.select(str(text), entities)
        except selector_dsl.SelectorError:
            continue
        out.update(e.name for e in matched)
    return out


def _pretty(k: str) -> str:
    axis, a, b = k.split("|")
    a, b = float(a), float(b)
    if axis == "X":
        return "edge along X at (y=%g, z=%g)" % (a, b)
    if axis == "Y":
        return "edge along Y at (x=%g, z=%g)" % (a, b)
    return "edge along Z at (x=%g, y=%g)" % (a, b)


def degenerate_fillets(ops: Sequence[object]) -> List[Finding]:
    """Every degenerate filleted edge in the op stream, or ``[]`` (incl. abstain).

    Chamfers are treated identically -- a straight chamfer of setback ``d`` eats a
    strip of width ``d`` from each adjacent face just as a fillet of radius ``d``
    does, so the same ceiling applies.
    """
    box = box_extents(ops)
    if box is None:
        return []
    entities, table = _edges(box)

    findings: List[Finding] = []
    for i, op in enumerate(ops):
        name = type(op).__name__
        if name == "Fillet":
            r = _positive(getattr(op, "radius", None))
        elif name == "Chamfer":
            r = _positive(getattr(op, "distance", None))
        else:
            continue
        if r is None:                      # non-positive / missing: not our case
            continue
        selected = _selected_keys(getattr(op, "edges", ()), entities)
        for entity in entities:                # built order -> deterministic
            k = entity.name
            if k not in selected:
                continue
            for extent, opp_key in table.get(k, ()):
                opp_selected = opp_key in selected
                consumed = 2.0 * r if opp_selected else r
                # STRICT `>`. At consumed == extent the two arcs are exactly
                # tangent at the mid-plane: the flat face has zero width but the
                # solid is a valid BULLNOSE (round all four long edges of a w x w
                # bar at r = w/2 and you get a cylinder). Only consumed > extent is
                # geometrically impossible -- the arcs would have to overlap. A
                # PROVEN rule claims infeasibility, so it must fire ONLY on the
                # impossible case; the exact bullnose builds (test_soundness's
                # filleted_thin_plate is r=3 on a 6 mm plate, and it is valid).
                if consumed > extent:
                    findings.append(Finding(
                        op_index=i, radius=r, edge=_pretty(k),
                        extent=extent, opposite_filleted=opp_selected,
                        consumed=consumed))
                    break              # one failing face is enough to condemn it
    return findings


def _positive(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0.0 else None
