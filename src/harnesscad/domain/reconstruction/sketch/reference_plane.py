"""Reference-plane finding for dependency-based sketch planes.

The baseline encoding defines every sketch plane from an origin and direction
vectors. On top of that we support *dependency-based* planes: a new sketch's
reference plane may be an existing face of a previously created extrude
(``add_sketchplane_ref``), which is more human-like and editable. What follows is
the deterministic geometric procedure that, given a desired target plane, searches
the existing extrude features for a face that coincides with it.

The procedure:

* ``t`` is the target plane's normal; for each existing extrude ``E`` with sketch
  normal ``n``:
  * if ``n`` is **parallel** to ``t``: the base face (signed distance 0) or the
    top face (signed distance == ``E``'s extent) is the reference -> found;
    otherwise ``E`` cannot contain it;
  * elif ``n`` is **perpendicular** to ``t``: any boundary line of ``E``'s sketch
    that lies on the target plane yields a **side face** reference -> found;
  * else ``E`` doesn't contain the reference plane.

Returns the first match, exiting early. Pure and deterministic; the coordinates
live in absolute space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from harnesscad.domain.programs.ast import openecad as oe

DEFAULT_TOL = 1e-6


# --- tiny 3-vector helpers --------------------------------------------------
def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a) -> float:
    return math.sqrt(_dot(a, a))


def is_parallel(a, b, tol: float = DEFAULT_TOL) -> bool:
    """True when *a* and *b* point along the same line (either direction)."""
    return _norm(_cross(a, b)) <= tol * max(1.0, _norm(a) * _norm(b))


def is_perpendicular(a, b, tol: float = DEFAULT_TOL) -> bool:
    return abs(_dot(a, b)) <= tol * max(1.0, _norm(a) * _norm(b))


@dataclass(frozen=True)
class ExtrudeFeature:
    """An existing extrude, as much of it as the search needs.

    ``normal`` is its base sketch plane's unit normal, ``origin`` a point on that
    base plane, ``extent`` the extrusion length along ``normal``, and ``lines``
    the sketch's boundary segments as ``((x,y,z),(x,y,z))`` pairs in 3D world
    space (used only for side-face detection).
    """

    normal: tuple
    origin: tuple
    extent: float
    lines: tuple = ()


@dataclass(frozen=True)
class RefPlaneResult:
    """Outcome of :func:`find_reference_plane`."""

    found: bool
    extrude_index: int = -1
    ref_type: str = ""            # oe.REF_SAMEPLANE / REF_TOPFACE / REF_SIDEFACE
    line_index: int = -1          # for a side face, which boundary line

    def as_call_kwargs(self) -> dict:
        """The keyword args an ``add_sketchplane_ref`` call would carry."""
        kw = {"type": self.ref_type}
        if self.ref_type == oe.REF_SIDEFACE:
            kw["line_index"] = self.line_index
        return kw


def _signed_distance(target_point, base_origin, normal) -> float:
    unit = normal
    length = _norm(normal)
    if length == 0:
        raise ValueError("degenerate (zero) normal")
    unit = tuple(c / length for c in normal)
    return _dot(_sub(target_point, base_origin), unit)


def _point_on_plane(point, target_point, target_normal, tol) -> bool:
    return abs(_dot(_sub(point, target_point), target_normal)) <= tol * max(
        1.0, _norm(target_normal))


def find_reference_plane(target_normal, target_point,
                         extrudes: list[ExtrudeFeature],
                         tol: float = DEFAULT_TOL) -> RefPlaneResult:
    """Search *extrudes* for a face coinciding with the target plane.

    Returns the first matching :class:`RefPlaneResult`, or ``found=False``.
    """
    for i, e in enumerate(extrudes):
        if is_parallel(e.normal, target_normal, tol):
            dist = _signed_distance(target_point, e.origin, e.normal)
            if abs(dist) <= tol:
                return RefPlaneResult(True, i, oe.REF_SAMEPLANE)
            if abs(abs(dist) - abs(e.extent)) <= tol:
                return RefPlaneResult(True, i, oe.REF_TOPFACE)
            # E can't contain the reference plane; try the next extrude.
            continue
        if is_perpendicular(e.normal, target_normal, tol):
            for j, (p0, p1) in enumerate(e.lines):
                if (_point_on_plane(p0, target_point, target_normal, tol)
                        and _point_on_plane(p1, target_point, target_normal, tol)):
                    return RefPlaneResult(True, i, oe.REF_SIDEFACE, j)
    return RefPlaneResult(False)
