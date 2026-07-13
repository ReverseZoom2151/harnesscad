"""Fluent geometric entity selection heuristic (deterministic, stdlib-only).

Ported from the ``EdgeSelector`` / ``FaceSelector`` classes in CascadeStudio's
``StandardLibrary.js``. In CascadeStudio the fillet/chamfer workflow never asks
the user for raw edge indices; instead a fluent query language selects edges or
faces by *geometric intent* -- "the edges parallel to Z", "the top face", "the
edges longer than 5mm inside this box" -- and the surviving indices are what get
filleted. The selection algebra is a pure, deterministic algorithm that only
needs three scalar attributes per entity (a representative position, an optional
direction, and an optional size); the OpenCascade kernel merely *supplies* those
attributes. This module reimplements the algebra over abstract entities so it is
usable with any geometry source.

The original class-per-kind design (EdgeSelector vs FaceSelector) is unified
here into one :class:`EntitySelector` whose ``direction`` slot means the edge
tangent (edges) or the face normal (faces) and whose ``size`` slot means length
(edges) or area (faces). Every filter returns a *new* selector, so a query chain
never mutates its source; terminal methods return plain lists.

Key algorithms preserved from the source:

* projection ordering -- entities are ordered by ``dot(position, axis_hat)``;
* tolerance grouping -- ``group_by`` walks the sorted projections and starts a
  new bucket whenever the gap exceeds a tolerance, so ``min``/``max`` return the
  *whole* extreme coplanar set (e.g. all four top edges), not a single entity;
* orientation filters -- ``parallel`` (|cos| ~ 1), ``perpendicular`` (|cos| ~ 0)
  and ``at_angle`` compare the (undirected) angle between direction and axis.

Deterministic: pure arithmetic; stable sorts; no clock, no randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]


def _as_vec3(v: Sequence[float]) -> Vec3:
    if len(v) == 2:
        return (float(v[0]), float(v[1]), 0.0)
    if len(v) == 3:
        return (float(v[0]), float(v[1]), float(v[2]))
    raise ValueError("expected a 2- or 3-component vector, got %d" % len(v))


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalize(v: Vec3) -> Vec3:
    n = _length(v)
    if n < 1e-10:
        raise ValueError("cannot normalize a zero-length vector; check the axis")
    return (v[0] / n, v[1] / n, v[2] / n)


@dataclass(frozen=True)
class Entity:
    """One selectable topological entity (an edge or a face).

    ``position``  -- representative point (edge midpoint / face centroid).
    ``direction`` -- edge tangent or face normal; ``None`` if undefined
                     (e.g. a non-linear edge has no single direction).
    ``size``      -- edge length or face area; ``None`` if not measured.
    ``kind``      -- free-form type tag ("Line", "Circle", "Plane", ...).
    ``index``     -- stable identity used by :meth:`EntitySelector.indices`.
    """

    index: int
    position: Vec3
    direction: Optional[Vec3] = None
    size: Optional[float] = None
    kind: Optional[str] = None


class EntitySelector:
    """Immutable fluent selector over a list of :class:`Entity`."""

    def __init__(self, entities: Sequence[Entity]):
        self._entries: List[Entity] = list(entities)

    # --- construction helpers -------------------------------------------

    @classmethod
    def from_tuples(cls, rows: Sequence[dict]) -> "EntitySelector":
        """Build from a list of dicts with keys position/direction/size/kind.

        ``index`` defaults to enumeration order when omitted.
        """
        ents = []
        for i, r in enumerate(rows):
            ents.append(
                Entity(
                    index=int(r.get("index", i)),
                    position=_as_vec3(r["position"]),
                    direction=_as_vec3(r["direction"]) if r.get("direction") is not None else None,
                    size=None if r.get("size") is None else float(r["size"]),
                    kind=r.get("kind"),
                )
            )
        return cls(ents)

    def _clone(self, entries: Sequence[Entity]) -> "EntitySelector":
        return EntitySelector(entries)

    # --- orientation filters --------------------------------------------

    def of_type(self, kind: str) -> "EntitySelector":
        return self._clone([e for e in self._entries if e.kind == kind])

    def parallel(self, axis: Sequence[float], tolerance: float = 1e-4) -> "EntitySelector":
        a = _normalize(_as_vec3(axis))
        out = []
        for e in self._entries:
            if e.direction is None:
                continue
            d = _normalize(e.direction)
            if abs(abs(_dot(d, a)) - 1.0) < tolerance:
                out.append(e)
        return self._clone(out)

    def perpendicular(self, axis: Sequence[float], tolerance: float = 1e-4) -> "EntitySelector":
        a = _normalize(_as_vec3(axis))
        out = []
        for e in self._entries:
            if e.direction is None:
                continue
            d = _normalize(e.direction)
            if abs(_dot(d, a)) < tolerance:
                out.append(e)
        return self._clone(out)

    def at_angle(self, axis: Sequence[float], degrees: float, tolerance: float = 1.0) -> "EntitySelector":
        a = _normalize(_as_vec3(axis))
        out = []
        for e in self._entries:
            if e.direction is None:
                continue
            d = _normalize(e.direction)
            dot = min(1.0, abs(_dot(d, a)))
            angle = math.degrees(math.acos(dot))
            if abs(angle - degrees) < tolerance:
                out.append(e)
        return self._clone(out)

    # --- sorting & positional -------------------------------------------

    def sort_by(self, axis: Sequence[float]) -> "EntitySelector":
        a = _normalize(_as_vec3(axis))
        ordered = sorted(self._entries, key=lambda e: _dot(e.position, a))
        return self._clone(ordered)

    def group_by(self, axis: Sequence[float], tolerance: float = 1e-3) -> List[List[Entity]]:
        a = _normalize(_as_vec3(axis))
        sorted_entries = self.sort_by(axis)._entries
        groups: List[Tuple[float, List[Entity]]] = []
        for e in sorted_entries:
            pos = _dot(e.position, a)
            if not groups or abs(pos - groups[-1][0]) > tolerance:
                groups.append((pos, [e]))
            else:
                groups[-1][1].append(e)
        return [g[1] for g in groups]

    def max(self, axis: Sequence[float], tolerance: float = 1e-3) -> "EntitySelector":
        groups = self.group_by(axis, tolerance)
        if not groups:
            return self._clone([])
        return self._clone(groups[-1])

    def min(self, axis: Sequence[float], tolerance: float = 1e-3) -> "EntitySelector":
        groups = self.group_by(axis, tolerance)
        if not groups:
            return self._clone([])
        return self._clone(groups[0])

    # --- size / box filters ---------------------------------------------

    def longer_than(self, size: float) -> "EntitySelector":
        return self._clone([e for e in self._entries if e.size is not None and e.size > size])

    def shorter_than(self, size: float) -> "EntitySelector":
        return self._clone([e for e in self._entries if e.size is not None and e.size < size])

    # aliases matching FaceSelector's area vocabulary
    larger_than = longer_than
    smaller_than = shorter_than

    def within_box(self, min_corner: Sequence[float], max_corner: Sequence[float]) -> "EntitySelector":
        lo = _as_vec3(min_corner)
        hi = _as_vec3(max_corner)
        out = []
        for e in self._entries:
            p = e.position
            if (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] and lo[2] <= p[2] <= hi[2]):
                out.append(e)
        return self._clone(out)

    # --- slicing --------------------------------------------------------

    def first(self, n: int = 1) -> "EntitySelector":
        return self._clone(self._entries[:n])

    def last(self, n: int = 1) -> "EntitySelector":
        if n <= 0:
            return self._clone([])
        return self._clone(self._entries[-n:])

    def at(self, position: int) -> int:
        """Return the stable index of the entity at list *position*, or -1."""
        if 0 <= position < len(self._entries):
            return self._entries[position].index
        return -1

    # --- terminals ------------------------------------------------------

    def indices(self) -> List[int]:
        return [e.index for e in self._entries]

    def entities(self) -> List[Entity]:
        return list(self._entries)

    def count(self) -> int:
        return len(self._entries)
