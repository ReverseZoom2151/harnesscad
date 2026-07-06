"""img2cadsvg_representation -- the Structured Visual Geometry (SVG) schema.

Img2CAD (Chen et al., "Conditioned 3D CAD Model Generation from Single Image
with Structured Visual Geometry") introduces an intermediate representation it
calls **Structured Visual Geometry (SVG)**: a *vectorized wireframe* of an
object, "which capture line segments and their associated endpoints (primarily
junctions)".  The learned parser that predicts the wireframe from an image is out
of scope, but the *representation itself* -- its schema, construction, validity
rules, and the canonical/normalised coordinate form the paper feeds to the
conditioning encoder (``L_n in R^{N x 2}``) -- is a deterministic data structure.

This module implements that structured-visual-geometry graph:

* a wireframe = a set of **junctions** (2D endpoint proposals) plus a set of
  **segments**, each segment referencing two distinct junctions;
* :func:`build_wireframe` de-duplicates near-coincident endpoints into shared
  junctions (tolerance ``eps``) so that segments that meet at a corner share a
  junction node -- turning a loose "line soup" into a topological wireframe;
* :func:`validity` reports the structural conditions the paper's SVG must
  satisfy (no zero-length segments, no duplicate segments, every segment endpoint
  a valid junction) that determine whether the wireframe can condition a CAD
  sequence;
* :func:`normalise` reproduces the paper's normalisation of endpoint coordinates
  into a common frame (``L_n in R^{N x 2}``) before they are embedded.

Naming ``img2cadsvg_`` = Img2CAD *Structured Visual Geometry* (paper index 109),
distinct from the separate "Img2CAD -- VLM-Assisted Conditional Factorization"
work.  Pure stdlib, deterministic; no learned components.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


Point = tuple[float, float]


@dataclass(frozen=True)
class Segment:
    """A wireframe line segment referencing two junction indices."""

    a: int
    b: int

    def key(self) -> tuple[int, int]:
        """Undirected canonical key (order-independent)."""
        return (self.a, self.b) if self.a <= self.b else (self.b, self.a)


@dataclass
class Wireframe:
    """Structured Visual Geometry: junctions + segments referencing them."""

    junctions: list[Point] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)

    def endpoints(self, seg: Segment) -> tuple[Point, Point]:
        return self.junctions[seg.a], self.junctions[seg.b]

    def degrees(self) -> list[int]:
        """Number of incident segments per junction."""
        n = len(self.junctions)
        deg = [0] * n
        for s in self.segments:
            if 0 <= s.a < n:
                deg[s.a] += 1
            if 0 <= s.b < n:
                deg[s.b] += 1
        return deg


def _dist2(p: Point, q: Point) -> float:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def build_wireframe(
    raw_segments: list[tuple[Point, Point]], eps: float = 1e-6
) -> Wireframe:
    """Construct a wireframe from raw ``(p0, p1)`` line segments.

    Endpoints within Euclidean distance ``eps`` collapse to a single shared
    junction, giving segments that meet at a corner a common node.  Junction
    order is deterministic (first-seen).  Degenerate (zero-length after merge)
    segments and exact duplicate segments are dropped.
    """
    if eps < 0:
        raise ValueError("eps must be non-negative")
    eps2 = eps * eps
    junctions: list[Point] = []

    def intern(p: Point) -> int:
        for i, q in enumerate(junctions):
            if _dist2(p, q) <= eps2:
                return i
        junctions.append((float(p[0]), float(p[1])))
        return len(junctions) - 1

    seen: set[tuple[int, int]] = set()
    segments: list[Segment] = []
    for p0, p1 in raw_segments:
        a, b = intern(p0), intern(p1)
        if a == b:
            continue  # zero-length after merge
        seg = Segment(a, b)
        k = seg.key()
        if k in seen:
            continue  # duplicate
        seen.add(k)
        segments.append(seg)
    return Wireframe(junctions=junctions, segments=segments)


@dataclass(frozen=True)
class Validity:
    ok: bool
    n_junctions: int
    n_segments: int
    zero_length: int
    duplicates: int
    out_of_range: int
    isolated_junctions: int


def validity(wf: Wireframe, eps: float = 1e-9) -> Validity:
    """Structural validity of a Structured-Visual-Geometry wireframe.

    A valid SVG has: every segment references two in-range, distinct junctions;
    no zero-length segments; no duplicate (undirected) segments; and no isolated
    junctions (every junction touched by >=1 segment).
    """
    n = len(wf.junctions)
    eps2 = eps * eps
    zero_length = 0
    out_of_range = 0
    seen: set[tuple[int, int]] = set()
    duplicates = 0
    for s in wf.segments:
        if not (0 <= s.a < n and 0 <= s.b < n):
            out_of_range += 1
            continue
        pa, pb = wf.junctions[s.a], wf.junctions[s.b]
        if s.a == s.b or _dist2(pa, pb) <= eps2:
            zero_length += 1
        k = s.key()
        if k in seen:
            duplicates += 1
        seen.add(k)
    isolated = sum(1 for d in wf.degrees() if d == 0)
    ok = (
        out_of_range == 0
        and zero_length == 0
        and duplicates == 0
        and isolated == 0
        and n > 0
        and len(wf.segments) > 0
    )
    return Validity(
        ok=ok,
        n_junctions=n,
        n_segments=len(wf.segments),
        zero_length=zero_length,
        duplicates=duplicates,
        out_of_range=out_of_range,
        isolated_junctions=isolated,
    )


def bounding_box(wf: Wireframe) -> tuple[Point, Point]:
    if not wf.junctions:
        raise ValueError("empty wireframe has no bounding box")
    xs = [p[0] for p in wf.junctions]
    ys = [p[1] for p in wf.junctions]
    return (min(xs), min(ys)), (max(xs), max(ys))


def normalise(wf: Wireframe) -> Wireframe:
    """Normalise junction coordinates into ``[-1, 1]`` preserving aspect ratio.

    Reproduces the paper's mapping of raw endpoint coordinates into a common
    reference frame (``L_n in R^{N x 2}``) prior to embedding: centre the
    wireframe at its bounding-box centre and scale by the largest half-extent so
    the longest axis spans ``[-1, 1]``.  Segments are unchanged.
    """
    (minx, miny), (maxx, maxy) = bounding_box(wf)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    half = max((maxx - minx) / 2.0, (maxy - miny) / 2.0)
    scale = 1.0 / half if half > 0 else 1.0
    new_j = [((x - cx) * scale, (y - cy) * scale) for x, y in wf.junctions]
    return Wireframe(junctions=new_j, segments=list(wf.segments))


def total_length(wf: Wireframe) -> float:
    """Sum of segment lengths -- a scale-summary of the wireframe."""
    out = 0.0
    for s in wf.segments:
        pa, pb = wf.junctions[s.a], wf.junctions[s.b]
        out += math.sqrt(_dist2(pa, pb))
    return out
