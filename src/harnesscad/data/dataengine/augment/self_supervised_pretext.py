"""Geometric self-supervision pretext task for CAD B-Rep faces.

Jones, Hu, Kim & Schulz, *Self-Supervised Representation Learning for CAD* (2022),
Section 3 ("Geometric Self-Supervision") and Section 3 "Training".

The paper's central, distinctive idea is to learn CAD-face embeddings **without any
labels** by training an encoder-decoder to *rasterize* local B-Rep geometry. A
B-Rep face is an explicitly represented parametric surface ``S : R^2 -> R^3``
bounded by an *implicit* clipping mask -- the region inside the neighbouring edges.
The decoder is asked to reproduce, per sampled ``(u, v)`` point, the joint field

    R^2 -> R^4,   (u, v) |-> (x, y, z, d)

where ``(x, y, z) = S(u, v)`` is the explicit surface position and ``d`` is the
*2D signed distance* to the clipping boundary (negative inside the clip, positive
outside). This module is the fully deterministic **pretext-target constructor**:
given a supporting surface and a clipping polygon in uv-space it produces the
self-supervision training targets. The learned encoder/decoder weights are out of
scope; the geometry-to-target map is closed form.

Distinct from the batch's contrastive (``bench.contrastcad_*``, InfoNCE + dropout
views) and masked-modelling (``dataengine.flexcad_masking``) pretext tasks: this
is a *reconstruct-the-geometry* pretext, not an instance-discrimination or
fill-in-the-blank pretext.

What the paper specifies, and this module implements:

* **uv reparameterization** so the clipping mask fits snugly inside the unit
  square ``[0, 1]^2`` (Section 4, "reparameterize the uv-space ... so that the
  clipping mask fits snugly within the unit square"). This normalises the field
  input range across faces of wildly different scale.
* the **implicit 2D SDF** of the clipping polygon: point-in-polygon (ray casting)
  for the sign and minimum point-to-edge distance for the magnitude.
* the **explicit surface** evaluators for the paper's fixed-parameter primitives
  (plane, cylinder, sphere, cone) -- ``R^2 -> R^3``.
* **boundary-biased sampling** (Section 3, "Training"): draw a large candidate
  pool over the reparameterized square (the paper samples in ``[-0.1, 1.1]^2`` to
  guarantee inside/outside coverage), keep ``N`` points, and force a fraction
  (0.40 in the paper) to be the ones nearest the 0-level set by sorting on
  ``|d|`` -- the rest sampled uniformly. Seeded ``random.Random`` only; no wall
  clock.
* the **nearest-opposite-set SDF approximation** the paper uses in place of a CAD
  kernel: label points inside/outside, then approximate each point's unsigned
  distance as the distance to its nearest neighbour in the *opposite* set (they
  accelerate with a KD-tree; here brute force, stdlib only).

Stdlib only, no numpy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]

# Paper's default boundary-bias fraction and sampling pad (Section 3 "Training").
DEFAULT_BOUNDARY_FRACTION = 0.40
DEFAULT_SAMPLE_PAD = 0.1  # samples drawn in [-0.1, 1.1]^2 around the unit square


# --------------------------------------------------------------------------- #
# uv reparameterization: fit the clipping mask snugly into the unit square.    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UVBox:
    """Axis-aligned bounding box of a clipping polygon in uv-space."""

    u_min: float
    v_min: float
    u_max: float
    v_max: float

    @property
    def u_span(self) -> float:
        return self.u_max - self.u_min

    @property
    def v_span(self) -> float:
        return self.v_max - self.v_min


def bounding_box(polygon: Sequence[Point2]) -> UVBox:
    """Axis-aligned uv bounding box of a clipping polygon."""
    if len(polygon) < 3:
        raise ValueError("clipping polygon needs at least 3 vertices")
    us = [p[0] for p in polygon]
    vs = [p[1] for p in polygon]
    return UVBox(min(us), min(vs), max(us), max(vs))


def reparameterize(polygon: Sequence[Point2]) -> List[Point2]:
    """Affinely map a clipping polygon so its bbox fills ``[0, 1]^2`` snugly.

    Each axis is scaled/translated independently so the mask's bounding box maps
    to the unit square (paper Section 4). A degenerate axis (zero span) maps to
    the constant 0.5 so the point stays inside the square. This is the exact
    normalisation applied *prior to training* so the decoder always sees a fixed
    input range.
    """
    box = bounding_box(polygon)
    return [_map_point(p, box) for p in polygon]


def _map_point(p: Point2, box: UVBox) -> Point2:
    u = 0.5 if box.u_span == 0.0 else (p[0] - box.u_min) / box.u_span
    v = 0.5 if box.v_span == 0.0 else (p[1] - box.v_min) / box.v_span
    return (u, v)


def reparameterizer(polygon: Sequence[Point2]):
    """Return a callable applying the same reparameterization to any uv point.

    Useful for mapping *surface-parameter* samples consistently with the polygon
    (the encoder-decoder must see surface and boundary in the same uv frame).
    """
    box = bounding_box(polygon)
    return lambda p: _map_point(p, box)


# --------------------------------------------------------------------------- #
# Implicit clipping mask: 2D signed distance to the polygon boundary.          #
# --------------------------------------------------------------------------- #
def point_in_polygon(p: Point2, polygon: Sequence[Point2]) -> bool:
    """Ray-casting (even-odd) point-in-polygon test; boundary counts as inside."""
    x, y = p
    n = len(polygon)
    inside = False
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        if _on_segment(p, (ax, ay), (bx, by)):
            return True
        if (ay > y) != (by > y):
            x_cross = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < x_cross:
                inside = not inside
    return inside


def _on_segment(p: Point2, a: Point2, b: Point2, eps: float = 1e-12) -> bool:
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if abs(cross) > eps:
        return False
    if min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and \
       min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps:
        return True
    return False


def _point_segment_distance(p: Point2, a: Point2, b: Point2) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 == 0.0:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / seg2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(p[0] - cx, p[1] - cy)


def distance_to_boundary(p: Point2, polygon: Sequence[Point2]) -> float:
    """Unsigned distance from ``p`` to the polygon boundary (nearest edge)."""
    n = len(polygon)
    return min(_point_segment_distance(p, polygon[i], polygon[(i + 1) % n])
               for i in range(n))


def signed_distance(p: Point2, polygon: Sequence[Point2]) -> float:
    """2D SDF of the clipping mask: negative inside, positive outside.

    This is exactly the implicit boundary field ``d`` the decoder learns
    (paper's "clipping mask encoded as an SDF over the parametric domain",
    ``R^2 -> R``). The sign convention -- negative inside the clip -- matches the
    common DeepSDF orientation.
    """
    mag = distance_to_boundary(p, polygon)
    return -mag if point_in_polygon(p, polygon) else mag


# --------------------------------------------------------------------------- #
# Explicit surface evaluators: fixed-parameter primitives (R^2 -> R^3).        #
# --------------------------------------------------------------------------- #
def eval_plane(u: float, v: float, *, origin: Point3 = (0.0, 0.0, 0.0),
               x_axis: Point3 = (1.0, 0.0, 0.0),
               y_axis: Point3 = (0.0, 1.0, 0.0)) -> Point3:
    """Plane ``S(u, v) = origin + u * x_axis + v * y_axis``."""
    return (origin[0] + u * x_axis[0] + v * y_axis[0],
            origin[1] + u * x_axis[1] + v * y_axis[1],
            origin[2] + u * x_axis[2] + v * y_axis[2])


def eval_cylinder(u: float, v: float, *, radius: float = 1.0,
                  center: Point3 = (0.0, 0.0, 0.0)) -> Point3:
    """Cylinder of given radius; ``u`` is the angle (radians), ``v`` the height."""
    return (center[0] + radius * math.cos(u),
            center[1] + radius * math.sin(u),
            center[2] + v)


def eval_sphere(u: float, v: float, *, radius: float = 1.0,
                center: Point3 = (0.0, 0.0, 0.0)) -> Point3:
    """Sphere; ``u`` azimuth (radians), ``v`` polar angle (radians)."""
    return (center[0] + radius * math.sin(v) * math.cos(u),
            center[1] + radius * math.sin(v) * math.sin(u),
            center[2] + radius * math.cos(v))


def eval_cone(u: float, v: float, *, half_angle: float = math.pi / 4,
              center: Point3 = (0.0, 0.0, 0.0)) -> Point3:
    """Cone; ``u`` angle (radians), ``v`` axial parameter (radius = v*tan(angle))."""
    r = v * math.tan(half_angle)
    return (center[0] + r * math.cos(u),
            center[1] + r * math.sin(u),
            center[2] + v)


_SURFACES = {
    "plane": eval_plane,
    "cylinder": eval_cylinder,
    "sphere": eval_sphere,
    "cone": eval_cone,
}


def eval_surface(kind: str, u: float, v: float, **params) -> Point3:
    """Dispatch to a named primitive surface evaluator."""
    try:
        fn = _SURFACES[kind]
    except KeyError:
        raise ValueError(f"unknown surface kind: {kind!r}")
    return fn(u, v, **params)


# --------------------------------------------------------------------------- #
# Boundary-biased sampling and the R^2 -> R^4 self-supervision targets.        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RasterSample:
    """One self-supervision target: uv point, explicit xyz, implicit SDF."""

    u: float
    v: float
    xyz: Point3
    sdf: float


def _uniform_pool(n: int, pad: float, rng: random.Random) -> List[Point2]:
    lo, hi = -pad, 1.0 + pad
    return [(rng.uniform(lo, hi), rng.uniform(lo, hi)) for _ in range(n)]


def boundary_biased_points(polygon: Sequence[Point2], n: int, seed,
                           *, pool_multiplier: int = 10,
                           boundary_fraction: float = DEFAULT_BOUNDARY_FRACTION,
                           pad: float = DEFAULT_SAMPLE_PAD) -> List[Point2]:
    """Sample ``n`` uv points biased toward the clipping 0-level set.

    Mirrors the paper's Training procedure: draw a large candidate pool over the
    padded square ``[-pad, 1+pad]^2``, then keep ``n`` points of which a fraction
    ``boundary_fraction`` are the *nearest* to the boundary (sorted by ``|d|``)
    and the remainder are sampled uniformly from the rest. Deterministic given
    ``seed``.

    The polygon is assumed already reparameterized into (near) the unit square.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= boundary_fraction <= 1.0:
        raise ValueError("boundary_fraction must be in [0, 1]")
    rng = random.Random(seed)
    pool = _uniform_pool(max(n * pool_multiplier, n), pad, rng)
    scored = sorted(pool, key=lambda p: abs(signed_distance(p, polygon)))
    n_boundary = min(int(round(n * boundary_fraction)), n, len(scored))
    near = scored[:n_boundary]
    rest_pool = scored[n_boundary:]
    n_rest = n - n_boundary
    if n_rest > 0:
        if n_rest >= len(rest_pool):
            rest = rest_pool
        else:
            idx = rng.sample(range(len(rest_pool)), n_rest)
            rest = [rest_pool[i] for i in sorted(idx)]
    else:
        rest = []
    return near + rest


def build_targets(polygon: Sequence[Point2], surface_kind: str, n: int, seed,
                  *, surface_params: dict | None = None,
                  boundary_fraction: float = DEFAULT_BOUNDARY_FRACTION,
                  reparam: bool = True) -> List[RasterSample]:
    """Construct the ``R^2 -> R^4`` self-supervision targets for one face.

    Steps (paper Section 3): optionally reparameterize the clipping polygon into
    the unit square, boundary-bias-sample uv points, then for each point emit the
    explicit surface position ``S(u, v)`` and the implicit clipping SDF ``d``.
    The result is the exact per-face target set the encoder-decoder regresses
    with an L2 loss. Deterministic given ``seed``.
    """
    params = surface_params or {}
    poly = list(reparameterize(polygon)) if reparam else list(polygon)
    pts = boundary_biased_points(poly, n, seed,
                                 boundary_fraction=boundary_fraction)
    samples: List[RasterSample] = []
    for (u, v) in pts:
        xyz = eval_surface(surface_kind, u, v, **params)
        d = signed_distance((u, v), poly)
        samples.append(RasterSample(u, v, xyz, d))
    return samples


def nearest_opposite_sdf(points: Sequence[Point2],
                         polygon: Sequence[Point2]) -> List[float]:
    """Kernel-free SDF approximation via nearest neighbour in the opposite set.

    The paper approximates the SDF without a CAD kernel by classifying sampled
    points inside/outside the clip and setting each point's *unsigned* distance
    to the distance to its nearest neighbour in the opposite set (they use a
    KD-tree; this is the brute-force stdlib equivalent). The sign is taken from
    the inside/outside membership. Returns one signed value per input point.
    """
    labels = [point_in_polygon(p, polygon) for p in points]
    inside = [p for p, lab in zip(points, labels) if lab]
    outside = [p for p, lab in zip(points, labels) if not lab]
    out: List[float] = []
    for p, is_in in zip(points, labels):
        opposite = outside if is_in else inside
        if not opposite:
            # Degenerate: fall back to the exact boundary distance.
            mag = distance_to_boundary(p, polygon)
        else:
            mag = min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in opposite)
        out.append(-mag if is_in else mag)
    return out


def l2_reconstruction_loss(predicted: Sequence[Sequence[float]],
                           targets: Sequence[RasterSample]) -> float:
    """Mean per-point L2 loss over the ``(x, y, z, d)`` field (paper Training).

    ``predicted`` supplies one 4-vector ``(x, y, z, d)`` per target sample. This
    is the exact objective the encoder-decoder minimises; given fixed predictions
    and targets it is byte-reproducible.
    """
    if len(predicted) != len(targets):
        raise ValueError("predicted and targets length mismatch")
    if not targets:
        raise ValueError("no targets supplied")
    total = 0.0
    for pred, tgt in zip(predicted, targets):
        if len(pred) != 4:
            raise ValueError("each prediction must be a 4-vector (x, y, z, d)")
        gx, gy, gz = tgt.xyz
        gt = (gx, gy, gz, tgt.sdf)
        total += sum((a - b) ** 2 for a, b in zip(pred, gt))
    return total / len(targets)
