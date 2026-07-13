"""Spatial-layout constraint solver for authored 3D scene composition.

Paper: *WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM
Agents* (Liu, Tang, Tai), Sec. 3.3 (ArrangeIt).

ArrangeIt formulates scene arrangement as a numerical optimization

    minimize  L({p_i, theta_i}) = sum_j lambda_j * L_j({p_i, theta_i})
    subject to c_1, ..., c_k

over object positions ``p_i`` and yaw orientations, using a *protocol* of spatial
relationship terms -- Distance, Relative Orientation, Alignment, Proximity,
Overlap, Symmetry -- that can act as soft score terms (weighted objective) or
hard constraints. The paper solves it with **simulated annealing** and the
Metropolis-Hastings acceptance criterion (Yu et al. 2011; Kirkpatrick 1984).

This module is the deterministic, stdlib-only realisation of that solver. It is
DISTINCT from ``reconstruction.scenegraph_model`` (which *reads* relations off
fixed geometry) and from the assembly/part-mating modules: here the poses are
free design variables and the solver *searches* for placements that satisfy the
authored constraints. It operates on the
:class:`reconstruction.worldcraft_layout_spec.LayoutSpec` representation.

Determinism: all randomness flows through an injected ``random.Random(seed)``;
no wall clock is read. Given the same spec, constraints and seed, the solved
layout is bit-for-bit reproducible.

Constraint kinds (each contributes a non-negative penalty; zero means satisfied):

* :class:`NonOverlap`      -- two objects' world footprints must not intersect;
* :class:`OnTopOf`         -- object rests on the top surface of its host
  (footprint contained, base at host top, centred);
* :class:`AlignAxis`       -- a set of objects share a common ``x`` or ``y``
  centre coordinate;
* :class:`MinDistance` / :class:`MaxDistance` -- bound the centre gap between two
  objects;
* :class:`Proximity`       -- two objects be immediately adjacent (gap <= eps);
* :class:`WithinRoom`      -- object footprint stays inside the room bounds;
* :class:`FacePoint`       -- object yaw points toward a target point.

The solver perturbs a mutable working copy of the placements (translate / yaw)
and accepts moves by Metropolis-Hastings, returning a new solved ``LayoutSpec``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.reconstruction.scene.worldcraft_layout_spec import (
    LayoutSpec,
    ObjectPlacement,
    Pose,
)

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# Geometry helpers                                                             #
# --------------------------------------------------------------------------- #
def _bounds(p: ObjectPlacement) -> Tuple[Vec3, Vec3]:
    return p.world_bounds()


def _overlap_volume(a: ObjectPlacement, b: ObjectPlacement) -> float:
    (alo, ahi) = _bounds(a)
    (blo, bhi) = _bounds(b)
    vol = 1.0
    for al, ah, bl, bh in zip(alo, ahi, blo, bhi):
        lo = max(al, bl)
        hi = min(ah, bh)
        if hi <= lo:
            return 0.0
        vol *= (hi - lo)
    return vol


def _center_distance(a: ObjectPlacement, b: ObjectPlacement) -> float:
    ac, bc = a.world_center(), b.world_center()
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(ac, bc)))


def _planar_gap(a: ObjectPlacement, b: ObjectPlacement) -> float:
    """Minimum xy-plane separation between the two footprints (0 if overlapping)."""
    (alo, ahi) = _bounds(a)
    (blo, bhi) = _bounds(b)
    gap_sq = 0.0
    for axis in (0, 1):
        if blo[axis] > ahi[axis]:
            d = blo[axis] - ahi[axis]
            gap_sq += d * d
        elif alo[axis] > bhi[axis]:
            d = alo[axis] - bhi[axis]
            gap_sq += d * d
    return math.sqrt(gap_sq)


# --------------------------------------------------------------------------- #
# Constraints                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Constraint:
    """Base spatial constraint. ``weight`` scales its penalty (``lambda_j``).

    ``hard`` marks a constraint the accepted solution must ultimately satisfy;
    it does not change the penalty but is reported by :meth:`SolveResult.satisfied`.
    """

    weight: float = 1.0
    hard: bool = False

    def penalty(self, spec: LayoutSpec) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def is_satisfied(self, spec: LayoutSpec, tol: float = 1e-6) -> bool:
        return self.penalty(spec) <= tol


@dataclass(frozen=True)
class NonOverlap(Constraint):
    a: str = ""
    b: str = ""

    def penalty(self, spec: LayoutSpec) -> float:
        return self.weight * _overlap_volume(spec.get(self.a), spec.get(self.b))


@dataclass(frozen=True)
class OnTopOf(Constraint):
    """``obj`` rests on the top face of ``host``: base flush, footprint inside."""

    obj: str = ""
    host: str = ""

    def penalty(self, spec: LayoutSpec) -> float:
        o = spec.get(self.obj)
        h = spec.get(self.host)
        (olo, ohi) = _bounds(o)
        (hlo, hhi) = _bounds(h)
        # vertical: object base should sit on host top.
        base_gap = abs(olo[2] - hhi[2])
        # horizontal: object footprint should lie within host footprint.
        contain = 0.0
        for axis in (0, 1):
            contain += max(0.0, hlo[axis] - olo[axis])
            contain += max(0.0, ohi[axis] - hhi[axis])
        return self.weight * (base_gap + contain)


@dataclass(frozen=True)
class AlignAxis(Constraint):
    """Objects share a common centre coordinate along ``axis`` (0=x, 1=y)."""

    objects: Tuple[str, ...] = ()
    axis: int = 0

    def penalty(self, spec: LayoutSpec) -> float:
        if len(self.objects) < 2:
            return 0.0
        coords = [spec.get(o).world_center()[self.axis] for o in self.objects]
        mean = sum(coords) / len(coords)
        return self.weight * sum(abs(c - mean) for c in coords)


@dataclass(frozen=True)
class MinDistance(Constraint):
    a: str = ""
    b: str = ""
    distance: float = 0.0

    def penalty(self, spec: LayoutSpec) -> float:
        d = _center_distance(spec.get(self.a), spec.get(self.b))
        return self.weight * max(0.0, self.distance - d)


@dataclass(frozen=True)
class MaxDistance(Constraint):
    a: str = ""
    b: str = ""
    distance: float = 0.0

    def penalty(self, spec: LayoutSpec) -> float:
        d = _center_distance(spec.get(self.a), spec.get(self.b))
        return self.weight * max(0.0, d - self.distance)


@dataclass(frozen=True)
class Proximity(Constraint):
    """Two objects be immediately adjacent (planar gap within ``eps``)."""

    a: str = ""
    b: str = ""
    eps: float = 0.05

    def penalty(self, spec: LayoutSpec) -> float:
        gap = _planar_gap(spec.get(self.a), spec.get(self.b))
        return self.weight * max(0.0, gap - self.eps)


@dataclass(frozen=True)
class WithinRoom(Constraint):
    """Object footprint stays inside the spec's room bounds."""

    obj: str = ""

    def penalty(self, spec: LayoutSpec) -> float:
        if spec.room_bounds is None:
            return 0.0
        rlo, rhi = spec.room_bounds
        (olo, ohi) = _bounds(spec.get(self.obj))
        out = 0.0
        for axis in range(3):
            out += max(0.0, rlo[axis] - olo[axis])
            out += max(0.0, ohi[axis] - rhi[axis])
        return self.weight * out


@dataclass(frozen=True)
class FacePoint(Constraint):
    """Object yaw should point its +x forward axis toward ``target`` in xy."""

    obj: str = ""
    target: Tuple[float, float] = (0.0, 0.0)

    def penalty(self, spec: LayoutSpec) -> float:
        o = spec.get(self.obj)
        cx, cy, _ = o.world_center()
        desired = math.atan2(self.target[1] - cy, self.target[0] - cx)
        diff = (o.pose.yaw - desired) % (2.0 * math.pi)
        if diff > math.pi:
            diff = 2.0 * math.pi - diff
        return self.weight * diff


# --------------------------------------------------------------------------- #
# Solver                                                                       #
# --------------------------------------------------------------------------- #
def total_penalty(spec: LayoutSpec, constraints: Sequence[Constraint]) -> float:
    return sum(c.penalty(spec) for c in constraints)


@dataclass
class SolveResult:
    """Outcome of :func:`solve_layout`."""

    layout: LayoutSpec
    initial_cost: float
    final_cost: float
    iterations: int
    accepted: int

    def satisfied(self, constraints: Sequence[Constraint], tol: float = 1e-6) -> bool:
        """True iff every *hard* constraint holds on the solved layout."""
        return all(c.is_satisfied(self.layout, tol) for c in constraints if c.hard)


def _clone(spec: LayoutSpec) -> LayoutSpec:
    return LayoutSpec.from_dict(spec.to_dict())


def solve_layout(
    spec: LayoutSpec,
    constraints: Sequence[Constraint],
    *,
    seed: int = 0,
    iterations: int = 2000,
    initial_temperature: float = 1.0,
    cooling: float = 0.995,
    move_scale: float = 1.0,
    movable: Optional[Sequence[str]] = None,
    rotate: bool = True,
) -> SolveResult:
    """Simulated-annealing constraint solver (Metropolis-Hastings acceptance).

    Perturbs a working copy of ``spec`` (translating an object in the xy-plane and
    optionally yawing it) and accepts each move with probability
    ``min(1, exp(-delta / T))``, cooling ``T`` geometrically. Returns the best
    layout found. Fully deterministic in ``seed``.

    ``movable`` restricts which object ids may be perturbed (default: all).
    """
    rng = random.Random(seed)
    work = _clone(spec)
    ids = list(movable) if movable is not None else list(work.object_ids)
    ids = [i for i in ids if work.has(i)]

    cur_cost = total_penalty(work, constraints)
    initial_cost = cur_cost
    best = _clone(work)
    best_cost = cur_cost

    temp = initial_temperature
    accepted = 0

    if not ids:
        return SolveResult(best, initial_cost, best_cost, 0, 0)

    for _ in range(iterations):
        oid = ids[rng.randrange(len(ids))]
        placement = work.get(oid)
        old_pose = placement.pose

        dx = rng.uniform(-move_scale, move_scale)
        dy = rng.uniform(-move_scale, move_scale)
        new_pose = old_pose.translated(dx, dy, 0.0)
        if rotate and rng.random() < 0.5:
            new_pose = new_pose.rotated_z(rng.uniform(-math.pi / 4.0, math.pi / 4.0))
        placement.pose = new_pose

        new_cost = total_penalty(work, constraints)
        delta = new_cost - cur_cost
        if delta <= 0.0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            cur_cost = new_cost
            accepted += 1
            if new_cost < best_cost:
                best_cost = new_cost
                best = _clone(work)
        else:
            placement.pose = old_pose  # reject: restore

        temp *= cooling

    return SolveResult(best, initial_cost, best_cost, iterations, accepted)
