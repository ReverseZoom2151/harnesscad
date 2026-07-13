"""Scene-validity and collision checking for authored 3D layouts.

Paper: *WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM
Agents* (Liu, Tang, Tai), Sec. 3.3 / Sec. 4.3.

WorldCraft aims for *physically feasible* scenes: the arrangement must respect
non-overlap and common-sense placement (the paper contrasts its functional
layouts with baselines that "violate common sense, such as placing a basketball
hoop over the bed", Sec. 4.3). A solved layout therefore needs a deterministic
**validity / collision pass** that flags interpenetrating objects, objects that
escape the room, floating objects that should rest on a surface, and children
that leave their host's footprint.

This module is DISTINCT from ``reconstruction.scenegraph_validity`` (paper 159),
which checks *graph structure* -- dangling edges, inverse consistency, relation
cycles. Here the checks are purely *geometric*, run over the placed world
footprints of a :class:`reconstruction.worldcraft_layout_spec.LayoutSpec`.

Checks (each yields typed :class:`CollisionIssue` records; nothing is mutated):

* **object collision** -- two objects' world AABBs interpenetrate by more than a
  tolerance (parent/child pairs are exempt from this check by default);
* **out of bounds** -- an object's footprint leaves the room bounds;
* **child escapes host** -- an object-tree child's footprint is not contained in
  its parent's footprint (the shelf/book relation);
* **floating object** -- an object flagged as needing support (via
  ``needs_support`` attribute) whose base does not rest on the floor or on
  another object's top surface;
* **stacking inversion** -- a child sits *below* the top of its host (a
  placement that would clip through the surface it should rest on).

:func:`check_scene` runs every check and returns a :class:`SceneReport` whose
``ok`` flag is true iff no error-severity issue was found. Stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from harnesscad.domain.reconstruction.scene.worldcraft_layout_spec import LayoutSpec, ObjectPlacement

Vec3 = Tuple[float, float, float]

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass(frozen=True)
class CollisionIssue:
    """A single geometric validity finding."""

    code: str
    severity: str
    message: str
    objects: Tuple[str, ...] = ()


@dataclass
class SceneReport:
    """Aggregate result of :func:`check_scene`."""

    issues: List[CollisionIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[CollisionIssue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> List[CollisionIssue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> List[str]:
        return [i.code for i in self.issues]


# --------------------------------------------------------------------------- #
# Geometry helpers                                                             #
# --------------------------------------------------------------------------- #
def _bounds(p: ObjectPlacement) -> Tuple[Vec3, Vec3]:
    return p.world_bounds()


def _overlap_extent(a: ObjectPlacement, b: ObjectPlacement) -> Vec3:
    """Per-axis positive overlap of two world AABBs (0 where they are apart)."""
    (alo, ahi) = _bounds(a)
    (blo, bhi) = _bounds(b)
    out = []
    for al, ah, bl, bh in zip(alo, ahi, blo, bhi):
        out.append(max(0.0, min(ah, bh) - max(al, bl)))
    return (out[0], out[1], out[2])


def _min_overlap(a: ObjectPlacement, b: ObjectPlacement) -> float:
    """Smallest per-axis overlap; > 0 means the boxes truly interpenetrate."""
    ox, oy, oz = _overlap_extent(a, b)
    return min(ox, oy, oz)


def _footprint_contained(child: ObjectPlacement, host: ObjectPlacement, tol: float) -> bool:
    (clo, chi) = _bounds(child)
    (hlo, hhi) = _bounds(host)
    for axis in (0, 1):
        if clo[axis] < hlo[axis] - tol or chi[axis] > hhi[axis] + tol:
            return False
    return True


# --------------------------------------------------------------------------- #
# Individual checks                                                            #
# --------------------------------------------------------------------------- #
def check_object_collisions(
    spec: LayoutSpec,
    *,
    tol: float = 1e-6,
    exempt_family: bool = True,
) -> List[CollisionIssue]:
    """Flag pairs of objects whose world AABBs interpenetrate.

    Parent/child (and, when ``exempt_family``, ancestor/descendant) pairs are
    exempt, since a book intentionally sits within its shelf's volume.
    """
    issues: List[CollisionIssue] = []
    placements = spec.placements
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            a, b = placements[i], placements[j]
            if exempt_family and _related(spec, a.object_id, b.object_id):
                continue
            if _min_overlap(a, b) > tol:
                issues.append(CollisionIssue(
                    "object_collision", ERROR,
                    f"objects {a.object_id!r} and {b.object_id!r} interpenetrate",
                    (a.object_id, b.object_id)))
    return issues


def _related(spec: LayoutSpec, a: str, b: str) -> bool:
    """True if a is an ancestor/descendant of b in the object tree."""
    if any(anc.object_id == b for anc in spec.ancestors(a)):
        return True
    if any(anc.object_id == a for anc in spec.ancestors(b)):
        return True
    return False


def check_out_of_bounds(spec: LayoutSpec, *, tol: float = 1e-6) -> List[CollisionIssue]:
    issues: List[CollisionIssue] = []
    if spec.room_bounds is None:
        return issues
    rlo, rhi = spec.room_bounds
    for p in spec.placements:
        (lo, hi) = _bounds(p)
        for axis in range(3):
            if lo[axis] < rlo[axis] - tol or hi[axis] > rhi[axis] + tol:
                issues.append(CollisionIssue(
                    "out_of_bounds", ERROR,
                    f"object {p.object_id!r} leaves room bounds on axis {axis}",
                    (p.object_id,)))
                break
    return issues


def check_child_containment(spec: LayoutSpec, *, tol: float = 1e-6) -> List[CollisionIssue]:
    issues: List[CollisionIssue] = []
    for p in spec.placements:
        if p.parent_id is None:
            continue
        host = spec.get(p.parent_id)
        if not _footprint_contained(p, host, tol):
            issues.append(CollisionIssue(
                "child_escapes_host", WARNING,
                f"object {p.object_id!r} footprint escapes host {p.parent_id!r}",
                (p.object_id, p.parent_id)))
    return issues


def check_stacking(spec: LayoutSpec, *, tol: float = 1e-6) -> List[CollisionIssue]:
    """A parented child whose base is below its host's top clips the surface."""
    issues: List[CollisionIssue] = []
    for p in spec.placements:
        if p.parent_id is None:
            continue
        host = spec.get(p.parent_id)
        (clo, _chi) = _bounds(p)
        (_hlo, hhi) = _bounds(host)
        if clo[2] < hhi[2] - tol:
            issues.append(CollisionIssue(
                "stacking_inversion", WARNING,
                f"object {p.object_id!r} base sits below host {p.parent_id!r} top",
                (p.object_id, p.parent_id)))
    return issues


def check_floating(
    spec: LayoutSpec,
    *,
    floor_z: float = 0.0,
    tol: float = 1e-6,
) -> List[CollisionIssue]:
    """Objects flagged ``needs_support`` must rest on the floor or a top surface.

    An object rests on the floor if its base is within ``tol`` of ``floor_z``. It
    rests on a support if some *other* object's top is within ``tol`` of its base
    and their footprints overlap in xy.
    """
    issues: List[CollisionIssue] = []
    placements = spec.placements
    for p in placements:
        if not bool(p.attributes.get("needs_support", False)):
            continue
        (plo, _phi) = _bounds(p)
        base = plo[2]
        if abs(base - floor_z) <= tol:
            continue
        supported = False
        for other in placements:
            if other.object_id == p.object_id:
                continue
            (olo, ohi) = _bounds(other)
            if abs(base - ohi[2]) <= tol and _xy_overlap(p, other):
                supported = True
                break
        if not supported:
            issues.append(CollisionIssue(
                "floating_object", WARNING,
                f"object {p.object_id!r} needs support but floats at z={base}",
                (p.object_id,)))
    return issues


def _xy_overlap(a: ObjectPlacement, b: ObjectPlacement) -> bool:
    ox, oy, _ = _overlap_extent(a, b)
    return ox > 0.0 and oy > 0.0


# --------------------------------------------------------------------------- #
# Aggregate                                                                    #
# --------------------------------------------------------------------------- #
def check_scene(
    spec: LayoutSpec,
    *,
    tol: float = 1e-6,
    floor_z: float = 0.0,
    check_collisions: bool = True,
    check_bounds: bool = True,
    check_containment: bool = True,
    check_stacking_order: bool = True,
    check_support: bool = True,
) -> SceneReport:
    """Run all geometric validity checks and aggregate into a :class:`SceneReport`."""
    issues: List[CollisionIssue] = []
    if check_collisions:
        issues += check_object_collisions(spec, tol=tol)
    if check_bounds:
        issues += check_out_of_bounds(spec, tol=tol)
    if check_containment:
        issues += check_child_containment(spec, tol=tol)
    if check_stacking_order:
        issues += check_stacking(spec, tol=tol)
    if check_support:
        issues += check_floating(spec, floor_z=floor_z, tol=tol)
    return SceneReport(issues=issues)
