"""Shape-level kernel preflight: typed failures with actionable suggestions.

OpenCAD's kernel refuses to hand a doomed operation to OCCT.  Before every boolean,
fillet, shell or draft it runs cheap *shape-level* checks (``core/checks.py``) --
non-zero volume, manifoldness, bounding-box overlap, near-tangency -- and, when one
trips, returns a structured :class:`Failure` rather than an exception: an error
``code`` from a closed taxonomy (``ERRORS.md``), a human message, the name of the
``failed_check``, and a *suggestion* telling the caller (or the LLM agent) how to
fix it.  That envelope is what makes an agent loop able to self-correct instead of
guessing.

The harness lints op *plans* symbolically (:mod:`verifiers.precheck`) and repairs
built geometry (:mod:`reliability.repair`), but has no numeric preflight on the
shapes themselves.  This module adds it:

* ``check_nonzero_volume`` / ``check_manifold`` -- input sanity;
* ``check_bbox_overlap`` -- booleans whose bounding boxes do not overlap (union of
  disjoint solids, cut that removes nothing) or overlap so little they are
  near-tangent, the classic OCCT boolean-failure trigger;
* ``check_containment`` -- a cut whose tool swallows the target entirely (empty
  result) or a union where one operand is redundant;
* ``check_fillet_radius`` / ``check_shell_thickness`` -- feature parameters larger
  than the geometry can carry;
* ``preflight_boolean`` / ``preflight_fillet`` / ``preflight_shell`` -- the ordered
  check batteries, returning the *first* failure (checks are ordered cheapest and
  most-diagnostic first) or ``None``.

Deterministic: pure arithmetic on shape metadata; no clock, no randomness.

Public API
----------
``ErrorCode``, ``Failure``, ``BoundingBox``, ``ShapeInfo``
``check_*`` predicates and ``preflight_*`` batteries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

__all__ = [
    "ErrorCode",
    "Failure",
    "BoundingBox",
    "ShapeInfo",
    "check_nonzero_volume",
    "check_manifold",
    "check_bbox_overlap",
    "check_containment",
    "check_fillet_radius",
    "check_shell_thickness",
    "preflight_boolean",
    "preflight_fillet",
    "preflight_shell",
]


class ErrorCode:
    ZERO_VOLUME = "ZERO_VOLUME"
    NON_MANIFOLD = "NON_MANIFOLD"
    BBOX_NO_OVERLAP = "BBOX_NO_OVERLAP"
    BBOX_NEAR_TANGENT = "BBOX_NEAR_TANGENT"
    EMPTY_RESULT = "EMPTY_RESULT"
    REDUNDANT_OPERAND = "REDUNDANT_OPERAND"
    RADIUS_TOO_LARGE = "RADIUS_TOO_LARGE"
    THICKNESS_TOO_LARGE = "THICKNESS_TOO_LARGE"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    suggestion: str
    failed_check: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "suggestion": self.suggestion,
            "failed_check": self.failed_check,
        }


@dataclass(frozen=True)
class BoundingBox:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @property
    def extents(self):
        return (
            max(0.0, self.max_x - self.min_x),
            max(0.0, self.max_y - self.min_y),
            max(0.0, self.max_z - self.min_z),
        )

    @property
    def volume(self) -> float:
        dx, dy, dz = self.extents
        return dx * dy * dz

    def min_extent(self) -> float:
        return min(self.extents)


@dataclass(frozen=True)
class ShapeInfo:
    id: str
    bbox: BoundingBox
    volume: float = 0.0
    manifold: bool = True
    min_edge_length: Optional[float] = None


def overlap_box(a: BoundingBox, b: BoundingBox) -> Optional[BoundingBox]:
    """Intersection box of two AABBs, or ``None`` when they do not overlap."""
    min_x = max(a.min_x, b.min_x)
    min_y = max(a.min_y, b.min_y)
    min_z = max(a.min_z, b.min_z)
    max_x = min(a.max_x, b.max_x)
    max_y = min(a.max_y, b.max_y)
    max_z = min(a.max_z, b.max_z)
    if max_x < min_x or max_y < min_y or max_z < min_z:
        return None
    return BoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)


def overlap_volume(a: BoundingBox, b: BoundingBox) -> float:
    box = overlap_box(a, b)
    return 0.0 if box is None else box.volume


def contains(outer: BoundingBox, inner: BoundingBox, tolerance: float = 0.0) -> bool:
    """True when *outer* fully encloses *inner* (within *tolerance*)."""
    return (
        outer.min_x - tolerance <= inner.min_x
        and outer.min_y - tolerance <= inner.min_y
        and outer.min_z - tolerance <= inner.min_z
        and outer.max_x + tolerance >= inner.max_x
        and outer.max_y + tolerance >= inner.max_y
        and outer.max_z + tolerance >= inner.max_z
    )


# ── individual checks ───────────────────────────────────────────────


def check_nonzero_volume(shape: ShapeInfo, tolerance: float = 1e-9) -> Optional[Failure]:
    if shape.volume <= tolerance:
        return Failure(
            code=ErrorCode.ZERO_VOLUME,
            message="Shape '%s' has zero or near-zero volume (%.3e)."
            % (shape.id, shape.volume),
            suggestion="Regenerate the shape with larger dimensions.",
            failed_check="nonzero_volume",
        )
    return None


def check_manifold(shape: ShapeInfo) -> Optional[Failure]:
    if not shape.manifold:
        return Failure(
            code=ErrorCode.NON_MANIFOLD,
            message="Shape '%s' is not manifold." % shape.id,
            suggestion="Heal or recreate the input shape before Boolean operations.",
            failed_check="manifold",
        )
    return None


def check_bbox_overlap(
    a: ShapeInfo, b: ShapeInfo, tolerance: float = 1e-9
) -> Optional[Failure]:
    volume = overlap_volume(a.bbox, b.bbox)
    if volume <= 0.0:
        return Failure(
            code=ErrorCode.BBOX_NO_OVERLAP,
            message="Bounding boxes for '%s' and '%s' do not overlap." % (a.id, b.id),
            suggestion="Move the shapes closer together before union/intersection/cut.",
            failed_check="bbox_overlap",
        )
    if volume <= tolerance:
        return Failure(
            code=ErrorCode.BBOX_NEAR_TANGENT,
            message="Bounding boxes for '%s' and '%s' are near-tangent (overlap %.3e)."
            % (a.id, b.id, volume),
            suggestion="Increase the overlap or relax the tolerance for near-tangent geometry.",
            failed_check="bbox_overlap",
        )
    return None


def check_containment(
    target: ShapeInfo, tool: ShapeInfo, operation: str
) -> Optional[Failure]:
    """Flag booleans whose bounding boxes make the result trivial."""
    if operation == "cut" and contains(tool.bbox, target.bbox):
        return Failure(
            code=ErrorCode.EMPTY_RESULT,
            message="Tool '%s' fully encloses target '%s'; the cut would remove everything."
            % (tool.id, target.id),
            suggestion="Shrink the tool or offset it so part of the target survives.",
            failed_check="containment",
        )
    if operation == "union" and contains(target.bbox, tool.bbox):
        return Failure(
            code=ErrorCode.REDUNDANT_OPERAND,
            message="Target '%s' already encloses tool '%s'; the union adds nothing."
            % (target.id, tool.id),
            suggestion="Drop the redundant operand or move the tool outside the target.",
            failed_check="containment",
        )
    return None


def check_fillet_radius(shape: ShapeInfo, radius: float) -> Optional[Failure]:
    """Reject a fillet radius the geometry cannot carry.

    The constraint is ``radius < limit / 2`` where ``limit`` is the smallest
    adjacent extent (``min_edge_length`` when known, else the smallest bbox
    extent): two fillets of radius ``r`` eat ``r`` from each side of that
    extent, so at ``2r == limit`` they meet and the face between them vanishes.
    ``2r == limit`` is therefore the *degenerate limit* and must fire — the old
    strict ``>`` let exactly that case through while flagging everything past
    it, so the rule disagreed with its own suggestion ("reduce below limit/2").
    """
    if radius <= 0.0:
        return Failure(
            code=ErrorCode.INVALID_INPUT,
            message="Fillet radius must be positive (got %.6g)." % radius,
            suggestion="Use a radius greater than zero.",
            failed_check="fillet_radius",
        )
    limit = shape.min_edge_length
    if limit is None:
        limit = shape.bbox.min_extent()
    if radius * 2.0 >= limit:
        return Failure(
            code=ErrorCode.RADIUS_TOO_LARGE,
            message="Fillet radius %.6g leaves no face on the smallest extent (%.6g) of '%s'."
            % (radius, limit, shape.id),
            suggestion="Reduce the radius below %.6g." % (limit / 2.0),
            failed_check="fillet_radius",
        )
    return None


def check_shell_thickness(shape: ShapeInfo, thickness: float) -> Optional[Failure]:
    if thickness <= 0.0:
        return Failure(
            code=ErrorCode.INVALID_INPUT,
            message="Shell thickness must be positive (got %.6g)." % thickness,
            suggestion="Use a thickness greater than zero.",
            failed_check="shell_thickness",
        )
    limit = shape.bbox.min_extent()
    if thickness * 2.0 >= limit:
        return Failure(
            code=ErrorCode.THICKNESS_TOO_LARGE,
            message="Shell thickness %.6g leaves no cavity in '%s' (smallest extent %.6g)."
            % (thickness, shape.id, limit),
            suggestion="Reduce the thickness below %.6g." % (limit / 2.0),
            failed_check="shell_thickness",
        )
    return None


# ── batteries ───────────────────────────────────────────────────────


def preflight_boolean(
    target: ShapeInfo,
    tool: ShapeInfo,
    operation: str,
    *,
    tolerance: float = 1e-9,
) -> Optional[Failure]:
    """Run the boolean check battery; returns the first failure, or ``None``."""
    if operation not in ("union", "cut", "intersection"):
        return Failure(
            code=ErrorCode.INVALID_INPUT,
            message="Unknown boolean operation '%s'." % operation,
            suggestion="Use one of: union, cut, intersection.",
            failed_check="operation_lookup",
        )
    for shape in (target, tool):
        failure = check_nonzero_volume(shape, tolerance) or check_manifold(shape)
        if failure is not None:
            return failure
    failure = check_bbox_overlap(target, tool, tolerance)
    if failure is not None:
        # A union of disjoint solids is legal (it makes a compound); everything
        # else genuinely needs overlapping geometry.
        if not (operation == "union" and failure.code == ErrorCode.BBOX_NO_OVERLAP):
            return failure
    return check_containment(target, tool, operation)


def preflight_fillet(
    shape: ShapeInfo, radius: float, *, tolerance: float = 1e-9
) -> Optional[Failure]:
    return (
        check_nonzero_volume(shape, tolerance)
        or check_manifold(shape)
        or check_fillet_radius(shape, radius)
    )


def preflight_shell(
    shape: ShapeInfo, thickness: float, *, tolerance: float = 1e-9
) -> Optional[Failure]:
    return (
        check_nonzero_volume(shape, tolerance)
        or check_manifold(shape)
        or check_shell_thickness(shape, thickness)
    )
