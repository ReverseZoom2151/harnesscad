"""The CADGenBench canonical-pose contract for submitted parts.

A grader always rigidly aligns a candidate to the ground truth before scoring
(rotation + translation, **never** scale). Alignment is reliable for most parts
but stays genuinely ambiguous for rotationally or mirror-symmetric shapes, where
several poses are equivalent. The contract removes the ambiguity at the source by
asking submissions to emit a canonical pose:

1. Bounding-box centre at the origin.
2. Bounding-box extents ordered ``Lx >= Ly >= Lz``: longest axis along X,
   intermediate along Y, shortest along Z.
3. If the part has a natural mounting/reference face, put it on the
   ``z = -Lz/2`` plane with its outward normal along ``-Z``. Rules 1-2 suffice
   for parts with no obvious reference face.

Two deterministic things this module provides that the harness does not already
have (``bench.cadrille_orientation_align`` brute-forces the 24 proper rotations
against a *target* by Chamfer distance; it has no notion of a target-free
canonical frame or of a compliance check):

- :func:`canonicalize` puts a point set into the canonical frame with a **proper**
  rotation (det = +1, never a reflection - a mirrored part is a different part).
- :func:`pose_report` audits an as-submitted part against the contract and, when
  two extents are within tolerance of each other, raises an **ambiguity** flag:
  that is exactly the case where the rules cannot disambiguate and where the
  grader's alignment is at risk of picking the wrong equivalent pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float, float]

DEFAULT_TOLERANCE = 1e-6
# Two extents closer than this fraction of the largest extent make the axis
# assignment ambiguous (a square plate, a cylinder): the canonical rules cannot
# choose between equivalent poses, so the report flags it rather than pretending.
AMBIGUITY_FRACTION = 1e-3


@dataclass(frozen=True)
class BBox:
    lo: Point
    hi: Point

    @property
    def extents(self) -> Point:
        return tuple(self.hi[k] - self.lo[k] for k in range(3))  # type: ignore[return-value]

    @property
    def center(self) -> Point:
        return tuple((self.hi[k] + self.lo[k]) / 2.0 for k in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class CanonicalPoseReport:
    compliant: bool
    centered: bool
    axes_ordered: bool
    reference_face_seated: Optional[bool]
    center_offset: Point
    extents: Point
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "compliant": self.compliant,
            "centered": self.centered,
            "axes_ordered": self.axes_ordered,
            "reference_face_seated": self.reference_face_seated,
            "center_offset": list(self.center_offset),
            "extents": list(self.extents),
            "flags": list(self.flags),
        }


def bounding_box(points: Sequence[Point]) -> BBox:
    if not points:
        raise ValueError("cannot bound an empty point set")
    lo = tuple(min(p[k] for p in points) for k in range(3))
    hi = tuple(max(p[k] for p in points) for k in range(3))
    return BBox(lo, hi)  # type: ignore[arg-type]


def canonicalize(points: Sequence[Point]) -> List[Point]:
    """Return *points* in the canonical frame: centred, extents ordered X>=Y>=Z.

    The axis permutation is completed into a **proper** rotation: an odd
    permutation is a reflection (det = -1), so one axis is negated to bring the
    composite back to det = +1 and the part is re-posed rather than mirrored (a
    mirrored part is a different part). Ties are broken by the original axis
    index, which keeps the mapping deterministic on cubes and square plates
    (whose pose is genuinely ambiguous - see :func:`pose_report`).
    """
    box = bounding_box(points)
    center = box.center
    extents = box.extents
    # Descending by extent, ties broken by original axis index.
    order = sorted(range(3), key=lambda k: (-extents[k], k))
    parity = _permutation_parity(order)
    signs = [1.0, 1.0, 1.0]
    if parity == -1:
        # The permutation matrix alone has det = -1 (a reflection). Negating one
        # axis brings the composite back to det = +1, a proper rotation, so the
        # part is re-posed rather than mirrored. Extents are unchanged by the flip.
        signs[0] = -1.0
    return [
        (
            signs[0] * (p[order[0]] - center[order[0]]),
            signs[1] * (p[order[1]] - center[order[1]]),
            signs[2] * (p[order[2]] - center[order[2]]),
        )
        for p in points
    ]


def _permutation_parity(order: Sequence[int]) -> int:
    """+1 for an even permutation, -1 for an odd one."""
    perm = list(order)
    swaps = 0
    for i in range(len(perm)):
        while perm[i] != i:
            j = perm[i]
            perm[i], perm[j] = perm[j], perm[i]
            swaps += 1
    return 1 if swaps % 2 == 0 else -1


def is_centered(points: Sequence[Point], *, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    center = bounding_box(points).center
    return all(abs(c) <= tolerance for c in center)


def axes_are_ordered(
    extents: Sequence[float], *, tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    """``Lx >= Ly >= Lz`` (within tolerance)."""
    return (
        extents[0] >= extents[1] - tolerance
        and extents[1] >= extents[2] - tolerance
    )


def reference_face_seated(
    face_z: float, extents: Sequence[float], *, tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    """Is the reference face on the ``z = -Lz/2`` plane?"""
    return abs(face_z - (-extents[2] / 2.0)) <= tolerance


def ambiguity_flags(
    extents: Sequence[float], *, fraction: float = AMBIGUITY_FRACTION
) -> List[str]:
    """Flag near-equal extents: the canonical rules cannot disambiguate there."""
    flags: List[str] = []
    scale = max(extents) if max(extents) > 0 else 1.0
    pairs = (("x", "y", 0, 1), ("y", "z", 1, 2), ("x", "z", 0, 2))
    for a, b, i, j in pairs:
        if abs(extents[i] - extents[j]) <= fraction * scale:
            flags.append(
                f"pose ambiguous: extents {a} and {b} agree to within "
                f"{fraction:g} of the part size; alignment may pick an "
                "equivalent-but-different pose"
            )
    return flags


def pose_report(
    points: Sequence[Point],
    *,
    reference_face_z: Optional[float] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> CanonicalPoseReport:
    """Audit an as-submitted part against the canonical-pose contract.

    ``reference_face_z`` is the z of the part's natural mounting face, when it has
    one; leave it ``None`` for parts where rules 1-2 are the whole contract (the
    reference-face check then reports ``None``, not ``False``).
    """
    box = bounding_box(points)
    extents = box.extents
    centered = all(abs(c) <= tolerance for c in box.center)
    ordered = axes_are_ordered(extents, tolerance=tolerance)
    seated = (
        None
        if reference_face_z is None
        else reference_face_seated(reference_face_z, extents, tolerance=tolerance)
    )
    compliant = centered and ordered and (seated is not False)
    return CanonicalPoseReport(
        compliant=compliant,
        centered=centered,
        axes_ordered=ordered,
        reference_face_seated=seated,
        center_offset=box.center,
        extents=extents,
        flags=ambiguity_flags(extents),
    )
