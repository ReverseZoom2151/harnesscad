"""Tri-state box contact / gap classifier -- the "collision problem" auditor.

From *How Can Large Language Models Help Humans in Design and Manufacturing?*
(Makatura et al., 2024), sections 4.1.2 and L.1. The paper reports that GPT-4,
asked to reason about the spatial relationship of two axis-aligned boxes, could
not reliably tell whether they *overlap* (interpenetrate), merely *touch* (share
a contact face without protruding), or have a *gap* between them -- the authors
call this the "collision problem". The most vivid failure is the **floating
tabletop**: a tabletop and its legs that were *intended* to be in contact were
placed with a gap, so the top hovers above the legs. The paper found that the
critical design relation is "in contact with (but not protruding into)".

This module is a **deterministic auditor** for exactly that relation. A box is
the paper's ``box(x, y, z, w, h, d)``: size ``w x h x d`` centred at
``(x, y, z)``. Given two boxes it classifies their relationship as one of:

  * ``SEPARATED``   -- at least one axis has a positive gap (they do not touch);
  * ``TOUCHING``    -- coincident faces / edges / corners, no protrusion; the
    important sub-case is a real supporting **face contact** (positive overlap
    on exactly two axes, touching on the third);
  * ``OVERLAPPING`` -- positive overlap on all three axes: the boxes protrude
    into one another (interpenetration).

On top of the pairwise test sit two assembly-level auditors:

  * :func:`audit_should_touch` -- given parts that are *meant* to touch, flag any
    that instead float with a gap (the floating-tabletop case) or protrude;
  * :func:`scan_protrusions` -- scan every pair and report interpenetrations.

This is deliberately **not** ``verifiers/interference.py`` (which only does
broad+narrow-phase clash detection): this module is a tri-state contact/gap
*classifier* plus a "should-touch but has a gap" auditor. It is standalone and
does not import the interference module.

Pure stdlib; fully deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_TOL",
    "AXES",
    "Box",
    "IntervalRelation",
    "classify_interval",
    "BoxRelation",
    "classify_boxes",
    "PairAudit",
    "audit_should_touch",
    "Protrusion",
    "scan_protrusions",
]

DEFAULT_TOL = 1e-9

# Human-readable axis names, indexed 0/1/2.
AXES = ("x", "y", "z")


# ---------------------------------------------------------------------------
# Box representation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Box:
    """An axis-aligned box: size ``w x h x d`` centred at ``(cx, cy, cz)``.

    This is exactly the paper's ``box(x, y, z, w, h, d)``. Sizes must be
    strictly positive (a degenerate zero/negative extent is rejected).
    """

    cx: float
    cy: float
    cz: float
    w: float
    h: float
    d: float

    def __post_init__(self) -> None:
        for name, value in (("w", self.w), ("h", self.h), ("d", self.d)):
            if value <= 0.0:
                raise ValueError(
                    "box size %s must be strictly positive, got %r" % (name, value)
                )

    @property
    def size(self) -> Tuple[float, float, float]:
        return (self.w, self.h, self.d)

    @property
    def center(self) -> Tuple[float, float, float]:
        return (self.cx, self.cy, self.cz)

    def min_corner(self) -> Tuple[float, float, float]:
        return (
            self.cx - self.w / 2.0,
            self.cy - self.h / 2.0,
            self.cz - self.d / 2.0,
        )

    def max_corner(self) -> Tuple[float, float, float]:
        return (
            self.cx + self.w / 2.0,
            self.cy + self.h / 2.0,
            self.cz + self.d / 2.0,
        )

    def aabb(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return ``(min_corner, max_corner)`` -- the axis-aligned bounds."""
        return (self.min_corner(), self.max_corner())

    def interval(self, axis: int) -> Tuple[float, float]:
        """Return the ``(lo, hi)`` extent of this box along ``axis`` (0/1/2)."""
        lo = self.min_corner()[axis]
        hi = self.max_corner()[axis]
        return (lo, hi)


# ---------------------------------------------------------------------------
# Per-axis 1D interval relation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IntervalRelation:
    """The relationship between two 1D intervals along one axis.

    ``kind`` is one of ``"separated"`` / ``"touching"`` / ``"overlapping"``.

    * ``overlap`` -- length of the shared span (>0 only when overlapping).
    * ``gap``     -- positive distance between the intervals (>0 only when
      separated); the closest approach.
    """

    kind: str
    overlap: float
    gap: float


def classify_interval(
    a: Tuple[float, float],
    b: Tuple[float, float],
    tol: float = DEFAULT_TOL,
) -> IntervalRelation:
    """Classify two 1D intervals ``a = (alo, ahi)`` and ``b = (blo, bhi)``.

    * ``gap > tol``               -> ``separated`` (positive gap reported);
    * ``|gap| <= tol`` (edges     -> ``touching`` (coincident edges,
      coincide)                       overlap ~ 0);
    * ``overlap > tol``           -> ``overlapping`` (positive overlap).
    """
    alo, ahi = a
    blo, bhi = b
    if alo > ahi:
        raise ValueError("interval a is inverted: %r" % (a,))
    if blo > bhi:
        raise ValueError("interval b is inverted: %r" % (b,))

    # Length of the (possibly negative) signed overlap of the two intervals.
    lo = max(alo, blo)
    hi = min(ahi, bhi)
    signed = hi - lo  # >0 overlap, <0 gap, ~0 touching

    if signed > tol:
        return IntervalRelation(kind="overlapping", overlap=signed, gap=0.0)
    if signed < -tol:
        return IntervalRelation(kind="separated", overlap=0.0, gap=-signed)
    # Within tolerance of zero: edges coincide.
    return IntervalRelation(kind="touching", overlap=0.0, gap=0.0)


# ---------------------------------------------------------------------------
# 3D box relation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoxRelation:
    """The classified relationship between two boxes A and B.

    ``classification`` is one of ``"OVERLAPPING"`` / ``"TOUCHING"`` /
    ``"SEPARATED"``.

    Fields populated depend on the classification:

    * ``overlap_volume``  -- shared volume (0 unless OVERLAPPING).
    * ``contact_axis``    -- for a face/edge/corner TOUCHING, the axis (or axes)
      along which the boxes are coincident; for a single-axis face contact this
      is a length-1 tuple.
    * ``contact_area``    -- for a face contact, the product of the two
      overlapping extents (0 for edge/corner degenerate contact).
    * ``separation_gap``  -- for SEPARATED, the minimum positive gap.
    * ``separation_axis`` -- for SEPARATED, the axis of that minimum gap.
    * ``per_axis``        -- the three :class:`IntervalRelation` results.
    """

    classification: str
    per_axis: Tuple[IntervalRelation, IntervalRelation, IntervalRelation]
    overlap_volume: float = 0.0
    contact_axis: Tuple[int, ...] = field(default_factory=tuple)
    contact_area: float = 0.0
    separation_gap: float = 0.0
    separation_axis: Optional[int] = None

    # -- boolean helpers ----------------------------------------------------
    @property
    def is_overlapping(self) -> bool:
        """True when the boxes interpenetrate (protrude) on all three axes."""
        return self.classification == "OVERLAPPING"

    #: alias -- the paper's language is "protruding into".
    @property
    def is_protruding(self) -> bool:
        return self.is_overlapping

    @property
    def is_touching(self) -> bool:
        """True for any coincident contact (face, edge, or corner)."""
        return self.classification == "TOUCHING"

    @property
    def is_face_contact(self) -> bool:
        """True only for a real supporting/mating face.

        A face contact is touching on *exactly one* axis with positive overlap
        on the other two -- a 2D contact patch, not a 1D edge or 0D corner.
        """
        if self.classification != "TOUCHING":
            return False
        return len(self.contact_axis) == 1 and self.contact_area > 0.0

    @property
    def is_separated(self) -> bool:
        """True when a positive gap keeps the boxes apart on some axis."""
        return self.classification == "SEPARATED"

    @property
    def contact_axis_names(self) -> Tuple[str, ...]:
        return tuple(AXES[i] for i in self.contact_axis)

    @property
    def separation_axis_name(self) -> Optional[str]:
        if self.separation_axis is None:
            return None
        return AXES[self.separation_axis]


def classify_boxes(a: Box, b: Box, tol: float = DEFAULT_TOL) -> BoxRelation:
    """Classify the 3D relationship between boxes ``a`` and ``b``.

    Returns a :class:`BoxRelation`. The logic:

    * any axis SEPARATED (positive gap) -> ``SEPARATED`` (report the minimum
      gap and its axis);
    * else all three axes OVERLAPPING -> ``OVERLAPPING`` (report shared volume);
    * else (no gap, at least one axis merely touching) -> ``TOUCHING`` (report
      the touching axis/axes and, for a single-axis face contact, the contact
      face area).
    """
    per_axis = tuple(
        classify_interval(a.interval(i), b.interval(i), tol=tol) for i in range(3)
    )

    separated_axes = [i for i in range(3) if per_axis[i].kind == "separated"]
    if separated_axes:
        # Minimum positive gap across all separated axes: the closest approach.
        gap_axis = min(separated_axes, key=lambda i: per_axis[i].gap)
        return BoxRelation(
            classification="SEPARATED",
            per_axis=per_axis,
            separation_gap=per_axis[gap_axis].gap,
            separation_axis=gap_axis,
        )

    overlapping_axes = [i for i in range(3) if per_axis[i].kind == "overlapping"]
    touching_axes = [i for i in range(3) if per_axis[i].kind == "touching"]

    if len(overlapping_axes) == 3:
        vol = (
            per_axis[0].overlap * per_axis[1].overlap * per_axis[2].overlap
        )
        return BoxRelation(
            classification="OVERLAPPING",
            per_axis=per_axis,
            overlap_volume=vol,
        )

    # No gap and not all-overlapping -> a coincident (touching) contact.
    # The contact axes are the ones that are merely touching; the remaining
    # axes overlap. Face contact = exactly one touching axis (the other two
    # overlap positively, giving a 2D patch). Edge = two touching axes,
    # corner = three touching axes -> zero contact area.
    if len(touching_axes) == 1:
        others = [i for i in range(3) if i != touching_axes[0]]
        area = per_axis[others[0]].overlap * per_axis[others[1]].overlap
    else:
        area = 0.0

    return BoxRelation(
        classification="TOUCHING",
        per_axis=per_axis,
        contact_axis=tuple(touching_axes),
        contact_area=area,
    )


# ---------------------------------------------------------------------------
# Assembly-level auditors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PairAudit:
    """Result of auditing one "should touch" pair of named parts.

    ``status`` is one of:

    * ``"face_contact"`` -- OK, the parts share a supporting face;
    * ``"contact"``      -- OK but only an edge/corner touch (no face patch);
    * ``"floating"``     -- ERROR, the parts float apart with a gap (the paper's
      floating-tabletop failure); ``gap`` and ``gap_axis`` describe it;
    * ``"protruding"``   -- ERROR, the parts interpenetrate; ``overlap_volume``
      is reported.
    """

    name_a: str
    name_b: str
    status: str
    relation: BoxRelation
    gap: float = 0.0
    gap_axis: Optional[str] = None
    overlap_volume: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in ("face_contact", "contact")

    @property
    def is_floating(self) -> bool:
        return self.status == "floating"

    @property
    def is_protruding(self) -> bool:
        return self.status == "protruding"


def audit_should_touch(
    boxes: Sequence[Tuple[str, Box]],
    should_touch: Sequence[Tuple[str, str]],
    tol: float = DEFAULT_TOL,
) -> List[PairAudit]:
    """Audit part pairs that are *intended* to be in contact.

    ``boxes`` is a sequence of ``(name, Box)``; ``should_touch`` is a sequence
    of ``(name_a, name_b)`` pairs that the design says should mate. For each
    pair the returned :class:`PairAudit` reports face contact (OK), a floating
    gap (the tabletop failure), or a protrusion (interpenetration error).
    """
    lookup = dict(boxes)
    if len(lookup) != len(boxes):
        raise ValueError("duplicate box name in 'boxes'")

    audits: List[PairAudit] = []
    for name_a, name_b in should_touch:
        if name_a not in lookup:
            raise ValueError("unknown box name %r" % (name_a,))
        if name_b not in lookup:
            raise ValueError("unknown box name %r" % (name_b,))

        rel = classify_boxes(lookup[name_a], lookup[name_b], tol=tol)
        if rel.is_separated:
            audits.append(
                PairAudit(
                    name_a=name_a,
                    name_b=name_b,
                    status="floating",
                    relation=rel,
                    gap=rel.separation_gap,
                    gap_axis=rel.separation_axis_name,
                )
            )
        elif rel.is_overlapping:
            audits.append(
                PairAudit(
                    name_a=name_a,
                    name_b=name_b,
                    status="protruding",
                    relation=rel,
                    overlap_volume=rel.overlap_volume,
                )
            )
        elif rel.is_face_contact:
            audits.append(
                PairAudit(
                    name_a=name_a,
                    name_b=name_b,
                    status="face_contact",
                    relation=rel,
                )
            )
        else:
            # Touching, but only an edge or corner -- no supporting patch.
            audits.append(
                PairAudit(
                    name_a=name_a,
                    name_b=name_b,
                    status="contact",
                    relation=rel,
                )
            )
    return audits


@dataclass(frozen=True)
class Protrusion:
    """An interpenetration between two named parts (an all-pairs violation)."""

    name_a: str
    name_b: str
    overlap_volume: float
    relation: BoxRelation


def scan_protrusions(
    boxes: Sequence[Tuple[str, Box]],
    tol: float = DEFAULT_TOL,
) -> List[Protrusion]:
    """Scan *all* unordered pairs and return every interpenetration.

    Results are ordered by descending overlap volume (worst first), with ties
    broken by name so the output is fully deterministic.
    """
    items = list(boxes)
    violations: List[Protrusion] = []
    for i in range(len(items)):
        name_a, box_a = items[i]
        for j in range(i + 1, len(items)):
            name_b, box_b = items[j]
            rel = classify_boxes(box_a, box_b, tol=tol)
            if rel.is_overlapping:
                violations.append(
                    Protrusion(
                        name_a=name_a,
                        name_b=name_b,
                        overlap_volume=rel.overlap_volume,
                        relation=rel,
                    )
                )
    violations.sort(key=lambda p: (-p.overlap_volume, p.name_a, p.name_b))
    return violations
