"""The brief type, and the two invariants a contaminated corpus cannot satisfy.

INVARIANT 1 -- PROVENANCE IS DECLARED AND IT IS NOT US.
:class:`Brief` carries a :class:`Source` naming where its numbers came from. The
enum has no member for "the harness said so", and there is deliberately no way to
write one: a brief whose expected volume was READ OFF a run of our own backend is
not a benchmark, it is a regression fixture, and it will defend whatever bug was
live the day it was recorded.

INVARIANT 2 -- THE ENVELOPE IS ALWAYS STATED.
``bbox`` is REQUIRED and validated. ``eval/pressure/briefs.py`` let it default to
``None``, and that single default is why a shell which dilated a 60x40x20 box
into a 63x43x23 box passed the benchmark: the benchmark never asked how big the
part was supposed to be. A brief that cannot say what shape the finished part
occupies cannot catch an envelope bug, so this class refuses to exist without it.

Tolerances are NOT carried per brief. They are derived from the engine that is
measuring (:mod:`harnesscad.eval.selftest.probe`, where every tolerance is a
physical consequence of what the engine IS -- an exact B-rep kernel, a
polygonising mesher, a sampled field on a grid of a known cell size). A per-brief
tolerance is a knob, and a knob is how a corpus gets tuned until it agrees with
the code it is supposed to be judging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op, canonical_json

__all__ = ["Source", "Split", "Brief", "Vec3"]

Vec3 = Tuple[float, float, float]


class Source:
    """Where a brief's ground truth came from. None of these is this repository."""

    #: Arithmetic. 60 * 40 * 10 = 24000, and no code here gets a vote.
    ANALYTIC = "analytic"
    #: A published engineering standard (ISO / DIN). A document we cannot edit.
    STANDARD = "standard"
    #: A relation between two runs of the same engine; needs no ground truth.
    METAMORPHIC = "metamorphic"
    #: Independent engines agreeing. Evidence, not proof -- see consensus.py.
    DIFFERENTIAL = "differential"

    ALL: Tuple[str, ...] = (ANALYTIC, STANDARD, METAMORPHIC, DIFFERENTIAL)


class Split:
    """DEV may be looked at. HELDOUT may only be scored. See ``score.py``."""

    DEV = "dev"
    HELDOUT = "heldout"
    ALL: Tuple[str, ...] = (DEV, HELDOUT)


@dataclass(frozen=True)
class Brief:
    """One part, its prompt, its reference op stream, and its independent truth.

    ``text``       the ONLY thing a model is ever shown.
    ``reference``  a hand-written op stream that satisfies the brief. It is the
                   SHAPE TARGET (see ``shape.py``) and it is never shown.
    ``volume``     the EXACT volume in mm3, in closed form. Not a band. The band
                   is supplied by the measuring engine's own physical tolerance.
    ``bbox``       the EXACT (dx, dy, dz) envelope. REQUIRED -- see the module
                   docstring; this is the check the pressure corpus did not have.
    ``genus``      exact topology when it is known in closed form (N through
                   holes => genus N). ``None`` means "not scored on genus".
    ``inside`` /   probe points, in world coordinates, derived by ARITHMETIC from
    ``outside``    the part's dimensions -- the middle of a wall, the axis of a
                   hole. They are what proves a feature landed where it was asked
                   for; volume and bbox cannot.
    """

    id: str
    split: str
    source: str
    citation: str
    text: str
    reference: Tuple[Op, ...]
    volume: float
    bbox: Vec3
    genus: Optional[int] = None
    inside: Tuple[Vec3, ...] = ()
    outside: Tuple[Vec3, ...] = ()
    note: str = ""
    #: The THINNEST FEATURE of the finished part -- a shell's wall, a tube's
    #: annulus -- and NOT the smallest bbox extent. For a solid plate they are the
    #: same thing; for a 80x60x25 box shelled to 2.5 mm they differ by a factor of
    #: ten, and the difference decides whether a SAMPLED engine can measure the
    #: part at all (see ``grade.resolvable``). ``None`` => min(bbox).
    min_feature: Optional[float] = None

    def __post_init__(self) -> None:
        if self.source not in Source.ALL:
            raise ValueError(
                "brief %r declares source %r; a brief must derive its ground "
                "truth from one of %s -- there is no member for 'the harness "
                "measured it', on purpose" % (self.id, self.source, Source.ALL))
        if self.split not in Split.ALL:
            raise ValueError("brief %r: unknown split %r" % (self.id, self.split))
        if not self.citation.strip():
            raise ValueError(
                "brief %r cites nothing. An unsourced number is the harness's "
                "opinion wearing a benchmark's clothes." % self.id)
        if self.bbox is None or len(self.bbox) != 3:
            raise ValueError(
                "brief %r has no expected bbox. THIS IS THE BUG THAT LET A SHELL "
                "DILATE A PART BY 3 mm ON EVERY FACE AND STILL SCORE A PASS "
                "(assets/pressure/report.md:92). A brief that does not state the "
                "envelope cannot catch an envelope bug." % self.id)
        if any(float(v) <= 0.0 for v in self.bbox):
            raise ValueError("brief %r: every bbox extent must be positive" % self.id)
        if not (self.volume > 0.0):
            raise ValueError("brief %r: volume must be positive" % self.id)
        if not self.reference:
            raise ValueError("brief %r carries no reference op stream" % self.id)

    @property
    def extent(self) -> float:
        return max(self.bbox)

    @property
    def min_extent(self) -> float:
        return min(self.bbox)

    @property
    def feature(self) -> float:
        """The thinnest thing in the part. What a sampled engine has to resolve."""
        if self.min_feature is not None and self.min_feature > 0.0:
            return float(self.min_feature)
        return min(self.bbox)

    def ops_json(self) -> Tuple[str, ...]:
        """The reference stream, canonically, so a finding is replayable by hand."""
        return tuple(canonical_json(o) for o in self.reference)

    def to_dict(self) -> dict:
        return {"id": self.id, "split": self.split, "source": self.source,
                "citation": self.citation, "text": self.text,
                "volume": self.volume, "bbox": list(self.bbox),
                "genus": self.genus,
                "inside": [list(p) for p in self.inside],
                "outside": [list(p) for p in self.outside],
                "note": self.note,
                "reference": [o.to_dict() for o in self.reference]}


def check_unique(briefs: Sequence[Brief]) -> None:
    """Two briefs with one id would silently overwrite each other in a report."""
    seen = {}
    for b in briefs:
        if b.id in seen:
            raise ValueError("duplicate brief id %r" % b.id)
        seen[b.id] = b
