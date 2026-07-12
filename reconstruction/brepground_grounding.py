"""Text-based B-Rep primitive grounding (FutureCAD / BRepGround).

Li et al., "Towards High-Fidelity CAD Generation via LLM-Driven Program
Generation and Text-Based B-Rep Primitive Grounding" (FutureCAD, 2026),
Sec. 3-4. In FutureCAD an LLM emits an executable CadQuery program in which,
whenever a feature (``fillet``/``chamfer``/``shell`` ...) needs an operand set
``pi_i``, the program embeds a *natural-language query* ``q_i`` -- for example
``cq.Query("All circular holes on the surface.")``. During execution BRepGround
takes the transient B-Rep ``B_{i-1}`` and grounds ``q_i`` to a subset of the
available primitives ``pi_i subseteq P(B_{i-1})`` (Eq. 5).

BRepGround itself is a trained BERT + UV-Net + cross-attention transformer; the
network and the LLM are external and out of scope. This module implements the
DETERMINISTIC counterpart the paper's task is defined against: given the set of
B-Rep primitives (faces and edges) available in a transient B-Rep, and a textual
reference such as "the top face", "the largest hole" or "all circular edges",
resolve the reference to specific primitives purely from their *geometric
properties* (type, sub-type, size, position, orientation).

This is a genuinely different grounding scheme from the neighbours in the repo:

  * ``reconstruction.pointercad_pointer`` resolves *index pointers* -- a numeric
    reference into an ordered primitive list (paper 145). Here the reference is
    free text describing geometry, not an index.
  * ``reconstruction.querycad_answer_engine`` grounds a *question* to segmented
    parts to compute an answer (paper 148). Here we ground a *selection query*
    to raw B-Rep faces/edges to produce an operand set for a CAD feature.

The grounder parses a query into (a) hard *type/sub-type/hole* predicates that
filter candidates and (b) soft *position / superlative-size* cues that rank the
survivors. Ranking is fully deterministic with an index tie-break, so the same
query over the same B-Rep always yields the same ordered result.

Pure, deterministic, stdlib-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]

# --------------------------------------------------------------------------- #
# Primitive representation
# --------------------------------------------------------------------------- #

# Face surface sub-types and edge curve sub-types we understand. Sub-types are
# grouped so a textual cue like "round"/"circular" matches every rounded form.
_FACE_SUBTYPES = frozenset(
    {"planar", "cylindrical", "conical", "spherical", "toroidal", "bspline"}
)
_EDGE_SUBTYPES = frozenset({"line", "circle", "arc", "ellipse", "bspline"})


@dataclass(frozen=True)
class BRepPrimitive:
    """A single B-Rep primitive available for grounding.

    ``index``    stable position in ``P(B)`` (used as the deterministic
                 tie-break and as the identifier a grounded operand refers to).
    ``kind``     ``"face"`` or ``"edge"``.
    ``subtype``  surface type for faces / curve type for edges (see the frozen
                 sets above); free-form strings are allowed but only the known
                 ones participate in sub-type cues.
    ``centroid`` geometric centre in world coordinates.
    ``size``     surface area for faces, curve length for edges (>= 0).
    ``normal``   outward face normal, or the edge tangent direction; optional.
    ``is_hole``  True for an inner cylindrical face or a hole-bounding circular
                 edge (an interior loop), i.e. material removed rather than the
                 outer wall.
    """

    index: int
    kind: str
    subtype: str = ""
    centroid: Vec3 = (0.0, 0.0, 0.0)
    size: float = 0.0
    normal: Optional[Vec3] = None
    is_hole: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("face", "edge"):
            raise ValueError("kind must be 'face' or 'edge', got %r" % (self.kind,))
        if self.size < 0:
            raise ValueError("size must be non-negative")


# --------------------------------------------------------------------------- #
# Query parsing
# --------------------------------------------------------------------------- #

# axis index (0=x,1=y,2=z) and sign: +1 selects the maximum coordinate, -1 the
# minimum. "front"/"back" follow the common CAD convention front = -Y.
_POSITION_CUES: Dict[str, Tuple[int, int]] = {
    "top": (2, +1),
    "upper": (2, +1),
    "highest": (2, +1),
    "bottom": (2, -1),
    "lower": (2, -1),
    "lowest": (2, -1),
    "right": (0, +1),
    "rightmost": (0, +1),
    "left": (0, -1),
    "leftmost": (0, -1),
    "back": (1, +1),
    "rear": (1, +1),
    "front": (1, -1),
}

# +1 == "largest first", -1 == "smallest first".
_SIZE_CUES: Dict[str, int] = {
    "largest": +1,
    "biggest": +1,
    "widest": +1,
    "longest": +1,
    "greatest": +1,
    "maximum": +1,
    "smallest": -1,
    "tiniest": -1,
    "shortest": -1,
    "narrowest": -1,
    "minimum": -1,
}

# sub-type keyword -> canonical sub-type set it matches.
_SUBTYPE_CUES: Dict[str, frozenset] = {
    "planar": frozenset({"planar"}),
    "flat": frozenset({"planar"}),
    "cylindrical": frozenset({"cylindrical"}),
    "circular": frozenset({"cylindrical", "circle", "arc"}),
    "round": frozenset({"cylindrical", "circle", "arc", "spherical"}),
    "conical": frozenset({"conical"}),
    "tapered": frozenset({"conical"}),
    "spherical": frozenset({"spherical"}),
    "linear": frozenset({"line"}),
    "straight": frozenset({"line"}),
}

_FACE_WORDS = frozenset({"face", "faces", "surface", "surfaces", "wall", "walls"})
_EDGE_WORDS = frozenset({"edge", "edges", "corner", "corners", "rim", "rims"})
_HOLE_WORDS = frozenset({"hole", "holes", "bore", "bores"})
_ALL_WORDS = frozenset({"all", "every", "each", "the"})

_WORD_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class ParsedQuery:
    """Structured form of a textual reference.

    ``kind``          "face", "edge" or None (no explicit type constraint).
    ``subtypes``      allowed sub-type set, or None (no constraint).
    ``require_hole``  True when the query says "hole"/"bore".
    ``size_dir``      +1/-1 superlative-size ranking, or 0.
    ``position``      (axis, sign) ranking cue, or None.
    ``wants_all``     True for plural / "all" / "every" references.
    """

    kind: Optional[str] = None
    subtypes: Optional[frozenset] = None
    require_hole: bool = False
    size_dir: int = 0
    position: Optional[Tuple[int, int]] = None
    wants_all: bool = False
    tokens: Tuple[str, ...] = field(default_factory=tuple)


def parse_query(text: str) -> ParsedQuery:
    """Parse a textual reference into a :class:`ParsedQuery`.

    Parsing is order-independent and case-insensitive; unknown words are
    ignored. When several cues of the same family appear the first-seen one
    wins (deterministic left-to-right scan).
    """
    tokens = tuple(_WORD_RE.findall(text.lower()))
    kind: Optional[str] = None
    subtypes: Optional[frozenset] = None
    require_hole = False
    size_dir = 0
    position: Optional[Tuple[int, int]] = None
    wants_all = False

    for tok in tokens:
        if tok in _ALL_WORDS and tok != "the":
            wants_all = True
        if tok in _HOLE_WORDS:
            require_hole = True
            if kind is None:
                kind = "face"  # a hole is grounded on its cylindrical face
            if tok.endswith("s"):
                wants_all = True
        if kind is None and tok in _FACE_WORDS:
            kind = "face"
        if kind is None and tok in _EDGE_WORDS:
            kind = "edge"
        if tok in _FACE_WORDS and tok.endswith("s"):
            wants_all = True
        if tok in _EDGE_WORDS and tok.endswith("s"):
            wants_all = True
        if subtypes is None and tok in _SUBTYPE_CUES:
            subtypes = _SUBTYPE_CUES[tok]
        if size_dir == 0 and tok in _SIZE_CUES:
            size_dir = _SIZE_CUES[tok]
        if position is None and tok in _POSITION_CUES:
            position = _POSITION_CUES[tok]

    return ParsedQuery(
        kind=kind,
        subtypes=subtypes,
        require_hole=require_hole,
        size_dir=size_dir,
        position=position,
        wants_all=wants_all,
        tokens=tokens,
    )


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #


def _matches_hard(prim: BRepPrimitive, q: ParsedQuery) -> bool:
    """Whether a primitive satisfies the hard (filter) predicates."""
    if q.kind is not None and prim.kind != q.kind:
        return False
    if q.require_hole and not prim.is_hole:
        return False
    if q.subtypes is not None and prim.subtype not in q.subtypes:
        return False
    return True


def _rank_key(prim: BRepPrimitive, q: ParsedQuery) -> Tuple:
    """Deterministic sort key; smaller sorts first (i.e. ranked higher).

    Priority: superlative size, then position extreme, then primitive index.
    """
    key: List[float] = []
    if q.size_dir != 0:
        # +1 -> largest first -> negate so larger size sorts first.
        key.append(-q.size_dir * prim.size)
    if q.position is not None:
        axis, sign = q.position
        coord = prim.centroid[axis]
        # sign +1 -> maximum coord first -> negate.
        key.append(-sign * coord)
    key.append(float(prim.index))
    return tuple(key)


def ground(
    text: str, primitives: Sequence[BRepPrimitive]
) -> List[BRepPrimitive]:
    """Ground ``text`` against ``primitives`` and return the ranked matches.

    All primitives satisfying the hard predicates are returned, ordered best
    first by :func:`_rank_key`. The list is empty when nothing matches. This is
    the general operation; :func:`ground_one` and :func:`ground_all` wrap it for
    the singular / plural cases.
    """
    q = parse_query(text)
    candidates = [p for p in primitives if _matches_hard(p, q)]
    candidates.sort(key=lambda p: _rank_key(p, q))
    return candidates


def ground_one(
    text: str, primitives: Sequence[BRepPrimitive]
) -> Optional[BRepPrimitive]:
    """Ground a singular reference; return the single best primitive or None."""
    ranked = ground(text, primitives)
    return ranked[0] if ranked else None


def ground_all(
    text: str, primitives: Sequence[BRepPrimitive]
) -> List[BRepPrimitive]:
    """Resolve a reference to an operand set ``pi_i`` (Eq. 5).

    Returns every hard-matching primitive when the query is plural/"all"; for a
    singular query with a ranking cue it returns just the top primitive, and for
    a singular query without any ranking cue it returns all matches (the caller
    is asking for "the face" but there may be several -- do not silently drop).
    """
    q = parse_query(text)
    ranked = ground(text, primitives)
    if not ranked:
        return []
    if q.wants_all:
        return ranked
    if q.size_dir != 0 or q.position is not None:
        return ranked[:1]
    return ranked


def index_set(primitives: Sequence[BRepPrimitive]) -> Tuple[int, ...]:
    """Return the sorted tuple of primitive indices (the grounded id set)."""
    return tuple(sorted(p.index for p in primitives))
