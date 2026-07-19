"""Multimodal refinement schema for sketch refinement (McCarthy, Vaduguru et al., 2025).

From *Multimodal Refinement of Computer-aided Designs* , Sec. 2.1-2.2.
This module encodes the deterministic grammar as immutable data:

State grammar::

    D      -> {Curve, ...}
    Curve  -> Line | Circle | Arc
    Line   -> l(P, P)        // end points
    Circle -> c(P, P)        // points on diameter
    Arc    -> a(P, P, P)     // start, mid, end

Action grammar / typed edit vocabulary::

    Action -> make_curve Curve | remove_curve Curve | move_curve Curve Vxy
            | move_point P P   | delete_point P

Message grammar::

    Message -> <Text, Drawing>
    Text    -> [char, ...] | empty
    Drawing -> [stroke, ...] | empty   (strokes are SVG-style polylines)

Only the deterministic representation is implemented here; the transition
function ``A(D)`` lives in :mod:`editing.mrcad_refinement`, and the distance
metrics in :mod:`bench.mrcad_metrics`. Pure stdlib, no learned model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Point = Tuple[float, float]

# Curve arity per the grammar.
_ARITY = {"line": 2, "circle": 2, "arc": 3}


@dataclass(frozen=True)
class Curve:
    """A CAD curve defined by its control points.

    ``kind`` is ``"line"`` (2 endpoints), ``"circle"`` (2 diameter points), or
    ``"arc"`` (start, mid, end). Points are stored as a canonical tuple so that
    curves are hashable and comparable by value.
    """

    kind: str
    points: Tuple[Point, ...]

    def __post_init__(self) -> None:
        if self.kind not in _ARITY:
            raise ValueError(f"unknown curve kind: {self.kind!r}")
        want = _ARITY[self.kind]
        pts = tuple((float(x), float(y)) for x, y in self.points)
        if len(pts) != want:
            raise ValueError(f"{self.kind} needs {want} control points, got {len(pts)}")
        object.__setattr__(self, "points", pts)

    def has_point(self, p: Point) -> bool:
        q = (float(p[0]), float(p[1]))
        return q in self.points

    def translate(self, vector: Point) -> "Curve":
        dx, dy = float(vector[0]), float(vector[1])
        return Curve(self.kind, tuple((x + dx, y + dy) for x, y in self.points))

    def replace_point(self, old: Point, new: Point) -> "Curve":
        o = (float(old[0]), float(old[1]))
        n = (float(new[0]), float(new[1]))
        if o not in self.points:
            return self
        return Curve(self.kind, tuple(n if q == o else q for q in self.points))


def line(a: Point, b: Point) -> Curve:
    return Curve("line", (a, b))


def circle(a: Point, b: Point) -> Curve:
    return Curve("circle", (a, b))


def arc(a: Point, m: Point, b: Point) -> Curve:
    return Curve("arc", (a, m, b))


@dataclass(frozen=True)
class Design:
    """A CAD state ``D -> {Curve, ...}`` as an ordered, de-duplicated tuple.

    Immutable: mutating helpers return a fresh :class:`Design`. Order is kept
    for determinism, but equality ignores order (set semantics of the grammar).
    """

    curves: Tuple[Curve, ...] = ()

    def __post_init__(self) -> None:
        seen: list[Curve] = []
        for c in self.curves:
            if c not in seen:
                seen.append(c)
        object.__setattr__(self, "curves", tuple(seen))

    @staticmethod
    def empty() -> "Design":
        return Design(())

    def __len__(self) -> int:
        return len(self.curves)

    def __iter__(self):
        return iter(self.curves)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Design):
            return NotImplemented
        return frozenset(self.curves) == frozenset(other.curves)

    def __hash__(self) -> int:
        return hash(frozenset(self.curves))

    def add(self, curve: Curve) -> "Design":
        return Design(self.curves + (curve,))

    def remove(self, curve: Curve) -> "Design":
        return Design(tuple(c for c in self.curves if c != curve))

    def points(self) -> Tuple[Point, ...]:
        out: list[Point] = []
        for c in self.curves:
            for p in c.points:
                if p not in out:
                    out.append(p)
        return tuple(out)


# ---------------------------------------------------------------------------
# Typed edit-operation vocabulary.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MakeCurve:
    curve: Curve
    op: str = "make_curve"


@dataclass(frozen=True)
class RemoveCurve:
    curve: Curve
    op: str = "remove_curve"


@dataclass(frozen=True)
class MoveCurve:
    curve: Curve
    vector: Point
    op: str = "move_curve"


@dataclass(frozen=True)
class MovePoint:
    old: Point
    new: Point
    op: str = "move_point"


@dataclass(frozen=True)
class DeletePoint:
    point: Point
    op: str = "delete_point"


#: The complete typed edit vocabulary.
EDIT_VOCABULARY = ("make_curve", "remove_curve", "move_curve", "move_point", "delete_point")


# ---------------------------------------------------------------------------
# Message: <Text, Drawing>.
# ---------------------------------------------------------------------------
Stroke = Tuple[Point, ...]


@dataclass(frozen=True)
class Message:
    """A multimodal designer instruction ``<Text, Drawing>``.

    ``text`` is a possibly-empty string; ``strokes`` is a possibly-empty tuple
    of polyline strokes (each a tuple of points), the deterministic stand-in for
    the SVG drawing.
    """

    text: str = ""
    strokes: Tuple[Stroke, ...] = ()

    def __post_init__(self) -> None:
        norm = tuple(
            tuple((float(x), float(y)) for x, y in s) for s in self.strokes
        )
        object.__setattr__(self, "strokes", norm)

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    @property
    def has_drawing(self) -> bool:
        return any(len(s) > 0 for s in self.strokes)

    def modality(self) -> str:
        """Classify as ``multimodal`` / ``text`` / ``drawing`` / ``empty``.

        Mirrors the modality tallies in Sec. 5.2.
        """
        t, d = self.has_text, self.has_drawing
        if t and d:
            return "multimodal"
        if t:
            return "text"
        if d:
            return "drawing"
        return "empty"

    def stroke_count(self) -> int:
        """Number of non-empty strokes."""
        return sum(1 for s in self.strokes if len(s) > 0)

    def ink(self) -> float:
        """Total drawn path length -- the amount of digital 'ink'."""
        total = 0.0
        for s in self.strokes:
            for (x0, y0), (x1, y1) in zip(s, s[1:]):
                total += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return total


# ---------------------------------------------------------------------------
# Deterministic multimodal-instruction parser / representation.
# ---------------------------------------------------------------------------
# Small closed-class verb set used to detect imperative refinement directives
# (Sec. 5.2, Fig. 6B: refinement text expresses actions/directives, mostly
# imperative verbs). This is a deterministic lexical heuristic, not a parser
# model; this approach used Stanza, which we deliberately do not depend on.
_IMPERATIVE_VERBS = frozenset({
    "make", "move", "remove", "delete", "draw", "add", "connect", "extend",
    "shrink", "enlarge", "rotate", "shift", "straighten", "curve", "close",
    "open", "erase", "place", "put", "resize", "align", "fix", "change",
})


@dataclass(frozen=True)
class Instruction:
    """A parsed representation of a :class:`Message`.

    Captures the deterministic features this approach studies: modality, tokenised
    text, the leading (root) word, whether it is an imperative verb, stroke
    count, and ink. ``is_refinement_like`` flags directive text -- the
    finding that refinement instructions head with imperative verbs whereas
    generation instructions usually do not.
    """

    modality: str
    tokens: Tuple[str, ...]
    root_word: str
    is_imperative: bool
    stroke_count: int
    ink: float

    @property
    def is_refinement_like(self) -> bool:
        return self.is_imperative


def tokenize(text: str) -> Tuple[str, ...]:
    """Lower-case alphanumeric word tokeniser (deterministic, stdlib)."""
    out: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return tuple(out)


def parse_instruction(message: Message) -> Instruction:
    """Parse a :class:`Message` into an :class:`Instruction` representation."""
    tokens = tokenize(message.text)
    root = tokens[0] if tokens else ""
    imperative = root in _IMPERATIVE_VERBS
    return Instruction(
        modality=message.modality(),
        tokens=tokens,
        root_word=root,
        is_imperative=imperative,
        stroke_count=message.stroke_count(),
        ink=message.ink(),
    )
