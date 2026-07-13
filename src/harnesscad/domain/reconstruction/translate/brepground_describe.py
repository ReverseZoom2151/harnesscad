"""B-Rep primitive text-description generator (FutureCAD / BRepGround).

Li et al., "Towards High-Fidelity CAD Generation via LLM-Driven Program
Generation and Text-Based B-Rep Primitive Grounding" (FutureCAD, 2026),
Sec. 5.1 "Textual description generation". Training BRepGround requires, for
each feature with non-empty operands, a *textual description that refers to the
target primitives* (the paper obtains these with Claude-4.5; the LLM step is
external). This module implements the inverse of
:mod:`reconstruction.brepground_grounding`: it turns a
:class:`reconstruction.brepground_grounding.BRepPrimitive` into a short natural
language phrase describing its *type, size, position and orientation* -- exactly
the geometric cues the grounder consumes.

The generator is deterministic and template-based (no LLM). It is designed to
round-trip with the grounder: for a primitive that is uniquely distinguishable
within its B-Rep, ``ground_one(describe(prim, brep), brep)`` returns that same
primitive (see the discriminative-phrase helper). This closes the loop that the
paper trains a network to approximate, and gives a reference oracle for the
grounding-accuracy metric.

Pure, deterministic, stdlib-only.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from harnesscad.domain.reconstruction.translate.brepground_grounding import (
    BRepPrimitive,
    Vec3,
    ground_one,
)

# Human-readable names for sub-types.
_FACE_NAMES = {
    "planar": "planar",
    "cylindrical": "cylindrical",
    "conical": "conical",
    "spherical": "spherical",
    "toroidal": "toroidal",
    "bspline": "freeform",
}
_EDGE_NAMES = {
    "line": "straight",
    "circle": "circular",
    "arc": "arc",
    "ellipse": "elliptical",
    "bspline": "freeform",
}

# Position words keyed by (axis, sign). Mirrors _POSITION_CUES in the grounder
# so descriptions are grounder-parseable.
_AXIS_WORDS = {
    (2, +1): "top",
    (2, -1): "bottom",
    (0, +1): "right",
    (0, -1): "left",
    (1, +1): "back",
    (1, -1): "front",
}

_SIZE_WORD = {+1: "largest", -1: "smallest"}


def _type_phrase(prim: BRepPrimitive) -> str:
    if prim.kind == "face":
        name = _FACE_NAMES.get(prim.subtype, prim.subtype or "")
        noun = "hole" if prim.is_hole else "face"
        return (name + " " + noun).strip()
    name = _EDGE_NAMES.get(prim.subtype, prim.subtype or "")
    return (name + " edge").strip()


def _extreme_axis(
    prim: BRepPrimitive, siblings: Sequence[BRepPrimitive]
) -> Optional[str]:
    """Return a position word if ``prim`` is the unique extreme along some axis.

    Checks z, x, y (both directions). The first axis on which the primitive is
    the strict, unique min or max among ``siblings`` yields the word.
    """
    for axis, sign in ((2, +1), (2, -1), (0, +1), (0, -1), (1, +1), (1, -1)):
        coord = prim.centroid[axis]
        others = [s.centroid[axis] for s in siblings if s.index != prim.index]
        if not others:
            continue
        if sign > 0 and coord > max(others):
            return _AXIS_WORDS[(axis, sign)]
        if sign < 0 and coord < min(others):
            return _AXIS_WORDS[(axis, sign)]
    return None


def _size_word(
    prim: BRepPrimitive, siblings: Sequence[BRepPrimitive]
) -> Optional[str]:
    """Return "largest"/"smallest" if ``prim`` is the unique size extreme."""
    others = [s.size for s in siblings if s.index != prim.index]
    if not others:
        return None
    if prim.size > max(others):
        return _SIZE_WORD[+1]
    if prim.size < min(others):
        return _SIZE_WORD[-1]
    return None


def describe(
    prim: BRepPrimitive, brep: Optional[Sequence[BRepPrimitive]] = None
) -> str:
    """Return a text description of ``prim``.

    Without ``brep`` the phrase is purely intrinsic (type + size + position of
    the centroid). With ``brep`` the description adds a discriminating cue
    (extreme position or extreme size relative to same-kind siblings) so the
    phrase points at this primitive specifically.
    """
    type_phrase = _type_phrase(prim)
    modifier = ""
    if brep is not None:
        siblings = [
            p for p in brep if p.kind == prim.kind and p.is_hole == prim.is_hole
        ]
        pos = _extreme_axis(prim, siblings)
        if pos is not None:
            modifier = pos
        else:
            sz = _size_word(prim, siblings)
            if sz is not None:
                modifier = sz
    phrase = ("the " + (modifier + " " if modifier else "") + type_phrase).strip()
    return phrase


def describe_detailed(prim: BRepPrimitive) -> str:
    """A richer, non-round-tripping sentence with numeric size and position.

    Mirrors the paper's "detailed description" style ("... a span of ~40 units
    ... extruded upwards by 15 units"): it states the measured size and the
    centroid coordinates. Numbers are rounded to keep the string deterministic.
    """
    x, y, z = prim.centroid
    measure = "area" if prim.kind == "face" else "length"
    return (
        _article(prim)
        + " with "
        + measure
        + " "
        + _fmt(prim.size)
        + " at ("
        + _fmt(x)
        + ", "
        + _fmt(y)
        + ", "
        + _fmt(z)
        + ")"
    )


def _article(prim: BRepPrimitive) -> str:
    tp = _type_phrase(prim)
    return ("an " if tp[:1] in "aeiou" else "a ") + tp


def _fmt(value: float) -> str:
    r = round(float(value), 3)
    if r == int(r):
        return str(int(r))
    return ("%.3f" % r).rstrip("0").rstrip(".")


def round_trips(
    prim: BRepPrimitive, brep: Sequence[BRepPrimitive]
) -> bool:
    """Whether ``describe(prim, brep)`` grounds back to ``prim`` uniquely."""
    phrase = describe(prim, brep)
    got = ground_one(phrase, brep)
    return got is not None and got.index == prim.index


def describe_all(
    brep: Sequence[BRepPrimitive],
) -> List[str]:
    """Describe every primitive in ``brep`` (order preserved)."""
    return [describe(p, brep) for p in brep]
