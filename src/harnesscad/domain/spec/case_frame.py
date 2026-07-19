"""Case-frame parser for natural-language CAD commands.

A deterministic parser built on *case-frames*: a verb is taken to
have a set of *semantic cases* (objective, locative, temporal, ...) each of
which may be filled by a *nominal* (a noun phrase or a prepositional phrase).
A nominal is admitted into a case only if it satisfies the case's constraint
(here expressed as declarative feature requirements) and matches the case's
preposition, if any.

This module supplies:

* a small CAD lexicon of nominals classified by *feature* (``shape``,
  ``quantity``, ``dimension-name``, ``location``);
* a verb lexicon mapping surface verbs (``draw``/``create``/``move``/...) to a
  canonical CAD ``action`` plus an ordered set of :class:`CaseSlot`s;
* :func:`parse_command`, which fills the frame of the sentence's verb, yielding
  a structured :class:`ParsedCommand` (e.g. ``draw a circle of radius 5 at the
  origin`` -> action=create, object=circle, dimensions={radius:5.0},
  location='origin').

Everything is pure/stdlib, so parsing is fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Nominal lexicon (surface word -> canonical value + feature)
# --------------------------------------------------------------------------- #
_SHAPE_NOUNS: Dict[str, str] = {
    "circle": "circle", "rectangle": "rectangle", "rect": "rectangle",
    "square": "square", "line": "line", "point": "point", "arc": "arc",
    "box": "box", "cube": "cube", "cuboid": "box", "block": "box",
    "cylinder": "cylinder", "sphere": "sphere", "cone": "cone",
    "hole": "hole", "slot": "slot", "fillet": "fillet", "chamfer": "chamfer",
    "polygon": "polygon", "ellipse": "ellipse",
}

# dimension descriptor word -> canonical dimension name
_DIMENSION_WORDS: Dict[str, str] = {
    "radius": "radius", "diameter": "diameter", "width": "width",
    "height": "height", "length": "length", "side": "side", "depth": "depth",
    "angle": "angle", "thickness": "thickness", "size": "size",
}

# location keywords that stand alone (no coordinates)
_LOCATION_WORDS = {"origin", "centre", "center"}

_DETERMINERS = {"a", "an", "the", "some", "this", "that"}
_FILLERS = {"of", "with", "and", "to"}


@dataclass(frozen=True)
class Nominal:
    """A classified surface token: a noun phrase / value candidate."""

    text: str
    feature: str            # 'shape' | 'quantity' | 'dimension-name' | 'location'
    value: object = None    # canonical value (float, str, tuple)


# --------------------------------------------------------------------------- #
# Case-frame model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CaseSlot:
    """One semantic case of a verb frame.

    ``prepositions`` names the prepositions that may introduce a filler for this
    case (empty tuple = the direct object, taken without a preposition).
    ``require`` is the set of features a nominal must carry to be admitted.
    ``multiple`` lets the case collect more than one filler (e.g. several
    dimensions).  ``required`` marks a case that must be filled for the command
    to be complete.
    """

    name: str
    require: Tuple[str, ...]
    prepositions: Tuple[str, ...] = ()
    multiple: bool = False
    required: bool = False


@dataclass(frozen=True)
class VerbFrame:
    verb: str
    action: str
    slots: Tuple[CaseSlot, ...]

    def slot(self, name: str) -> Optional[CaseSlot]:
        for s in self.slots:
            if s.name == name:
                return s
        return None


_OBJECT = CaseSlot("object", require=("shape",), required=True)
_LOCATION = CaseSlot("location", require=("location", "quantity"),
                     prepositions=("at", "on", "in"))
_DIMENSION = CaseSlot("dimension", require=("dimension-name",),
                      prepositions=("of", "with"), multiple=True)
_TARGET = CaseSlot("target", require=("location", "quantity"),
                   prepositions=("to", "by"))

# verb -> frame.  Synonyms share a canonical action.
_VERB_FRAMES: Dict[str, VerbFrame] = {}


def _register(action: str, verbs: Tuple[str, ...], slots: Tuple[CaseSlot, ...]):
    for v in verbs:
        _VERB_FRAMES[v] = VerbFrame(v, action, slots)


_register("create", ("draw", "create", "make", "add", "place", "sketch",
                     "insert"),
          (_OBJECT, _DIMENSION, _LOCATION))
_register("delete", ("delete", "remove", "erase"),
          (_OBJECT, _LOCATION))
_register("translate", ("move", "translate", "shift"),
          (_OBJECT, _TARGET))
_register("rotate", ("rotate", "turn", "spin"),
          (_OBJECT, _DIMENSION, _TARGET))
_register("scale", ("scale", "resize", "grow", "shrink"),
          (_OBJECT, _DIMENSION))


def verb_frame(verb: str) -> Optional[VerbFrame]:
    return _VERB_FRAMES.get(verb.lower())


def known_verbs() -> Tuple[str, ...]:
    return tuple(sorted(_VERB_FRAMES))


# --------------------------------------------------------------------------- #
# Tokenisation / nominal classification
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> List[str]:
    out: List[str] = []
    tok = ""
    for ch in text:
        if ch.isalnum() or ch in ".-_":
            tok += ch
        elif ch in "(),":
            if tok:
                out.append(tok)
                tok = ""
            out.append(ch)
        else:
            if tok:
                out.append(tok)
                tok = ""
    if tok:
        out.append(tok)
    return out


def _as_number(tok: str) -> Optional[float]:
    try:
        return float(tok)
    except ValueError:
        return None


def classify(tok: str) -> Optional[Nominal]:
    """Classify one surface token as a nominal, or ``None`` if it is not one."""
    low = tok.lower()
    if low in _SHAPE_NOUNS:
        return Nominal(tok, "shape", _SHAPE_NOUNS[low])
    if low in _DIMENSION_WORDS:
        return Nominal(tok, "dimension-name", _DIMENSION_WORDS[low])
    if low in _LOCATION_WORDS:
        return Nominal(tok, "location", "origin" if low == "origin" else "center")
    num = _as_number(low)
    if num is not None:
        return Nominal(tok, "quantity", num)
    return None


# --------------------------------------------------------------------------- #
# Parsed command
# --------------------------------------------------------------------------- #
@dataclass
class ParsedCommand:
    action: str
    verb: str
    obj: Optional[str] = None
    dimensions: Dict[str, float] = field(default_factory=dict)
    location: object = None
    target: object = None
    missing: Tuple[str, ...] = ()
    text: str = ""

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict:
        d = {"action": self.action, "verb": self.verb}
        if self.obj is not None:
            d["object"] = self.obj
        if self.dimensions:
            d["dimensions"] = dict(self.dimensions)
        if self.location is not None:
            d["location"] = self.location
        if self.target is not None:
            d["target"] = self.target
        if self.missing:
            d["missing"] = list(self.missing)
        return d


def _parse_coordinate(tokens: List[str], i: int) -> Optional[Tuple[object, int]]:
    """Try to read a ``( x , y [, z] )`` coordinate starting at ``tokens[i]``.

    Returns ``(coord_tuple, next_index)`` or ``None``.
    """
    if tokens[i] != "(":
        return None
    nums: List[float] = []
    j = i + 1
    while j < len(tokens) and tokens[j] != ")":
        n = _as_number(tokens[j])
        if n is not None:
            nums.append(n)
        j += 1
    if j < len(tokens) and tokens[j] == ")" and nums:
        return tuple(nums), j + 1
    return None


def parse_command(text: str) -> Optional[ParsedCommand]:
    """Parse an imperative CAD command into a filled case frame.

    Returns ``None`` when no known verb is present.  Otherwise returns a
    :class:`ParsedCommand`; ``missing`` lists any *required* case (only the
    direct object, in practice) left unfilled.
    """
    tokens = _tokenize(text)
    lowered = [t.lower() for t in tokens]

    # locate the verb (first token that heads a frame)
    verb_idx = next((i for i, t in enumerate(lowered) if t in _VERB_FRAMES), None)
    if verb_idx is None:
        return None
    frame = _VERB_FRAMES[lowered[verb_idx]]
    cmd = ParsedCommand(action=frame.action, verb=lowered[verb_idx], text=text)

    pending_dim: Optional[str] = None      # a seen dimension-name awaiting a value
    active_prep: Optional[str] = None      # last preposition read

    i = verb_idx + 1
    n = len(tokens)
    while i < n:
        low = lowered[i]

        # coordinate literal -> fills the prep-selected case, else location
        coord = _parse_coordinate(tokens, i)
        if coord is not None:
            value, i = coord
            slot = _slot_for_prep(frame, active_prep)
            _assign(cmd, slot.name if slot else "location", value)
            active_prep = None
            continue

        # preposition? (checked before fillers: 'of'/'with'/'to' may head a case)
        if _is_preposition(frame, low):
            active_prep = low
            i += 1
            continue

        if low in _DETERMINERS or low in _FILLERS or low in (",",):
            i += 1
            continue

        nom = classify(low)
        if nom is None:
            i += 1
            continue

        if nom.feature == "shape":
            if cmd.obj is None:
                cmd.obj = nom.value
            active_prep = None
        elif nom.feature == "dimension-name":
            pending_dim = nom.value
            active_prep = None
        elif nom.feature == "quantity":
            if pending_dim is not None:
                cmd.dimensions[pending_dim] = nom.value
                pending_dim = None
            else:
                slot = _slot_for_prep(frame, active_prep)
                _assign(cmd, slot.name if slot else "location", nom.value)
            active_prep = None
        elif nom.feature == "location":
            slot = _slot_for_prep(frame, active_prep) or _LOCATION
            _assign(cmd, slot.name, nom.value)
            active_prep = None
        i += 1

    cmd.missing = tuple(s.name for s in frame.slots
                        if s.required and not _filled(cmd, s.name))
    return cmd


def _is_preposition(frame: VerbFrame, word: str) -> bool:
    return any(word in s.prepositions for s in frame.slots)


def _slot_for_prep(frame: VerbFrame, prep: Optional[str]) -> Optional[CaseSlot]:
    if prep is None:
        return None
    for s in frame.slots:
        if prep in s.prepositions:
            return s
    return None


def _assign(cmd: ParsedCommand, slot_name: str, value: object) -> None:
    if slot_name == "location":
        if cmd.location is None:
            cmd.location = value
    elif slot_name == "target":
        if cmd.target is None:
            cmd.target = value
    else:  # unexpected slot -> location fallback
        if cmd.location is None:
            cmd.location = value


def _filled(cmd: ParsedCommand, slot_name: str) -> bool:
    if slot_name == "object":
        return cmd.obj is not None
    if slot_name == "location":
        return cmd.location is not None
    if slot_name == "target":
        return cmd.target is not None
    if slot_name == "dimension":
        return bool(cmd.dimensions)
    return False
