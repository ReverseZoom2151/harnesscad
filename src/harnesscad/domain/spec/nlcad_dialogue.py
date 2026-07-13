"""Dialogue state: ellipsis + reference resolution (Cleopatra, 1985).

The sample session in "Towards a Natural Language Interface for CAD" shows
Cleopatra sustaining a *conversation*: after a full query
``what is the voltage at nl at 10 ns`` the user may type only the fragment
``at 20 ns`` and Cleopatra reuses the prior frame, overriding just the supplied
time -- annotated in the paper as "Simple ellipsis is handled".  Section 4.1
also flags pronoun handling ("it", "the last one") as a needed capability of a
CAD language interface.

This module supplies the two dialogue mechanisms, deterministically:

* :class:`DialogueState` -- keeps the last fully-parsed command and interprets
  each new utterance.  A verbless fragment is resolved by *ellipsis*: clone the
  previous command and override only the cases the fragment supplies.
* :class:`EntityRegistry` / :func:`resolve_reference` -- resolve a referring
  phrase (``it``, ``the last one``, ``the first one``, ``the big circle``,
  ``the circle``) against the entities created so far, reporting ambiguity when
  a description matches more than one entity.

Built on :mod:`spec.nlcad_case_frame` for the underlying command parse.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from harnesscad.domain.spec.nlcad_case_frame import (
    ParsedCommand, classify, parse_command, _parse_coordinate, _tokenize,
    _SHAPE_NOUNS, _DIMENSION_WORDS,
)

_ADJECTIVES = {"big", "large", "small", "tiny", "wide", "narrow", "tall",
               "short", "red", "blue", "green", "first", "last"}
_ANAPHORA = {"it", "that", "this", "one", "them"}
_RECENCY_LAST = {"last", "latest", "previous", "recent"}
_RECENCY_FIRST = {"first", "oldest"}


# --------------------------------------------------------------------------- #
# Entity registry + reference resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Entity:
    eid: int
    etype: str
    attributes: Tuple[str, ...] = ()


class EntityRegistry:
    """The set of CAD entities created so far, in insertion order."""

    def __init__(self) -> None:
        self._items: List[Entity] = []
        self._next = 1

    def add(self, etype: str, attributes: Tuple[str, ...] = ()) -> Entity:
        ent = Entity(self._next, etype, tuple(attributes))
        self._items.append(ent)
        self._next += 1
        return ent

    def all(self) -> List[Entity]:
        return list(self._items)

    def get(self, eid: int) -> Optional[Entity]:
        return next((e for e in self._items if e.eid == eid), None)

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class Resolution:
    entity: Optional[Entity]
    candidates: Tuple[Entity, ...]
    reason: str

    @property
    def resolved(self) -> bool:
        return self.entity is not None

    @property
    def ambiguous(self) -> bool:
        return self.entity is None and len(self.candidates) > 1


def resolve_reference(phrase: str, registry: EntityRegistry) -> Resolution:
    """Resolve a referring ``phrase`` to a single :class:`Entity`.

    Handles anaphora (``it``/``that``/``the last one``/``the first one``) and
    definite descriptions (``the big circle``, ``the circle``).  Returns a
    :class:`Resolution`; if the description matches several entities it is left
    unresolved with the matches in ``candidates`` (``ambiguous``).
    """
    items = registry.all()
    if not items:
        return Resolution(None, (), "no-entities")

    words = [w.lower() for w in _tokenize(phrase)]
    wordset = set(words)

    # pure anaphora ("it", "the last one", "the first one")
    etypes = {w for w in words if w in _SHAPE_NOUNS}
    if wordset & _RECENCY_FIRST and not (wordset & _RECENCY_LAST):
        pool = _filter(items, etypes, wordset)
        if pool:
            return Resolution(pool[0], (), "recency-first")
    if (wordset & _ANAPHORA or wordset & _RECENCY_LAST) and not etypes:
        return Resolution(items[-1], (), "anaphora-recent")

    # definite description: filter by type + adjectives
    adjs = {w for w in words if w in _ADJECTIVES} - _RECENCY_FIRST - _RECENCY_LAST
    pool = _filter(items, etypes, wordset)
    pool = [e for e in pool if adjs.issubset(set(e.attributes))]

    if len(pool) == 1:
        return Resolution(pool[0], (), "unique-description")
    if len(pool) > 1:
        if wordset & _RECENCY_LAST:
            return Resolution(pool[-1], (), "description-recent")
        if wordset & _RECENCY_FIRST:
            return Resolution(pool[0], (), "description-first")
        return Resolution(None, tuple(pool), "ambiguous")
    return Resolution(None, (), "no-match")


def _filter(items: List[Entity], etypes, wordset) -> List[Entity]:
    canon = {_SHAPE_NOUNS[t] for t in etypes if t in _SHAPE_NOUNS}
    if canon:
        return [e for e in items if e.etype in canon]
    return list(items)


# --------------------------------------------------------------------------- #
# Ellipsis / dialogue state
# --------------------------------------------------------------------------- #
@dataclass
class Fragment:
    """The cases recovered from a verbless follow-up utterance."""

    obj: Optional[str] = None
    dimensions: Dict[str, float] = field(default_factory=dict)
    location: object = None
    target: object = None

    @property
    def empty(self) -> bool:
        return (self.obj is None and not self.dimensions
                and self.location is None and self.target is None)


def extract_fragment(text: str) -> Fragment:
    """Pull cases from a verbless fragment (``at 20 ns``, ``radius 8``, ``(3,4)``)."""
    tokens = _tokenize(text)
    lowered = [t.lower() for t in tokens]
    frag = Fragment()
    pending_dim: Optional[str] = None
    active_prep: Optional[str] = None
    i, n = 0, len(tokens)
    while i < n:
        low = lowered[i]
        coord = _parse_coordinate(tokens, i)
        if coord is not None:
            value, i = coord
            if active_prep in ("to", "by"):
                frag.target = value
            else:
                frag.location = value
            active_prep = None
            continue
        if low in ("at", "on", "in"):
            active_prep = low
            i += 1
            continue
        if low in ("to", "by"):
            active_prep = low
            i += 1
            continue
        if low in ("of", "with"):
            active_prep = low
            i += 1
            continue
        nom = classify(low)
        if nom is None:
            i += 1
            continue
        if nom.feature == "shape":
            frag.obj = nom.value
        elif nom.feature == "dimension-name":
            pending_dim = nom.value
        elif nom.feature == "location":
            frag.location = nom.value
        elif nom.feature == "quantity":
            if pending_dim is not None:
                frag.dimensions[pending_dim] = nom.value
                pending_dim = None
            elif active_prep in ("to", "by"):
                frag.target = nom.value
            else:
                frag.location = nom.value
        active_prep = None
        i += 1
    return frag


def _adjectives_in(text: str) -> Tuple[str, ...]:
    return tuple(w.lower() for w in _tokenize(text)
                 if w.lower() in _ADJECTIVES
                 and w.lower() not in _RECENCY_FIRST | _RECENCY_LAST)


class DialogueState:
    """Track the running command and resolve ellipsis against it."""

    def __init__(self) -> None:
        self.last: Optional[ParsedCommand] = None
        self.registry = EntityRegistry()
        self.history: List[ParsedCommand] = []

    def interpret(self, text: str) -> Optional[ParsedCommand]:
        """Interpret one utterance -- a full command or an elliptical fragment."""
        cmd = parse_command(text)
        if cmd is not None:
            self._record(cmd, text)
            return cmd
        # no verb: ellipsis relative to the previous command
        frag = extract_fragment(text)
        if self.last is None or frag.empty:
            return None
        merged = self._merge(self.last, frag)
        self._record(merged, text, register=False)
        return merged

    def _merge(self, base: ParsedCommand, frag: Fragment) -> ParsedCommand:
        dims = dict(base.dimensions)
        dims.update(frag.dimensions)          # supplied dims override
        return replace(
            base,
            obj=frag.obj if frag.obj is not None else base.obj,
            dimensions=dims,
            location=frag.location if frag.location is not None else base.location,
            target=frag.target if frag.target is not None else base.target,
            text=frag.__class__.__name__,
        )

    def _record(self, cmd: ParsedCommand, text: str, register: bool = True) -> None:
        self.last = cmd
        self.history.append(cmd)
        if register and cmd.action == "create" and cmd.obj:
            self.registry.add(cmd.obj, _adjectives_in(text))
