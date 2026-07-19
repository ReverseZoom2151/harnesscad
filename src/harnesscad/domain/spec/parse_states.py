"""Parallel parse-states with confidence-levels.

Parsing is modelled as a *parallel* search over competing *parse-states*.  As
each word is read, as many new parse-states are instantiated as there are
possible interpretations of the input -- one per *lexical sense* of the word
(``output`` is both a verb and a noun, and ``what`` is both a determiner and an
interrogative pronoun).  A parse-state carries a numeric *confidence-level* (a
1..10 scale) reflecting how likely it is to be the correct reading; confidence
is dynamic and driven by sense frequency and word-order preference.  States that
violate a local constraint are *terminated* (a determiner cannot be followed by
a verb, so the determiner reading of ``what`` dies at ``is``), and if at any
point too many parse-states are in contention, the less-likely ones are
suspended.

This module is a deterministic realisation of that engine:

* :data:`LEXICON` -- word -> tuple of :class:`Sense` (part-of-speech + relative
  frequency), with an unknown word defaulting to a low-confidence noun;
* :func:`parse` -- left-to-right, one word at a time, expanding every surviving
  state by every admissible sense, terminating states that break the adjacency
  constraint, and suspending the weakest when the beam width is exceeded;
* :class:`ParseResult` -- the ranked survivors, the best reading, and counts of
  terminated / suspended states for inspection.

No randomness, no wall clock: identical input yields identical rankings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Lexical senses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Sense:
    """One lexical sense of a word: a part of speech + relative frequency 1..10."""

    pos: str
    freq: int


LEXICON: Dict[str, Tuple[Sense, ...]] = {
    # function words
    "the": (Sense("det", 9),),
    "a": (Sense("det", 9),), "an": (Sense("det", 9),),
    "what": (Sense("pron", 6), Sense("det", 4)),  # ambiguous, per paper
    "which": (Sense("pron", 5), Sense("det", 5)),
    "is": (Sense("verb", 8),), "are": (Sense("verb", 8),),
    "at": (Sense("prep", 9),), "on": (Sense("prep", 7),),
    "of": (Sense("prep", 8),), "with": (Sense("prep", 7),),
    "to": (Sense("prep", 7),), "in": (Sense("prep", 7),),
    "after": (Sense("prep", 6),), "before": (Sense("prep", 6),),
    # adjectives / quantities
    "maximum": (Sense("adj", 6), Sense("noun", 4)),
    "minimum": (Sense("adj", 6), Sense("noun", 4)),
    "big": (Sense("adj", 7),), "small": (Sense("adj", 7),),
    "large": (Sense("adj", 7),),
    # domain nouns / verbs
    "voltage": (Sense("noun", 8),),
    "node": (Sense("noun", 8),),
    "output": (Sense("noun", 6), Sense("verb", 4)),  # ambiguous, per paper
    "circle": (Sense("noun", 8),), "rectangle": (Sense("noun", 8),),
    "square": (Sense("noun", 8),), "line": (Sense("noun", 8),),
    "hole": (Sense("noun", 8),), "radius": (Sense("noun", 7),),
    "origin": (Sense("noun", 7),),
    "draw": (Sense("verb", 8),), "create": (Sense("verb", 8),),
    "move": (Sense("verb", 7),), "delete": (Sense("verb", 8),),
    "rotate": (Sense("verb", 7),),
}

# an unknown word: assume a common noun of middling confidence
_UNKNOWN = (Sense("noun", 3),)


def senses_of(word: str) -> Tuple[Sense, ...]:
    if word and word.replace(".", "", 1).lstrip("-").isdigit():
        return (Sense("num", 8),)
    return LEXICON.get(word.lower(), _UNKNOWN)


# --------------------------------------------------------------------------- #
# Adjacency constraint (redundant, bottom-up + top-down encoding, simplified)
# --------------------------------------------------------------------------- #
# which part of speech may legally follow which.  START is the sentence head.
_ALLOWED: Dict[str, frozenset] = {
    "START": frozenset({"det", "pron", "verb", "num", "adj", "noun"}),
    "det": frozenset({"noun", "adj", "num"}),          # NOT verb -> kills S1
    "pron": frozenset({"verb", "prep"}),
    "verb": frozenset({"det", "noun", "num", "prep", "pron", "adj"}),
    "noun": frozenset({"prep", "verb", "noun", "num"}),
    "adj": frozenset({"noun", "adj"}),
    "num": frozenset({"noun", "prep", "num"}),
    "prep": frozenset({"det", "noun", "num", "adj", "pron"}),
}

# a parse-state is *complete* only if its final word can end a sentence
_CAN_END: frozenset = frozenset({"noun", "num"})

# word-order preference bonus (0..2) for a transition; default 1
_PREF: Dict[Tuple[str, str], int] = {
    ("START", "verb"): 2,   # imperative CAD commands start with a verb
    ("det", "noun"): 2,
    ("adj", "noun"): 2,
    ("prep", "noun"): 2,
    ("verb", "det"): 2,
    ("pron", "verb"): 2,
}


def _allowed(prev_pos: str, next_pos: str) -> bool:
    return next_pos in _ALLOWED.get(prev_pos, frozenset())


# --------------------------------------------------------------------------- #
# Parse states
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParseState:
    """A surviving analysis of the words read so far."""

    tags: Tuple[Tuple[str, str], ...]   # (word, pos) pairs
    score_sum: int                      # accumulated (freq + pref)
    steps: int

    @property
    def last_pos(self) -> str:
        return self.tags[-1][1] if self.tags else "START"

    @property
    def confidence(self) -> int:
        """Confidence on a 1..10 scale (mean per-step score)."""
        if self.steps == 0:
            return 1
        raw = round(self.score_sum / self.steps)
        return max(1, min(10, raw))

    @property
    def complete(self) -> bool:
        return bool(self.tags) and self.tags[-1][1] in _CAN_END

    def pos_sequence(self) -> Tuple[str, ...]:
        return tuple(pos for _, pos in self.tags)


@dataclass
class ParseResult:
    ranked: List[ParseState] = field(default_factory=list)
    terminated_count: int = 0
    suspended_count: int = 0

    @property
    def best(self):
        return self.ranked[0] if self.ranked else None

    @property
    def complete_states(self) -> List[ParseState]:
        return [s for s in self.ranked if s.complete]


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> List[str]:
    return [t for t in text.replace("?", " ").replace(",", " ").split() if t]


def _rank(states: List[ParseState]) -> List[ParseState]:
    # deterministic: confidence desc, then more-explained (steps) desc, then
    # lexicographic pos sequence for a total order.
    return sorted(
        states,
        key=lambda s: (-s.confidence, -s.steps, s.pos_sequence()),
    )


def parse(text: str, beam: int = 8) -> ParseResult:
    """Parse ``text`` into ranked parse-states via parallel sense expansion.

    ``beam`` bounds the number of live states; when exceeded, the lowest-
    confidence states are *suspended* (dropped from contention but counted).
    """
    words = _tokenize(text)
    live: List[ParseState] = [ParseState(tags=(), score_sum=0, steps=0)]
    terminated = 0
    suspended = 0

    for word in words:
        senses = senses_of(word)
        nxt: List[ParseState] = []
        for st in live:
            for sense in senses:
                if not _allowed(st.last_pos, sense.pos):
                    terminated += 1
                    continue
                pref = _PREF.get((st.last_pos, sense.pos), 1)
                nxt.append(ParseState(
                    tags=st.tags + ((word, sense.pos),),
                    score_sum=st.score_sum + sense.freq + pref,
                    steps=st.steps + 1,
                ))
        nxt = _rank(nxt)
        if len(nxt) > beam:
            suspended += len(nxt) - beam
            nxt = nxt[:beam]
        live = nxt
        if not live:  # every reading terminated; keep nothing further
            break

    ranked = _rank(live)
    return ParseResult(ranked=ranked, terminated_count=terminated,
                       suspended_count=suspended)
