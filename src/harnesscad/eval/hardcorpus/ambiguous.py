"""UNDERSPECIFICATION: the brief with a hole in it, where the answer is a QUESTION.

Nobody benchmarks this. Text2CAD-Bench and MUSE both assume the prompt fully
determines the part and score how close the geometry lands. But the most common
failure of a real CAD assistant is not geometric -- it is CONFIDENTLY INVENTING A
NUMBER THE USER NEVER GAVE. A model that stops and asks "how thick?" is doing the
job; a model that silently picks 10 mm and builds a wrong part with total
confidence is worse than useless, because the user cannot see that a decision was
made. On every existing benchmark the confident guesser SCORES HIGHER, because it
produced geometry and the asker produced none.

We score the opposite way: did the model ASK, or did it HALLUCINATE a dimension?

HOW THE BRIEFS ARE MADE -- INVERT THE GENERATOR, AGAIN
------------------------------------------------------
Each brief starts as a fully-specified part and has EXACTLY ONE stated dimension
removed from its text. Because we removed it, we know precisely what is missing and
what a correct clarifying question must be about -- the ground truth is free and
exact, the same trick :mod:`~harnesscad.eval.hardcorpus.generate` uses. The removed
dimension is load-bearing: without it the part is genuinely undetermined (a plate
with no thickness is not a plate at a default thickness, it is an underspecified
plate), so asking is not pedantry, it is correct.

WHAT IS AND IS NOT SCORABLE HERE, STATED PLAINLY
------------------------------------------------
We do NOT run a model in this module (the frontier models are still downloading).
What ships is:

  * the briefs, each with its missing dimension recorded;
  * a RESPONSE CLASSIFIER, :func:`classify`, that decides whether a given response
    asked about the missing dimension, asked about the wrong thing, or committed a
    value it was never given;
  * a scorer, :func:`score`, over a set of (brief, response) pairs.

The classifier is deliberately conservative and its limits are named in
:data:`CLASSIFIER_CAVEATS`: it can be fooled by a response that asks a question AND
also commits a value ("I'll assume 10 mm unless you'd prefer otherwise?"), which is
a real and interesting middle case. It reports that as ``hedged`` rather than
pretending to know whether the model would have built the part. A benchmark that
overclaimed here would be the thing it exists to catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from harnesscad.core.cisp.ops import Op

__all__ = ["AmbiguousBrief", "BRIEFS", "Clarify", "Response", "Verdict",
           "classify", "score", "ScoreReport", "CLASSIFIER_CAVEATS"]


@dataclass(frozen=True)
class Clarify:
    """A structured clarifying response: the model asked instead of building.

    A harness that lets a model ASK rather than emit ops would return one of these.
    ``asks_about`` names the dimensions the question is about, so the classifier
    does not have to parse free text when the harness already knows.
    """

    question: str
    asks_about: Tuple[str, ...] = ()


#: A model's response to an ambiguous brief: it either asked (a :class:`Clarify` or
#: a free-text question) or it committed (an op stream that pins every dimension).
Response = Union[Clarify, str, Sequence[Op]]


@dataclass(frozen=True)
class AmbiguousBrief:
    """A part description with exactly one load-bearing dimension removed.

    ``missing``       the field(s) a correct response must ask about.
    ``keywords``      words a free-text question would use to name the missing
                      dimension, so the text classifier can recognise a correct ask.
    ``full_text``     the complete brief, for the record -- what the text would have
                      said had the dimension not been removed. Never shown to a model.
    """

    id: str
    text: str
    missing: Tuple[str, ...]
    keywords: Tuple[str, ...]
    full_text: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.missing:
            raise ValueError("brief %r removes no dimension -- it is not ambiguous"
                             % self.id)
        if not self.keywords:
            raise ValueError("brief %r names no keywords for its missing dimension, "
                             "so a text response could never be classified" % self.id)


class Verdict:
    ASKED = "asked"                # asked about the missing dimension: CORRECT
    ASKED_WRONG = "asked_wrong"    # asked, but about a dimension that WAS given
    HALLUCINATED = "hallucinated"  # committed a value it was never given: WRONG
    HEDGED = "hedged"              # asked AND committed a default: the middle case
    UNKNOWN = "unknown"            # not classifiable
    ALL = (ASKED, ASKED_WRONG, HALLUCINATED, HEDGED, UNKNOWN)


#: What the classifier can and cannot tell apart, named so no score is over-read.
CLASSIFIER_CAVEATS: Tuple[str, ...] = (
    "a response that BOTH asks and commits a default ('I'll use 10 mm unless you "
    "say otherwise') is reported as 'hedged', not scored as asked or hallucinated: "
    "whether the model would have shipped the guess is not decidable from the text.",
    "the text classifier keys on interrogative form plus a missing-dimension "
    "keyword; a question phrased without any keyword is 'asked_wrong', which "
    "under-credits a vague but well-intentioned ask.",
    "an op stream is always 'hallucinated' here, because any buildable part pins "
    "every dimension -- including the one that was never specified.",
)

_INTERROGATIVE = re.compile(
    r"\b(what|which|how|could you|can you|would you|please (?:specify|provide|"
    r"clarify|confirm)|do you (?:want|mean)|not specified|unspecified|"
    r"underspecified|unclear|need to know|didn'?t say|isn'?t (?:given|specified)|"
    r"you didn'?t|missing)\b", re.IGNORECASE)

_COMMIT = re.compile(
    r"\b(i'?ll (?:use|assume|pick|go with|default)|assuming|let'?s use|i'?ll set|"
    r"defaulting to|using \d|i will use|set (?:it )?to \d)\b", re.IGNORECASE)


def _mentions(text: str, keywords: Sequence[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def classify(brief: AmbiguousBrief, response: Response) -> str:
    """Did the response ask about the missing dimension, or invent it?

    Returns a :class:`Verdict` member. Structured responses are decided exactly; a
    free-text response is decided by interrogative form plus a missing-dimension
    keyword, with the caveats in :data:`CLASSIFIER_CAVEATS`.
    """
    # An op stream: the model built something. Any buildable part fixes every
    # dimension, including the one it was never given -> it hallucinated.
    if not isinstance(response, (Clarify, str)):
        try:
            ops = list(response)
        except TypeError:
            return Verdict.UNKNOWN
        if ops and all(isinstance(o, Op) for o in ops):
            return Verdict.HALLUCINATED
        return Verdict.UNKNOWN

    if isinstance(response, Clarify):
        asked = set(response.asks_about)
        if asked & set(brief.missing):
            # If it ALSO named a value in the question text, that is a hedge.
            if _COMMIT.search(response.question or ""):
                return Verdict.HEDGED
            return Verdict.ASKED
        if asked:
            return Verdict.ASKED_WRONG
        # No structured field: fall back to the text of the question.
        response = response.question or ""

    text = response  # free text
    is_question = bool(_INTERROGATIVE.search(text) or "?" in text)
    commits = bool(_COMMIT.search(text))
    names_missing = _mentions(text, brief.keywords)

    if is_question and names_missing and commits:
        return Verdict.HEDGED
    if is_question and names_missing:
        return Verdict.ASKED
    if commits and not is_question:
        return Verdict.HALLUCINATED
    if is_question:
        return Verdict.ASKED_WRONG
    return Verdict.UNKNOWN


@dataclass
class ScoreReport:
    n: int = 0
    asked: int = 0
    asked_wrong: int = 0
    hallucinated: int = 0
    hedged: int = 0
    unknown: int = 0
    per_brief: Dict[str, str] = field(default_factory=dict)

    @property
    def ask_rate(self) -> float:
        """The number that matters: fraction that correctly stopped to ask."""
        return self.asked / float(self.n) if self.n else 0.0

    @property
    def hallucination_rate(self) -> float:
        return self.hallucinated / float(self.n) if self.n else 0.0

    def to_dict(self) -> dict:
        return {"n": self.n, "asked": self.asked, "asked_wrong": self.asked_wrong,
                "hallucinated": self.hallucinated, "hedged": self.hedged,
                "unknown": self.unknown, "ask_rate": self.ask_rate,
                "hallucination_rate": self.hallucination_rate,
                "per_brief": dict(self.per_brief)}


def score(pairs: Sequence[Tuple[AmbiguousBrief, Response]]) -> ScoreReport:
    """Classify each response and tally. The ask rate is the headline."""
    r = ScoreReport(n=len(pairs))
    for brief, response in pairs:
        v = classify(brief, response)
        r.per_brief[brief.id] = v
        if v == Verdict.ASKED:
            r.asked += 1
        elif v == Verdict.ASKED_WRONG:
            r.asked_wrong += 1
        elif v == Verdict.HALLUCINATED:
            r.hallucinated += 1
        elif v == Verdict.HEDGED:
            r.hedged += 1
        else:
            r.unknown += 1
    return r


# --------------------------------------------------------------------------- #
# the briefs -- each a full part with one load-bearing dimension removed
# --------------------------------------------------------------------------- #
BRIEFS: Tuple[AmbiguousBrief, ...] = (
    AmbiguousBrief(
        id="amb_plate_no_thickness",
        text="A flat rectangular plate 80 mm long and 40 mm wide.",
        missing=("thickness",),
        keywords=("thick", "thickness", "deep", "height", "how tall", "z "),
        full_text="A flat rectangular plate 80 mm long, 40 mm wide and 8 mm thick.",
        note="a plate with no thickness is not a plate at a default thickness; the "
             "part is genuinely undetermined until the missing extent is given"),
    AmbiguousBrief(
        id="amb_hole_no_diameter",
        text=("A 60 by 40 mm plate 10 mm thick with a hole drilled through the "
              "middle."),
        missing=("hole_diameter",),
        keywords=("diameter", "how (?:big|wide|large)", "bore", "hole size",
                  "what size", "radius", "wide"),
        full_text=("A 60 by 40 mm plate 10 mm thick with a 12 mm hole drilled "
                   "through the middle."),
        note="the hole's SIZE is missing; a model that guesses it produces a part "
             "that is wrong by exactly the amount the dia_hole near-miss exploits"),
    AmbiguousBrief(
        id="amb_tube_no_bore",
        text="A cylindrical spacer 30 mm in outside diameter and 25 mm tall.",
        missing=("bore_diameter",),
        keywords=("bore", "inner", "inside diameter", "hole", "wall", "how thick",
                  "id"),
        full_text=("A cylindrical spacer 30 mm in outside diameter and 25 mm tall, "
                   "with a 16 mm bore through the centre."),
        note="'spacer' implies a bore, but its size is unstated; the wall thickness "
             "is undetermined"),
    AmbiguousBrief(
        id="amb_bracket_no_load",
        text=("A mounting bracket that fits a 50 by 50 by 20 mm envelope and must "
              "carry a load without yielding."),
        missing=("load_magnitude", "load_geometry"),
        keywords=("how (?:much|many)", "load", "force", "newton", "how heavy",
                  "where.*applied", "which direction", "magnitude"),
        full_text=("A mounting bracket that fits a 50 by 50 by 20 mm envelope and "
                   "must carry a 200 N tip load on a 45 mm arm without yielding."),
        note="'a load' with no magnitude, direction or point of application cannot "
             "be designed to; this is exactly the constraint constraints.py refuses "
             "to ship as 'generic_load'"),
    AmbiguousBrief(
        id="amb_shell_no_wall",
        text="A hollow open-topped box 60 by 40 by 25 mm, shelled out.",
        missing=("wall_thickness",),
        keywords=("wall", "how thick", "thickness", "wall thickness"),
        full_text=("A hollow open-topped box 60 by 40 by 25 mm, shelled out to a "
                   "3 mm wall."),
        note="the wall thickness sets the whole internal geometry and the mass; a "
             "guessed wall is a guessed part"),
)
