"""Typed CAD-QA query schema (Kienle et al., "QueryCAD: Grounded Question
Answering for CAD Models", Sec. III-C / Fig. 4).

QueryCAD answers free-text questions about a CAD model, but the paper shows
(Fig. 4) that every benchmark question is *structured*: it is composed of

  * a **part** it asks about (inner circle) -- e.g. "hole", "shaft", or the
    whole model;
  * a **property** to retrieve (middle circle) -- a measurement (radius,
    diameter, width, ...), a position (center), a count, or existence;
  * optionally one or more **sides** the part must be visible from (outer
    circle) -- a subset of {top, bottom, left, right, front, back};
  * optionally an additional **filter** on part properties, e.g. "retrieving
    only parts with a radius of 5 mm" (Fig. 4 caption).

The natural-language -> structured-query step is done by an LLM in the paper
(external, skipped here). This module provides the DETERMINISTIC target of that
step: a typed, validated query object that a grounded answer engine
(:mod:`reconstruction.querycad_answer_engine`) can execute against a structured
CAD model. It is the schema the paper's "43 diverse parts" x "11 properties"
question space (Sec. III-C) is drawn from.

Pure, deterministic, stdlib-only. No geometry kernel, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
# The kind of answer a question expects.
QUESTION_TYPES = (
    "count",        # how many X               -> integer
    "measurement",  # what is the <prop> of X  -> number
    "existence",    # does X exist / is there  -> boolean
    "position",     # where is X / its center  -> vector
    "comparison",   # which X is largest/...    -> a part + its property
)

# Retrievable scalar/vector properties (the middle circle of Fig. 4). A subset
# maps directly onto :mod:`fabrication.mfgfeat_taxonomy` feature attributes.
MEASUREMENT_PROPERTIES = (
    "diameter", "radius", "width", "height", "depth", "length",
    "thickness", "angle", "pitch", "area", "volume",
)
POSITION_PROPERTIES = ("center", "position", "tip")

# The six canonical rendered viewing directions (Sec. III-A.4).
VIEWS = ("top", "bottom", "left", "right", "front", "back")
_VIEWS_SET = frozenset(VIEWS)

# Comparison operators usable in a property filter and in comparison questions.
FILTER_OPS = ("eq", "ne", "gt", "lt", "gte", "lte", "approx")
# Aggregations for comparison questions ("the largest bore", "the smallest hole").
AGGREGATIONS = ("max", "min", "largest", "smallest", "widest", "narrowest",
                "deepest", "shallowest")
_AGG_CANON = {
    "max": "max", "largest": "max", "widest": "max", "deepest": "max",
    "min": "min", "smallest": "min", "narrowest": "min", "shallowest": "min",
}


def canonical_aggregation(agg):
    """Map an aggregation word onto ``"max"`` or ``"min"``. Raises KeyError."""
    return _AGG_CANON[str(agg).strip().lower()]


# --------------------------------------------------------------------------- #
# Property filter (Fig. 4: "additional filtering based on the part properties")
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PropertyFilter:
    """A predicate on a part's measured property, e.g. radius == 5 mm."""
    prop: str
    op: str
    value: float
    tol: float = 1e-6  # absolute tolerance for the ``approx`` operator

    def __post_init__(self):
        if self.op not in FILTER_OPS:
            raise ValueError("unknown filter op: %r" % (self.op,))
        if self.tol < 0.0:
            raise ValueError("tol must be non-negative")

    def matches(self, measured):
        """True iff ``measured`` (a number, or None => no match) satisfies this."""
        if measured is None:
            return False
        m = float(measured)
        v = float(self.value)
        if self.op == "eq":
            return m == v
        if self.op == "ne":
            return m != v
        if self.op == "gt":
            return m > v
        if self.op == "lt":
            return m < v
        if self.op == "gte":
            return m >= v
        if self.op == "lte":
            return m <= v
        if self.op == "approx":
            return abs(m - v) <= self.tol
        raise ValueError("unknown filter op: %r" % (self.op,))  # pragma: no cover


# --------------------------------------------------------------------------- #
# The query
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CadQaQuestion:
    """A structured CAD question (the deterministic target of the LLM parse).

    ``part``            free-text part description (inner circle of Fig. 4);
                        the empty string / ``"model"`` targets the whole model.
    ``question_type``   one of :data:`QUESTION_TYPES`.
    ``prop``            the property to retrieve (required for measurement /
                        position / comparison; ignored for count / existence).
    ``views``           tuple of required viewing sides (subset of :data:`VIEWS`).
    ``filters``         tuple of :class:`PropertyFilter` restricting valid parts.
    ``aggregation``     for comparison questions: "max"/"min" (or a synonym).
    """
    part: str
    question_type: str
    prop: Optional[str] = None
    views: Tuple[str, ...] = ()
    filters: Tuple[PropertyFilter, ...] = ()
    aggregation: Optional[str] = None

    def __post_init__(self):
        if self.question_type not in QUESTION_TYPES:
            raise ValueError("unknown question_type: %r" % (self.question_type,))

        # Normalise + validate views (order-insensitive, dedup, canonical order).
        seen = set()
        for v in self.views:
            vv = str(v).strip().lower()
            if vv not in _VIEWS_SET:
                raise ValueError("unknown view: %r" % (v,))
            seen.add(vv)
        object.__setattr__(self, "views",
                           tuple(v for v in VIEWS if v in seen))

        for f in self.filters:
            if not isinstance(f, PropertyFilter):
                raise TypeError("filters must be PropertyFilter instances")

        qt = self.question_type
        if qt in ("measurement", "position", "comparison"):
            if not self.prop:
                raise ValueError("%s question requires a property" % (qt,))
            prop = str(self.prop).strip().lower()
            object.__setattr__(self, "prop", prop)
            if qt == "position":
                if prop not in POSITION_PROPERTIES:
                    raise ValueError("not a position property: %r" % (prop,))
            else:
                if prop not in MEASUREMENT_PROPERTIES:
                    raise ValueError("not a measurement property: %r" % (prop,))
        else:
            # count / existence carry no property.
            object.__setattr__(self, "prop", None)

        if qt == "comparison":
            if self.aggregation is None:
                raise ValueError("comparison question requires an aggregation")
            # normalise to canonical max/min (raises on unknown)
            object.__setattr__(self, "aggregation",
                               canonical_aggregation(self.aggregation))
        else:
            object.__setattr__(self, "aggregation", None)

    # -- convenience ------------------------------------------------------- #
    @property
    def targets_whole_model(self):
        p = (self.part or "").strip().lower()
        return p in ("", "model", "object", "part", "the object", "the model")

    def view_set(self):
        return frozenset(self.views)

    def answer_kind(self):
        """The Python answer kind this question expects."""
        return {
            "count": "int",
            "measurement": "number",
            "existence": "bool",
            "position": "vector",
            "comparison": "part_property",
        }[self.question_type]


# --------------------------------------------------------------------------- #
# Construction from a plain dict (e.g. a benchmark row or an LLM's JSON parse)
# --------------------------------------------------------------------------- #
def filter_from_dict(d):
    return PropertyFilter(
        prop=str(d["prop"]).strip().lower(),
        op=str(d["op"]).strip().lower(),
        value=float(d["value"]),
        tol=float(d.get("tol", 1e-6)),
    )


def question_from_dict(d):
    """Build a :class:`CadQaQuestion` from a plain mapping."""
    filters = tuple(filter_from_dict(f) for f in d.get("filters", ()))
    return CadQaQuestion(
        part=str(d.get("part", "")),
        question_type=str(d["question_type"]).strip().lower(),
        prop=(str(d["prop"]).strip().lower() if d.get("prop") else None),
        views=tuple(d.get("views", ())),
        filters=filters,
        aggregation=(d.get("aggregation") or None),
    )
