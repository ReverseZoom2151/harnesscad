"""clarify_ambiguity -- proactive ambiguity detection for text-to-CAD.

Implements an ambiguity *detection taxonomy* and rule-based classifier. The
clarifying agent audits a prompt *before* code synthesis and decides
``is_misleading`` by checking for three issue classes:

  1. Ambiguous / under-specified dimensions -- vague size descriptions without
     specific measurements, or a required slot left blank.
  2. Conflicting dimensions -- two or more measurements assigned to the same
     feature that contradict each other.
  3. Geometrically impossible dimensions -- measurements that cannot form a
     valid solid.

This is deterministic and stdlib-only. It is *not* the existing
``spec.interview`` (which ranks generic requirement gaps like material/tolerance
for a fabrication brief). Here we audit a structured CAD build specification for
the three geometric ambiguity classes and emit the *minimum* set of
targeted clarification questions, using a two-round policy.

The canonical structured spec (:class:`CADSpec`) and the issue / question value
objects are defined here and reused by the sibling ``clarify_*`` modules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Canonical structured CAD specification
# --------------------------------------------------------------------------- #

#: Every valid workplane string the descriptions use.
VALID_PLANES = ("XY", "YZ", "ZX", "XZ", "YX", "ZY")

#: Extrusion directions a "Build description" always states.
VALID_DIRECTIONS = ("positive_normal", "negative_normal", "both")


@dataclass
class Feature:
    """One geometric feature in a build description.

    ``params`` maps a parameter name (e.g. ``"radius"``, ``"width"``) to its
    *stated values*. A value of ``None`` means the slot exists but was left
    unspecified (under-specification). A list/tuple of two or more distinct
    numbers means the same slot was assigned conflicting values (the
    "radius 8 ... radius 10" case).
    """

    kind: str
    name: str
    params: Dict[str, object] = field(default_factory=dict)

    def stated(self, key: str) -> List[float]:
        """Return the distinct numeric values stated for ``key`` (may be []). """
        raw = self.params.get(key, None)
        if raw is None:
            return []
        if isinstance(raw, (list, tuple, set)):
            vals = [float(v) for v in raw if v is not None]
        else:
            vals = [float(raw)]
        # de-dup preserving order
        out: List[float] = []
        for v in vals:
            if v not in out:
                out.append(v)
        return out


@dataclass
class CADSpec:
    """A structured CAD build spec (General shape / Setup / Build description)."""

    general_shape: str = ""
    workplane: Optional[str] = None
    origin: Optional[Tuple[float, float, float]] = None
    extrude_direction: Optional[str] = None
    extrude_distance: Optional[object] = None  # None / float / list (conflict)
    features: List[Feature] = field(default_factory=list)

    def copy(self) -> "CADSpec":
        return replace(
            self,
            origin=None if self.origin is None else tuple(self.origin),
            features=[Feature(f.kind, f.name, dict(f.params)) for f in self.features],
        )


# --------------------------------------------------------------------------- #
# Issue taxonomy
# --------------------------------------------------------------------------- #

UNDER_SPECIFIED = "under_specified"
CONFLICTING = "conflicting"
IMPOSSIBLE = "geometrically_impossible"

ISSUE_TYPES = (UNDER_SPECIFIED, CONFLICTING, IMPOSSIBLE)


@dataclass(frozen=True)
class Issue:
    """One detected specification issue.

    ``key`` is a stable feature/parameter identifier (e.g. ``"hole.radius"``)
    used by :mod:`clarify_metrics` to align questions across agents without an
    LLM. ``values`` holds the conflicting/observed numbers when relevant.
    """

    type: str
    key: str
    message: str
    values: Tuple[float, ...] = ()


@dataclass(frozen=True)
class ClarQuestion:
    """A single targeted clarification question tied to an issue ``key``."""

    key: str
    text: str
    type: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.text


# --------------------------------------------------------------------------- #
# Vague-language detection (ambiguous dimensions without measurements)
# --------------------------------------------------------------------------- #

_VAGUE_TERMS = (
    "some", "several", "a few", "many", "large", "small", "big", "tiny",
    "appropriate", "suitable", "reasonable", "roughly", "about", "approximately",
    "around", "moderate", "standard", "typical", "sufficient",
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def vague_phrases(text: str) -> List[str]:
    """Return vague size terms in ``text`` that are not pinned to a number.

    Mirrors the rule "Ambiguous dimensions: vague size descriptions without
    specific measurements". A term is only flagged when no numeric measurement
    appears in the same clause.
    """
    found: List[str] = []
    if not text:
        return found
    for clause in re.split(r"[.;,\n]", text.lower()):
        if _NUMBER_RE.search(clause):
            continue
        for term in _VAGUE_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", clause):
                found.append(term)
    return found


# --------------------------------------------------------------------------- #
# Required-slot model ("Always include sketch plane, extrusion
# direction, and extrusion distance").
# --------------------------------------------------------------------------- #

#: Parameters that each feature kind must specify to be fully determined.
_REQUIRED_PARAMS = {
    "rectangle": ("width", "height"),
    "circle": ("radius",),
    "polygon": ("vertices",),
    "slot": ("length", "width"),
    "hole": ("radius",),
}


def _feature_key(feat: Feature, param: str) -> str:
    return "{0}.{1}".format(feat.name or feat.kind, param)


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #

class AmbiguityDetector:
    """Deterministic auditor implementing the three-class taxonomy."""

    def detect(self, spec: CADSpec) -> List[Issue]:
        issues: List[Issue] = []
        issues.extend(self._under_specified(spec))
        issues.extend(self._conflicting(spec))
        issues.extend(self._impossible(spec))
        # Deterministic, stable order: by (type-rank, key).
        rank = {UNDER_SPECIFIED: 0, CONFLICTING: 1, IMPOSSIBLE: 2}
        issues.sort(key=lambda i: (rank[i.type], i.key))
        return issues

    # -- class 1: under-specification / vagueness ------------------------ #
    def _under_specified(self, spec: CADSpec) -> List[Issue]:
        out: List[Issue] = []
        if spec.workplane is None:
            out.append(Issue(UNDER_SPECIFIED, "setup.workplane",
                             "The sketch plane (workplane) is not specified."))
        elif spec.workplane.upper() not in VALID_PLANES:
            out.append(Issue(UNDER_SPECIFIED, "setup.workplane",
                             "The sketch plane '{0}' is not a recognised plane."
                             .format(spec.workplane)))
        if spec.origin is None:
            out.append(Issue(UNDER_SPECIFIED, "setup.origin",
                             "The workplane origin (shift vector) is missing."))
        if spec.extrude_direction is None:
            out.append(Issue(UNDER_SPECIFIED, "build.extrude_direction",
                             "The extrusion direction is not specified."))
        if _is_missing(spec.extrude_distance):
            out.append(Issue(UNDER_SPECIFIED, "build.extrude_distance",
                             "The extrusion distance (thickness) is omitted."))

        for feat in spec.features:
            for param in _REQUIRED_PARAMS.get(feat.kind, ()):
                raw = feat.params.get(param, None)
                if _is_missing(raw):
                    key = _feature_key(feat, param)
                    out.append(Issue(UNDER_SPECIFIED, key,
                                     "The {0} of {1} is not specified."
                                     .format(param, feat.name or feat.kind)))

        for term in vague_phrases(spec.general_shape):
            out.append(Issue(UNDER_SPECIFIED, "shape.vague:" + term,
                             "Vague size term '{0}' without a measurement."
                             .format(term)))
        return out

    # -- class 2: conflicting dimensions --------------------------------- #
    def _conflicting(self, spec: CADSpec) -> List[Issue]:
        out: List[Issue] = []
        # Global extrusion distance conflict.
        dvals = _stated_values(spec.extrude_distance)
        if len(dvals) > 1:
            out.append(Issue(CONFLICTING, "build.extrude_distance",
                             "Conflicting extrusion distances: {0}."
                             .format(_fmt(dvals)), tuple(dvals)))
        for feat in spec.features:
            for param in feat.params:
                vals = feat.stated(param)
                if len(vals) > 1:
                    key = _feature_key(feat, param)
                    out.append(Issue(CONFLICTING, key,
                                     "Conflicting {0} for {1}: {2}."
                                     .format(param, feat.name or feat.kind,
                                             _fmt(vals)), tuple(vals)))
        return out

    # -- class 3: geometrically impossible ------------------------------- #
    def _impossible(self, spec: CADSpec) -> List[Issue]:
        out: List[Issue] = []
        # Non-positive extrusion distance.
        for d in _stated_values(spec.extrude_distance):
            if d <= 0:
                out.append(Issue(IMPOSSIBLE, "build.extrude_distance",
                                 "Extrusion distance {0} is not positive."
                                 .format(_fmt([d])), (d,)))
                break
        for feat in spec.features:
            for param, raw in feat.params.items():
                for v in _stated_values(raw):
                    if param in ("radius", "width", "height", "length",
                                 "diameter", "thickness") and v <= 0:
                        out.append(Issue(IMPOSSIBLE, _feature_key(feat, param),
                                         "{0} {1} is not positive."
                                         .format(param, _fmt([v])), (v,)))
                        break
            # Concentric annulus: inner radius must be < outer radius.
            outer = feat.params.get("outer_radius")
            inner = feat.params.get("inner_radius")
            ov = _stated_values(outer)
            iv = _stated_values(inner)
            if ov and iv and iv[0] >= ov[0]:
                out.append(Issue(IMPOSSIBLE, _feature_key(feat, "inner_radius"),
                                 "Inner radius {0} >= outer radius {1}; no wall."
                                 .format(_fmt([iv[0]]), _fmt([ov[0]])),
                                 (iv[0], ov[0])))
        return out


# --------------------------------------------------------------------------- #
# Question synthesis + top-level audit
# --------------------------------------------------------------------------- #

def question_for(issue: Issue) -> ClarQuestion:
    """Produce a single targeted, minimal clarification question for an issue."""
    if issue.type == CONFLICTING and issue.values:
        opts = " or ".join(_fmt([v]) for v in issue.values)
        feature = issue.key.split(".")[0]
        param = issue.key.split(".")[-1]
        text = "For the {0}, should the {1} be {2}?".format(feature, param, opts)
    elif issue.type == IMPOSSIBLE:
        text = ("The stated dimensions cannot form a valid solid ({0}). "
                "What are the corrected values?").format(issue.message)
    else:  # under-specified
        text = "What is the value of {0}? ({1})".format(
            issue.key.split(":")[0], issue.message)
    return ClarQuestion(issue.key, text, issue.type)


@dataclass(frozen=True)
class Audit:
    """Result of a proactive audit: the ``is_misleading`` envelope."""

    is_misleading: bool
    issues: Tuple[Issue, ...]
    questions: Tuple[ClarQuestion, ...]

    def envelope(self, standardized_prompt: str = "") -> dict:
        """Return the JSON-shaped output of the Clarification-Generation prompt."""
        if self.is_misleading:
            return {"is_misleading": True,
                    "questions": [q.text for q in self.questions]}
        return {"is_misleading": False,
                "standardized_prompt": standardized_prompt}

    def under_specification_score(self, spec: CADSpec) -> float:
        return under_specification_score(spec)


def audit(spec: CADSpec) -> Audit:
    """Audit ``spec`` and emit the minimal set of clarification questions.

    Only one question is asked per distinct issue ``key`` (the "ask the
    minimum number of clarifying questions necessary").
    """
    issues = AmbiguityDetector().detect(spec)
    seen = set()
    questions: List[ClarQuestion] = []
    for iss in issues:
        if iss.key in seen:
            continue
        seen.add(iss.key)
        questions.append(question_for(iss))
    return Audit(bool(issues), tuple(issues), tuple(questions))


# --------------------------------------------------------------------------- #
# Under-specification scorer
# --------------------------------------------------------------------------- #

def _required_slots(spec: CADSpec) -> List[Tuple[str, object]]:
    slots: List[Tuple[str, object]] = [
        ("setup.workplane", spec.workplane),
        ("setup.origin", spec.origin),
        ("build.extrude_direction", spec.extrude_direction),
        ("build.extrude_distance", spec.extrude_distance),
    ]
    for feat in spec.features:
        for param in _REQUIRED_PARAMS.get(feat.kind, ()):
            slots.append((_feature_key(feat, param), feat.params.get(param)))
    return slots


def under_specification_score(spec: CADSpec) -> float:
    """Fraction of required geometric slots that are filled in ``[0, 1]``.

    ``1.0`` means fully specified; lower values indicate more missing
    dimensions (the under-specified prompts).
    """
    slots = _required_slots(spec)
    if not slots:
        return 1.0
    filled = sum(0 if _is_missing(v) else 1 for _, v in slots)
    return filled / len(slots)


def missing_slots(spec: CADSpec) -> List[str]:
    """Return the keys of required slots that are unspecified."""
    return [k for k, v in _required_slots(spec) if _is_missing(v)]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _is_missing(raw: object) -> bool:
    if raw is None:
        return True
    if isinstance(raw, (list, tuple, set)) and len(raw) == 0:
        return True
    return False


def _stated_values(raw: object) -> List[float]:
    if _is_missing(raw):
        return []
    if isinstance(raw, (list, tuple, set)):
        vals = [float(v) for v in raw if v is not None]
    else:
        try:
            vals = [float(raw)]
        except (TypeError, ValueError):
            return []
    out: List[float] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def _fmt(vals: List[float]) -> str:
    def one(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else str(v)
    return ", ".join(one(v) for v in vals)
