"""The spec surface -- the FRONT DOOR of the harness.

The harness could already plan, build, verify and ingest. What it could not do
was turn a *brief* into a *checked spec*: the ``domain/spec`` tree carried brief
formalisers, case-frame command parsers, clarification planners, ambiguity
audits, EXPRESS/STEP schema machinery and half a dozen structured spec formats,
and almost none of it was reachable. This module is the dispatcher that turns
that tree into one route:

    brief (text)  ->  interpret  ->  clarify + lint  ->  SpecResult
    SpecResult    ->  .constraints / .parameters      (the planner reads these)
    SpecResult    ->  verifiers()                     (the loop runs these)

and one schema route:

    EXPRESS schema text  ->  parse_schema  ->  Schema
    Part-21 file + Schema ->  validate_part21 -> ValidationReport  (ok / issues)

RIVALS ARE SELECTED BY NAME, NEVER BLENDED
------------------------------------------
There are several genuinely different formalisms for "read this brief":

*   ``formalize``        -- typed requirement extraction (count/dimension/
                            material/tolerance/feature). Produces a machine
                            checkable :class:`RequirementSet`.
*   ``case_frame``       -- verb/case-frame parse of a single imperative CAD
                            command ("add a 10 mm hole to the top face").
*   ``command_recovery`` -- the same case-frame parse, but redundancy-based
                            repair of terse / errorful input first.
*   ``parse_states``     -- parallel POS parse-states with confidence (Cleopatra).
*   ``dialogue_state``   -- fragment + reference resolution against a running
                            entity registry (multi-turn ellipsis).

These answer different questions and their outputs are not commensurable. A
caller picks one by name; nothing here averages them. Likewise the two
clarifiers (``interview`` is a gap-driven question planner over a typed
requirement set; ``clarify_ambiguity`` is ProCAD's static under-specification
audit over a CADSpec) are a rival family, selected, never merged.

Structured spec FORMATS (urdf / srdf / plate / rim / express / part21) are not
rivals at all -- they are different input languages, keyed by name, exactly like
:mod:`harnesscad.domain.programs.registry` keys by language.

Discovery goes through :mod:`harnesscad.registry` (the static AST index).
Adapters live here; the spec modules are never modified. Stdlib-only,
deterministic, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "SpecError",
    "UnknownRoute",
    "Constraint",
    "SpecResult",
    "Interpretation",
    "interpreters",
    "interpret",
    "clarifiers",
    "clarify",
    "linters",
    "lint",
    "formats",
    "parse_format",
    "parse_schema",
    "validate_part21",
    "compile_brief",
    "constraints_of",
    "parameters_of",
    "verifiers",
    "size",
    "skeleton",
    "to_ops",
    "coverage",
    "score_clarification",
    "RIVAL_FAMILIES",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

SPEC_PACKAGE = "spec"
_PKG = "harnesscad.domain.spec."


class SpecError(ValueError):
    """Base class for every spec-surface failure."""


class UnknownRoute(SpecError):
    """A route name that is not registered."""


# --------------------------------------------------------------------------- #
# The checked spec a planner / verifier consumes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Constraint:
    """One machine-checkable ask, lifted out of the brief.

    ``kind`` is the requirement kind (count / dimension / material / tolerance /
    feature). ``label`` names what the target describes ('length', 'hole', ...).
    ``source`` keeps the span of the brief it came from -- provenance survives
    the whole route.
    """

    kind: str
    label: str
    target: Any
    unit: Optional[str] = None
    tol: Optional[float] = None
    source: str = ""

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "label": self.label, "target": self.target}
        if self.unit is not None:
            d["unit"] = self.unit
        if self.tol is not None:
            d["tol"] = self.tol
        if self.source:
            d["source"] = self.source
        return d


@dataclass
class SpecResult:
    """A brief, compiled and checked.

    ``ok`` is False when a *blocking* lint fired (code leakage in what should be
    a natural-language brief) -- unanswered questions are gaps, not errors.
    """

    brief: str = ""
    interpreter: str = ""
    requirements: List[dict] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    contract: Dict[str, Any] = field(default_factory=dict)
    questions: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    ok: bool = True

    def to_dict(self) -> dict:
        return {
            "brief": self.brief,
            "interpreter": self.interpreter,
            "ok": self.ok,
            "requirements": list(self.requirements),
            "constraints": [c.to_dict() for c in self.constraints],
            "parameters": dict(self.parameters),
            "contract": dict(self.contract),
            "questions": list(self.questions),
            "missing": list(self.missing),
            "issues": list(self.issues),
        }


@dataclass
class Interpretation:
    """What one brief-interpreter made of the text. Shape depends on the route."""

    name: str
    payload: Any
    requirements: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Route 1: brief interpreters (RIVALS -- selected by name, never blended)
# --------------------------------------------------------------------------- #
def _interp_formalize(brief: str) -> Interpretation:
    from harnesscad.domain.spec.formalize import formalize

    reqset = formalize(brief)
    return Interpretation(
        name="formalize",
        payload=reqset,
        requirements=[r.to_dict() for r in reqset.requirements],
    )


def _interp_case_frame(brief: str) -> Interpretation:
    from harnesscad.domain.spec.case_frame import parse_command

    parsed = parse_command(brief)
    notes = ["missing slot: %s" % s for s in parsed.missing]
    return Interpretation("case_frame", parsed, _reqs_from_command(parsed), notes)


def _interp_command_recovery(brief: str) -> Interpretation:
    from harnesscad.domain.spec.command_recovery import parse_with_recovery

    parsed, recovery = parse_with_recovery(brief)
    notes = ["repair: %s %r at %d" % (r.kind, r.inserted, r.at)
             for r in recovery.repairs]
    notes += ["unknown word: %s" % w for w in recovery.unknown_words]
    return Interpretation("command_recovery", (parsed, recovery),
                          _reqs_from_command(parsed), notes)


def _interp_parse_states(brief: str) -> Interpretation:
    from harnesscad.domain.spec.parse_states import parse

    result = parse(brief)
    best = result.best
    notes = []
    if best is not None:
        notes.append("pos: %s" % " ".join(best.pos_sequence()))
        notes.append("confidence: %.3f" % best.confidence())
    notes.append("terminated: %d" % result.terminated_count)
    notes.append("suspended: %d" % result.suspended_count)
    return Interpretation("parse_states", result, [], notes)


def _interp_dialogue_state(brief: str) -> Interpretation:
    """Multi-turn: each LINE of the brief is a turn against a live registry."""
    from harnesscad.domain.spec.dialogue_state import DialogueState

    state = DialogueState()
    parsed_turns = []
    notes = []
    reqs: List[dict] = []
    for line in [ln.strip() for ln in brief.splitlines() if ln.strip()]:
        parsed = state.interpret(line)
        parsed_turns.append(parsed)
        reqs.extend(_reqs_from_command(parsed))
        notes.append("turn: %s -> %s" % (line, parsed.action or "?"))
    return Interpretation("dialogue_state", parsed_turns, reqs, notes)


def _reqs_from_command(parsed: Any) -> List[dict]:
    """A case-frame ParsedCommand -> requirement dicts (same shape as formalize)."""
    out: List[dict] = []
    obj = getattr(parsed, "obj", None)
    if obj:
        out.append({"kind": "feature", "target": str(obj), "label": str(obj)})
    dims = getattr(parsed, "dimensions", None) or {}
    for label in sorted(dims):
        out.append({"kind": "dimension", "target": dims[label], "label": label})
    return out


_INTERPRETERS: Dict[str, Tuple[Callable[[str], Interpretation], str, str]] = {
    "formalize": (
        _interp_formalize,
        "harnesscad.domain.spec.formalize",
        "typed requirement extraction from a free-text brief (the default)"),
    "case_frame": (
        _interp_case_frame,
        "harnesscad.domain.spec.case_frame",
        "verb/case-frame parse of one imperative CAD command"),
    "command_recovery": (
        _interp_command_recovery,
        "harnesscad.domain.spec.command_recovery",
        "case-frame parse with redundancy-based repair of terse/errorful input"),
    "parse_states": (
        _interp_parse_states,
        "harnesscad.domain.spec.parse_states",
        "parallel POS parse-states with confidence levels (Cleopatra)"),
    "dialogue_state": (
        _interp_dialogue_state,
        "harnesscad.domain.spec.dialogue_state",
        "multi-turn fragment + reference resolution (one turn per line)"),
}

DEFAULT_INTERPRETER = "formalize"


def interpreters() -> Tuple[str, ...]:
    """The selectable brief interpreters. RIVALS: pick one, never blend."""
    return tuple(sorted(n for n in _INTERPRETERS if _available(_INTERPRETERS[n][1])))


def interpret(brief: str, name: str = DEFAULT_INTERPRETER) -> Interpretation:
    """Run ONE named brief interpreter. No fallback to another formalism."""
    try:
        fn, dotted, _doc = _INTERPRETERS[name]
    except KeyError:
        raise UnknownRoute(
            "unknown interpreter %r; known: %s"
            % (name, ", ".join(sorted(_INTERPRETERS)))) from None
    if not _available(dotted):
        raise UnknownRoute("interpreter %r is not present in this tree" % name)
    return fn(brief or "")


# --------------------------------------------------------------------------- #
# Route 2: clarifiers (RIVALS)
# --------------------------------------------------------------------------- #
def _clarify_interview(brief: str, interp: Interpretation) -> Tuple[List[str], List[str]]:
    """Gap-driven question planner over a typed RequirementSet."""
    from harnesscad.domain.spec.formalize import RequirementSet
    from harnesscad.domain.spec.interview import RequirementsInterview

    subject = interp.payload if isinstance(interp.payload, RequirementSet) else brief
    iv = RequirementsInterview()
    missing = list(iv.missing_fields(subject))
    questions = [q.text for q in iv.next_questions(subject)]
    return questions, missing


def _clarify_ambiguity(brief: str, interp: Interpretation) -> Tuple[List[str], List[str]]:
    """ProCAD static under-specification audit over a CADSpec built from the brief."""
    from harnesscad.domain.spec.clarify_ambiguity import CADSpec, audit

    spec = CADSpec(general_shape=brief.strip())
    result = audit(spec)
    questions = [q.text for q in result.questions]
    missing = sorted({i.key for i in result.issues})
    return questions, missing


_CLARIFIERS: Dict[str, Tuple[Callable[[str, Interpretation], Tuple[List[str], List[str]]], str, str]] = {
    "interview": (
        _clarify_interview,
        "harnesscad.domain.spec.interview",
        "rank the under-specified fields of a typed requirement set into questions"),
    "ambiguity": (
        _clarify_ambiguity,
        "harnesscad.domain.spec.clarify_ambiguity",
        "ProCAD proactive under-specification / misleading-prompt audit"),
}

DEFAULT_CLARIFIER = "interview"


def clarifiers() -> Tuple[str, ...]:
    return tuple(sorted(n for n in _CLARIFIERS if _available(_CLARIFIERS[n][1])))


def clarify(brief: str, interp: Optional[Interpretation] = None,
            name: str = DEFAULT_CLARIFIER) -> Tuple[List[str], List[str]]:
    """(questions, missing-field keys) from ONE named clarifier."""
    try:
        fn, dotted, _doc = _CLARIFIERS[name]
    except KeyError:
        raise UnknownRoute(
            "unknown clarifier %r; known: %s"
            % (name, ", ".join(sorted(_CLARIFIERS)))) from None
    if not _available(dotted):
        raise UnknownRoute("clarifier %r is not present in this tree" % name)
    if interp is None:
        interp = interpret(brief)
    return fn(brief or "", interp)


# --------------------------------------------------------------------------- #
# Route 3: brief linters (NOT rivals -- every one runs, each finds its own thing)
# --------------------------------------------------------------------------- #
def _lint_leakage(brief: str) -> List[str]:
    """A natural-language brief that has CadQuery/Python leaking into it is BAD input."""
    from harnesscad.domain.spec.clarify_leakage import check_leakage, style_warnings

    out: List[str] = []
    res = check_leakage(brief)
    if res.contains_code:
        for snip in res.detected_code_snippets:
            out.append("leakage: code in a natural-language brief: %s" % snip)
    out.extend("style: %s" % w for w in style_warnings(brief))
    return out


def _lint_scaling(brief: str) -> List[str]:
    """The Text2CAD 'scale the sketch by k' failure mode: k is not a real edit."""
    from harnesscad.domain.spec.clarify_scaling import detect_scaling, parse_steps

    lines = [ln.strip() for ln in brief.splitlines() if ln.strip()]
    if not lines:
        return []
    issues = detect_scaling(parse_steps(lines))
    return ["scaling: step %d factor %s -- %s"
            % (i.index, i.factor, i.reason) for i in issues]


def _lint_intent(brief: str) -> List[str]:
    """Prompt/constraint lint + brick-category routing (AlphaCAD)."""
    from harnesscad.domain.spec.intent_categories import validate_prompt

    res = validate_prompt(brief)
    return ["intent: %s" % w for w in res.warnings]


_LINTERS: Dict[str, Tuple[Callable[[str], List[str]], str, str, bool]] = {
    "leakage": (_lint_leakage, "harnesscad.domain.spec.clarify_leakage",
                "code leaking into a natural-language brief", True),
    "scaling": (_lint_scaling, "harnesscad.domain.spec.clarify_scaling",
                "the 'scale the sketch' hallucination-risk failure mode", False),
    "intent": (_lint_intent, "harnesscad.domain.spec.intent_categories",
               "prompt/constraint lint against the known object categories", False),
}

#: The linters whose own findings make :attr:`SpecResult.ok` False, derived from
#: the blocking flag each linter declares in :data:`_LINTERS`. Never hardcode a
#: linter name against this: :func:`_blocking` reads the set, so registering a
#: blocking linter is enough to make it block.
_BLOCKING = frozenset(n for n, v in _LINTERS.items() if v[3])


def linters() -> Tuple[str, ...]:
    return tuple(sorted(n for n in _LINTERS if _available(_LINTERS[n][1])))


def lint(brief: str, only: Optional[Sequence[str]] = None) -> List[str]:
    """Every linter's findings, in deterministic (linter, finding) order."""
    wanted = tuple(only) if only is not None else linters()
    out: List[str] = []
    for name in sorted(wanted):
        entry = _LINTERS.get(name)
        if entry is None or not _available(entry[1]):
            continue
        try:
            out.extend(entry[0](brief or ""))
        except Exception as exc:  # noqa: BLE001 - a lint must never break the route
            out.append("lint-error: %s raised %s: %s"
                       % (name, type(exc).__name__, exc))
    return out


def _blocking(issues: Sequence[str]) -> bool:
    """True when any issue was raised BY a linter declared blocking.

    A finding is attributed to a linter by its ``'<linter>: '`` prefix -- the
    convention every linter above follows for its own findings. The attribution
    is deliberately by prefix and not "everything this linter returned":
    ``leakage`` also emits advisory ``'style: '`` lines, which are not code
    leakage and must not block a brief. Likewise a ``'lint-error: '`` line is
    the route reporting that a linter crashed, not that linter's verdict.

    This reads :data:`_BLOCKING` rather than naming a linter, so a second
    blocking linter starts blocking the moment it declares the flag.
    """
    prefixes = tuple("%s:" % n for n in sorted(_BLOCKING))
    return any(i.startswith(prefixes) for i in issues)


# --------------------------------------------------------------------------- #
# Route 4: structured spec FORMATS (input languages, keyed by name -- not rivals)
# --------------------------------------------------------------------------- #
def _fmt_urdf(text: str, **kw):
    from harnesscad.domain.spec.urdf import parse_urdf

    return parse_urdf(text)


def _fmt_srdf(text: str, **kw):
    """SRDF only means something against the URDF it annotates."""
    from harnesscad.domain.spec.srdf import parse_srdf
    from harnesscad.domain.spec.urdf import parse_urdf

    urdf_text = kw.get("urdf")
    if not urdf_text:
        raise SpecError("the 'srdf' format needs its companion urdf=<text>")
    return parse_srdf(text, parse_urdf(urdf_text))


def _fmt_plate(text: str, **kw):
    """A plate-stack building: JSON list of plate dicts."""
    from harnesscad.domain.spec.plate_spec import validate_building

    data = json.loads(text)
    if not isinstance(data, list):
        raise SpecError("the 'plate' format expects a JSON array of plates")
    validate_building(data)
    return data


def _fmt_rim(text: str, **kw):
    from harnesscad.domain.spec.rim_spec import parse_rim_spec, spec_summary

    spec = parse_rim_spec(text)
    return {"spec": spec, "summary": spec_summary(spec)}


def _fmt_express(text: str, **kw):
    return parse_schema(text)


def _fmt_nurbs(text: str, **kw):
    """A NURBGen NURBS-surface JSON B-rep document -> per-face validation errors."""
    from harnesscad.domain.spec.nurbs_json_validator import validate_model

    data = json.loads(text)
    if not isinstance(data, dict):
        raise SpecError("the 'nurbs' format expects a JSON object of faces")
    errors = validate_model(data)
    return {"faces": sorted(data), "errors": errors,
            "ok": all(not e for e in errors.values())}


def _fmt_scad(text: str, **kw):
    """OpenSCAD source -> its Customizer-annotated editable parameters (Zoo/CADAM)."""
    from harnesscad.domain.spec.scad_parameters import parse_parameters

    return parse_parameters(text)


def _fmt_cadmium(text: str, **kw):
    """A CADmium CAD-op sequence (text) -> a canonical, checkable CadSequence."""
    from harnesscad.domain.spec.cadmium_sequence import normalise, parse

    seq = parse(text)
    return {"sequence": seq, "normal_form": normalise(seq)}


_FORMATS: Dict[str, Tuple[Callable[..., Any], str, str]] = {
    "urdf": (_fmt_urdf, "harnesscad.domain.spec.urdf",
             "URDF robot description -> link/joint tree (validated)"),
    "srdf": (_fmt_srdf, "harnesscad.domain.spec.srdf",
             "SRDF semantic robot description, cross-validated against urdf=<text>"),
    "plate": (_fmt_plate, "harnesscad.domain.spec.plate_spec",
              "plate-stack building DSL (JSON array of plates)"),
    "rim": (_fmt_rim, "harnesscad.domain.spec.rim_spec",
            "ISO wheel-rim specification code -> derived geometry"),
    "express": (_fmt_express, "harnesscad.domain.spec.express_schema_parser",
                "EXPRESS schema (ISO 10303-11) -> Schema + inheritance graph"),
    "nurbs": (_fmt_nurbs, "harnesscad.domain.spec.nurbs_json_validator",
              "NURBGen NURBS-surface JSON B-rep -> per-face structural validity"),
    "scad": (_fmt_scad, "harnesscad.domain.spec.scad_parameters",
             "OpenSCAD source -> its Customizer-annotated editable parameters"),
    "cadmium": (_fmt_cadmium, "harnesscad.domain.spec.cadmium_sequence",
                "CADmium CAD-op sequence text -> a canonical, checkable CadSequence"),
}


def formats() -> Tuple[str, ...]:
    return tuple(sorted(n for n in _FORMATS if _available(_FORMATS[n][1])))


def parse_format(fmt: str, text: str, **kw) -> Any:
    """Parse+validate one structured spec format. NEVER guesses the format."""
    try:
        fn, dotted, _doc = _FORMATS[fmt]
    except KeyError:
        raise UnknownRoute(
            "unknown spec format %r; known: %s"
            % (fmt, ", ".join(sorted(_FORMATS)))) from None
    if not _available(dotted):
        raise UnknownRoute("format %r is not present in this tree" % fmt)
    return fn(text, **kw)


# --------------------------------------------------------------------------- #
# Route 5: EXPRESS schema -> Part-21 validation (the real STEP-conformance route)
# --------------------------------------------------------------------------- #
def parse_schema(schema_text: str):
    """Parse an EXPRESS (ISO 10303-11) schema and build its inheritance graph.

    Returns ``(schema, graph)``-carrying object: a small namespace-ish tuple
    subclass would be over-engineering, so the pair is returned directly.
    """
    from harnesscad.domain.spec.express_inheritance import build_inheritance
    from harnesscad.domain.spec.express_schema_parser import parse_schema as _parse

    schema = _parse(schema_text)
    graph = build_inheritance(schema)
    return schema, graph


def validate_part21(p21_text: str, schema_text: str):
    """Validate a Part-21 (``.step``) DATA section against an EXPRESS schema.

    This is the pairing the harness could not previously make: an arbitrary
    ``.exp`` schema defines the entities, and every ``#N = FOO(...)`` record in
    the ``.step`` file is checked against the FLATTENED (inherited + own)
    attribute list of ``FOO``. Returns the module's own
    :class:`~harnesscad.domain.spec.express_p21_validator.ValidationReport`
    (``.ok``, ``.issues``, ``.summary()``).
    """
    from harnesscad.domain.spec.express_p21_validator import validate_data
    from harnesscad.io.formats import step as step_format

    schema, graph = parse_schema(schema_text)
    step_file = step_format.parse(p21_text)
    return validate_data(step_file, schema, graph)


# --------------------------------------------------------------------------- #
# Route 6: spec coverage (decompose a spec into tiles, score exemplar coverage)
# --------------------------------------------------------------------------- #
def coverage(query_spec: str, exemplar_specs: Sequence[str]):
    """Knowledge-sufficiency coverage of a spec against retrieved exemplars."""
    from harnesscad.domain.spec.spec_coverage import coverage_report

    return coverage_report(query_spec, list(exemplar_specs))


def score_clarification(generated_keys: Sequence[str],
                        ground_truth_keys: Sequence[str]):
    """Efficiency (P/R/F1) of a clarifier's questions against the true gaps."""
    from harnesscad.domain.spec.clarify_metrics import efficiency

    return efficiency(list(generated_keys), list(ground_truth_keys))


# --------------------------------------------------------------------------- #
# The end-to-end route: brief -> validated spec -> constraints / parameters
# --------------------------------------------------------------------------- #
_AXIS_OF_LABEL = {
    "length": "x", "long": "x",
    "width": "y", "wide": "y",
    "height": "z", "tall": "z", "high": "z",
    "depth": "z", "deep": "z",
    "thickness": "z", "thick": "z",
}


def constraints_of(requirements: Sequence[dict]) -> List[Constraint]:
    """Requirement dicts -> the ordered, deduplicated Constraint list."""
    out: List[Constraint] = []
    seen = set()
    for r in requirements:
        kind = str(r.get("kind") or "")
        label = str(r.get("label") or kind or "requirement")
        key = (kind, label, repr(r.get("target")))
        if key in seen:
            continue
        seen.add(key)
        out.append(Constraint(
            kind=kind,
            label=label,
            target=r.get("target"),
            unit=r.get("unit"),
            tol=r.get("tolerance"),
            source=str(r.get("source_phrase") or ""),
        ))
    out.sort(key=lambda c: (c.kind, c.label, repr(c.target)))
    return out


def parameters_of(constraints: Sequence[Constraint]) -> Dict[str, Any]:
    """The named numeric knobs a planner can bind straight to a template.

    Dimensions become ``length``/``width``/``height`` (and the ``x``/``y``/``z``
    axis aliases the geometry side speaks); counts become ``n_<label>``;
    material and the default tolerance come through by name.
    """
    params: Dict[str, Any] = {}
    for c in constraints:
        if c.kind == "dimension" and isinstance(c.target, (int, float)):
            label = c.label.lower()
            params.setdefault(label, float(c.target))
            axis = _AXIS_OF_LABEL.get(label)
            if axis is not None:
                params.setdefault(axis, float(c.target))
        elif c.kind == "count" and isinstance(c.target, int):
            params.setdefault("n_%s" % c.label.lower(), int(c.target))
        elif c.kind == "material":
            params.setdefault("material", c.target)
        elif c.kind == "tolerance" and isinstance(c.target, (int, float)):
            params.setdefault("tolerance", float(c.target))
    return dict(sorted(params.items()))


def compile_brief(brief: str,
                  interpreter: str = DEFAULT_INTERPRETER,
                  clarifier: Optional[str] = DEFAULT_CLARIFIER,
                  run_lints: bool = True) -> SpecResult:
    """The front door: a brief in, a CHECKED spec out.

    ``interpreter`` and ``clarifier`` are RIVAL families -- exactly one of each
    runs, chosen by name. Nothing is averaged, nothing falls back to a different
    formalism when the chosen one comes up empty.
    """
    interp = interpret(brief, interpreter)
    issues = list(interp.notes)
    if run_lints:
        issues.extend(lint(brief))

    questions: List[str] = []
    missing: List[str] = []
    if clarifier:
        try:
            questions, missing = clarify(brief, interp, clarifier)
        except SpecError:
            raise
        except Exception as exc:  # noqa: BLE001 - a clarifier gap is not fatal
            issues.append("clarifier-error: %s raised %s: %s"
                          % (clarifier, type(exc).__name__, exc))

    cons = constraints_of(interp.requirements)
    params = parameters_of(cons)

    contract: Dict[str, Any] = {}
    payload = interp.payload
    if type(payload).__name__ == "RequirementSet":
        from harnesscad.domain.spec.formalize import to_contract

        contract = to_contract(payload)

    return SpecResult(
        brief=brief,
        interpreter=interpreter,
        requirements=list(interp.requirements),
        constraints=cons,
        parameters=params,
        contract=contract,
        questions=questions,
        missing=missing,
        issues=issues,
        ok=not _blocking(issues),
    )


def size(requirements: Sequence[dict]) -> List[dict]:
    """Engineering sizing: named formulas turn a requirement into a DIMENSION.

    Each requirement names a formula (``{"formula": "beam_thickness", ...}``) and
    supplies its inputs; the result is a ``{dimension, value, citation, ...}``
    record that :func:`skeleton` merges straight into the master layout. A bad
    requirement raises -- silent mis-sizing is worse than a loud failure.
    """
    from harnesscad.domain.sizing.calc import SizingCalc

    calc = SizingCalc()
    return [calc.size(dict(r)) for r in requirements]


def skeleton(result: SpecResult, sizing: Optional[Sequence[dict]] = None):
    """A checked spec -> the top-down MASTER LAYOUT (envelope + named datums).

    This closes the front door onto the geometry side: the parameters the brief
    yielded become an envelope and a datum reference frame, and
    :meth:`Skeleton.to_ops` emits the CISP that realises the master sketch. Feed
    ``sizing`` (from :func:`size`) to let engineering formulas -- not the brief's
    round numbers -- drive the dimensions.
    """
    from harnesscad.domain.skeleton.layout import build_skeleton

    p = result.parameters
    spec: Dict[str, Any] = {}
    for key, src in (("width", "width"), ("height", "height"),
                     ("depth", "depth")):
        if src in p:
            spec[key] = float(p[src])
    # A plate's brief says length/width/thickness; the skeleton wants w/h/d.
    spec.setdefault("width", float(p.get("length", p.get("x", 0.0)) or 0.0))
    spec.setdefault("height", float(p.get("width", p.get("y", 0.0)) or 0.0))
    spec.setdefault("depth", float(p.get("thickness", p.get("z", 0.0)) or 0.0))
    if "n_hole" in p:
        spec["hole_count"] = int(p["n_hole"])
    if result.contract.get("name"):
        spec["name"] = result.contract["name"]
    if not any(spec.get(k) for k in ("width", "height", "depth")):
        # Nothing dimensional came out of the brief: hand the raw text over and
        # let the layout's own heuristic parser have its go.
        return build_skeleton(result.brief, sizing=list(sizing or []) or None)
    return build_skeleton(spec, sizing=list(sizing or []) or None)


def to_ops(result: SpecResult, sizing: Optional[Sequence[dict]] = None) -> List[Any]:
    """The CISP ops for the master layout a checked spec implies."""
    return list(skeleton(result, sizing).to_ops())


def verifiers(result: SpecResult) -> List[Any]:
    """The verifier objects that CHECK a built model against this spec.

    ``RequirementsCheck`` (each typed ask, measured) and, when the brief carried
    an envelope/count, ``ContractCheck`` (the acceptance contract). Both speak
    the harness verifier protocol -- hand them to
    :class:`~harnesscad.core.loop.HarnessSession` or run them directly.
    """
    out: List[Any] = []
    if result.contract:
        from harnesscad.core.contract import Contract, ContractCheck

        out.append(ContractCheck(Contract.from_dict(result.contract)))
    if result.requirements:
        from harnesscad.domain.spec.formalize import RequirementSet
        from harnesscad.eval.verifiers.requirements import RequirementsCheck

        out.append(RequirementsCheck(
            RequirementSet.from_dict({"requirements": list(result.requirements)})))
    return out


# --------------------------------------------------------------------------- #
# Rivals
# --------------------------------------------------------------------------- #
RIVAL_FAMILIES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("brief-interpreter",
     "Different formalisms for reading a brief. Their outputs are not "
     "commensurable -- a typed requirement set is not a POS lattice. Select one.",
     ("formalize", "case_frame", "command_recovery", "parse_states",
      "dialogue_state")),
    ("clarifier",
     "Gap analysis over a typed requirement set (interview) vs ProCAD's static "
     "under-specification audit over a CADSpec (ambiguity). Different question "
     "spaces; never merged.",
     ("interview", "ambiguity")),
)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    return {e.dotted: e for e in capability_registry.find(package=SPEC_PACKAGE)}


def _available(dotted: str) -> bool:
    return dotted in _index()


def routed_modules() -> Tuple[str, ...]:
    """Every spec module this dispatcher actually reaches (directly or via one)."""
    direct = set()
    for table in (_INTERPRETERS, _CLARIFIERS, _FORMATS):
        for _fn, dotted, *_rest in table.values():
            direct.add(dotted)
    for _fn, dotted, _doc, _blocking_flag in _LINTERS.values():
        direct.add(dotted)
    direct.update({
        _PKG + "express_schema_parser",
        _PKG + "express_inheritance",
        _PKG + "express_p21_validator",
        _PKG + "spec_coverage",
        _PKG + "spec_decompose",
        _PKG + "clarify_metrics",
        _PKG + "formalize",
    })
    return tuple(sorted(d for d in direct if _available(d)))


def discover() -> List[dict]:
    """Every registered route, deterministically ordered. >5 real modules."""
    rows: List[dict] = []
    for name in sorted(_INTERPRETERS):
        _fn, dotted, doc = _INTERPRETERS[name]
        rows.append({"route": "interpret", "name": name, "module": dotted,
                     "doc": doc, "present": _available(dotted)})
    for name in sorted(_CLARIFIERS):
        _fn, dotted, doc = _CLARIFIERS[name]
        rows.append({"route": "clarify", "name": name, "module": dotted,
                     "doc": doc, "present": _available(dotted)})
    for name in sorted(_LINTERS):
        _fn, dotted, doc, _b = _LINTERS[name]
        rows.append({"route": "lint", "name": name, "module": dotted,
                     "doc": doc, "present": _available(dotted)})
    for name in sorted(_FORMATS):
        _fn, dotted, doc = _FORMATS[name]
        rows.append({"route": "format", "name": name, "module": dotted,
                     "doc": doc, "present": _available(dotted)})
    rows.append({"route": "validate", "name": "part21",
                 "module": _PKG + "express_p21_validator",
                 "doc": "validate a Part-21 file against an EXPRESS schema",
                 "present": _available(_PKG + "express_p21_validator")})
    rows.append({"route": "coverage", "name": "coverage",
                 "module": _PKG + "spec_coverage",
                 "doc": "spec tiles (spec_decompose) + exemplar coverage ratio",
                 "present": _available(_PKG + "spec_coverage")})
    rows.append({"route": "layout", "name": "skeleton",
                 "module": "harnesscad.domain.skeleton.layout",
                 "doc": "checked spec -> master layout (envelope + datums) -> CISP ops",
                 "present": True})
    rows.append({"route": "layout", "name": "sizing",
                 "module": "harnesscad.domain.sizing.calc",
                 "doc": "engineering formulas: a requirement -> a sized dimension",
                 "present": True})
    rows.append({"route": "metric", "name": "clarification",
                 "module": _PKG + "clarify_metrics",
                 "doc": "efficiency (P/R/F1) of a clarifier against the true gaps",
                 "present": _available(_PKG + "clarify_metrics")})
    return rows


#: Spec modules deliberately left with no route here, and why.
UNADAPTED_REASONS: Dict[str, str] = {
    "harnesscad.domain.spec.clarify_perturb":
        "ambiguity SYNTHESIS for training data -- a datagen concern, not a "
        "front-door route; owned by the data engine, not the spec surface",
    "harnesscad.domain.spec.clarify_dialogue":
        "the two-round clarification MDP needs an ORACLE (a human or the ground "
        "truth spec) to answer the questions -- reachable via clarify_metrics, "
        "not a standalone route",
    "harnesscad.domain.spec.contract":
        "the Measured Geometric Contract (MGC) is the Specify-phase artifact of "
        "Parts-Driven Development: predicates compiled from a parsed part brief "
        "and checked by a differential oracle against a re-measured file -- "
        "reached through the PDD pipeline (`harnesscad pdd`), not as a standalone "
        "spec-surface route",
    "harnesscad.domain.spec.contract_split":
        "the hidden/visible MGC partition is an anti-gaming EVALUATION step: it "
        "splits a compiled contract into a visible half handed to the generator "
        "and a hidden half kept only for scoring -- reached inside the PDD "
        "evaluation pipeline (`harnesscad pdd`), never a front-door route",
    "harnesscad.domain.spec.caid_artifact":
        "the raw-JSON half of the same SimCorrect/OpenCAD design-artifact "
        "handshake `design_patch` restates with dataclasses: payload validation, "
        "bidirectional tag name resolution, and patch construction straight from "
        "a fault-identification result -- loaded and applied by the physics eval "
        "loop (`harnesscad.eval.quality.physics.sim_correction_loop` and "
        "`fault_identification`), which owns the corrector; the spec front door "
        "never reads it",
    "harnesscad.domain.spec.design_patch":
        "the versioned design-artifact / design-patch handshake (OpenCAD's "
        "`caid-design-artifact-v1` / `caid-design-patch-v1`): it carries an "
        "already-built parametric model's feature tree and named parameters out "
        "to an external simulator and applies the structured, compare-and-swap "
        "guarded parameter corrections that come back -- a downstream edit "
        "transport for the correction loop that consumes a spec, never a brief "
        "reader that produces one",
    "harnesscad.domain.spec.design_brief":
        "an alternative note-taking brief IR (the text-to-cad skill template) with "
        "documented default resolution; the harness's brief front door is "
        "`compile_brief` / `formalize`, and this IR is consumed by that spec "
        "pipeline rather than exposed as its own rival interpreter",
    "harnesscad.domain.spec.part_brief_parser":
        "a deterministic dimensional-brief -> PartSpec -> OpenSCAD path (cad-agent); "
        "its brief-reading half is subsumed by the `formalize` / `case_frame` "
        "interpreters the spec pipeline already routes, and its SCAD synthesis is a "
        "generator concern, not a spec-surface route",
    "harnesscad.domain.spec.part_metadata_contract":
        "forgent3d/aicad's three checkable rules over a GENERATED part's "
        "metadata -- selector copy form, assembly-level vs per-part parameter "
        "placement, and `__viewer` preview isolation -- an output validator that "
        "audits a model the harness has already built (the metadata counterpart "
        "of `harnesscad.domain.programs.validate`, which audits generated code), "
        "not a reader of briefs",
    "harnesscad.domain.spec.project_object":
        "Forma-OSS's addressable, versioned namespace VIEW of a whole project "
        "document (`product.geometry` at its current version, with per-attribute "
        "identity/kind inference and data-URL redaction) -- the navigation layer "
        "the iteration engine `harnesscad.agents.agent.project_iteration` targets "
        "its edits through; it slices a document that already exists rather than "
        "compiling a brief into one",
    "harnesscad.domain.spec.prompt_spec_extract":
        "Studio-OSS's deterministic extraction of a SCORING target from a prompt: "
        "its symmetry confidence, per-family ideal aspect-ratio bounds and texture "
        "hint are scorer tolerances rather than constraints a planner could bind, "
        "and its one consumer is the spec-conditioned half of "
        "`harnesscad.eval.quality.geometry.two_stage_score` -- an eval-surface "
        "target, not a rival of the `formalize` / `case_frame` interpreters",
    "harnesscad.domain.spec.safety_scope":
        "Forma-OSS's brief-time scope POLICY gate (weapons, medical/life-support, "
        "automotive control, mains AC, high-power battery): it yields a refusal "
        "verdict, never a spec, and its own docs demand enforcement before any "
        "agent runs -- which `harnesscad.core.pipeline._scope_refuse` already does "
        "as the first act of `build`, a hard BuildError upstream of this surface "
        "rather than an advisory lint inside it",
    "harnesscad.domain.spec.kcl_grammar":
        "a checked, importable model of Zoo/KittyCAD's KCL lexical grammar, keyword "
        "set and AST node vocabulary -- reference data for a Zoo-backend author "
        "(`harnesscad.io.adapters.zoo_api`), not a brief-to-spec route",
    "harnesscad.domain.spec.kcl_productions":
        "the syntactic layer over `kcl_grammar`: KCL's production-rule table and "
        "precedence block transliterated from Zoo modeling-app's `kcl.grammar`, "
        "plus a recursive-descent structural checker that lints KCL source into "
        "typed diagnostics -- a checker a Zoo-backend author "
        "(`harnesscad.io.adapters.zoo_api`) runs over KCL it emits, not a route "
        "that reads a brief",
    "harnesscad.domain.spec.representation_completeness":
        "a schema/scoring utility that ranks CAD data-representation formats by "
        "which engineering-semantic layers they preserve -- an analysis metric "
        "reached via the eval surface, not a front door that turns input into a "
        "checked spec",
    "harnesscad.domain.spec.zoo_catalog":
        "inert reference data (KCL stdlib, engine op set, file-conversion matrix) "
        "for a Zoo-backend / codec author to read what Zoo supports -- consulted by "
        "the Zoo adapter, not a spec-surface route",
    "harnesscad.domain.spec.zoo_cli_catalog":
        "a static catalogue of the Zoo CLI verb surface (geometry queries, convert, "
        "ml endpoints) for an agent driving the `zoo` binary -- reference data for "
        "the Zoo adapter, not a brief-to-spec route",
    "harnesscad.domain.spec.zoo_ml_feedback":
        "the Zoo text-to-CAD response/feedback model and its thumbs-up acceptance "
        "metric -- an offline eval metric over completed generations, reached via "
        "the eval surface, not a spec front door",
}


def unadapted() -> List[Tuple[str, str]]:
    """(module, reason) for every spec module with no route."""
    routed = set(routed_modules())
    out = []
    for dotted in sorted(_index()):
        if dotted in routed or dotted.endswith(".registry"):
            continue
        out.append((dotted, UNADAPTED_REASONS.get(dotted, "no route yet")))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every registered spec route")
    parser.add_argument("--rivals", action="store_true",
                        help="list the rival families (selected by name, never blended)")
    parser.add_argument("--unadapted", action="store_true",
                        help="list spec modules with no route, and why")
    parser.add_argument("--brief", default=None,
                        help="a natural-language brief to compile into a checked spec")
    parser.add_argument("--brief-file", default=None,
                        help="read the brief from a file instead of --brief")
    parser.add_argument("--interpreter", default=DEFAULT_INTERPRETER,
                        help="which brief interpreter to use (a RIVAL choice)")
    parser.add_argument("--clarifier", default=DEFAULT_CLARIFIER,
                        help="which clarifier to use (a RIVAL choice), or 'none'")
    parser.add_argument("--format", dest="fmt", default=None,
                        help="parse a structured spec format instead (urdf/srdf/plate/rim/express)")
    parser.add_argument("--file", default=None,
                        help="the file the --format / --part21 route reads")
    parser.add_argument("--part21", default=None,
                        help="a Part-21 .step file to validate against --schema")
    parser.add_argument("--schema", default=None,
                        help="an EXPRESS .exp schema file (with --part21)")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "rivals", False):
        for family, doc, members in RIVAL_FAMILIES:
            print("%s: (selected by name, NEVER blended)" % family)
            print("    %s" % doc)
            for m in members:
                print("    - %s" % m)
        return 0

    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "part21", None):
        schema_path = getattr(args, "schema", None)
        if not schema_path:
            print("--part21 needs --schema <express file>", file=sys.stderr)
            return 2
        report = validate_part21(_read(args.part21), _read(schema_path))
        if getattr(args, "json", False):
            print(json.dumps({
                "ok": report.ok,
                "checked": report.checked,
                "issues": [str(i) for i in report.issues],
            }, indent=2, sort_keys=True))
        else:
            print(report.summary())
            for issue in report.issues:
                print("  %s" % issue)
        return 0 if report.ok else 1

    if getattr(args, "fmt", None):
        path = getattr(args, "file", None)
        if not path:
            print("--format needs --file <path>", file=sys.stderr)
            return 2
        try:
            parsed = parse_format(args.fmt, _read(path))
        except SpecError as exc:
            print("error: %s" % exc, file=sys.stderr)
            return 2
        print(repr(parsed))
        return 0

    brief = getattr(args, "brief", None)
    if getattr(args, "brief_file", None):
        brief = _read(args.brief_file)
    if brief:
        clarifier = getattr(args, "clarifier", DEFAULT_CLARIFIER)
        if clarifier in ("none", ""):
            clarifier = None
        result = compile_brief(brief, getattr(args, "interpreter",
                                              DEFAULT_INTERPRETER), clarifier)
        if getattr(args, "json", False):
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print("interpreter: %s   ok: %s" % (result.interpreter, result.ok))
            print("constraints:")
            for c in result.constraints:
                print("    %-10s %-12s %s%s" % (
                    c.kind, c.label, c.target, (" " + c.unit) if c.unit else ""))
            print("parameters: %s" % json.dumps(result.parameters, sort_keys=True))
            if result.questions:
                print("open questions:")
                for q in result.questions:
                    print("    - %s" % q)
            if result.issues:
                print("issues:")
                for i in result.issues:
                    print("    ! %s" % i)
        return 0 if result.ok else 1

    # default: --list
    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["name"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-9s %-*s  %s" % (mark, r["route"], width, r["name"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad spec",
        description="spec surface: brief -> checked spec -> constraints; "
                    "EXPRESS/Part-21 validation; structured spec formats")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
