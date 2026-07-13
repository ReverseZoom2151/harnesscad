"""Program-analysis surface: parse / validate / emit / review, dispatched by LANGUAGE.

The ``domain/programs`` tree carries ASTs, validators, emitters, expression
engines, code review and parameter schemas for several *different* code-CAD
languages -- CadQuery, OpenSCAD, OpenECAD, a typed CSG language, OpenMC-style
CSG, the FreeCAD expression language, Blender ``bpy`` scripts. Almost none of it
was reachable: the modules were correct and tested, and nothing dispatched to
them.

This module is that dispatcher.

    parse(source, lang)     -> Program   (a language-tagged AST)
    validate(program)       -> Findings  (diagnostics, never an exception storm)
    emit(ops, lang)         -> source    (the neutral op IR -> that language)
    review(source, lang)    -> Findings  (static review of somebody else's code)

plus the capabilities that only some languages have: :func:`extract` (recover
source from an LLM reply), :func:`params` (the Customizer / parameter schema),
:func:`annotate` (CADTalk commentable blocks), :func:`repair`, :func:`quantize`,
:func:`diagnostics` (parse a toolchain's error output), :func:`bom`.

LANGUAGE IS THE KEY. THE LANGUAGES ARE NOT RIVALS AND ARE NEVER BLENDED.
------------------------------------------------------------------------
``ast/cadquery.py``, ``ast/openscad.py`` and ``ast/typed_csg.py`` are not three
answers to one question -- they are three *languages*. An OpenSCAD parser handed
CadQuery source does not "mostly work"; it produces a syntax error or, worse,
a plausible wrong tree. So:

*   every entry point takes an explicit ``lang=``;
*   a parsed :class:`Program` carries the language it was parsed as, and
    :func:`validate` / :func:`serialize` refuse a program from another language
    (:class:`LanguageMismatch`);
*   a language that cannot do a capability says so (:class:`Unsupported`) instead
    of falling back to another language's implementation.

Discovery goes through :mod:`harnesscad.registry` (the static AST index): a
language is only offered when the modules it dispatches to are actually in the
tree. Adapters live here; the program modules are never modified.

Stdlib-only, absolute imports, deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "LANGUAGES",
    "CAPABILITIES",
    "ProgramError",
    "UnknownLanguage",
    "LanguageMismatch",
    "Unsupported",
    "Finding",
    "Program",
    "Language",
    "languages",
    "language",
    "capabilities",
    "supports",
    "parse",
    "validate",
    "serialize",
    "emit",
    "review",
    "extract",
    "params",
    "annotate",
    "repair",
    "quantize",
    "diagnostics",
    "bom",
    "operations",
    "validate_ops",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

PROGRAMS_PACKAGE = "programs"
_PKG = "harnesscad.domain.programs."

#: The selectable languages. Never inferred from the source text.
CADQUERY = "cadquery"
OPENSCAD = "openscad"
OPENECAD = "openecad"
TYPED_CSG = "typed_csg"
OPENMC_CSG = "openmc_csg"
FREECAD_EXPR = "freecad_expr"
BPY = "bpy"

LANGUAGES: Tuple[str, ...] = (
    CADQUERY, OPENSCAD, OPENECAD, TYPED_CSG, OPENMC_CSG, FREECAD_EXPR, BPY,
)

#: The capabilities a language may implement.
CAPABILITIES: Tuple[str, ...] = (
    "parse", "validate", "serialize", "emit", "review", "extract", "params",
    "annotate", "repair", "quantize", "diagnostics", "bom", "handles",
)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ProgramError(ValueError):
    """Base class for every program-surface failure."""


class UnknownLanguage(ProgramError):
    """A language name outside :data:`LANGUAGES`."""


class LanguageMismatch(ProgramError):
    """A program parsed as one language was handed to another's dispatcher."""


class Unsupported(ProgramError):
    """This language genuinely does not implement this capability (no fallback)."""


def _check_lang(name: str) -> str:
    if name not in LANGUAGES:
        raise UnknownLanguage(
            "unknown code-CAD language %r; the selectable languages are %s"
            % (name, ", ".join(LANGUAGES)))
    return name


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass(frozen=True)
class Finding:
    """One diagnostic. Uniform across languages; the ``source`` names its origin."""

    language: str
    severity: str
    code: str
    message: str
    line: Optional[int] = None
    source: str = ""          # the dotted module that produced it

    def to_dict(self) -> dict:
        return {"language": self.language, "severity": self.severity,
                "code": self.code, "message": self.message, "line": self.line,
                "source": self.source}


@dataclass(frozen=True)
class Program:
    """A LANGUAGE-TAGGED parse tree. The tag is what makes dispatch honest."""

    lang: str
    tree: Any
    source: str = ""

    def __post_init__(self) -> None:
        _check_lang(self.lang)


@dataclass(frozen=True)
class Language:
    """One code-CAD language and the modules that implement its capabilities."""

    name: str
    description: str
    modules: Tuple[str, ...]                      # dotted, all must be indexed
    caps: Dict[str, Callable[..., Any]] = field(default_factory=dict)

    def supports(self, capability: str) -> bool:
        return capability in self.caps

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "capabilities": sorted(self.caps), "modules": list(self.modules)}


def _err(lang: str, code: str, message: str, line=None, source="") -> Finding:
    return Finding(lang, ERROR, code, str(message), line, source)


def _warn(lang: str, code: str, message: str, line=None, source="") -> Finding:
    return Finding(lang, WARNING, code, str(message), line, source)


def _info(lang: str, code: str, message: str, line=None, source="") -> Finding:
    return Finding(lang, INFO, code, str(message), line, source)


# --------------------------------------------------------------------------- #
# The neutral operation IR (the input to emit()).
#
# `validate.operation_schema` already defines a backend-agnostic code-CAD
# operation vocabulary (draw / solid / boolean / modify calls with typed params).
# It is the emit surface's input language: ops are validated against it FIRST,
# then lowered into whichever target language was asked for.
# --------------------------------------------------------------------------- #
def operations(category: Optional[str] = None) -> List[str]:
    """The neutral operation vocabulary emit() accepts (from ``validate.operation_schema``)."""
    from harnesscad.domain.programs.validate import operation_schema as m

    return m.operation_names(category)


def _as_calls(ops: Sequence[Any]) -> List[Any]:
    from harnesscad.domain.programs.validate import operation_schema as m

    out = []
    for op in ops:
        if isinstance(op, m.Call):
            out.append(op)
        elif isinstance(op, dict):
            out.append(m.Call(operation=str(op["operation"]),
                              args=dict(op.get("args") or {}),
                              result=op.get("result")))
        else:
            raise ProgramError("an op must be an operation_schema.Call or a dict "
                               "with 'operation'/'args'/'result' (got %r)"
                               % type(op).__name__)
    return out


def validate_ops(ops: Sequence[Any]) -> List[str]:
    """Validate a neutral op program (language-independent). [] == valid."""
    from harnesscad.domain.programs.validate import operation_schema as m

    return m.validate_program(_as_calls(ops))


def _op_args(call: Any, *names: str) -> Tuple[Any, ...]:
    missing = [n for n in names if n not in call.args]
    if missing:
        raise ProgramError("op %r is missing %s" % (call.operation, ", ".join(missing)))
    return tuple(call.args[n] for n in names)


# --------------------------------------------------------------------------- #
# CadQuery
# --------------------------------------------------------------------------- #
def _cq_parse(source: str):
    from harnesscad.domain.programs.ast import cadquery as m

    return m.parse_program(source)


def _cq_serialize(tree) -> str:
    from harnesscad.domain.programs.ast import cadquery as m

    return m.serialize(tree)


def _cq_validate(tree) -> List[Finding]:
    from harnesscad.domain.programs.ast import cadquery as m

    return [_err(CADQUERY, "cq-ast", msg, source=_PKG + "ast.cadquery")
            for msg in m.validate(tree)]


def _cq_validate_source(source: str) -> List[Finding]:
    """Execution-free validity: API + arity (cadquery_validity) and the Workplane
    state machine (cadquery_workplane). Both are static -- nothing is executed."""
    from harnesscad.domain.programs.validate import cadquery_validity as cv
    from harnesscad.domain.programs.validate import cadquery_workplane as cw

    out: List[Finding] = []
    for issue in cv.check_code(source):
        out.append(_err(CADQUERY, issue.code, issue.message, issue.line,
                        _PKG + "validate.cadquery_validity"))
    for diag in cw.validate_code(source):
        severity = ERROR if getattr(diag, "severity", "error") == "error" else WARNING
        out.append(Finding(CADQUERY, severity, "workplane-" + diag.method,
                           diag.message, None, _PKG + "validate.cadquery_workplane"))
    return out


def _cq_review(source: str, **_kw) -> List[Finding]:
    """Static review: safety/quality analysis plus an API-usage profile."""
    from harnesscad.domain.programs.validate import cadquery_analysis as ca
    from harnesscad.domain.programs.validate import cadquery_api_profile as ap

    out: List[Finding] = []
    report = ca.analyze(source)
    for f in report.findings:
        severity = {"error": ERROR, "warning": WARNING}.get(f.severity, INFO)
        out.append(Finding(CADQUERY, severity, f.code, f.message, f.line,
                           _PKG + "validate.cadquery_analysis"))
    profile = ap.profile_source(source)
    for err in profile.parse_errors:
        out.append(_err(CADQUERY, "parse-error", str(err), None,
                        _PKG + "validate.cadquery_api_profile"))
    if profile.methods:
        out.append(_info(CADQUERY, "api-profile",
                         "methods used: " + ", ".join(sorted(profile.method_names())),
                         None, _PKG + "validate.cadquery_api_profile"))
    out.extend(_cq_validate_source(source))
    return out


def _cq_extract(reply: str, *, export_path: str = "part.step", require_eos: bool = True):
    """Recover a runnable CadQuery script from a raw LLM reply."""
    from harnesscad.domain.programs.extract import cadquery_clean as m

    result = m.clean_output(reply, export_path, require_eos=require_eos)
    if not result.ok:
        raise ProgramError("cadquery extraction failed: %s" % result.reason)
    return result.code


def _cq_params(defs):
    """CadQuery parameter definitions -> the unified cross-language schema."""
    from harnesscad.domain.programs.params import param_schema as m

    return m.normalize("cadquery", list(defs))


def _cq_diagnostics(text: str):
    from harnesscad.domain.programs.validate import diagnostics as m

    return [Finding(CADQUERY, d.severity, "toolchain", d.message, d.line,
                    _PKG + "validate.diagnostics")
            for d in m.parse("cadquery", text)]


def _cq_emit(ops: Sequence[Any]) -> str:
    """The neutral op IR -> CadQuery source (via ``ast.cadquery``).

    Sketch ops become the workplane primitive that starts a chain; ``extrude``
    closes it; booleans become ``.cut`` / ``.union`` / ``.intersect`` on the
    named results. Anything the CadQuery subset cannot express raises
    :class:`Unsupported` -- it is never silently dropped.
    """
    from harnesscad.domain.programs.ast import cadquery as m

    calls = _as_calls(ops)
    errors = validate_ops(calls)
    if errors:
        raise ProgramError("neutral op program is invalid: " + "; ".join(errors))

    sketches: Dict[str, Tuple[str, List[Any]]] = {}   # result -> (plane, calls)
    statements: List[Any] = []
    last: Optional[str] = None
    for call in calls:
        op = call.operation
        result = call.result or ("v%d" % (len(statements) + 1))
        if op == "rectangle":
            center, w, h = _op_args(call, "center", "width", "height")
            plane = str(call.args.get("plane", "xy")).upper()
            sketches[result] = (plane, [m.Call("center", [float(center[0]),
                                                          float(center[1])]),
                                        m.Call("rect", [float(w), float(h)])])
        elif op == "circle":
            center, r = _op_args(call, "center", "radius")
            plane = str(call.args.get("plane", "xy")).upper()
            sketches[result] = (plane, [m.Call("center", [float(center[0]),
                                                          float(center[1])]),
                                        m.Call("circle", [float(r)])])
        elif op == "polygon":
            center, r, sides = _op_args(call, "center", "radius", "sides")
            plane = str(call.args.get("plane", "xy")).upper()
            sketches[result] = (plane, [m.Call("center", [float(center[0]),
                                                          float(center[1])]),
                                        m.Call("polygon", [int(sides), float(r)])])
        elif op == "extrude":
            profile, height = _op_args(call, "profile", "height")
            if profile not in sketches:
                raise ProgramError("extrude names an unknown profile %r" % (profile,))
            plane, chain_calls = sketches[profile]
            chain = m.Chain(m.Workplane(plane),
                            chain_calls + [m.Call("extrude", [float(height)])])
            statements.append(m.Assign(result, chain))
            last = result
        elif op in ("union", "subtract", "intersect"):
            this, that = _op_args(call, "this", "that")
            method = {"union": "union", "subtract": "cut", "intersect": "intersect"}[op]
            chain = m.Chain(m.VarRef(str(this)), [m.Call(method, [m.VarRef(str(that))])])
            statements.append(m.Assign(result, chain))
            last = result
        else:
            raise Unsupported(
                "the CadQuery emitter has no lowering for the neutral op %r "
                "(supported: rectangle, circle, polygon, extrude, union, "
                "subtract, intersect)" % op)
    if not statements:
        raise ProgramError("no CadQuery statements were produced (no solid ops)")
    return m.serialize(m.CqProgram(statements, last))


# --------------------------------------------------------------------------- #
# OpenSCAD
# --------------------------------------------------------------------------- #
def _scad_parse(source: str):
    from harnesscad.domain.programs.ast import openscad as m

    return m.parse(source)


def _scad_serialize(tree) -> str:
    from harnesscad.domain.programs.ast import openscad as m

    return m.unparse(tree)


def _scad_validate(tree) -> List[Finding]:
    """Validate an OpenSCAD AST by unparsing it and running the static checker."""
    from harnesscad.domain.programs.ast import openscad as ast_m

    return _scad_validate_source(ast_m.unparse(tree))


def _scad_validate_source(source: str) -> List[Finding]:
    from harnesscad.domain.programs.validate import openscad_check as m

    return [Finding(OPENSCAD, issue.severity, issue.code, issue.message,
                    issue.line, _PKG + "validate.openscad_check")
            for issue in m.check(source)]


def _scad_review(source: str, *, reference: Optional[str] = None, seed: int = 0
                 ) -> List[Finding]:
    """CADReview: segment into blocks, locate the wrong one, name the error type.

    Without a ``reference`` there is nothing to review *against*, so the review
    degrades honestly to the static check plus the block segmentation.
    """
    from harnesscad.domain.programs.review import blocks as bl

    out: List[Finding] = _scad_validate_source(source)
    segments = bl.segment(source)
    out.append(_info(OPENSCAD, "blocks", "%d reviewable block(s)" % len(segments),
                     None, _PKG + "review.blocks"))
    if reference is None:
        return out

    from harnesscad.domain.programs.review import report as rp

    built = rp.build_report(source, reference, seed=int(seed))
    if built.error_type is None:
        out.append(_info(OPENSCAD, "review-clean",
                         "no error detected against the reference", None,
                         _PKG + "review.report"))
        return out
    label = getattr(built.error_type, "label", str(built.error_type))
    out.append(_warn(OPENSCAD, "review-" + str(getattr(built.error_type, "id", "error")),
                     "%s in block %s: %s" % (label, built.block_id, built.feedback),
                     None, _PKG + "review.report"))
    for suggestion in built.suggestions:
        out.append(_info(OPENSCAD, "review-fix", suggestion.instruction, None,
                         _PKG + "review.correct"))
    return out


def _scad_extract(reply: str, *, strict: bool = False) -> str:
    from harnesscad.domain.programs.extract import openscad_extract as m

    return m.normalise_scad(m.extract_scad(reply, strict=strict))


def _scad_params(source: str):
    """OpenSCAD Customizer declarations -> the unified cross-language schema."""
    from harnesscad.domain.programs.params import openscad_customizer as oc
    from harnesscad.domain.programs.params import param_schema as ps

    defs = []
    for p in oc.parse_parameters(source):
        raw: Dict[str, Any] = {"name": p.name, "type": p.type, "value": p.value,
                               "caption": p.display_name}
        raw.update({k: v for k, v in (p.range or {}).items()})
        if p.options:
            raw["options"] = [{"name": o.label, "value": o.value} for o in p.options]
        defs.append(raw)
    return ps.normalize("openscad", defs)


def _scad_annotate(source: str, *, labels=None, block_points=None,
                   point_labels=None, evidence=None, tbc: str = "TBC") -> str:
    """CADTalk: segment into commentable blocks and comment each one.

    ``labels`` may be given directly, derived from a point-cloud label transfer
    (``block_points`` + ``point_labels``), or aggregated from multi-view
    ``evidence`` -- three real sources, never invented.
    """
    from harnesscad.domain.programs.annotate import block_parser as bp

    if labels is None and block_points is not None and point_labels is not None:
        from harnesscad.domain.programs.annotate import label_transfer as lt

        labels = lt.majority_label(block_points, point_labels)
    if labels is None and evidence is not None:
        from harnesscad.domain.programs.annotate import voting as vo

        labels = vo.vote(evidence)
    return bp.annotate(source, labels, tbc=tbc)


def _scad_repair(source: str) -> str:
    from harnesscad.domain.programs.review import syntax_repair as m

    return m.repair(source).code


def _scad_quantize(source: str, *, lo: float = 0.0, hi: float = 256.0) -> str:
    from harnesscad.domain.programs.review import quantize as m

    return m.quantize_program(source, lo=float(lo), hi=float(hi))


def _scad_diagnostics(text: str):
    from harnesscad.domain.programs.validate import diagnostics as m

    return [Finding(OPENSCAD, d.severity, "toolchain", d.message, d.line,
                    _PKG + "validate.diagnostics")
            for d in m.parse("openscad", text)]


def _scad_bom(root, *, headers=(), csv: bool = False) -> str:
    """Bill of materials from a SCAD object tree (``emit.openscad_emit`` nodes)."""
    from harnesscad.domain.programs.runtime import bill_of_materials as m

    return m.bill_of_materials(root, headers=tuple(headers), csv=bool(csv))


def _scad_emit(ops: Sequence[Any], *, modularize: bool = False) -> str:
    """The neutral op IR -> OpenSCAD source (via ``emit.openscad_emit``)."""
    from harnesscad.domain.programs.emit import openscad_emit as m

    calls = _as_calls(ops)
    errors = validate_ops(calls)
    if errors:
        raise ProgramError("neutral op program is invalid: " + "; ".join(errors))

    profiles: Dict[str, Any] = {}
    solids: Dict[str, Any] = {}
    last: Optional[str] = None
    for call in calls:
        op = call.operation
        result = call.result or ("v%d" % (len(solids) + len(profiles) + 1))
        if op == "rectangle":
            center, w, h = _op_args(call, "center", "width", "height")
            node = m.translate([float(center[0]), float(center[1])])
            node.add(m.square([float(w), float(h)], center=True))
            profiles[result] = node
        elif op == "circle":
            center, r = _op_args(call, "center", "radius")
            node = m.translate([float(center[0]), float(center[1])])
            node.add(m.circle(r=float(r)))
            profiles[result] = node
        elif op == "polygon":
            center, r, sides = _op_args(call, "center", "radius", "sides")
            node = m.translate([float(center[0]), float(center[1])])
            node.add(m.circle(r=float(r), segments=int(sides)))
            profiles[result] = node
        elif op == "extrude":
            profile, height = _op_args(call, "profile", "height")
            if profile not in profiles:
                raise ProgramError("extrude names an unknown profile %r" % (profile,))
            node = m.linear_extrude(height=float(height))
            node.add(profiles[profile])
            solids[result] = node
            last = result
        elif op in ("union", "subtract", "intersect"):
            this, that = _op_args(call, "this", "that")
            for name in (this, that):
                if name not in solids:
                    raise ProgramError("boolean names an unknown solid %r" % (name,))
            node = {"union": m.union, "subtract": m.difference,
                    "intersect": m.intersection}[op]()
            node.add(solids[str(this)])
            node.add(solids[str(that)])
            solids[result] = node
            last = result
        else:
            raise Unsupported(
                "the OpenSCAD emitter has no lowering for the neutral op %r "
                "(supported: rectangle, circle, polygon, extrude, union, "
                "subtract, intersect)" % op)
    if last is None:
        raise ProgramError("no OpenSCAD solid was produced (no solid ops)")
    if modularize:
        raise Unsupported(
            "module CSE runs on a csg_algebra term, not on a ScadNode tree; "
            "call emit_modules(term) instead")
    return m.scad_render(solids[last])


def emit_modules(term, *, min_size: int = 2, min_count: int = 2) -> str:
    """OpenSCAD source with shared subtrees hoisted into ``module mdl_N`` (CSE).

    Operates on a :mod:`geometry.sdf.csg_algebra` term -- the shared CSG term
    type -- because that is what content-addressed CSE needs (a ScadNode tree is
    already flattened for rendering).
    """
    from harnesscad.domain.programs.emit import module_cse as m

    builder = m.ModuleBuilder()
    body = m.auto_modularize(term, builder, min_size=int(min_size),
                             min_count=int(min_count))
    return m.render(body, builder)


def emit_primitives(primitives, *, group_consecutive: bool = True,
                    category: Optional[str] = None):
    """CADTalk machine-made program synthesis: labelled primitives -> OpenSCAD."""
    from harnesscad.domain.programs.emit import primitive_program as m

    return m.synthesize(list(primitives), group_consecutive=group_consecutive,
                        category=category)


def inject_error(source: str, error_type, *, seed: int = 0):
    """CADReview error generator: deterministically inject one of the eight errors."""
    from harnesscad.domain.programs.review import errorgen as m

    return m.inject(source, error_type, seed=int(seed))


def injectable_errors(source: str):
    from harnesscad.domain.programs.review import errorgen as m

    return m.injectable_types(source)


def annotation_score(predicted, ground_truth, synonyms=None):
    """CADTalk block accuracy + semantic IoU for an annotation run."""
    from harnesscad.domain.programs.annotate import metrics as m

    return m.evaluate(dict(predicted), dict(ground_truth), synonyms or {})


# --------------------------------------------------------------------------- #
# OpenECAD
# --------------------------------------------------------------------------- #
def _oe_parse(source: str):
    from harnesscad.domain.programs.ast import openecad as m

    return m.parse(source)


def _oe_serialize(tree) -> str:
    from harnesscad.domain.programs.ast import openecad as m

    return m.emit(tree)


def _oe_validate(tree) -> List[Finding]:
    """Loop closure + profile validity for the sketches in an OpenECAD program."""
    from harnesscad.domain.programs.validate import openecad_validity as m

    out: List[Finding] = []
    if not m.program_profiles_valid(tree):
        out.append(_err(OPENECAD, "open-loop",
                        "at least one sketch profile does not close",
                        source=_PKG + "validate.openecad_validity"))
    for key in m.loops_from_program(tree):
        pass
    return out


def _oe_params(tree, *, target: Optional[str] = None, **edits):
    """Editability: rename / reparametrize an OpenECAD script's variables."""
    from harnesscad.domain.programs.params import openecad_edit as m

    if target is None:
        return {"variables": m.variable_names(tree)}
    return m.reparametrize(tree, target, **edits)


# --------------------------------------------------------------------------- #
# Typed CSG (AngelCAD-style: dimension-checked)
# --------------------------------------------------------------------------- #
def _tcsg_validate(tree) -> List[Finding]:
    from harnesscad.domain.programs.ast import typed_csg as m

    return [Finding(TYPED_CSG, d.severity if hasattr(d, "severity") else ERROR,
                    d.key(), str(d), None, _PKG + "ast.typed_csg")
            for d in m.check(tree)]


def _tcsg_emit(ops: Sequence[Any]):
    """The neutral op IR -> a TYPE-CHECKED CSG tree (2D profiles must be extruded)."""
    from harnesscad.domain.programs.ast import typed_csg as m

    calls = _as_calls(ops)
    errors = validate_ops(calls)
    if errors:
        raise ProgramError("neutral op program is invalid: " + "; ".join(errors))

    profiles: Dict[str, Any] = {}
    solids: Dict[str, Any] = {}
    last: Optional[str] = None
    for call in calls:
        op = call.operation
        result = call.result or ("v%d" % (len(solids) + len(profiles) + 1))
        if op == "rectangle":
            center, w, h = _op_args(call, "center", "width", "height")
            node = m.rectangle(float(w), float(h), center=True)
            if float(center[0]) or float(center[1]):
                node = m.transform(m.translate(float(center[0]), float(center[1])), node)
            profiles[result] = node
        elif op == "circle":
            center, r = _op_args(call, "center", "radius")
            node = m.circle(float(r))
            if float(center[0]) or float(center[1]):
                node = m.transform(m.translate(float(center[0]), float(center[1])), node)
            profiles[result] = node
        elif op == "extrude":
            profile, height = _op_args(call, "profile", "height")
            if profile not in profiles:
                raise ProgramError("extrude names an unknown profile %r" % (profile,))
            solids[result] = m.linear_extrude(profiles[profile], float(height))
            last = result
        else:
            raise Unsupported(
                "the typed-CSG emitter has no lowering for the neutral op %r "
                "(supported: rectangle, circle, extrude)" % op)
    if last is None:
        raise ProgramError("no typed-CSG solid was produced")
    tree = solids[last]
    m.type_check(tree)          # dimension checking is the whole point
    return tree


# --------------------------------------------------------------------------- #
# OpenMC-style CSG scripts
# --------------------------------------------------------------------------- #
def _mc_parse(source: str):
    from harnesscad.domain.programs.ast import openmc_csg as m

    return m.parse(source)


def _mc_serialize(tree) -> str:
    from harnesscad.domain.programs.ast import openmc_csg as m

    model = getattr(tree, "model", tree)
    return m.serialize(model)


def _mc_review(source: str, **_kw) -> List[Finding]:
    from harnesscad.domain.programs.ast import openmc_csg as m

    fixed = m.correct_syntax(source)
    if fixed != source:
        return [_warn(OPENMC_CSG, "syntax-corrected",
                      "the script needed syntax correction to parse", None,
                      _PKG + "ast.openmc_csg")]
    return [_info(OPENMC_CSG, "syntax-ok", "the script parses as written", None,
                  _PKG + "ast.openmc_csg")]


# --------------------------------------------------------------------------- #
# FreeCAD expression language
# --------------------------------------------------------------------------- #
def _fc_parse(source: str):
    from harnesscad.domain.programs.expressions import freecad_expressions as m

    return m.Expression(source, m.parse(source))


def _fc_validate(tree) -> List[Finding]:
    """An expression is valid when every reference it makes is nameable."""
    from harnesscad.domain.programs.expressions import freecad_expressions as m

    out: List[Finding] = []
    expr = tree if isinstance(tree, m.Expression) else m.Expression("", tree)
    for key in expr.reference_keys():
        out.append(_info(FREECAD_EXPR, "reference", "references %s" % (key,), None,
                         _PKG + "expressions.freecad_expressions"))
    return out


def _fc_review(source: str, **_kw) -> List[Finding]:
    """Classify the expression into the paper's C1..C5 formative categories."""
    from harnesscad.domain.programs.expressions import classify as m

    result = m.classify_expression(source)
    return [_info(FREECAD_EXPR, "category-" + result.category.label(), result.reason,
                  None, _PKG + "expressions.classify")]


def _fc_handles(root, *, node_id: str, handle: str):
    """Derive a parametric handle position by walking a CSG tree (paper Sec. 4.1-4.2)."""
    from harnesscad.domain.programs.expressions import handle_position as m

    vec = m.derive_position(root, node_id, handle)
    return {"position": vec, "statement": m.translate_statement(vec)}


# --------------------------------------------------------------------------- #
# Blender bpy scripts
# --------------------------------------------------------------------------- #
def _bpy_validate_source(source: str) -> List[Finding]:
    from harnesscad.domain.programs.validate import bpy_script as m

    check = m.check_syntax(source)
    out: List[Finding] = []
    if not check.ok:
        out.append(_err(BPY, "syntax-error", check.error or "syntax error",
                        check.lineno, _PKG + "validate.bpy_script"))
        return out
    coverage = m.vocabulary_coverage(source)
    out.append(_info(BPY, "vocabulary-coverage", "recognised bpy vocabulary: %r"
                     % (coverage,), None, _PKG + "validate.bpy_script"))
    return out


def _bpy_review(source: str, **_kw) -> List[Finding]:
    return _bpy_validate_source(source)


# --------------------------------------------------------------------------- #
# The language table.
#
# `modules` are the dotted modules the language dispatches to -- discovery drops
# a language whose modules are not in the capability index.
# --------------------------------------------------------------------------- #
_LANGUAGE_TABLE: Tuple[Tuple[str, str, Tuple[str, ...], Dict[str, Callable]], ...] = (
    (CADQUERY,
     "CadQuery: a fluent Python solid-modelling API. Structured AST + "
     "execution-free validity, Workplane state machine, API profiling, LLM-reply "
     "cleaning, and the unified parameter schema.",
     ("ast.cadquery", "validate.cadquery_validity", "validate.cadquery_workplane",
      "validate.cadquery_analysis", "validate.cadquery_api_profile",
      "extract.cadquery_clean", "params.param_schema", "validate.diagnostics",
      "validate.operation_schema"),
     {"parse": _cq_parse, "serialize": _cq_serialize, "validate": _cq_validate,
      "validate_source": _cq_validate_source, "review": _cq_review,
      "emit": _cq_emit, "extract": _cq_extract, "params": _cq_params,
      "diagnostics": _cq_diagnostics}),

    (OPENSCAD,
     "OpenSCAD: a declarative CSG scripting language. Tokenizer + "
     "recursive-descent parser, static compile gate, SolidPython-style emitter, "
     "CADReview block review, Customizer parameters, CADTalk annotation.",
     ("ast.openscad", "validate.openscad_check", "emit.openscad_emit",
      "review.blocks", "review.report", "review.syntax_repair", "review.quantize",
      "review.errorgen", "extract.openscad_extract", "params.openscad_customizer",
      "params.param_schema", "annotate.block_parser", "annotate.label_transfer",
      "annotate.voting", "runtime.bill_of_materials", "validate.diagnostics",
      "validate.operation_schema"),
     {"parse": _scad_parse, "serialize": _scad_serialize, "validate": _scad_validate,
      "validate_source": _scad_validate_source, "review": _scad_review,
      "emit": _scad_emit, "extract": _scad_extract, "params": _scad_params,
      "annotate": _scad_annotate, "repair": _scad_repair, "quantize": _scad_quantize,
      "diagnostics": _scad_diagnostics, "bom": _scad_bom}),

    (OPENECAD,
     "OpenECAD: an editable CAD-script format (Yuan, Shi & Huang 2024). Parser + "
     "emitter, loop-closure/profile validity, and variable reparametrisation.",
     ("ast.openecad", "validate.openecad_validity", "params.openecad_edit"),
     {"parse": _oe_parse, "serialize": _oe_serialize, "validate": _oe_validate,
      "params": _oe_params}),

    (TYPED_CSG,
     "Typed CSG (AngelCAD-style): a DIMENSION-CHECKED CSG language -- a 2D profile "
     "cannot be unioned with a solid. It has an emitter and a type checker, and "
     "deliberately no source parser.",
     ("ast.typed_csg", "validate.operation_schema"),
     {"validate": _tcsg_validate, "emit": _tcsg_emit}),

    (OPENMC_CSG,
     "OpenMC-style CSG scripts: half-space surfaces + cell regions. Parse, "
     "serialise, and a syntax-correction review pass.",
     ("ast.openmc_csg",),
     {"parse": _mc_parse, "serialize": _mc_serialize, "review": _mc_review}),

    (FREECAD_EXPR,
     "The FreeCAD expression language: a units-aware arithmetic expression with "
     "object references. Parse + evaluate, C1..C5 formative classification, and "
     "handle-position derivation over a CSG tree.",
     ("expressions.freecad_expressions", "expressions.classify",
      "expressions.handle_position"),
     {"parse": _fc_parse, "validate": _fc_validate, "review": _fc_review,
      "handles": _fc_handles}),

    (BPY,
     "Blender bpy scripts (BlenderLLM): static syntax + primitive/transform "
     "vocabulary coverage. Analysis only -- nothing is executed.",
     ("validate.bpy_script",),
     {"validate": _bpy_validate_source, "validate_source": _bpy_validate_source,
      "review": _bpy_review}),
)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
_LANGS: Optional[Dict[str, Language]] = None
_UNADAPTED: Tuple[str, ...] = ()

#: Capabilities that live on the surface rather than on one language.
_SURFACE_MODULES: Tuple[str, ...] = (
    "emit.module_cse", "emit.primitive_program", "annotate.metrics",
)


def _build_languages() -> Dict[str, Language]:
    """Join the language table onto the AST capability index (package='programs')."""
    global _UNADAPTED
    entries = {e.dotted: e for e in capability_registry.find(package=PROGRAMS_PACKAGE)}
    adapted = set(_PKG + m for m in _SURFACE_MODULES)
    out: Dict[str, Language] = {}
    for name, description, mods, caps in _LANGUAGE_TABLE:
        dotted = tuple(_PKG + m for m in mods)
        missing = [d for d in dotted if d not in entries]
        if missing:
            continue                      # a language is only offered if it exists
        for cap in caps:
            if cap not in CAPABILITIES and cap != "validate_source":
                raise ValueError("language %r declares unknown capability %r"
                                 % (name, cap))
        adapted.update(dotted)
        out[name] = Language(name=name, description=description, modules=dotted,
                             caps=dict(caps))
    _UNADAPTED = tuple(sorted(d for d in entries if d not in adapted))
    return out


def _all() -> Dict[str, Language]:
    global _LANGS
    if _LANGS is None:
        _LANGS = _build_languages()
    return _LANGS


def languages() -> Tuple[str, ...]:
    """Every language whose modules are actually in the tree."""
    return tuple(sorted(_all()))


def language(name: str) -> Language:
    _check_lang(name)
    try:
        return _all()[name]
    except KeyError:
        raise UnknownLanguage(
            "language %r has no discoverable implementation (its modules are not "
            "indexed)" % name) from None


def capabilities(name: str) -> Tuple[str, ...]:
    return tuple(sorted(c for c in language(name).caps if c in CAPABILITIES))


def supports(name: str, capability: str) -> bool:
    return language(name).supports(capability)


def _cap(name: str, capability: str) -> Callable:
    lang = language(name)
    fn = lang.caps.get(capability)
    if fn is None:
        raise Unsupported(
            "the %r language does not implement %r (it implements: %s); it is NOT "
            "delegated to another language" % (name, capability,
                                               ", ".join(capabilities(name))))
    return fn


def unadapted() -> Tuple[str, ...]:
    """Program modules the index knows but no language/surface capability binds."""
    _all()
    return _UNADAPTED


# --------------------------------------------------------------------------- #
# The surface. Every entry point takes an explicit language.
# --------------------------------------------------------------------------- #
def parse(source: str, lang: str) -> Program:
    """Parse ``source`` AS ``lang``. The language is never inferred from the text."""
    tree = _cap(lang, "parse")(source)
    return Program(lang=lang, tree=tree, source=source)


def validate(program: Any, lang: Optional[str] = None) -> Tuple[Finding, ...]:
    """Validate a parsed :class:`Program` (or raw source with ``lang=``).

    Handing a :class:`Program` parsed as one language to another language's
    validator raises :class:`LanguageMismatch` -- an OpenSCAD checker run over a
    CadQuery tree does not produce weaker findings, it produces wrong ones.
    """
    if isinstance(program, Program):
        if lang is not None and lang != program.lang:
            raise LanguageMismatch(
                "this program was parsed as %r but the %r validator was requested; "
                "code-CAD languages are dispatched, never blended"
                % (program.lang, lang))
        return tuple(_cap(program.lang, "validate")(program.tree) or ())
    if lang is None:
        raise ProgramError("validate() needs a parsed Program, or raw source plus lang=")
    fn = language(lang).caps.get("validate_source")
    if fn is None:
        return tuple(_cap(lang, "validate")(parse(program, lang).tree) or ())
    return tuple(fn(program) or ())


def serialize(program: Program, lang: Optional[str] = None) -> str:
    """A parsed program -> source, in ITS language."""
    if not isinstance(program, Program):
        raise ProgramError("serialize() needs a parsed Program")
    if lang is not None and lang != program.lang:
        raise LanguageMismatch(
            "this program is %r; the %r serialiser will not be used on it"
            % (program.lang, lang))
    return _cap(program.lang, "serialize")(program.tree)


def emit(ops: Sequence[Any], lang: str, **kwargs: Any):
    """The neutral op IR -> ``lang`` source (or, for typed CSG, a checked tree)."""
    return _cap(lang, "emit")(ops, **kwargs)


def review(source: str, lang: str, **kwargs: Any) -> Tuple[Finding, ...]:
    """Static review of somebody else's source, in ``lang``. Nothing is executed."""
    return tuple(_cap(lang, "review")(source, **kwargs) or ())


def extract(reply: str, lang: str, **kwargs: Any) -> str:
    """Recover runnable ``lang`` source from a raw LLM reply."""
    return _cap(lang, "extract")(reply, **kwargs)


def params(source_or_defs: Any, lang: str, **kwargs: Any):
    """The parameter schema of a program (or of a raw parameter manifest)."""
    return _cap(lang, "params")(source_or_defs, **kwargs)


def annotate(source: str, lang: str, **kwargs: Any) -> str:
    """CADTalk-style annotation: comment every commentable block."""
    return _cap(lang, "annotate")(source, **kwargs)


def repair(source: str, lang: str, **kwargs: Any) -> str:
    """Pattern-template syntax repair of generated source."""
    return _cap(lang, "repair")(source, **kwargs)


def quantize(source: str, lang: str, **kwargs: Any) -> str:
    """Snap the spatial literals of a program onto the quantisation grid."""
    return _cap(lang, "quantize")(source, **kwargs)


def diagnostics(text: str, lang: str) -> Tuple[Finding, ...]:
    """Parse a toolchain's error output into uniform findings."""
    return tuple(_cap(lang, "diagnostics")(text) or ())


def bom(root: Any, lang: str = OPENSCAD, **kwargs: Any) -> str:
    """Bill of materials from an emitted object tree."""
    return _cap(lang, "bom")(root, **kwargs)


def handles(root: Any, lang: str = FREECAD_EXPR, **kwargs: Any):
    """Derive a parametric handle position by walking a CSG tree."""
    return _cap(lang, "handles")(root, **kwargs)


# --------------------------------------------------------------------------- #
# CLI (wired into core.cli as `harnesscad program`)
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list the languages and what each can do")
    parser.add_argument("--lang", default=None, choices=list(LANGUAGES),
                        help="the language -- MANDATORY for any real work, never guessed")
    parser.add_argument("--parse", default=None, metavar="FILE",
                        help="parse FILE as --lang and print the round-tripped source")
    parser.add_argument("--validate", default=None, metavar="FILE",
                        help="validate FILE as --lang")
    parser.add_argument("--review", default=None, metavar="FILE",
                        help="statically review FILE as --lang")
    parser.add_argument("--reference", default=None, metavar="FILE",
                        help="the reference program to review against (openscad)")
    parser.add_argument("--emit", default=None, metavar="OPS.JSON",
                        help="emit a neutral op program (JSON array) as --lang source")
    parser.add_argument("--operations", action="store_true",
                        help="list the neutral operation vocabulary emit() accepts")
    parser.add_argument("--unadapted", action="store_true",
                        help="list program modules with no capability binding yet")
    parser.add_argument("--json", action="store_true", help="emit findings as JSON")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _print_findings(findings: Sequence[Finding], as_json: bool) -> int:
    if as_json:
        print(json.dumps([f.to_dict() for f in findings], sort_keys=True, indent=2))
    else:
        for f in findings:
            where = f" line {f.line}" if f.line else ""
            print(f"[{f.severity}] {f.code}: {f.message}{where}")
            if f.source:
                print(f"    {f.source}")
        if not findings:
            print("no findings")
    return 1 if any(f.severity == ERROR for f in findings) else 0


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "operations", False):
        for name in operations():
            print(name)
        return 0

    if getattr(args, "unadapted", False):
        for dotted in unadapted():
            print(dotted)
        print(f"-- {len(unadapted())} program modules without a capability binding")
        return 0

    action = next((a for a in ("parse", "validate", "review", "emit")
                   if getattr(args, a, None)), None)

    if getattr(args, "list", False) or action is None:
        for name in languages():
            lang = language(name)
            print(f"{name}")
            print(f"    {lang.description}")
            print(f"    capabilities: {', '.join(capabilities(name))}")
        print(f"-- {len(languages())} languages / {len(unadapted())} program "
              f"modules unbound")
        return 0

    if not getattr(args, "lang", None):
        print("error: --lang is mandatory and is never guessed "
              f"(one of: {', '.join(LANGUAGES)})", file=sys.stderr)
        return 2

    try:
        if action == "parse":
            program = parse(_read(args.parse), args.lang)
            print(serialize(program))
            return 0
        if action == "validate":
            return _print_findings(validate(_read(args.validate), lang=args.lang),
                                   args.json)
        if action == "review":
            kwargs: Dict[str, Any] = {}
            if getattr(args, "reference", None):
                kwargs["reference"] = _read(args.reference)
            return _print_findings(review(_read(args.review), args.lang, **kwargs),
                                   args.json)
        if action == "emit":
            ops = json.loads(_read(args.emit))
            if not isinstance(ops, list):
                print("error: --emit needs a JSON array of ops", file=sys.stderr)
                return 2
            out = emit(ops, args.lang)
            print(out if isinstance(out, str) else repr(out))
            return 0
    except ProgramError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (SyntaxError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad program",
        description="language-dispatched code-CAD program surface "
                    "(parse / validate / emit / review)")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
