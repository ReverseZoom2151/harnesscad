"""Deterministic error-to-hint-to-autofix rules for generated CAD code.

The module applies a robustness rule:
*deterministic* pre-execution validator plus an error-classification table that
turns a raw kernel exception into (a) an actionable hint and (b), where the fix
is unambiguous, a rewritten source string the loop can re-run **without another
model call**. The module has no CAD-kernel imports and is fully testable in
isolation.

That property is exactly what makes it a harness piece: the repair is a pure
function of ``(exception_type, exception_message, source)``, so it is
reproducible, unit-testable, and free. It complements the harness's
model-driven refine loops (compiler_refine, the CADSmith dual loop) by handling
the large class of failures that need no reasoning at all -- a missing ``Part.``
prefix, an in-place ``translate`` whose result was assigned, a boolean op whose
new shape was dropped.

The rule set is generalized over the two CAD dialects the harness targets:

* **CadQuery** (``cq.Workplane`` chains): method-not-standalone, wrong translate
  arity (tuple vs args), extrude-before-2D, cylinder ``(height, radius)`` order.
* **FreeCAD Part** scripting: ``Part.makeBox`` needs the ``Part.`` prefix,
  ``FreeCAD.Vector`` capital V, boolean ops (cut/fuse/common) return new shapes,
  ``translate`` returns None, ``makePipe`` needs a Wire, non-existent
  ``Part.makeEllipse`` / ``.sweep()``.

Each rule yields a :class:`RepairSuggestion` carrying the hint and, when safe, a
``fixed_code`` that differs from the input. A caller applies ``fixed_code`` and
re-runs; if it still fails, the next rule (or a model call) takes over.

stdlib-only (``re``, ``dataclasses``), deterministic, absolute imports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

__all__ = [
    "RepairSuggestion",
    "precheck_syntax",
    "suggest_repair",
    "apply_first_repair",
]

_BOOL_OPS = r"(?:cut|fuse|common)"
_MAKE_FNS = ("makeBox", "makeCylinder", "makeCone", "makeSphere", "makeTorus")
_CQ_METHODS = (
    "box", "cylinder", "cone", "sphere", "torus", "extrude", "cut", "union",
    "intersect", "translate", "rotate", "mirror", "circle", "rect", "polygon",
    "workplane", "solid", "val",
)


@dataclass(frozen=True)
class RepairSuggestion:
    """One repair proposal.

    ``rule_id`` names the rule that fired; ``hint`` is human/agent guidance;
    ``fixed_code`` is a rewritten source ready to re-run, or ``None`` when the
    fix is not mechanical (hint only).
    """

    rule_id: str
    hint: str
    fixed_code: Optional[str] = None

    @property
    def has_autofix(self) -> bool:
        return self.fixed_code is not None


def precheck_syntax(code: str) -> Tuple[bool, str]:
    """Compile-check ``code`` without executing it.

    Returns ``(ok, message)``; on a SyntaxError the message points at the line
    and offset with a caret, so a repair loop can report *where* before spending
    a kernel run on code that cannot parse.
    """
    try:
        compile(code, "<cad_code>", "exec")
        return True, ""
    except SyntaxError as exc:
        line_info = f"line {exc.lineno}" if exc.lineno else "unknown line"
        detail = f": {exc.msg}" if exc.msg else ""
        pointer = ""
        if exc.text and exc.offset:
            pointer = f"\n  {exc.text.rstrip()}\n  {' ' * (max(exc.offset - 1, 0))}^"
        return False, f"SyntaxError at {line_info}{detail}{pointer}"


# ---------------------------------------------------------------------------
# Individual rules. Each takes (error_type, error_str, code) and returns a
# RepairSuggestion or None. Order matters: the first match wins.
# ---------------------------------------------------------------------------

def _rule_missing_part_prefix(etype: str, estr: str, code: str) -> Optional[RepairSuggestion]:
    if etype != "NameError":
        return None
    for fn in _MAKE_FNS:
        if fn in estr:
            pat = re.compile(r"(?<!\w)(?<!\.)" + fn + r"\s*\(")
            fixed = pat.sub(f"Part.{fn}(", code)
            if fixed != code:
                return RepairSuggestion(
                    "missing_part_prefix",
                    f"'{fn}' is not a standalone function; call Part.{fn}(...).",
                    fixed,
                )
            return RepairSuggestion(
                "missing_part_prefix",
                f"'{fn}' needs the Part. prefix (Part.{fn}). "
                f"Or use cq.Workplane('XY').{fn[4:].lower()}(...).",
            )
    return None


def _rule_cq_method_not_standalone(etype: str, estr: str, code: str) -> Optional[RepairSuggestion]:
    if etype != "NameError":
        return None
    for m in _CQ_METHODS:
        if f"'{m}'" in estr or f'"{m}"' in estr:
            return RepairSuggestion(
                "cq_method_not_standalone",
                f"'{m}' is a method on cq.Workplane, not a standalone function. "
                f"Use cq.Workplane('XY').{m}(...).",
            )
    return None


def _rule_freecad_vector_case(etype: str, estr: str, code: str) -> Optional[RepairSuggestion]:
    if etype != "AttributeError" or "vector" not in estr.lower():
        return None
    pat = re.compile(r"FreeCAD\.vector\s*\(", re.IGNORECASE)
    fixed = pat.sub("FreeCAD.Vector(", code)
    if fixed != code:
        return RepairSuggestion(
            "freecad_vector_case",
            "FreeCAD.Vector has a capital V. Fixed FreeCAD.vector -> FreeCAD.Vector.",
            fixed,
        )
    return None


def _rule_inplace_translate_assigned(etype: str, estr: str, code: str) -> Optional[RepairSuggestion]:
    # `shape = shape.translate(...)` -- translate() mutates in place, returns None.
    if etype != "AttributeError" or "'NoneType'" not in estr:
        return None
    pat = re.compile(r"^(\s*)(\w+)\s*=\s*\2\.translate\s*\(", re.MULTILINE)
    if pat.search(code):
        fixed = pat.sub(r"\1\2.translate(", code)
        return RepairSuggestion(
            "inplace_translate_assigned",
            "translate() modifies in place and returns None; removed the assignment.",
            fixed,
        )
    return None


def _rule_boolean_result_dropped(etype: str, estr: str, code: str) -> Optional[RepairSuggestion]:
    # `body.cut(hole)` on its own line -- boolean ops return a NEW shape.
    if etype != "AttributeError" or "'NoneType'" not in estr:
        return None
    pat = re.compile(r"^(\s*)(\w+)\.(" + _BOOL_OPS + r")\s*\((.+?)\)\s*$", re.MULTILINE)
    if pat.search(code):
        def _fix(m: "re.Match[str]") -> str:
            indent, var, op, args = m.group(1), m.group(2), m.group(3), m.group(4)
            return f"{indent}{var} = {var}.{op}({args})"

        fixed = pat.sub(_fix, code)
        if fixed != code:
            return RepairSuggestion(
                "boolean_result_dropped",
                "Boolean ops (cut/fuse/common) return NEW shapes; assigned the result.",
                fixed,
            )
    return None


def _rule_translate_arity(etype: str, estr: str, code: str) -> Optional[RepairSuggestion]:
    if etype != "TypeError" or "translate" not in code:
        return None
    is_cq = bool(re.search(r"\bcq\.|Workplane\(|cq_show", code))
    pat = re.compile(r"\.translate\s*\(\s*([^)]+)\)")
    m = pat.search(code)
    if not m:
        return None
    inner = m.group(1)
    if is_cq:
        if "(" not in inner:
            fixed = pat.sub(f".translate(({inner.strip()}))", code)
            return RepairSuggestion(
                "cq_translate_tuple",
                "cq translate() takes a tuple (x, y, z); wrapped the arguments.",
                fixed,
            )
        return RepairSuggestion(
            "cq_translate_tuple",
            "cq.Workplane.translate() takes a tuple: .translate((x, y, z)).",
        )
    if "Vector" not in inner and "FreeCAD" not in inner:
        fixed = pat.sub(f".translate(FreeCAD.Vector({inner.strip()}))", code)
        return RepairSuggestion(
            "freecad_translate_vector",
            "translate() takes a FreeCAD.Vector; wrapped the arguments in Vector().",
            fixed,
        )
    return RepairSuggestion(
        "freecad_translate_vector",
        "translate() takes a FreeCAD.Vector, not separate x, y, z.",
    )


def _rule_extrude_before_2d(etype: str, estr: str, code: str) -> Optional[RepairSuggestion]:
    if etype == "ValueError" and "No pending" in estr and "extrude" in estr:
        return RepairSuggestion(
            "extrude_before_2d",
            "extrude() needs a 2D op first: add .circle(R)/.rect(L,W)/.polygon(n,R) "
            "before .extrude(H).",
        )
    return None


def _rule_make_ellipse_missing(etype: str, estr: str, code: str) -> Optional[RepairSuggestion]:
    if "makeEllipse" in estr:
        return RepairSuggestion(
            "make_ellipse_missing",
            "Part.makeEllipse does NOT exist. Use Part.Ellipse(); set MajorRadius/"
            "MinorRadius; edge = e.toShape(); wire = Part.Wire([edge]).",
        )
    return None


def _rule_make_pipe_needs_wire(etype: str, estr: str, code: str) -> Optional[RepairSuggestion]:
    if "makePipe" not in estr or "has no attribute" not in estr:
        return None
    m = re.search(r"(\w+)\.makePipe\s*\((.+)\)", code)
    if m:
        var, profile = m.group(1), m.group(2)
        fixed = code.replace(m.group(0), f"Part.Wire([{var}]).makePipe({profile})")
        return RepairSuggestion(
            "make_pipe_needs_wire",
            "makePipe() is a method on Part.Wire, not Part.Edge; wrapped the edge in a Wire.",
            fixed,
        )
    return RepairSuggestion(
        "make_pipe_needs_wire",
        "makePipe() requires Part.Wire: wire = Part.Wire([edge]); wire.makePipe(profile).",
    )


def _rule_null_shape(etype: str, estr: str, code: str) -> Optional[RepairSuggestion]:
    if "null shape" in estr.lower():
        return RepairSuggestion(
            "null_shape",
            "Operation produced a null/empty shape: ensure boolean operands overlap "
            "by >= 0.1mm, profiles are closed (wire.isClosed()), and each step's output "
            "is actually used.",
        )
    return None


def _rule_cylinder_arg_order(etype: str, estr: str, code: str) -> Optional[RepairSuggestion]:
    if etype == "AttributeError" and "cylinder" in estr and (
        "positional" in estr.lower() or "argument" in estr.lower()
    ):
        return RepairSuggestion(
            "cq_cylinder_arg_order",
            "cq.Workplane.cylinder uses (height, radius) -- HEIGHT first. "
            "e.g. cq.Workplane('XY').cylinder(80, 40).",
        )
    return None


# Ordered rule table -- first match wins.
_RULES: Tuple[Callable[[str, str, str], Optional[RepairSuggestion]], ...] = (
    _rule_cylinder_arg_order,
    _rule_freecad_vector_case,
    _rule_inplace_translate_assigned,
    _rule_boolean_result_dropped,
    _rule_make_pipe_needs_wire,
    _rule_missing_part_prefix,
    _rule_cq_method_not_standalone,
    _rule_translate_arity,
    _rule_extrude_before_2d,
    _rule_make_ellipse_missing,
    _rule_null_shape,
)


def suggest_repair(error_type: str, error_message: str, code: str) -> List[RepairSuggestion]:
    """All repair suggestions that match, best-first.

    ``error_type`` is the exception class name (e.g. ``"NameError"``);
    ``error_message`` is ``str(exc)``; ``code`` is the source that failed.
    Returns an ordered list (possibly empty). Suggestions with an autofix sort
    before hint-only ones at equal rule rank.
    """
    out: List[RepairSuggestion] = []
    for rule in _RULES:
        s = rule(error_type, error_message, code)
        if s is not None:
            out.append(s)
    out.sort(key=lambda s: 0 if s.has_autofix else 1)
    return out


def apply_first_repair(error_type: str, error_message: str, code: str) -> Optional[RepairSuggestion]:
    """The single best repair, or ``None`` when nothing matches.

    Prefers a mechanical autofix over a hint-only suggestion so a repair loop
    can re-run immediately when possible.
    """
    suggestions = suggest_repair(error_type, error_message, code)
    return suggestions[0] if suggestions else None
