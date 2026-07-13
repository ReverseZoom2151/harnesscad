"""Execution-free static analysis of generated CadQuery code.

Paper 171 -- *Text-to-CadQuery* -- notes two distinct failure modes for generated
CadQuery scripts that this module detects *without executing anything*:

  1. **Missing / mis-rooted workplane.** Sec. 3.1 reports that "many common errors,
     such as failing to define a workplane, can be automatically fixed" by the
     annotation feedback loop. Statically, this is a chain of sketch/solid methods
     whose root is neither ``cq.Workplane(...)`` nor a previously-assigned variable.

  2. **Numerically-unstable arcs (Appendix A.4).** The paper's worked failure case
     is a ``threePointArc`` whose three control points are "nearly colinear and
     extremely close in magnitude", which makes the OpenCascade kernel raise
     ``GC_MakeArcOfCircle::Value() - no result``. The recommended mitigations are to
     increase the sketch scale or switch to ``radiusArc`` / splines. We reproduce
     the *detection* half deterministically: tracking the sketch pen position
     through ``moveTo`` / ``lineTo``, we flag ``threePointArc`` calls whose
     start/mid/end points are near-colinear and small in magnitude.

On top of these, it performs the ordinary execution-free hygiene checks that a
static analyser should: **safety** (no ``exec`` / ``eval`` / ``__import__`` / ``os``
/ ``subprocess`` / ``open`` -- CadQuery scripts should be pure geometry) and
**use-before-assignment** of variables. Each finding is a :class:`Finding` with a
severity and a stable code, aggregated into an :class:`AnalysisReport`.

Pure stdlib (``ast`` + ``math``). Distinct from :mod:`programs.t2cq_validity`
(which checks API names / arity): this module reasons about *dataflow and geometry*
-- variable definedness, chain rooting, degenerate-geometry risk, and safety.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field

from harnesscad.domain.programs.ast.cadquery import CHAIN_METHODS

# Sketch/solid methods that require a rooted chain (Workplane or a defined var).
_GEOMETRY_METHODS = frozenset(CHAIN_METHODS) - {"workplane"}

# Names that must never appear in a pure-geometry CadQuery script.
_FORBIDDEN_NAMES = frozenset({
    "exec", "eval", "compile", "__import__", "open", "os", "sys",
    "subprocess", "shutil", "socket", "input", "globals", "locals",
})

# Magnitude below which a coordinate is considered "extremely close" (Appendix A.4
# cites failures at scales on the order of 1e-2); colinearity area threshold.
_SMALL_MAGNITUDE = 0.05
_COLINEAR_AREA = 1e-3


@dataclass(frozen=True)
class Finding:
    """A single static-analysis finding."""

    severity: str      # "error" | "warning"
    code: str          # stable machine-readable category
    message: str
    line: int


@dataclass
class AnalysisReport:
    """Aggregated result of :func:`analyze`."""

    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def has_code(self, code: str) -> bool:
        return any(f.code == code for f in self.findings)

    @property
    def ok(self) -> bool:
        """True when there are no *error*-severity findings (warnings allowed)."""
        return not self.errors


def _tuple_point(node):
    """Return ``(x, y)`` floats if ``node`` is a 2-number tuple literal, else None."""
    if not isinstance(node, ast.Tuple) or len(node.elts) != 2:
        return None
    vals = []
    for elt in node.elts:
        try:
            v = ast.literal_eval(elt)
        except (ValueError, SyntaxError, TypeError):
            return None
        if not isinstance(v, (int, float)):
            return None
        vals.append(float(v))
    return (vals[0], vals[1])


def _triangle_area(p, q, r) -> float:
    """Unsigned area of the triangle ``p q r``."""
    return abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])) / 2.0


def _flatten_chain(node):
    """Yield ``(method, call_node)`` for a chained expression, root-first."""
    calls = []
    cur = node
    while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
        calls.append((cur.func.attr, cur))
        cur = cur.func.value
    calls.reverse()
    return cur, calls


def _analyze_arcs(calls, issues: list[Finding]) -> None:
    """Track the sketch pen position and flag degenerate threePointArc calls."""
    pen = None  # current (x, y)
    for method, call in calls:
        line = getattr(call, "lineno", 0)
        if method == "moveTo":
            pt = _literal_xy(call.args)
            pen = pt if pt else pen
        elif method == "lineTo":
            pt = _literal_xy(call.args)
            pen = pt if pt else pen
        elif method == "threePointArc" and len(call.args) == 2:
            mid = _tuple_point(call.args[0])
            end = _tuple_point(call.args[1])
            if pen is not None and mid is not None and end is not None:
                pts = (pen, mid, end)
                area = _triangle_area(*pts)
                max_mag = max(abs(c) for p in pts for c in p)
                if area < _COLINEAR_AREA and max_mag < _SMALL_MAGNITUDE:
                    issues.append(Finding(
                        "warning", "degenerate_arc",
                        "threePointArc points are near-colinear and small in "
                        "magnitude; OpenCascade may fail (see paper A.4 -- prefer "
                        "radiusArc or increase sketch scale)", line))
                pen = end
            elif end is not None:
                pen = end


def _literal_xy(args):
    """Two positional numeric literals -> (x, y), else None."""
    if len(args) != 2:
        return None
    vals = []
    for a in args:
        try:
            v = ast.literal_eval(a)
        except (ValueError, SyntaxError, TypeError):
            return None
        if not isinstance(v, (int, float)):
            return None
        vals.append(float(v))
    return (vals[0], vals[1])


def analyze(code: str) -> AnalysisReport:
    """Run all execution-free analyses over CadQuery source.

    Returns an :class:`AnalysisReport`. Syntax errors are reported as a single
    ``error`` finding (further analysis is skipped). This never executes the code.
    """
    report = AnalysisReport()
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        report.findings.append(Finding("error", "syntax",
                                       f"syntax error: {exc.msg}", exc.lineno or 0))
        return report

    # --- safety: forbidden names anywhere in the module ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            report.findings.append(Finding(
                "error", "unsafe", f"forbidden name {node.id!r} in CadQuery script",
                getattr(node, "lineno", 0)))
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
            report.findings.append(Finding(
                "error", "unsafe", f"forbidden attribute {node.attr!r}",
                getattr(node, "lineno", 0)))

    # --- dataflow: variable definedness + chain rooting + arc geometry ---
    defined: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                defined.add((alias.asname or alias.name).split(".")[0])
            continue
        if not isinstance(stmt, ast.Assign):
            continue
        root, calls = _flatten_chain(stmt.value)
        line = getattr(stmt, "lineno", 0)

        uses_geometry = any(m in _GEOMETRY_METHODS for m, _ in calls)
        # Chain rooting: geometry chains must start at cq.Workplane(...) or a
        # previously-defined variable.
        if uses_geometry:
            rooted = False
            if isinstance(root, ast.Call) and isinstance(root.func, ast.Attribute) \
                    and root.func.attr == "Workplane":
                rooted = True
            elif isinstance(root, ast.Name) and root.id in defined:
                rooted = True
            elif isinstance(root, ast.Name) and root.id not in defined:
                report.findings.append(Finding(
                    "error", "undefined_var",
                    f"chain root {root.id!r} used before assignment", line))
                rooted = True  # already reported; don't double-flag as unrooted
            if not rooted:
                report.findings.append(Finding(
                    "error", "missing_workplane",
                    "geometry chain is not rooted at a Workplane or defined "
                    "variable (paper A: 'failing to define a workplane')", line))

        # Variable-argument definedness (e.g. .union(other_part)).
        for _, call in calls:
            for arg in call.args:
                if isinstance(arg, ast.Name) and arg.id not in defined:
                    report.findings.append(Finding(
                        "error", "undefined_var",
                        f"variable {arg.id!r} used before assignment", line))

        _analyze_arcs(calls, report.findings)

        for target in stmt.targets:
            if isinstance(target, ast.Name):
                defined.add(target.id)

    return report


def is_safe(code: str) -> bool:
    """True iff static analysis finds no error-severity issues."""
    return analyze(code).ok
