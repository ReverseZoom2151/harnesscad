"""Text2CAD-Bench complexity-level classifier (L1-L3) for CadQuery programs.

Paper: *Text2CAD-Bench -- A Benchmark for LLM-based Text-to-Parametric CAD
Generation* (Wang et al., 2026). The benchmark's organizing contribution is a
deterministic **geometric-complexity hierarchy** used to stratify examples
(Sec. 3.3.1), plus per-sample complexity proxies (Appendix A.1):

  * **L1 (Basic)**   -- a single primitive with basic finishing features
    (chamfer, fillet, through-hole);
  * **L2 (Intermediate)** -- compositional complexity via *boolean operations*
    (cut/union/intersect of separate solids) and standard CAD features;
  * **L3 (Advanced)** -- sophisticated operations: sweep, loft, shell, twist,
    and freeform/parametric surfaces.

The paper stratifies by "the complexity of required CAD operations and geometric
features", and reports lines-of-code and CadQuery-API-call counts as complexity
proxies (Tab. 4: L1 7.9 lines / 10.8 calls, L3 70.7 / 26.8). This module makes
that classification executable and deterministic: it parses a CadQuery program
with the stdlib ``ast`` module, collects every chained method name, buckets them
into operation categories, and assigns the level by the highest category present
(advanced -> L2-boolean -> L1). It also returns the two complexity proxies.

Unlike :mod:`harnesscad.domain.programs.ast.cadquery` (which validates a *fixed*
subset and rejects advanced ops), this classifier accepts arbitrary CadQuery
source so it can recognise the L3 operations the benchmark is built to test.

Stdlib-only (``ast``) and deterministic.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

__all__ = [
    "PRIMITIVE_OPS",
    "FINISHING_OPS",
    "BOOLEAN_OPS",
    "ADVANCED_OPS",
    "LevelReport",
    "method_calls",
    "classify_level",
    "complexity_report",
]

#: Solid/profile primitives -- their presence alone is an L1 signal.
PRIMITIVE_OPS = frozenset({
    "box", "cylinder", "sphere", "wedge", "circle", "rect", "polygon",
    "ellipse", "text", "sketch", "cone", "makeCone", "makeBox",
})

#: Basic finishing features -- still L1 when applied to a single primitive.
FINISHING_OPS = frozenset({
    "fillet", "chamfer", "hole", "cutThruAll", "cutBlind", "cboreHole",
    "cskHole", "extrude", "workplane", "faces", "edges", "vertices",
    "rarray", "polarArray", "mirror",
})

#: Boolean combinations of separate solids -- the L2 signal.
BOOLEAN_OPS = frozenset({"cut", "union", "intersect", "combine", "add"})

#: Advanced constructive operations -- the L3 signal.
ADVANCED_OPS = frozenset({
    "sweep", "loft", "twistExtrude", "shell", "revolve", "spline",
    "parametricCurve", "ellipseArc", "makeSplineApprox", "sweep_multi",
    "interpPlate", "makeLoft",
})


def method_calls(code: str) -> list:
    """Every chained method name in ``code``, in source order (with repeats).

    Uses the stdlib AST: any ``<expr>.name(...)`` call contributes ``name``.
    Raises :class:`SyntaxError` for unparsable source (a caller may treat that as
    an invalid sample).
    """
    tree = ast.parse(code)
    names: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    return names


def _categories(names) -> dict:
    nameset = set(names)
    return {
        "primitives": sorted(nameset & PRIMITIVE_OPS),
        "finishing": sorted(nameset & FINISHING_OPS),
        "boolean": sorted(nameset & BOOLEAN_OPS),
        "advanced": sorted(nameset & ADVANCED_OPS),
    }


def classify_level(code: str) -> str:
    """Return ``"L1"``, ``"L2"`` or ``"L3"`` for a CadQuery program.

    L3 if any advanced op is present, else L2 if any boolean op is present, else
    L1. Programs with no recognised ops still classify as L1 (basic).
    """
    cats = _categories(method_calls(code))
    if cats["advanced"]:
        return "L3"
    if cats["boolean"]:
        return "L2"
    return "L1"


@dataclass
class LevelReport:
    """Classification plus the paper's complexity proxies."""

    level: str
    categories: dict          # category -> sorted list of ops seen
    api_calls: int            # total chained method calls (Tab. 4 proxy)
    code_lines: int           # non-blank source lines (Tab. 4 proxy)


def complexity_report(code: str) -> LevelReport:
    """Full deterministic report: level, operation categories, and proxies.

    ``api_calls`` counts every chained method call (the paper's "API Calls");
    ``code_lines`` counts non-blank lines (its "Code Lines"). Both grow with
    level, validating the stratification (Tab. 4).
    """
    names = method_calls(code)
    cats = _categories(names)
    if cats["advanced"]:
        level = "L3"
    elif cats["boolean"]:
        level = "L2"
    else:
        level = "L1"
    lines = sum(1 for ln in code.splitlines() if ln.strip())
    return LevelReport(
        level=level,
        categories=cats,
        api_calls=len(names),
        code_lines=lines,
    )
