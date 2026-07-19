"""FreeCAD execution-error taxonomy.

Classifies the failures produced by executing an LLM-generated
FreeCAD script in headless mode into three families:

  * SYNTAX          invalid Python (indentation, missing imports, ...)
  * GEOMETRIC       invalid boolean ops, degenerate / null-shape geometry,
                    overconstraint
  * EXECUTION       incorrect API calls, missing object dependencies

Concrete signatures include things such as
``module 'Part' has no attribute 'makeGear'`` (unsupported API -> EXECUTION) and
``Null shape`` (GEOMETRIC).  This module is a deterministic classifier over a
FreeCAD stderr string plus a convenience tally for a batch of runs.  No LLM,
pure pattern matching, reproducible.

Stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

SYNTAX = "syntax"
GEOMETRIC = "geometric"
EXECUTION = "execution"
NONE = "none"

# Ordered (family, keyword) rules; first match wins.  Geometric and execution
# signatures are checked before generic syntax so that, e.g. a Python traceback
# whose final line names a null shape is classed geometric.
_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (EXECUTION, ("has no attribute", "unsupported", "not callable",
                 "attributeerror", "nameerror", "no module named",
                 "referenced before assignment", "object dependency",
                 "missing object")),
    (GEOMETRIC, ("null shape", "nullshape", "degenerate", "self-intersect",
                 "boolean", "overconstrain", "over-constrain", "invalid shape",
                 "empty wire", "wire is not closed", "non-manifold")),
    (SYNTAX, ("syntaxerror", "indentationerror", "unexpected indent",
              "invalid syntax", "eol while scanning", "taberror",
              "missing import")),
)


@dataclass
class ErrorClassification:
    """A classified FreeCAD error."""
    family: str
    signature: str  # the matched keyword, or "" when none/unknown


def classify(stderr: str) -> ErrorClassification:
    """Classify a FreeCAD stderr string into the error taxonomy above.

    Empty / whitespace stderr means the script executed cleanly (E = 0).  An
    unrecognised non-empty error is reported as EXECUTION with empty signature
    (a generic catch-all "execution failure").
    """
    if not stderr or not stderr.strip():
        return ErrorClassification(NONE, "")
    low = stderr.lower()
    for family, keys in _RULES:
        for k in keys:
            if k in low:
                return ErrorClassification(family, k)
    return ErrorClassification(EXECUTION, "")


def tally(stderrs: List[str]) -> Dict[str, int]:
    """Count errors by family across a batch of runs."""
    counts = {SYNTAX: 0, GEOMETRIC: 0, EXECUTION: 0, NONE: 0}
    for s in stderrs:
        counts[classify(s).family] += 1
    return counts
