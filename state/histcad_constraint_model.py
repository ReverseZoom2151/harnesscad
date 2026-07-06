"""HistCAD ten-type geometric-constraint model with DOF / consistency check.

HistCAD encodes exactly ten explicit constraint types (coincident, parallel,
perpendicular, horizontal, vertical, tangent, equal, concentric, fix, normal).
This module is the *deterministic* constraint model for that flat sketch
representation: it assigns each primitive a degrees-of-freedom (DOF) budget and
each constraint a DOF-removal weight and reference arity, then performs a
net-DOF analysis and a rule-based conflict/redundancy scan.

It is intentionally distinct from:

  * ``constraints.py`` — a rank-style union-find DOF solver over the CISP
    abstract sketch model (``cisp.ops``);
  * ``dataengine/sketch_constraint_ontology.py`` — the CadVLM constraint token
    ontology (names/tokens/arity only, no DOF or conflict logic).

Here the vocabulary is HistCAD's exact ten types, the DOF weights follow the
2D sketch conventions, and the conflict scan captures the specific
contradictions HistCAD's flat sketches can express (e.g. horizontal AND
vertical on one line, parallel AND perpendicular on one pair). Stdlib-only,
deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
CONSTRAINT_TYPES: Tuple[str, ...] = (
    "coincident", "parallel", "perpendicular", "horizontal", "vertical",
    "tangent", "equal", "concentric", "fix", "normal",
)

#: minimum number of primitive references each constraint needs.
MIN_REFS: Dict[str, int] = {
    "coincident": 2, "parallel": 2, "perpendicular": 2, "horizontal": 1,
    "vertical": 1, "tangent": 2, "equal": 2, "concentric": 2, "fix": 1,
    "normal": 2,
}

#: nominal DOF removed by each constraint type (fix handled specially).
CONSTRAINT_DOF: Dict[str, int] = {
    "coincident": 2, "parallel": 1, "perpendicular": 1, "horizontal": 1,
    "vertical": 1, "tangent": 1, "equal": 1, "concentric": 2, "fix": 0,
    "normal": 1,
}

#: DOF each primitive kind contributes.
PRIMITIVE_DOF: Dict[str, int] = {"line": 4, "circle": 3, "arc": 6}


class SketchStatus(str, Enum):
    EMPTY = "empty"
    UNDER = "under-constrained"
    WELL = "well-constrained"
    OVER = "over-constrained"


@dataclass(frozen=True)
class ConstraintAnalysis:
    status: SketchStatus
    total_dof: int
    removed_dof: int
    net_dof: int
    conflicts: Tuple[Tuple[str, Tuple[int, ...]], ...]
    redundant: Tuple[int, ...]
    arity_errors: Tuple[int, ...]

    @property
    def consistent(self) -> bool:
        return not self.conflicts and not self.arity_errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prim_kind(prim) -> str:
    # accept objects with a ``.kind`` attribute or plain ("line", ...) tuples
    k = getattr(prim, "kind", None)
    if k is None and isinstance(prim, (tuple, list)) and prim:
        k = prim[0]
    if k not in PRIMITIVE_DOF:
        raise ValueError(f"unknown primitive kind: {k!r}")
    return k


def _fix_weight(prim) -> int:
    """``fix`` removes all DOF of the referenced primitive."""
    return PRIMITIVE_DOF[_prim_kind(prim)]


# ---------------------------------------------------------------------------
# Conflict rules
# ---------------------------------------------------------------------------
def _scan_conflicts(constraints: Sequence) -> List[Tuple[str, Tuple[int, ...]]]:
    """Detect contradictory constraint combinations.

    Rules (each references a *set* of primitive indices):
      * horizontal AND vertical on the same line;
      * parallel AND perpendicular on the same unordered pair;
      * horizontal AND perpendicular / vertical AND parallel are *consistent*
        (not flagged) — only direct contradictions are reported.
    """
    horiz: Dict[int, int] = {}
    vert: Dict[int, int] = {}
    par: Dict[frozenset, int] = {}
    perp: Dict[frozenset, int] = {}
    conflicts: List[Tuple[str, Tuple[int, ...]]] = []
    for c in constraints:
        ctype = getattr(c, "ctype", None) or c[0]
        refs = tuple(getattr(c, "refs", None) if getattr(c, "refs", None) is not None else c[1])
        if ctype == "horizontal":
            for r in refs:
                horiz[r] = 1
        elif ctype == "vertical":
            for r in refs:
                vert[r] = 1
        elif ctype == "parallel":
            par[frozenset(refs)] = 1
        elif ctype == "perpendicular":
            perp[frozenset(refs)] = 1
    for r in sorted(set(horiz) & set(vert)):
        conflicts.append(("horizontal-vertical", (r,)))
    for pair in par:
        if pair in perp:
            conflicts.append(("parallel-perpendicular", tuple(sorted(pair))))
    return conflicts


def _scan_redundant(constraints: Sequence) -> List[int]:
    """Return indices of exact-duplicate constraints (same type + ref-set)."""
    seen: Dict[Tuple, int] = {}
    redundant: List[int] = []
    for i, c in enumerate(constraints):
        ctype = getattr(c, "ctype", None) or c[0]
        refs = tuple(getattr(c, "refs", None) if getattr(c, "refs", None) is not None else c[1])
        key = (ctype, frozenset(refs))
        if key in seen:
            redundant.append(i)
        else:
            seen[key] = i
    return redundant


# ---------------------------------------------------------------------------
# Public analysis
# ---------------------------------------------------------------------------
def analyze(primitives: Sequence, constraints: Sequence) -> ConstraintAnalysis:
    """Full DOF + consistency analysis of a flat HistCAD sketch.

    ``primitives`` — objects with a ``.kind`` or ("line"/"circle"/"arc", ...).
    ``constraints`` — objects with ``.ctype``/``.refs`` or (ctype, refs) tuples.
    """
    prims = list(primitives)
    cons = list(constraints)

    if not prims:
        return ConstraintAnalysis(SketchStatus.EMPTY, 0, 0, 0, (), (), ())

    total_dof = sum(PRIMITIVE_DOF[_prim_kind(p)] for p in prims)

    arity_errors: List[int] = []
    removed = 0
    for i, c in enumerate(cons):
        ctype = getattr(c, "ctype", None) or c[0]
        refs = tuple(getattr(c, "refs", None) if getattr(c, "refs", None) is not None else c[1])
        if ctype not in CONSTRAINT_TYPES:
            raise ValueError(f"unknown constraint type: {ctype!r}")
        if len(refs) < MIN_REFS[ctype]:
            arity_errors.append(i)
            continue
        if ctype == "fix":
            for r in refs:
                if 0 <= r < len(prims):
                    removed += _fix_weight(prims[r])
        else:
            removed += CONSTRAINT_DOF[ctype]

    conflicts = _scan_conflicts(cons)
    redundant = _scan_redundant(cons)

    net = total_dof - removed
    if net > 0:
        status = SketchStatus.UNDER
    elif net == 0:
        status = SketchStatus.WELL
    else:
        status = SketchStatus.OVER
    # A redundant / conflicting constraint over-constrains regardless of net DOF.
    if (conflicts or redundant) and status != SketchStatus.UNDER:
        status = SketchStatus.OVER
    elif (conflicts or redundant) and status == SketchStatus.UNDER:
        # conflict present even while under-constrained -> still over in the
        # affected sub-system; report OVER to surface the contradiction.
        if conflicts:
            status = SketchStatus.OVER

    return ConstraintAnalysis(
        status=status,
        total_dof=total_dof,
        removed_dof=removed,
        net_dof=net,
        conflicts=tuple(conflicts),
        redundant=tuple(redundant),
        arity_errors=tuple(arity_errors),
    )


def constraint_histogram(constraints: Sequence) -> Dict[str, int]:
    """Count constraints per type (all ten keys present, zero-filled)."""
    hist = {t: 0 for t in CONSTRAINT_TYPES}
    for c in constraints:
        ctype = getattr(c, "ctype", None) or c[0]
        if ctype in hist:
            hist[ctype] += 1
    return hist
