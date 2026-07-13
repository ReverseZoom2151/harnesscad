"""traceability — a requirements-traceability matrix for a built HarnessCAD part.

The causal join at the heart of design QA: for every typed
:class:`spec.formalize.Requirement` extracted from a brief, walk *forward*
through the model and answer "what actually satisfied this ask?" —

    Requirement  ->  the op(s) that satisfy it  ->  the resulting feature /
    element  ->  its mass / cost impact (from :mod:`quality.estimate`).

It reads the same event-sourced op history the rest of the harness does
(:class:`state.opdag.OpDAG`, anything exposing ``ops()``), replaying op
semantics deterministically with the *same* id scheme the backends and
:mod:`quality.featuregraph` use (``sk1``, ``f1``, ...), so element ids line up.

Two failure modes are surfaced explicitly, because they are the two ways a
traceability audit fails:

  * **orphan requirements** — an ask with no satisfying op (nothing in the model
    even attempts it: a fillet was requested, none was built);
  * **orphan elements** — a produced feature no requirement asked for (scope
    creep: a chamfer nobody specified).

Pure, stdlib-only, deterministic, and gracefully degrading: with no ``backend``
(or a backend that reports no mass properties) the mass/cost columns are simply
``None`` — the causal join still stands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from harnesscad.core.cisp.ops import (
    Op, NewSketch, Extrude, Fillet, Boolean, Revolve, Chamfer, Hole, Shell,
    Draft, Loft, Sweep, LinearPattern, CircularPattern, Mirror,
)
from harnesscad.domain.spec.formalize import RequirementSet, Requirement
from harnesscad.eval.quality.estimate import estimate_part, PartEstimate

# op class -> feature-graph node type (mirrors quality.featuregraph)
_FEATURE_OP_TYPE = {
    Extrude: "extrude", Revolve: "revolve", Loft: "loft", Sweep: "sweep",
    Fillet: "fillet", Chamfer: "chamfer", Shell: "shell", Draft: "draft",
    Boolean: "boolean", Hole: "hole", Mirror: "mirror",
    LinearPattern: "linear_pattern", CircularPattern: "circular_pattern",
}

# body families whose creation carries the part's mass.
_BODY_TYPES = ("extrude", "revolve", "loft", "sweep")

# a count/feature requirement noun -> the feature type that satisfies it.
_NOUN_TO_TYPE = {
    "hole": "hole", "counterbore": "hole", "countersink": "hole",
    "pocket": "hole", "cutout": "hole",
    "fillet": "fillet", "chamfer": "chamfer", "shell": "shell",
    "draft": "draft",
}

# canonical dimension label -> bounding-box axis index (measure['bbox'] order).
_AXIS_INDEX = {
    "length": 0, "long": 0,
    "width": 1, "wide": 1,
    "height": 2, "tall": 2, "high": 2,
    "depth": 2, "deep": 2,
    "thickness": 2, "thick": 2,
}

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Element replay (mirrors featuregraph / backend id scheme)
# --------------------------------------------------------------------------- #
@dataclass
class _Element:
    """One produced element: a sketch or a feature, with the op that made it."""

    id: str
    type: str
    op_index: int
    op: Op
    params: Dict[str, Any] = field(default_factory=dict)


def _replay_elements(ops: List[Op]) -> List[_Element]:
    """Assign deterministic element ids (``sk1``/``f1``/...) to producing ops.

    Sketch ids depend only on the sketch counter and feature ids only on the
    feature counter (independent of entity primitives), exactly as the backends'
    ``_new_id`` advances them, so the ids match :func:`build_feature_graph`.
    """
    n_sk = 0
    n_f = 0
    out: List[_Element] = []
    for i, op in enumerate(ops):
        if isinstance(op, NewSketch):
            n_sk += 1
            out.append(_Element(f"sk{n_sk}", "sketch", i, op,
                                {"plane": op.plane}))
            continue
        ftype = _FEATURE_OP_TYPE.get(type(op))
        if ftype is None:
            continue
        n_f += 1
        params = {k: (list(v) if isinstance(v, tuple) else v)
                  for k, v in op.to_dict().items() if k != "op"}
        out.append(_Element(f"f{n_f}", ftype, i, op, params))
    return out


# --------------------------------------------------------------------------- #
# Matrix model
# --------------------------------------------------------------------------- #
@dataclass
class TraceRow:
    """One row of the matrix: a requirement and everything downstream of it."""

    requirement: Requirement
    ops: List[Dict[str, Any]] = field(default_factory=list)   # {index, op}
    elements: List[str] = field(default_factory=list)          # element ids
    mass: Optional[float] = None                               # grams (impact)
    cost: Optional[float] = None                               # currency (impact)
    satisfied: bool = False

    def to_dict(self) -> dict:
        return {
            "requirement": self.requirement.to_dict(),
            "ops": [dict(o) for o in self.ops],
            "elements": list(self.elements),
            "mass": self.mass,
            "cost": self.cost,
            "satisfied": self.satisfied,
        }


@dataclass
class TraceabilityMatrix:
    """Requirement -> op -> element -> mass/cost, plus the two orphan sets."""

    rows: List[TraceRow] = field(default_factory=list)
    orphans_req: List[Requirement] = field(default_factory=list)
    orphans_elem: List[str] = field(default_factory=list)
    total_mass: Optional[float] = None
    total_cost: Optional[float] = None

    # -- rollups ------------------------------------------------------------ #
    @property
    def satisfied_count(self) -> int:
        return sum(1 for r in self.rows if r.satisfied)

    @property
    def coverage(self) -> float:
        """Fraction of requirements with at least one satisfying op (0..1)."""
        if not self.rows:
            return 1.0
        linked = sum(1 for r in self.rows if r.ops)
        return linked / len(self.rows)

    def to_dict(self) -> dict:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "orphans_req": [r.to_dict() for r in self.orphans_req],
            "orphans_elem": list(self.orphans_elem),
            "totals": {
                "mass": self.total_mass,
                "cost": self.total_cost,
                "coverage": self.coverage,
                "satisfied": self.satisfied_count,
                "requirements": len(self.rows),
            },
        }

    # -- render ------------------------------------------------------------- #
    def render(self) -> str:
        header = ["Requirement", "Ops", "Elements", "Mass(g)", "Cost", "OK"]
        table: List[List[str]] = [header]
        for r in self.rows:
            table.append([
                _req_label(r.requirement),
                ", ".join(o["op"] for o in r.ops) or "-",
                ", ".join(r.elements) or "-",
                _num(r.mass),
                _num(r.cost),
                "yes" if r.satisfied else "no",
            ])
        widths = [max(len(row[c]) for row in table) for c in range(len(header))]
        lines = []
        for ri, row in enumerate(table):
            lines.append("  ".join(cell.ljust(widths[c])
                                   for c, cell in enumerate(row)).rstrip())
            if ri == 0:
                lines.append("  ".join("-" * widths[c]
                                       for c in range(len(header))))
        if self.orphans_req:
            lines.append("")
            lines.append("Orphan requirements (no satisfying op):")
            for r in self.orphans_req:
                lines.append("  - " + _req_label(r))
        if self.orphans_elem:
            lines.append("")
            lines.append("Orphan elements (no requirement):")
            lines.append("  " + ", ".join(self.orphans_elem))
        totals = "Coverage %.0f%%  (%d/%d satisfied)" % (
            100.0 * self.coverage, self.satisfied_count, len(self.rows))
        if self.total_mass is not None:
            totals += "  mass=%s g" % _num(self.total_mass)
        if self.total_cost is not None:
            totals += "  cost=%s" % _num(self.total_cost)
        lines.append("")
        lines.append(totals)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build_traceability(requirement_set: RequirementSet,
                       opdag: Any,
                       backend: Any = None) -> TraceabilityMatrix:
    """Join a :class:`RequirementSet` to the op history that satisfies it.

    ``opdag`` is any object exposing ``ops()`` (e.g. :class:`state.opdag.OpDAG`).
    ``backend``, when given and able to report mass properties, prices the
    mass/cost impact of the body element(s) via :func:`quality.estimate.estimate_part`.
    """
    ops = list(opdag.ops()) if (opdag is not None
                                and hasattr(opdag, "ops")) else []
    elements = _replay_elements(ops)
    features = [e for e in elements if e.type != "sketch"]
    bodies = [e for e in features if e.type in _BODY_TYPES]

    part_est = _estimate(backend)
    default_tol = _default_tol(requirement_set)
    metrics = _metrics(backend)

    rows: List[TraceRow] = []
    referenced: set = set()

    for req in requirement_set.requirements:
        row = _row_for(req, features, bodies, metrics, default_tol)
        for eid in row.elements:
            referenced.add(eid)
        rows.append(row)

    # Mass/cost impact: attribute the part estimate to the rows that reference a
    # body element (the body carries the mass), split evenly across them.
    body_ids = {e.id for e in bodies}
    body_rows = [r for r in rows if any(eid in body_ids for eid in r.elements)]
    if part_est is not None and part_est.measured and body_rows:
        share_mass = (None if part_est.mass is None
                      else part_est.mass / len(body_rows))
        tc = part_est.total_cost
        share_cost = None if tc is None else tc / len(body_rows)
        for r in body_rows:
            r.mass = share_mass
            r.cost = share_cost

    orphans_req = [r.requirement for r in rows if not r.ops]
    orphans_elem = [e.id for e in features if e.id not in referenced]

    total_mass = part_est.mass if (part_est and part_est.measured) else None
    total_cost = (part_est.total_cost
                  if (part_est and part_est.measured) else None)

    return TraceabilityMatrix(
        rows=rows,
        orphans_req=orphans_req,
        orphans_elem=orphans_elem,
        total_mass=total_mass,
        total_cost=total_cost,
    )


def _row_for(req: Requirement,
             features: List[_Element],
             bodies: List[_Element],
             metrics: Optional[dict],
             default_tol: float) -> TraceRow:
    if req.kind in ("count", "feature"):
        ftype = _NOUN_TO_TYPE.get((req.label or "").lower())
        if ftype is None:
            return TraceRow(req, satisfied=False)  # orphan (unmappable noun)
        matched = [e for e in features if e.type == ftype]
        ops = [{"index": e.op_index, "op": e.op.OP} for e in matched]
        elements = [e.id for e in matched]
        if req.kind == "count":
            target = int(req.target) if req.target is not None else 0
            satisfied = len(matched) >= target and target > 0
        else:
            satisfied = len(matched) >= 1
        return TraceRow(req, ops=ops, elements=elements, satisfied=satisfied)

    if req.kind in ("dimension", "envelope"):
        # A dimension is realised by the part's primary body; satisfaction is
        # confirmed against measured geometry when the backend reports it.
        if not bodies:
            return TraceRow(req, satisfied=False)  # orphan (nothing built)
        body = bodies[0]
        ops = [{"index": body.op_index, "op": body.op.OP}]
        satisfied = _dimension_satisfied(req, metrics, default_tol)
        return TraceRow(req, ops=ops, elements=[body.id], satisfied=satisfied)

    # material / tolerance and anything else: not causally tied to a single op.
    return TraceRow(req, satisfied=False)


def _dimension_satisfied(req: Requirement, metrics: Optional[dict],
                         default_tol: float) -> bool:
    if req.target is None:
        return True
    idx = _AXIS_INDEX.get((req.label or "").lower())
    bbox = metrics.get("bbox") if metrics else None
    if idx is None or not isinstance(bbox, (list, tuple)) or idx >= len(bbox):
        return True  # op exists but not measurable here -> op-level satisfied
    actual = bbox[idx]
    if not isinstance(actual, (int, float)):
        return True
    tol = req.tolerance if req.tolerance is not None else default_tol
    return abs(float(actual) - float(req.target)) <= tol + _EPS


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _estimate(backend: Any) -> Optional[PartEstimate]:
    if backend is None:
        return None
    try:
        return estimate_part(backend)
    except Exception:  # noqa: BLE001 - estimate must never break traceability
        return None


def _metrics(backend: Any) -> Optional[dict]:
    if backend is None or not hasattr(backend, "query"):
        return None
    for key in ("metrics", "measure"):
        try:
            result = backend.query(key)
        except Exception:  # noqa: BLE001
            result = None
        if isinstance(result, dict) and result:
            return result
    return None


def _default_tol(reqset: RequirementSet) -> float:
    tols = reqset.by_kind("tolerance")
    if tols and tols[0].target is not None:
        return float(tols[0].target)
    return 0.0


def _req_label(req: Requirement) -> str:
    if req.source_phrase:
        return req.source_phrase
    if req.label:
        return f"{req.kind}:{req.label}={req.target}"
    return f"{req.kind}:{req.target}"


def _num(v: Optional[float]) -> str:
    return "-" if v is None else f"{float(v):.4g}"
