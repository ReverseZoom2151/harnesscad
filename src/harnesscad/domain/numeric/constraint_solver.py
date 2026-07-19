"""Numeric constraint-graph diagnostics and a Gauss-Newton sketch solver.

A sketch solver can expose something most sketch solvers hide: a full
*constraint-graph introspection* payload (``ConstraintDiagnostics``) built from
the numeric Jacobian of the residual system -- variable index -> (entity,
parameter) map, constraint -> residual-row-span map, Jacobian shape + rank +
sparsity pattern, remaining degrees of freedom (``dof = n_vars - rank``), the
IDs of conflicting constraints, and the indices of variables no constraint
touches.  A typical implementation this needs NumPy/SciPy (``matrix_rank``, ``minimize``); the
harness already has a *combinatorial* DOF model (:mod:`constraints` --
union-find over nominal constraint weights), but no numeric one.

This module supplies the numeric half in pure stdlib:

* residual functions for the ten the solver constraint kinds (horizontal, vertical,
  parallel, perpendicular, equal, coincident, tangent, fixed, distance, angle)
  over point / line / circle entities;
* a variable codec (entity parameters <-> flat vector) with an index map;
* a central-difference Jacobian;
* numeric rank via Gaussian elimination with partial pivoting, giving
  ``dof``, over-constrained detection (rank deficiency *plus* unsatisfiable
  residual) and under-constrained variables (all-zero Jacobian columns);
* a damped Gauss-Newton (Levenberg-Marquardt style) solve of the normal
  equations, so a sketch can actually be solved, not merely classified.

Deterministic: fixed iteration counts, fixed step sizes, no randomness, no clock.
The rank tolerance is explicit so classification is reproducible.

Public API
----------
``Point``, ``Line``, ``Circle``, ``Constraint``, ``Sketch``
``residuals(sketch)``, ``jacobian(sketch)``, ``diagnose(sketch)``, ``solve(sketch)``
``Diagnostics``, ``SolveResult``, ``SolveStatus``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Point",
    "Line",
    "Circle",
    "Constraint",
    "Sketch",
    "VariableInfo",
    "ConstraintInfo",
    "Diagnostics",
    "SolveStatus",
    "SolveResult",
    "residuals",
    "jacobian",
    "matrix_rank",
    "diagnose",
    "solve",
]

EPS = 1e-9


# -- entities --------------------------------------------------------


@dataclass(frozen=True)
class Point:
    id: str
    x: float = 0.0
    y: float = 0.0

    params: Tuple[str, ...] = field(default=("x", "y"), init=False, repr=False)


@dataclass(frozen=True)
class Line:
    id: str
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0

    params: Tuple[str, ...] = field(
        default=("x1", "y1", "x2", "y2"), init=False, repr=False
    )


@dataclass(frozen=True)
class Circle:
    id: str
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 1.0

    params: Tuple[str, ...] = field(
        default=("cx", "cy", "radius"), init=False, repr=False
    )


Entity = object  # Point | Line | Circle


CONSTRAINT_KINDS = (
    "horizontal",
    "vertical",
    "parallel",
    "perpendicular",
    "equal",
    "coincident",
    "tangent",
    "fixed",
    "distance",
    "angle",
)


@dataclass(frozen=True)
class Constraint:
    id: str
    kind: str
    a: str
    b: Optional[str] = None
    value: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in CONSTRAINT_KINDS:
            raise ValueError("Unknown constraint kind '%s'" % self.kind)


@dataclass
class Sketch:
    entities: Dict[str, Entity] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)

    def add(self, entity: Entity) -> None:
        self.entities[getattr(entity, "id")] = entity

    def constrain(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)


# -- codec -----------------------------------------------------------


@dataclass(frozen=True)
class VariableInfo:
    index: int
    entity_id: str
    parameter_name: str


class Codec:
    """Flat variable vector <-> entity dict, with a stable index map."""

    def __init__(self, entities: Dict[str, Entity]) -> None:
        self.order: List[str] = list(entities.keys())
        self.variables: List[VariableInfo] = []
        self.slices: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        for eid in self.order:
            names = getattr(entities[eid], "params")
            self.slices[eid] = (cursor, cursor + len(names))
            for name in names:
                self.variables.append(
                    VariableInfo(index=cursor, entity_id=eid, parameter_name=name)
                )
                cursor += 1
        self.dimension = cursor

    def to_vector(self, entities: Dict[str, Entity]) -> List[float]:
        out: List[float] = []
        for eid in self.order:
            ent = entities[eid]
            for name in getattr(ent, "params"):
                out.append(float(getattr(ent, name)))
        return out

    def from_vector(
        self, vector: Sequence[float], template: Dict[str, Entity]
    ) -> Dict[str, Entity]:
        out: Dict[str, Entity] = {}
        for eid in self.order:
            start, end = self.slices[eid]
            ent = template[eid]
            names = getattr(ent, "params")
            updates = {}
            for offset, name in enumerate(names):
                val = float(vector[start + offset])
                if name == "radius":
                    val = max(1e-6, abs(val))
                updates[name] = val
            out[eid] = replace(ent, **updates)  # type: ignore[type-var]
        return out


# -- residuals -------------------------------------------------------


def _dir(line: Line) -> Tuple[float, float]:
    return (line.x2 - line.x1, line.y2 - line.y1)


def _norm(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _unit(v: Tuple[float, float]) -> Tuple[float, float]:
    n = _norm(v)
    if n < EPS:
        return (0.0, 0.0)
    return (v[0] / n, v[1] / n)


def _point_line_distance(px: float, py: float, line: Line) -> float:
    dx, dy = _dir(line)
    denom = math.hypot(dx, dy)
    if denom < EPS:
        return math.hypot(px - line.x1, py - line.y1)
    cross = dx * (py - line.y1) - dy * (px - line.x1)
    return abs(cross) / denom


PENALTY = 1e3


def constraint_residual(
    constraint: Constraint,
    entities: Dict[str, Entity],
    initial: Dict[str, Tuple[float, ...]],
) -> List[float]:
    """Residual rows for one constraint (zero when satisfied)."""
    a = entities.get(constraint.a)
    b = entities.get(constraint.b) if constraint.b else None
    if a is None or (constraint.b is not None and b is None):
        return [PENALTY]

    kind = constraint.kind

    if kind == "fixed":
        base = initial[constraint.a]
        current = tuple(float(getattr(a, n)) for n in getattr(a, "params"))
        return [c - i for c, i in zip(current, base)]

    if kind == "horizontal":
        if isinstance(a, Line):
            return [a.y1 - a.y2]
        if isinstance(a, Point) and isinstance(b, Point):
            return [a.y - b.y]
        return [PENALTY]

    if kind == "vertical":
        if isinstance(a, Line):
            return [a.x1 - a.x2]
        if isinstance(a, Point) and isinstance(b, Point):
            return [a.x - b.x]
        return [PENALTY]

    if kind == "parallel":
        if isinstance(a, Line) and isinstance(b, Line):
            ua, ub = _unit(_dir(a)), _unit(_dir(b))
            return [ua[0] * ub[1] - ua[1] * ub[0]]
        return [PENALTY]

    if kind == "perpendicular":
        if isinstance(a, Line) and isinstance(b, Line):
            ua, ub = _unit(_dir(a)), _unit(_dir(b))
            return [ua[0] * ub[0] + ua[1] * ub[1]]
        return [PENALTY]

    if kind == "equal":
        if isinstance(a, Line) and isinstance(b, Line):
            return [_norm(_dir(a)) - _norm(_dir(b))]
        if isinstance(a, Circle) and isinstance(b, Circle):
            return [a.radius - b.radius]
        return [PENALTY]

    if kind == "coincident":
        if isinstance(a, Point) and isinstance(b, Point):
            return [a.x - b.x, a.y - b.y]
        if isinstance(a, Point) and isinstance(b, Line):
            return [_point_line_distance(a.x, a.y, b)]
        if isinstance(a, Point) and isinstance(b, Circle):
            return [math.hypot(a.x - b.cx, a.y - b.cy) - b.radius]
        return [PENALTY]

    if kind == "tangent":
        if isinstance(a, Line) and isinstance(b, Circle):
            return [_point_line_distance(b.cx, b.cy, a) - b.radius]
        if isinstance(a, Circle) and isinstance(b, Line):
            return [_point_line_distance(a.cx, a.cy, b) - a.radius]
        if isinstance(a, Circle) and isinstance(b, Circle):
            centre = math.hypot(a.cx - b.cx, a.cy - b.cy)
            return [centre - (a.radius + b.radius)]
        return [PENALTY]

    if kind == "distance":
        if constraint.value is None:
            return [PENALTY]
        target = float(constraint.value)
        if isinstance(a, Line) and b is None:
            return [_norm(_dir(a)) - target]
        if isinstance(a, Point) and isinstance(b, Point):
            return [math.hypot(a.x - b.x, a.y - b.y) - target]
        if isinstance(a, Point) and isinstance(b, Line):
            return [_point_line_distance(a.x, a.y, b) - target]
        return [PENALTY]

    if kind == "angle":
        if constraint.value is None:
            return [PENALTY]
        if isinstance(a, Line) and isinstance(b, Line):
            ua, ub = _unit(_dir(a)), _unit(_dir(b))
            dot = max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1]))
            return [math.acos(dot) - float(constraint.value)]
        return [PENALTY]

    return [PENALTY]


def _initial_params(entities: Dict[str, Entity]) -> Dict[str, Tuple[float, ...]]:
    return {
        eid: tuple(float(getattr(ent, n)) for n in getattr(ent, "params"))
        for eid, ent in entities.items()
    }


def residuals(sketch: Sketch) -> List[float]:
    """Full residual vector of the sketch at its current geometry."""
    init = _initial_params(sketch.entities)
    out: List[float] = []
    for constraint in sketch.constraints:
        out.extend(constraint_residual(constraint, sketch.entities, init))
    return out


@dataclass(frozen=True)
class ConstraintInfo:
    constraint_id: str
    row_start: int
    row_count: int
    residual_norm: float


def _constraint_infos(
    entities: Dict[str, Entity],
    initial: Dict[str, Tuple[float, ...]],
    constraints: Sequence[Constraint],
) -> List[ConstraintInfo]:
    infos: List[ConstraintInfo] = []
    cursor = 0
    for constraint in constraints:
        rows = constraint_residual(constraint, entities, initial)
        norm = math.sqrt(sum(r * r for r in rows))
        infos.append(
            ConstraintInfo(
                constraint_id=constraint.id,
                row_start=cursor,
                row_count=len(rows),
                residual_norm=norm,
            )
        )
        cursor += len(rows)
    return infos


# -- linear algebra (stdlib) -----------------------------------------


def jacobian(sketch: Sketch, step: float = 1e-6) -> List[List[float]]:
    """Central-difference Jacobian d(residual)/d(variable)."""
    codec = Codec(sketch.entities)
    init = _initial_params(sketch.entities)
    x0 = codec.to_vector(sketch.entities)

    def resid(vec: Sequence[float]) -> List[float]:
        ents = codec.from_vector(vec, sketch.entities)
        out: List[float] = []
        for constraint in sketch.constraints:
            out.extend(constraint_residual(constraint, ents, init))
        return out

    base = resid(x0)
    rows = len(base)
    cols = codec.dimension
    jac = [[0.0] * cols for _ in range(rows)]
    for j in range(cols):
        up = list(x0)
        down = list(x0)
        up[j] += step
        down[j] -= step
        r_up = resid(up)
        r_down = resid(down)
        for i in range(rows):
            jac[i][j] = (r_up[i] - r_down[i]) / (2.0 * step)
    return jac


def matrix_rank(matrix: Sequence[Sequence[float]], tol: float = 1e-7) -> int:
    """Numeric rank via Gaussian elimination with partial pivoting."""
    rows = [list(map(float, r)) for r in matrix]
    if not rows or not rows[0]:
        return 0
    n_rows, n_cols = len(rows), len(rows[0])
    rank = 0
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        best = pivot_row
        for r in range(pivot_row, n_rows):
            if abs(rows[r][col]) > abs(rows[best][col]):
                best = r
        if abs(rows[best][col]) <= tol:
            continue
        rows[pivot_row], rows[best] = rows[best], rows[pivot_row]
        pivot = rows[pivot_row][col]
        for r in range(pivot_row + 1, n_rows):
            factor = rows[r][col] / pivot
            if factor == 0.0:
                continue
            for c in range(col, n_cols):
                rows[r][c] -= factor * rows[pivot_row][c]
        pivot_row += 1
        rank += 1
    return rank


def _solve_linear(
    matrix: List[List[float]], rhs: List[float], tol: float = 1e-12
) -> Optional[List[float]]:
    """Solve a square system by Gaussian elimination; ``None`` if singular."""
    n = len(matrix)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        best = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[best][col]) <= tol:
            return None
        aug[col], aug[best] = aug[best], aug[col]
        pivot = aug[col][col]
        for r in range(col + 1, n):
            factor = aug[r][col] / pivot
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = aug[i][n]
        for j in range(i + 1, n):
            acc -= aug[i][j] * x[j]
        x[i] = acc / aug[i][i]
    return x


# -- diagnostics -----------------------------------------------------


class SolveStatus:
    SOLVED = "SOLVED"
    UNDERCONSTRAINED = "UNDERCONSTRAINED"
    OVERCONSTRAINED = "OVERCONSTRAINED"


@dataclass
class Diagnostics:
    dof: int
    status: str
    rows: int
    cols: int
    rank: int
    max_residual: float
    variables: List[VariableInfo] = field(default_factory=list)
    constraints: List[ConstraintInfo] = field(default_factory=list)
    nonzero_entries: List[Tuple[int, int]] = field(default_factory=list)
    over_constrained_ids: List[str] = field(default_factory=list)
    under_constrained_variables: List[int] = field(default_factory=list)


def diagnose(sketch: Sketch, tolerance: float = 1e-6) -> Diagnostics:
    """Classify the constraint system and expose its full index/rank structure."""
    codec = Codec(sketch.entities)
    init = _initial_params(sketch.entities)
    infos = _constraint_infos(sketch.entities, init, sketch.constraints)
    res = residuals(sketch)
    jac = jacobian(sketch)

    rows = len(jac)
    cols = codec.dimension
    rank = matrix_rank(jac) if rows and cols else 0
    dof = max(0, cols - rank)

    nonzero = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if abs(jac[i][j]) > 1e-8
    ]
    under = [
        j
        for j in range(cols)
        if all(abs(jac[i][j]) <= 1e-8 for i in range(rows))
    ]
    max_residual = max((abs(r) for r in res), default=0.0)
    over = [
        info.constraint_id
        for info in infos
        if info.residual_norm > tolerance * 10
    ]

    # Rank deficiency with more rows than rank means at least one constraint is
    # redundant; it is only a *conflict* when the residual cannot be driven to 0.
    if dof > 0:
        status = SolveStatus.UNDERCONSTRAINED
    elif max_residual > tolerance * 10:
        status = SolveStatus.OVERCONSTRAINED
    else:
        status = SolveStatus.SOLVED

    return Diagnostics(
        dof=dof,
        status=status,
        rows=rows,
        cols=cols,
        rank=rank,
        max_residual=max_residual,
        variables=list(codec.variables),
        constraints=infos,
        nonzero_entries=nonzero,
        under_constrained_variables=under,
        over_constrained_ids=over,
    )


# -- solve -----------------------------------------------------------


@dataclass
class SolveResult:
    status: str
    sketch: Sketch
    iterations: int
    max_residual: float
    message: str
    conflict_constraint_id: Optional[str] = None
    diagnostics: Optional[Diagnostics] = None


def solve(
    sketch: Sketch,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    damping: float = 1e-6,
) -> SolveResult:
    """Damped Gauss-Newton solve of the sketch constraint system."""
    codec = Codec(sketch.entities)
    init = _initial_params(sketch.entities)
    x = codec.to_vector(sketch.entities)

    def resid_at(vec: Sequence[float]) -> List[float]:
        ents = codec.from_vector(vec, sketch.entities)
        out: List[float] = []
        for constraint in sketch.constraints:
            out.extend(constraint_residual(constraint, ents, init))
        return out

    if not sketch.constraints:
        return SolveResult(
            status=SolveStatus.UNDERCONSTRAINED,
            sketch=sketch,
            iterations=0,
            max_residual=0.0,
            message="Sketch has no constraints.",
            diagnostics=diagnose(sketch, tolerance),
        )

    iterations = 0
    lam = damping
    for iterations in range(1, max_iterations + 1):
        current = Sketch(
            entities=codec.from_vector(x, sketch.entities),
            constraints=list(sketch.constraints),
        )
        r = resid_at(x)
        if max(abs(v) for v in r) <= tolerance:
            break
        jac = jacobian(current)
        cols = codec.dimension
        # Normal equations: (J^T J + lam I) dx = -J^T r
        jtj = [[0.0] * cols for _ in range(cols)]
        jtr = [0.0] * cols
        for i, row in enumerate(jac):
            for a_idx in range(cols):
                va = row[a_idx]
                if va == 0.0:
                    continue
                jtr[a_idx] += va * r[i]
                for b_idx in range(a_idx, cols):
                    vb = row[b_idx]
                    if vb == 0.0:
                        continue
                    jtj[a_idx][b_idx] += va * vb
        for a_idx in range(cols):
            for b_idx in range(a_idx + 1, cols):
                jtj[b_idx][a_idx] = jtj[a_idx][b_idx]
            jtj[a_idx][a_idx] += lam
        step = _solve_linear(jtj, [-v for v in jtr])
        if step is None:
            lam *= 10.0
            if lam > 1e6:
                break
            continue
        candidate = [x[i] + step[i] for i in range(cols)]
        new_r = resid_at(candidate)
        if max(abs(v) for v in new_r) < max(abs(v) for v in r):
            x = candidate
            lam = max(damping, lam / 3.0)
        else:
            lam *= 10.0
            if lam > 1e6:
                break

    solved = Sketch(
        entities=codec.from_vector(x, sketch.entities),
        constraints=list(sketch.constraints),
    )
    diag = diagnose(solved, tolerance)
    conflict = diag.over_constrained_ids[0] if diag.over_constrained_ids else None
    if diag.status == SolveStatus.OVERCONSTRAINED:
        message = "Constraints conflict or cannot be simultaneously satisfied."
    elif diag.status == SolveStatus.UNDERCONSTRAINED:
        message = "Sketch has remaining degrees of freedom."
        conflict = None
    else:
        message = "Solved."
        conflict = None
    return SolveResult(
        status=diag.status,
        sketch=solved,
        iterations=iterations,
        max_residual=diag.max_residual,
        message=message,
        conflict_constraint_id=conflict,
        diagnostics=diag,
    )
