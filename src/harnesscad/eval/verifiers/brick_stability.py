"""Physical stability analysis for brick structures.

Each brick is assigned a stability score ``s_i in [0, 1]`` by solving for
a set of connection forces that put every brick in *static equilibrium* under
gravity. A structure is physically stable iff every brick has ``s_i > 0``.

Force model implemented here (a deterministic, stdlib linearisation of that
force model; an exact formulation solves a nonlinear program with a proprietary
solver, which is external):

* Bricks are 1-unit tall, axis-aligned. Each shared cell between a lower brick
  and the brick directly above it (or between a ground-layer brick and the
  baseplate) is a *connection point* carrying a force with three components:

  - a compressive normal ``N >= 0`` (the lower brick / baseplate pressing up on
    the upper brick -- "supporting/pressing"),
  - a "dragging" hold ``T >= 0``: the stud-clutch tension that resists pull-out,
  - horizontal shear ``(fx, fy)``: the knob/adjacency friction, also "dragging".

* Newton's third law is enforced structurally: one force
  variable per connection is added with ``+`` to the upper brick's balance and
  ``-`` to the lower brick's balance.

* Static equilibrium: for every brick, ``sum F = 0`` and
  ``sum tau = 0`` about the brick's centre of mass, with gravity ``-weight`` in
  ``z`` (weight proportional to stud count).

The "dragging" magnitude on a brick is ``D_i = max over its connections of
(T + |fx| + |fy|)`` and ``FT`` is the connection friction capacity. We minimise
the total internal stress (sum of dragging forces -- the ``beta * sum D`` term)
subject to equilibrium, then score:

    s_i = 0                       if no equilibrium exists or D_i > FT,
        = (FT - D_i) / FT          otherwise.

A perfectly-supported vertical stack needs no dragging (``D_i = 0``, score 1); a
cantilever needs increasing dragging force and fails (``s_i = 0``) once the
required force exceeds ``FT``; a floating brick has no equilibrium (score 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from harnesscad.domain.geometry.assembly.brick_structure import Brick, BrickStructure
from harnesscad.domain.geometry.assembly.brick_connectivity import grounded_bricks

_EPS = 1e-7

# Default friction capacity. A reference measurement gives FT = 0.98 N. Weight here is in
# stud-count units, so we pick a capacity of the same order as a small brick's
# weight; callers may override.
DEFAULT_FRICTION_CAPACITY = 4.0


# ---------------------------------------------------------------------------
# Minimal two-phase simplex LP solver (stdlib, deterministic via Bland's rule).
# Solves:  minimise c . x   subject to   A x = b,  x >= 0.
# ---------------------------------------------------------------------------


def solve_lp(
    c: Sequence[float],
    a_eq: Sequence[Sequence[float]],
    b_eq: Sequence[float],
    max_iter: int = 20000,
) -> tuple[bool, list[float], float]:
    """Return ``(feasible, x, objective)`` for min ``c.x`` s.t. ``A x = b, x>=0``.

    A dense two-phase simplex with Bland's anti-cycling pivot rule, so the
    result is fully deterministic. Intended for the small LPs produced by brick
    stability analysis (tens to a few hundred variables).
    """
    m = len(a_eq)
    n = len(c)
    if m == 0:
        # No constraints: optimum is x = 0 when costs are non-negative.
        return True, [0.0] * n, 0.0

    # Copy A, b and make b >= 0 by flipping rows.
    A = [[float(v) for v in row] for row in a_eq]
    b = [float(v) for v in b_eq]
    for i in range(m):
        if b[i] < 0:
            b[i] = -b[i]
            A[i] = [-v for v in A[i]]

    total = n + m  # structural + artificial variables
    # Tableau rows: A | I(artificials) | b
    tableau = []
    for i in range(m):
        row = A[i] + [1.0 if j == i else 0.0 for j in range(m)] + [b[i]]
        tableau.append(row)
    basis = [n + i for i in range(m)]

    def pivot(
        rows: list[list[float]],
        basis_vars: list[int],
        obj: list[float],
        forbid_from: int = total,
    ) -> bool:
        for _ in range(max_iter):
            # Bland's rule: choose the lowest-index eligible column with negative
            # reduced cost. Columns >= ``forbid_from`` (artificials in phase 2)
            # are not allowed to enter.
            entering = -1
            for j in range(forbid_from):
                if obj[j] < -_EPS:
                    entering = j
                    break
            if entering == -1:
                return True  # optimal
            # Ratio test, ties broken by lowest basis index (Bland).
            leaving = -1
            best_ratio = None
            for i in range(m):
                coeff = rows[i][entering]
                if coeff > _EPS:
                    ratio = rows[i][-1] / coeff
                    if (
                        best_ratio is None
                        or ratio < best_ratio - _EPS
                        or (
                            abs(ratio - best_ratio) <= _EPS
                            and basis_vars[i] < basis_vars[leaving]
                        )
                    ):
                        best_ratio = ratio
                        leaving = i
            if leaving == -1:
                return False  # unbounded
            # Normalise pivot row and eliminate.
            piv = rows[leaving][entering]
            rows[leaving] = [v / piv for v in rows[leaving]]
            for i in range(m):
                if i != leaving and abs(rows[i][entering]) > _EPS:
                    factor = rows[i][entering]
                    rows[i] = [
                        rows[i][k] - factor * rows[leaving][k]
                        for k in range(total + 1)
                    ]
            factor = obj[entering]
            if abs(factor) > _EPS:
                obj[:] = [obj[k] - factor * rows[leaving][k] for k in range(total + 1)]
            basis_vars[leaving] = entering
        return False  # iteration limit

    # Phase 1: minimise sum of artificial variables.
    phase1_obj = [0.0] * n + [1.0] * m + [0.0]
    # Reduce objective row against the initial (artificial) basis.
    for i in range(m):
        phase1_obj = [phase1_obj[k] - tableau[i][k] for k in range(total + 1)]
    if not pivot(tableau, basis, phase1_obj):
        return False, [], 0.0
    if -phase1_obj[-1] > 1e-6:
        return False, [], 0.0  # infeasible: artificials not driven to zero

    # Drive any artificial still in the basis out (degenerate case).
    for i in range(m):
        if basis[i] >= n:
            for j in range(n):
                if abs(tableau[i][j]) > _EPS:
                    piv = tableau[i][j]
                    tableau[i] = [v / piv for v in tableau[i]]
                    for r in range(m):
                        if r != i and abs(tableau[r][j]) > _EPS:
                            f = tableau[r][j]
                            tableau[r] = [
                                tableau[r][k] - f * tableau[i][k]
                                for k in range(total + 1)
                            ]
                    basis[i] = j
                    break

    # Phase 2: minimise the real objective over structural variables only.
    phase2_obj = [float(c[j]) if j < n else 0.0 for j in range(total)] + [0.0]
    for i in range(m):
        if abs(phase2_obj[basis[i]]) > _EPS:
            factor = phase2_obj[basis[i]]
            phase2_obj = [
                phase2_obj[k] - factor * tableau[i][k] for k in range(total + 1)
            ]
    if not pivot(tableau, basis, phase2_obj, forbid_from=n):
        return False, [], 0.0

    x = [0.0] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = tableau[i][-1]
    obj_val = sum(c[j] * x[j] for j in range(n))
    return True, x, obj_val


# ---------------------------------------------------------------------------
# Force model assembly.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Connection:
    upper: int  # brick index of the upper party
    lower: int  # brick index of the lower party, or -1 for the baseplate/ground
    px: float
    py: float
    pz: float


def _corner_points(cells: frozenset, zc: float) -> list[tuple[float, float, float]]:
    """The distinct corner points of a set of shared unit stud-cells.

    A stud-into-tube contact is a small *area*, not a point. Modelling each
    shared cell by its four corners lets a single-cell contact resist a tipping
    moment (compression on the far corner, clutch on the near corner), which a
    single centre point cannot. Corners shared between adjacent cells are merged.
    """
    corners: set[tuple[float, float, float]] = set()
    for cx, cy in cells:
        corners.add((float(cx), float(cy), zc))
        corners.add((float(cx + 1), float(cy), zc))
        corners.add((float(cx), float(cy + 1), zc))
        corners.add((float(cx + 1), float(cy + 1), zc))
    return sorted(corners)


def _connections(structure: BrickStructure) -> list[_Connection]:
    """Enumerate stud connection points (corner points of each contact patch)."""
    bricks = structure.bricks
    by_layer: dict[int, list[int]] = {}
    for i, b in enumerate(bricks):
        by_layer.setdefault(b.z, []).append(i)

    conns: list[_Connection] = []
    # Ground connections for layer-0 bricks (the baseplate contact patch).
    for i, b in enumerate(bricks):
        if b.z == 0:
            for px, py, pz in _corner_points(b.cell_set(), 0.0):
                conns.append(_Connection(i, -1, px, py, pz))
    # Brick-brick vertical connections over the shared contact patch.
    for z, lower_indices in by_layer.items():
        upper_indices = by_layer.get(z + 1, [])
        for li in lower_indices:
            lcells = bricks[li].cell_set()
            for ui in upper_indices:
                shared = lcells & bricks[ui].cell_set()
                if not shared:
                    continue
                for px, py, pz in _corner_points(shared, float(z + 1)):
                    conns.append(_Connection(ui, li, px, py, pz))
    return conns


@dataclass(frozen=True)
class StabilityResult:
    stable: bool
    scores: tuple[float, ...]
    feasible: bool
    friction_capacity: float

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 1.0

    @property
    def min_score(self) -> float:
        return min(self.scores) if self.scores else 1.0

    def unstable_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.scores) if s <= _EPS]


def _score_structure(
    structure: BrickStructure,
    friction_capacity: float,
) -> tuple[bool, list[float]]:
    """Solve the equilibrium LP for a structure whose bricks are all grounded.

    Returns ``(feasible, scores)``. Assumes every brick has a support path to
    the baseplate (so equilibrium via compression + clutch always exists).
    """
    bricks = structure.bricks
    n = len(bricks)
    if n == 0:
        return True, []

    conns = _connections(structure)

    # Variable layout: each connection contributes 5 non-negative variables
    #   N (>=0 normal), T (>=0 clutch tension), fxp, fxn, fyp, fyn.
    # We use fx = fxp - fxn, fy = fyp - fyn. That's 6 vars per connection.
    per = 6
    num_vars = len(conns) * per

    def vidx(k: int) -> tuple[int, int, int, int, int, int]:
        base = k * per
        return (base, base + 1, base + 2, base + 3, base + 4, base + 5)

    # 6 equilibrium equations per brick: Fx, Fy, Fz, Mx, My, Mz.
    rows: list[list[float]] = []
    rhs: list[float] = []
    for _ in range(6 * n):
        rows.append([0.0] * num_vars)
        rhs.append(0.0)

    def add(eq: int, var: int, coeff: float) -> None:
        rows[eq][var] += coeff

    for k, conn in enumerate(conns):
        n_v, t_v, fxp, fxn, fyp, fyn = vidx(k)
        # Force on the UPPER brick: Fz = +(N - T), Fx = +(fx), Fy = +(fy).
        # Applied at point (px, py, pz), lever about the upper brick's COM.
        for sign, bi in ((1.0, conn.upper), (-1.0, conn.lower)):
            if bi < 0:
                continue  # ground has no equilibrium equation
            b = bricks[bi]
            cx, cy, cz = b.center
            rx, ry, rz = conn.px - cx, conn.py - cy, conn.pz - cz
            fx_eq, fy_eq, fz_eq = 6 * bi, 6 * bi + 1, 6 * bi + 2
            mx_eq, my_eq, mz_eq = 6 * bi + 3, 6 * bi + 4, 6 * bi + 5
            # Fx = fxp - fxn
            add(fx_eq, fxp, sign)
            add(fx_eq, fxn, -sign)
            # Fy = fyp - fyn
            add(fy_eq, fyp, sign)
            add(fy_eq, fyn, -sign)
            # Fz = N - T
            add(fz_eq, n_v, sign)
            add(fz_eq, t_v, -sign)
            # Torque tau = r x F.
            # tau_x = ry*Fz - rz*Fy
            add(mx_eq, n_v, sign * ry)
            add(mx_eq, t_v, -sign * ry)
            add(mx_eq, fyp, -sign * rz)
            add(mx_eq, fyn, sign * rz)
            # tau_y = rz*Fx - rx*Fz
            add(my_eq, fxp, sign * rz)
            add(my_eq, fxn, -sign * rz)
            add(my_eq, n_v, -sign * rx)
            add(my_eq, t_v, sign * rx)
            # tau_z = rx*Fy - ry*Fx
            add(mz_eq, fyp, sign * rx)
            add(mz_eq, fyn, -sign * rx)
            add(mz_eq, fxp, -sign * ry)
            add(mz_eq, fxn, sign * ry)

    # Gravity: each brick's Fz equation must balance -weight.
    # sum(Fz) - weight = 0  ->  sum(Fz) = weight.
    for bi, b in enumerate(bricks):
        rhs[6 * bi + 2] = float(b.stud_count)

    # Objective: minimise total dragging (internal stress): sum(T + fxp + fxn +
    # fyp + fyn) over all connections. Normal force N is free of cost.
    obj = [0.0] * num_vars
    for k in range(len(conns)):
        _, t_v, fxp, fxn, fyp, fyn = vidx(k)
        for v in (t_v, fxp, fxn, fyp, fyn):
            obj[v] = 1.0

    feasible, x, _ = solve_lp(obj, rows, rhs)
    if not feasible:
        return False, [0.0] * n

    # Per-brick dragging magnitude D_i = max over its connections of
    # (T + |fx| + |fy|).
    drag = [0.0] * n
    for k, conn in enumerate(conns):
        _, t_v, fxp, fxn, fyp, fyn = vidx(k)
        d = x[t_v] + abs(x[fxp] - x[fxn]) + abs(x[fyp] - x[fyn])
        for bi in (conn.upper, conn.lower):
            if bi >= 0 and d > drag[bi]:
                drag[bi] = d

    scores = []
    for bi in range(n):
        d = drag[bi]
        if d > friction_capacity + _EPS:
            scores.append(0.0)
        else:
            scores.append(max(0.0, (friction_capacity - d) / friction_capacity))
    return True, scores


def analyze_stability(
    structure: BrickStructure,
    friction_capacity: float = DEFAULT_FRICTION_CAPACITY,
) -> StabilityResult:
    """Compute per-brick stability scores by solving the equilibrium LP.

    A brick that is not connected to the baseplate (a floating island) can never
    reach equilibrium and is scored 0 directly; the remaining grounded bricks are
    scored by solving the static-equilibrium LP over that substructure so the
    instability is *localised* to the offending bricks (needed by rollback).

    Returns a :class:`StabilityResult`. ``stable`` is True iff every brick has a
    positive stability score.
    """
    bricks = structure.bricks
    n = len(bricks)
    if n == 0:
        return StabilityResult(True, (), True, friction_capacity)

    grounded = grounded_bricks(structure)
    if len(grounded) == n:
        feasible, scores = _score_structure(structure, friction_capacity)
        scores_t = tuple(scores)
        stable = feasible and all(s > _EPS for s in scores_t)
        return StabilityResult(stable, scores_t, feasible, friction_capacity)

    # Some bricks float: score the grounded substructure, mark the rest 0.
    grounded_order = sorted(grounded)
    sub = BrickStructure(
        tuple(bricks[i] for i in grounded_order),
        structure.grid_h,
        structure.grid_w,
        structure.grid_d,
    )
    sub_feasible, sub_scores = _score_structure(sub, friction_capacity)
    score_by_index = {gi: sub_scores[k] for k, gi in enumerate(grounded_order)}
    scores_t = tuple(score_by_index.get(i, 0.0) for i in range(n))
    # Overall structure is not stable: at least one brick floats (score 0).
    return StabilityResult(False, scores_t, False, friction_capacity)


def is_stable(
    structure: BrickStructure,
    friction_capacity: float = DEFAULT_FRICTION_CAPACITY,
) -> bool:
    """True iff every brick has a positive stability score."""
    return analyze_stability(structure, friction_capacity).stable
