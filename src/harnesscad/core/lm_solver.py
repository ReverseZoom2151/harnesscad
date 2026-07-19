"""Levenberg-Marquardt 2D constraint solver.

This third independent constraint method is deliberately not the same algorithm
as either existing method:

* :class:`core.constraints.ConstraintGraph` is a **union-find rank** analysis
  over an abstract DOF model -- combinatorial, never touches real coordinates.
* :class:`core.constraints.SolveSpaceSketch` wraps SolveSpace's own solver.
* This module is a **numerical Levenberg-Marquardt** least-squares solver that
  actually moves 2D coordinates to drive a vector of residual functions to zero,
  plus a **Jacobian-rank freedom analysis** that names the individual variables
  that are still under-constrained.

Independence is the point: three methods that agree on a sketch's status is far
stronger evidence than one, the same way an independent Boolean kernel
strengthens the geometry oracle.  The residual formulation here also makes the
solver constraint-agnostic -- any differentiable residual can be added, not only
the fixed vocabulary a rank model understands.

Algorithm:

* residuals ``r(x)`` are stacked from every constraint; the objective is
  ``sum r_i^2``.
* each step solves the damped normal equations
  ``(JᵀJ + λ·diag(JᵀJ)) Δ = -Jᵀr`` (finite-difference Jacobian ``J``) and adapts
  the LM damping ``λ`` -- shrink on an accepted step, grow on a rejected one --
  exactly as ezpz adapts its ``lambda``.
* convergence on residual norm, step norm, or an iteration cap (ezpz's
  ``Config`` defaults: 35 iterations, ``1e-8`` residual tol, ``1e-12`` step tol).

Freedom analysis (ezpz ``analysis.rs`` ``FreedomAnalysis``): the Jacobian at the
solution is reduced to row-echelon form; every variable whose column carries no
pivot is *under-constrained* -- its value depends on the initial guess rather
than the constraints -- and its index is reported so a caller can tell the user
which points still need pinning.

Pure stdlib (small dense linear algebra by Gaussian elimination), deterministic:
no randomness, no wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot, sqrt
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "SolveStatus",
    "Residual",
    "LMConfig",
    "LMResult",
    "FreedomReport",
    "System2D",
    "solve_residuals",
    "matrix_rank",
]

# residual callback: given the flat parameter vector, return a list of residuals.
Residual = Callable[[Sequence[float]], List[float]]


class SolveStatus(str, Enum):
    """Outcome of a Levenberg-Marquardt solve."""

    CONVERGED = "converged"          # residual driven below tolerance
    STALLED = "stalled"              # step became negligible before convergence
    MAX_ITERATIONS = "max-iterations"


# --------------------------------------------------------------------------- #
# small dense linear algebra (stdlib)                                          #
# --------------------------------------------------------------------------- #
def _matvec(m: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    return [sum(row[j] * v[j] for j in range(len(v))) for row in m]


def _solve_linear(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Solve ``a x = b`` for a square matrix by Gaussian elimination w/ pivoting.

    Returns ``None`` if the matrix is singular to working precision.
    """
    n = len(a)
    # augment (deep copy so the caller's matrix is untouched).
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        # partial pivot
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-15:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        inv = 1.0 / m[col][col]
        for j in range(col, n + 1):
            m[col][j] *= inv
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor != 0.0:
                for j in range(col, n + 1):
                    m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def matrix_rank(m: Sequence[Sequence[float]], tol: float = 1e-9) -> Tuple[int, List[int]]:
    """Rank of ``m`` and the list of pivot column indices (row-echelon, pivoting).

    Deterministic Gaussian elimination.  The pivot columns identify the
    constrained variables; the complement is the free/under-constrained set.
    """
    if not m:
        return 0, []
    rows = [list(r) for r in m]
    n_rows = len(rows)
    n_cols = len(rows[0])
    pivot_cols: List[int] = []
    pivot_row = 0
    for col in range(n_cols):
        # find a row at or below pivot_row with the largest magnitude in this col.
        best = pivot_row
        best_val = abs(rows[pivot_row][col]) if pivot_row < n_rows else 0.0
        for r in range(pivot_row, n_rows):
            if abs(rows[r][col]) > best_val:
                best_val = abs(rows[r][col])
                best = r
        if pivot_row >= n_rows or best_val <= tol:
            continue
        rows[pivot_row], rows[best] = rows[best], rows[pivot_row]
        inv = 1.0 / rows[pivot_row][col]
        for j in range(col, n_cols):
            rows[pivot_row][j] *= inv
        for r in range(n_rows):
            if r == pivot_row:
                continue
            factor = rows[r][col]
            if abs(factor) > 0.0:
                for j in range(col, n_cols):
                    rows[r][j] -= factor * rows[pivot_row][j]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row >= n_rows:
            break
    return len(pivot_cols), pivot_cols


# --------------------------------------------------------------------------- #
# LM configuration / results                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LMConfig:
    """Levenberg-Marquardt settings (ezpz ``Config`` defaults)."""

    max_iterations: int = 35
    residual_tolerance: float = 1e-8
    step_tolerance: float = 1e-12
    initial_lambda: float = 1e-9
    lambda_down: float = 0.3       # accepted step -> shrink damping
    lambda_up: float = 10.0        # rejected step -> grow damping


@dataclass(frozen=True)
class FreedomReport:
    """Which variables remain under-constrained (ezpz ``FreedomAnalysis``)."""

    n_variables: int
    rank: int
    underconstrained: Tuple[int, ...]

    @property
    def free_dof(self) -> int:
        return self.n_variables - self.rank

    @property
    def is_underconstrained(self) -> bool:
        return len(self.underconstrained) > 0


@dataclass(frozen=True)
class LMResult:
    """Outcome of a Levenberg-Marquardt solve."""

    status: SolveStatus
    params: Tuple[float, ...]
    residual_norm: float
    iterations: int
    freedom: FreedomReport

    @property
    def solved(self) -> bool:
        return self.status is SolveStatus.CONVERGED


# --------------------------------------------------------------------------- #
# core solver                                                                  #
# --------------------------------------------------------------------------- #
def _stack(residuals: Sequence[Residual], x: Sequence[float]) -> List[float]:
    out: List[float] = []
    for r in residuals:
        out.extend(r(x))
    return out


def _jacobian(residuals: Sequence[Residual], x: Sequence[float], h: float = 1e-7):
    """Finite-difference Jacobian ``J[i][j] = d r_i / d x_j`` (central difference)."""
    n = len(x)
    base = _stack(residuals, x)
    m = len(base)
    jac = [[0.0] * n for _ in range(m)]
    for j in range(n):
        xp = list(x)
        xm = list(x)
        xp[j] += h
        xm[j] -= h
        rp = _stack(residuals, xp)
        rm = _stack(residuals, xm)
        for i in range(m):
            jac[i][j] = (rp[i] - rm[i]) / (2.0 * h)
    return jac, base


def _norm(v: Sequence[float]) -> float:
    return sqrt(sum(c * c for c in v))


def solve_residuals(
    residuals: Sequence[Residual],
    initial: Sequence[float],
    config: Optional[LMConfig] = None,
) -> LMResult:
    """Solve ``r(x) = 0`` in the least-squares sense by Levenberg-Marquardt.

    ``residuals`` is a list of callables, each returning one or more residual
    values for the current parameter vector; ``initial`` is the starting guess.
    Returns the solution, its residual norm, and the Jacobian-rank freedom report.
    """
    cfg = config or LMConfig()
    x = list(initial)
    n = len(x)
    lam = cfg.initial_lambda

    jac, r = _jacobian(residuals, x)
    cost = _norm(r)
    status = SolveStatus.MAX_ITERATIONS
    iterations = 0

    for iterations in range(1, cfg.max_iterations + 1):
        if cost < cfg.residual_tolerance:
            status = SolveStatus.CONVERGED
            break
        # normal equations JtJ and gradient Jt r.
        m = len(r)
        jtj = [[0.0] * n for _ in range(n)]
        jtr = [0.0] * n
        for a in range(n):
            for b in range(a, n):
                s = 0.0
                for i in range(m):
                    s += jac[i][a] * jac[i][b]
                jtj[a][b] = s
                jtj[b][a] = s
            g = 0.0
            for i in range(m):
                g += jac[i][a] * r[i]
            jtr[a] = g

        # damped: (JtJ + lam*diag(JtJ)) delta = -Jt r
        accepted = False
        for _ in range(30):  # inner damping search
            damped = [list(row) for row in jtj]
            for a in range(n):
                damped[a][a] += lam * (jtj[a][a] if jtj[a][a] != 0.0 else 1.0)
            delta = _solve_linear(damped, [-g for g in jtr])
            if delta is None:
                lam *= cfg.lambda_up
                continue
            x_new = [x[j] + delta[j] for j in range(n)]
            r_new = _stack(residuals, x_new)
            cost_new = _norm(r_new)
            if cost_new < cost:
                # accept
                x = x_new
                if _norm(delta) < cfg.step_tolerance:
                    jac, r = _jacobian(residuals, x)
                    cost = cost_new
                    status = SolveStatus.STALLED
                    accepted = True
                    lam = max(lam * cfg.lambda_down, 1e-15)
                    break
                lam = max(lam * cfg.lambda_down, 1e-15)
                jac, r = _jacobian(residuals, x)
                cost = cost_new
                accepted = True
                break
            else:
                lam *= cfg.lambda_up
        if not accepted:
            status = SolveStatus.STALLED
            break
        if status is SolveStatus.STALLED:
            break

    if cost < cfg.residual_tolerance:
        status = SolveStatus.CONVERGED

    # freedom analysis on the Jacobian at the solution.
    jac_final, _ = _jacobian(residuals, x)
    rank, pivots = matrix_rank(jac_final)
    pivot_set = set(pivots)
    under = tuple(j for j in range(n) if j not in pivot_set)
    freedom = FreedomReport(n_variables=n, rank=rank, underconstrained=under)

    return LMResult(
        status=status,
        params=tuple(x),
        residual_norm=cost,
        iterations=iterations,
        freedom=freedom,
    )


# --------------------------------------------------------------------------- #
# 2D sketch front-end (ezpz-style entities + constraints)                      #
# --------------------------------------------------------------------------- #
@dataclass
class System2D:
    """A 2D point-and-constraint system solved by Levenberg-Marquardt.

    Points are stored as consecutive ``(x, y)`` pairs in a flat parameter vector;
    :meth:`add_point` returns the point's index.  Constraint builders append
    residual callables that read that vector.  ezpz's constraint vocabulary is
    reproduced with residuals: ``fix`` (anchor), ``distance``, ``coincident``,
    ``horizontal``, ``vertical`` and axis pins.
    """

    _guess: List[float] = field(default_factory=list)
    _residuals: List[Residual] = field(default_factory=list)

    def add_point(self, x: float, y: float) -> int:
        idx = len(self._guess) // 2
        self._guess.extend([float(x), float(y)])
        return idx

    def _xy(self, x: Sequence[float], p: int) -> Tuple[float, float]:
        return x[2 * p], x[2 * p + 1]

    # -- constraints (each appends a residual) ------------------------------
    def fix(self, p: int, x: float, y: float) -> None:
        """Anchor point ``p`` at ``(x, y)`` (ezpz ``dragged``/fixed)."""
        def r(v: Sequence[float]) -> List[float]:
            px, py = self._xy(v, p)
            return [px - x, py - y]
        self._residuals.append(r)

    def distance(self, a: int, b: int, d: float) -> None:
        """Constrain the distance between points ``a`` and ``b`` to ``d``."""
        def r(v: Sequence[float]) -> List[float]:
            ax, ay = self._xy(v, a)
            bx, by = self._xy(v, b)
            return [hypot(ax - bx, ay - by) - d]
        self._residuals.append(r)

    def coincident(self, a: int, b: int) -> None:
        """Make points ``a`` and ``b`` coincide."""
        def r(v: Sequence[float]) -> List[float]:
            ax, ay = self._xy(v, a)
            bx, by = self._xy(v, b)
            return [ax - bx, ay - by]
        self._residuals.append(r)

    def horizontal(self, a: int, b: int) -> None:
        """Constrain the segment ``a-b`` to be horizontal (equal y)."""
        def r(v: Sequence[float]) -> List[float]:
            _, ay = self._xy(v, a)
            _, by = self._xy(v, b)
            return [ay - by]
        self._residuals.append(r)

    def vertical(self, a: int, b: int) -> None:
        """Constrain the segment ``a-b`` to be vertical (equal x)."""
        def r(v: Sequence[float]) -> List[float]:
            ax, _ = self._xy(v, a)
            bx, _ = self._xy(v, b)
            return [ax - bx]
        self._residuals.append(r)

    def pin_x(self, p: int, x: float) -> None:
        """Pin only the x coordinate of point ``p``."""
        def r(v: Sequence[float]) -> List[float]:
            return [v[2 * p] - x]
        self._residuals.append(r)

    def pin_y(self, p: int, y: float) -> None:
        """Pin only the y coordinate of point ``p``."""
        def r(v: Sequence[float]) -> List[float]:
            return [v[2 * p + 1] - y]
        self._residuals.append(r)

    # -- solve --------------------------------------------------------------
    def solve(self, config: Optional[LMConfig] = None) -> LMResult:
        """Run the Levenberg-Marquardt solve over the accumulated residuals."""
        return solve_residuals(self._residuals, self._guess, config)

    def point(self, result: LMResult, p: int) -> Tuple[float, float]:
        """Read the solved coordinates of point ``p`` from a result."""
        return result.params[2 * p], result.params[2 * p + 1]
