"""Compliance minimisation on a structured 2D grid under manufacturing rules.

This module answers one question: given a rectangular design domain, a set of
load cases and a material budget, which elements should carry material so that
the structure is as stiff as possible while staying manufacturable?

Method
------
The design variable is one density per element, ``x[e]`` in ``[1e-3, 1]``.
Stiffness is interpolated by the usual power law (solid-isotropic material with
penalisation)::

    E(x) = Emin + x**penal * (1 - Emin)      with E0 = 1

A penalty exponent above one makes intermediate densities structurally
inefficient, so the optimiser is pushed towards a near-binary solid/void
layout.  Physics comes from a bilinear four-node quadrilateral in plane stress
on a unit grid; the element stiffness is obtained by two-by-two Gauss
quadrature of ``B^T D B`` rather than a table of magic constants, so the
material law is visible in the code.

Each design cycle performs:

1. **Filtering** -- the raw densities are convolved with a conical (linear-hat)
   kernel of radius ``rmin``.  The convolution is a convex average, so a
   constant field is a fixed point, and any feature thinner than the kernel is
   averaged away.  This is what gives a minimum member size and what stops the
   well-known checkerboard instability.
2. **Analysis** -- for every load case the restrained stiffness system is
   assembled over free degrees of freedom only and solved.  The system is
   symmetric positive definite once the body is properly restrained, so it is
   factorised as ``L D L^T`` inside a fixed half-band; this costs
   ``O(n * band**2)`` instead of the ``O(n**3)`` of a general solve, which is
   what makes a hermetic selfcheck affordable.  Should the factorisation meet a
   non-positive pivot -- the signature of an under-restrained model -- the
   caller falls back to pivoting Gaussian elimination.
3. **Sensitivities** -- compliance is self-adjoint, so the derivative of the
   weighted objective with respect to element density is available in closed
   form from the element strain energy, with no adjoint solve.  The
   sensitivities are pushed through the same filter as the densities to keep
   the update consistent with the filtered physics.
4. **Update** -- an optimality-criteria step scales each density by the square
   root of its sensitivity-to-multiplier ratio, clipped by a move limit and the
   box bounds.  The Lagrange multiplier of the volume constraint is found by
   bisection, exploiting the fact that the delivered volume falls monotonically
   as the multiplier grows.
5. **Manufacturing repair** -- optional mirror symmetry, mould draw direction
   (densities made monotone along the extraction axis) and additive overhang
   support (each cell capped by the strongest cell inside the self-support cone
   underneath).  Repair only ever removes material, so the volume is rescaled
   afterwards to keep the budget binding.

The loop stops when the objective and the design both stop moving.

Verification helpers
--------------------
:func:`count_overhang_violations`, :func:`isolated_island_count` and
:func:`min_member_ok` judge a finished density field on its own, without any
knowledge of how it was produced, so they can be used as independent oracles
for the constraints the optimiser claims to satisfy.

Contract
--------
Every public entry point is total: malformed input yields
``{"ok": False, "reason": ...}`` instead of an exception.  The two linear
solvers are the deliberate exception -- they raise ``ValueError``, which the
analysis step catches to select its fallback.

Deterministic (no randomness, fixed iteration counts, bisection to a fixed
tolerance), stdlib only, ASCII only.  Run ``python -m
harnesscad.domain.geometry.volumes.topology_optimize --selfcheck``.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "Mesh2D",
    "ke_q4",
    "solve_dense",
    "solve_spd_banded",
    "build_filter",
    "apply_filter",
    "mbb_problem",
    "optimize",
    "pareto_sweep",
    "count_overhang_violations",
    "isolated_island_count",
    "min_member_ok",
]

# Box bounds on the design variable.  A strictly positive floor keeps the
# stiffness matrix invertible and the power law differentiable at x -> 0.
_X_MIN = 1e-3
_X_MAX = 1.0

# Two-point Gauss-Legendre rule mapped onto the unit interval [0, 1].
_GAUSS_1D = (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0))


# ==========================================================================
# Linear algebra
# ==========================================================================

def solve_dense(A: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting.

    The caller's data is never touched: the augmented system is copied first.
    A pivot column that is numerically empty means the matrix is singular and
    raises ``ValueError``.
    """
    n = len(b)
    # Augmented rows [A | b] so the right-hand side follows the row swaps for
    # free instead of being tracked separately.
    aug = [list(A[i]) + [b[i]] for i in range(n)]

    for step in range(n):
        # Largest magnitude in the remaining column controls round-off growth.
        pivot_row = max(range(step, n), key=lambda r: abs(aug[r][step]))
        if abs(aug[pivot_row][step]) < 1e-300:
            raise ValueError("singular matrix")
        if pivot_row != step:
            aug[step], aug[pivot_row] = aug[pivot_row], aug[step]

        top = aug[step]
        recip = 1.0 / top[step]
        for r in range(step + 1, n):
            row = aug[r]
            mult = row[step] * recip
            if mult == 0.0:
                continue
            for c in range(step, n + 1):
                row[c] -= mult * top[c]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        row = aug[i]
        acc = row[n]
        for c in range(i + 1, n):
            acc -= row[c] * x[c]
        x[i] = acc / row[i]
    return x


def solve_spd_banded(rows: Sequence[Dict[int, float]], b: Sequence[float],
                     bandwidth: int) -> List[float]:
    """Solve a symmetric positive definite banded system, given sparse rows.

    ``rows[i][j]`` holds ``A[i][j]``; only entries within ``bandwidth`` of the
    diagonal participate.  The factorisation is ``A = L D L^T`` with a unit
    lower triangular ``L``, which needs no square roots and no pivoting -- both
    are safe because a properly restrained elastic body gives a positive
    definite operator.  A pivot that is not positive means that assumption was
    violated, and ``ValueError`` is raised so the caller can switch to a
    general solver.

    The factor is held in dense band storage: ``band[i]`` is a list of
    ``bandwidth`` floats whose slot ``p`` means column ``i - bandwidth + p``.
    Contiguous lists beat dictionaries here because the inner product below is
    the hot loop of the whole module.
    """
    n = len(b)
    bw = max(1, int(bandwidth))
    band: List[List[float]] = [[0.0] * bw for _ in range(n)]
    diag = [0.0] * n

    for i in range(n):
        row_i = rows[i]
        band_i = band[i]
        base_i = i - bw  # column of band_i[0]
        for j in range(max(0, base_i), i):
            band_j = band[j]
            base_j = j - bw
            acc = row_i.get(j, 0.0)
            # Both factors must be inside their own band for the term to exist.
            for k in range(max(0, base_i, base_j), j):
                lik = band_i[k - base_i]
                if lik == 0.0:
                    continue
                ljk = band_j[k - base_j]
                if ljk != 0.0:
                    acc -= lik * ljk * diag[k]
            band_i[j - base_i] = acc / diag[j]

        pivot = row_i.get(i, 0.0)
        for p in range(max(0, base_i) - base_i, bw):
            lik = band_i[p]
            if lik != 0.0:
                pivot -= lik * lik * diag[p + base_i]
        if pivot <= 1e-300:
            raise ValueError("non-SPD pivot in banded factorisation")
        diag[i] = pivot

    # L z = b
    z = list(b)
    for i in range(n):
        band_i = band[i]
        base_i = i - bw
        acc = z[i]
        for j in range(max(0, base_i), i):
            lij = band_i[j - base_i]
            if lij != 0.0:
                acc -= lij * z[j]
        z[i] = acc
    # D y = z
    for i in range(n):
        z[i] /= diag[i]
    # L^T x = y, solved in place from the bottom up.
    for i in range(n - 1, -1, -1):
        band_i = band[i]
        base_i = i - bw
        xi = z[i]
        for j in range(max(0, base_i), i):
            lij = band_i[j - base_i]
            if lij != 0.0:
                z[j] -= lij * xi
    return z


# ==========================================================================
# Element physics
# ==========================================================================

def _plane_stress_moduli(E: float, nu: float) -> Tuple[float, float, float]:
    """Return the three distinct entries of the plane-stress elasticity matrix.

    ``D = [[d11, d12, 0], [d12, d11, 0], [0, 0, d33]]`` relating the strain
    triple ``(exx, eyy, gxy)`` to stress.
    """
    scale = E / (1.0 - nu * nu)
    return scale, scale * nu, scale * (1.0 - nu) * 0.5


def _shape_gradients(xi: float, eta: float) -> Tuple[List[float], List[float]]:
    """Cartesian derivatives of the four bilinear shape functions.

    Evaluated on the unit square with corners taken in the counter-clockwise
    order ``(0,0), (1,0), (1,1), (0,1)``.  Because the element is the unit
    square the Jacobian is the identity, so the derivatives with respect to the
    reference coordinates are already the physical ones and the differential
    volume is one.
    """
    # N1 = (1-xi)(1-eta), N2 = xi(1-eta), N3 = xi*eta, N4 = (1-xi)eta
    d_dx = [-(1.0 - eta), (1.0 - eta), eta, -eta]
    d_dy = [-(1.0 - xi), -xi, xi, (1.0 - xi)]
    return d_dx, d_dy


def ke_q4(E: float, nu: float) -> List[List[float]]:
    """Stiffness of a unit-square bilinear quadrilateral in plane stress.

    Eight degrees of freedom, ordered ``(ux, uy)`` per corner in the same
    counter-clockwise order the mesh uses.  Integrated exactly by a two-by-two
    Gauss rule: the integrand is bilinear in each coordinate, so two points per
    direction is not an approximation here.

    The result is independent of the physical element size because the element
    is a unit square, which is what keeps the benchmark below mesh-independent.
    """
    d11, d12, d33 = _plane_stress_moduli(float(E), float(nu))
    K = [[0.0] * 8 for _ in range(8)]

    for xi in _GAUSS_1D:
        for eta in _GAUSS_1D:
            d_dx, d_dy = _shape_gradients(xi, eta)
            # Strain-displacement rows for this quadrature point, laid out per
            # degree of freedom: exx picks up d/dx of the x-displacements, eyy
            # picks up d/dy of the y-displacements, and the shear row mixes.
            b_xx = [0.0] * 8
            b_yy = [0.0] * 8
            b_xy = [0.0] * 8
            for c in range(4):
                b_xx[2 * c] = d_dx[c]
                b_yy[2 * c + 1] = d_dy[c]
                b_xy[2 * c] = d_dy[c]
                b_xy[2 * c + 1] = d_dx[c]
            # Accumulate B^T D B; the Gauss weight is 0.25 per point on a unit
            # square (weight 0.5 per direction after the [-1,1] -> [0,1] map).
            for a in range(8):
                sxx = d11 * b_xx[a] + d12 * b_yy[a]
                syy = d12 * b_xx[a] + d11 * b_yy[a]
                sxy = d33 * b_xy[a]
                row = K[a]
                for b in range(8):
                    row[b] += 0.25 * (sxx * b_xx[b] + syy * b_yy[b]
                                      + sxy * b_xy[b])

    # Force exact symmetry: the assembly above is symmetric mathematically, but
    # floating-point accumulation order can leave a few ulps of asymmetry, and
    # the banded factorisation assumes a symmetric operator.
    for a in range(8):
        for b in range(a + 1, 8):
            mean = 0.5 * (K[a][b] + K[b][a])
            K[a][b] = mean
            K[b][a] = mean
    return K


# ==========================================================================
# Structured grid
# ==========================================================================

class Mesh2D:
    """A ``nelx`` by ``nely`` grid of unit quadrilateral elements.

    Nodes are numbered column-major, ``node(i, j) = i * (nely + 1) + j``, with
    ``i`` running along x and ``j`` along y; that ordering is what keeps the
    stiffness matrix banded.  Node ``n`` owns degrees of freedom ``2n`` (x) and
    ``2n + 1`` (y).  Elements follow the same column-major convention.
    """

    def __init__(self, nelx: int, nely: int) -> None:
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.nnodes = (self.nelx + 1) * (self.nely + 1)
        self.ndof = 2 * self.nnodes
        self.nel = self.nelx * self.nely
        # Connectivity is queried once per element per assembly, so it is
        # cheaper to build the whole table once at construction.
        self._edofs: List[List[int]] = [[] for _ in range(self.nel)]
        for ex in range(self.nelx):
            for ey in range(self.nely):
                corners = (self.node(ex, ey), self.node(ex + 1, ey),
                           self.node(ex + 1, ey + 1), self.node(ex, ey + 1))
                dofs: List[int] = []
                for c in corners:
                    dofs.append(2 * c)
                    dofs.append(2 * c + 1)
                self._edofs[self.elem(ex, ey)] = dofs

    def node(self, i: int, j: int) -> int:
        return i * (self.nely + 1) + j

    def elem(self, ex: int, ey: int) -> int:
        return ex * self.nely + ey

    def cell(self, e: int) -> Tuple[int, int]:
        """Inverse of :meth:`elem`: the ``(column, row)`` of element ``e``."""
        return divmod(e, self.nely)

    def elem_centroid(self, e: int) -> Tuple[float, float]:
        ex, ey = self.cell(e)
        return (ex + 0.5, ey + 0.5)

    def edof(self, e: int) -> List[int]:
        return self._edofs[e]

    def cells(self) -> Iterator[Tuple[int, int, int]]:
        """Iterate ``(element, column, row)`` in element-index order."""
        for e in range(self.nel):
            ex, ey = divmod(e, self.nely)
            yield e, ex, ey

    def half_bandwidth(self) -> int:
        """Upper bound on the distance between coupled free degrees of freedom.

        Two nodes only interact when they share an element, so their column
        indices differ by at most one; column-major numbering turns that into a
        node-index gap of at most ``nely + 1``, hence twice that in degrees of
        freedom plus a small margin for the intra-node pairing.
        """
        return 2 * (self.nely + 1) + 4


# ==========================================================================
# Density filter
# ==========================================================================

def build_filter(mesh: Mesh2D, rmin: float) -> List[List[Tuple[int, float]]]:
    """Neighbour weights of a conical filter of radius ``rmin``.

    Entry ``e`` lists ``(neighbour, weight)`` pairs with weight ``rmin - d``
    over the disc of radius ``rmin`` around element ``e``, measured between
    cell centres.  The weight is strictly positive at the centre, so an element
    is always its own neighbour, and the support grows monotonically with the
    radius.  A radius below one cell is meaningless on this grid and is lifted
    to one.
    """
    radius = max(1.0, float(rmin))
    reach = int(math.ceil(radius))

    # The kernel is translation invariant, so its offsets and weights are
    # computed once and then clipped against the domain boundary per element.
    stencil: List[Tuple[int, int, float]] = []
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            weight = radius - math.hypot(dx, dy)
            if weight > 0.0:
                stencil.append((dx, dy, weight))

    table: List[List[Tuple[int, float]]] = [[] for _ in range(mesh.nel)]
    for e, ex, ey in mesh.cells():
        neighbours: List[Tuple[int, float]] = []
        for dx, dy, weight in stencil:
            kx = ex + dx
            ky = ey + dy
            if 0 <= kx < mesh.nelx and 0 <= ky < mesh.nely:
                neighbours.append((mesh.elem(kx, ky), weight))
        table[e] = neighbours
    return table


def apply_filter(vec: Sequence[float],
                 weights: Sequence[Sequence[Tuple[int, float]]]) -> List[float]:
    """Convolve ``vec`` with a weight table, renormalised per element.

    Dividing by the local weight sum makes the operator a convex average even
    where the kernel is clipped by the boundary, so constants are preserved and
    no material is invented at the edges of the domain.
    """
    out: List[float] = []
    for e, neighbours in enumerate(weights):
        numer = 0.0
        denom = 0.0
        for j, w in neighbours:
            numer += w * vec[j]
            denom += w
        out.append(numer / denom if denom > 0.0 else vec[e])
    return out


# ==========================================================================
# Manufacturing rules
# ==========================================================================

def _mirror_pairs(mesh: Mesh2D) -> List[Tuple[int, int]]:
    """Element pairs reflected through the vertical mid-plane of the domain."""
    pairs: List[Tuple[int, int]] = []
    for ex in range(mesh.nelx // 2):
        for ey in range(mesh.nely):
            pairs.append((mesh.elem(ex, ey),
                          mesh.elem(mesh.nelx - 1 - ex, ey)))
    return pairs


def _symmetrise(values: List[float], pairs: Sequence[Tuple[int, int]]) -> None:
    """Average every mirrored pair in place, giving exact reflection symmetry."""
    for left, right in pairs:
        mid = 0.5 * (values[left] + values[right])
        values[left] = mid
        values[right] = mid


def _support_reach(angle_deg: float) -> int:
    """Sideways cell span a self-supporting cone covers per layer.

    At a build angle measured from the base plate, one layer of height buys
    ``1 / tan(angle)`` of horizontal offset; a steep angle therefore admits no
    lateral offset at all and demands support directly underneath.
    """
    angle = min(89.999, max(1e-6, float(angle_deg)))
    return max(0, int(round(1.0 / math.tan(math.radians(angle)))))


def _unsupported_cells(mesh: Mesh2D, density: Sequence[float],
                       angle_deg: float, threshold: float) -> int:
    """Count solid cells with nothing solid inside the cone one layer below."""
    reach = _support_reach(angle_deg)
    count = 0
    for e, ex, ey in mesh.cells():
        if ey == 0 or density[e] <= threshold:
            continue  # the bottom layer rests on the base plate
        for kx in range(max(0, ex - reach), min(mesh.nelx, ex + reach + 1)):
            if density[mesh.elem(kx, ey - 1)] > threshold:
                break
        else:
            count += 1
    return count


def _cap_by_support(mesh: Mesh2D, density: List[float], angle_deg: float) -> None:
    """Clamp each cell to the strongest cell in the cone below it.

    Sweeping upward means the row underneath is already conformant when it is
    read, so one pass suffices and the result is self-supporting everywhere.
    Material is only ever removed.
    """
    reach = _support_reach(angle_deg)
    for ey in range(1, mesh.nely):
        for ex in range(mesh.nelx):
            lo = max(0, ex - reach)
            hi = min(mesh.nelx, ex + reach + 1)
            ceiling = max(density[mesh.elem(kx, ey - 1)] for kx in range(lo, hi))
            e = mesh.elem(ex, ey)
            if density[e] > ceiling:
                density[e] = ceiling


def _make_drawable(mesh: Mesh2D, density: List[float]) -> None:
    """Remove undercuts along the -y extraction axis.

    A mould can only release if no cell is thinner than what sits above it, so
    each column is turned into a running maximum taken downward.
    """
    for ex in range(mesh.nelx):
        running = 0.0
        for ey in range(mesh.nely - 1, -1, -1):
            e = mesh.elem(ex, ey)
            if density[e] > running:
                running = density[e]
            density[e] = running


def _repair(mesh: Mesh2D, density: List[float],
            pairs: Sequence[Tuple[int, int]],
            draw: bool, overhang: Optional[float]) -> bool:
    """Apply every enabled manufacturing rule; report whether volume may drift.

    Symmetry preserves the mean, but draw direction and overhang capping do
    not, so the caller is told when a volume rescale is owed.
    """
    if pairs:
        _symmetrise(density, pairs)
    if draw:
        _make_drawable(mesh, density)
    if overhang is not None:
        _cap_by_support(mesh, density, overhang)
    return draw or overhang is not None


def _rescale_to_volume(density: List[float], volfrac: float) -> List[float]:
    """Scale a field back onto its volume budget, respecting the box bounds."""
    mean = sum(density) / len(density)
    if mean <= 1e-12:
        return density
    factor = volfrac / mean
    return [min(_X_MAX, max(_X_MIN, v * factor)) for v in density]


# ==========================================================================
# Analysis
# ==========================================================================

def _stiffness_scale(x: float, penal: float, Emin: float) -> float:
    """Power-law modulus of an element at density ``x`` (with ``E0 = 1``)."""
    return Emin + (x ** penal) * (1.0 - Emin)


def _analyse(mesh: Mesh2D, density: Sequence[float],
             KE: Sequence[Sequence[float]], penal: float, Emin: float,
             fixed: Sequence[int],
             load: Sequence[float]) -> Tuple[float, List[float]]:
    """Solve one load case; return its compliance and per-element strain energy.

    The energy returned is that of the *unscaled* element, i.e. ``u_e^T KE
    u_e``; multiplying by the power law gives the element's contribution to
    compliance, and differentiating the power law gives the sensitivity.  That
    split lets a multi-load-case caller reuse one solve for both.
    """
    restrained = set(fixed)
    free = [d for d in range(mesh.ndof) if d not in restrained]
    slot = {d: i for i, d in enumerate(free)}

    # Assemble directly in the reduced (free-free) numbering; the restrained
    # rows and columns never exist, which is what makes the operator definite.
    reduced: List[Dict[int, float]] = [dict() for _ in free]
    for e in range(mesh.nel):
        scale = _stiffness_scale(density[e], penal, Emin)
        dofs = mesh.edof(e)
        local = [slot.get(d) for d in dofs]
        for a in range(8):
            ra = local[a]
            if ra is None:
                continue
            target = reduced[ra]
            ke_row = KE[a]
            for b in range(8):
                cb = local[b]
                if cb is None:
                    continue
                contrib = scale * ke_row[b]
                if contrib != 0.0:
                    target[cb] = target.get(cb, 0.0) + contrib

    rhs = [load[d] for d in free]
    try:
        reduced_u = solve_spd_banded(reduced, rhs, mesh.half_bandwidth())
    except ValueError:
        # Under-restrained or otherwise indefinite: pay for a general solve.
        n = len(free)
        square = [[0.0] * n for _ in range(n)]
        for i, sparse_row in enumerate(reduced):
            dense_row = square[i]
            for j, v in sparse_row.items():
                dense_row[j] = v
        reduced_u = solve_dense(square, rhs)

    u = [0.0] * mesh.ndof
    for i, d in enumerate(free):
        u[d] = reduced_u[i]

    energy = [0.0] * mesh.nel
    compliance = 0.0
    for e in range(mesh.nel):
        ue = [u[d] for d in mesh.edof(e)]
        quad = 0.0
        for a in range(8):
            ua = ue[a]
            if ua == 0.0:
                continue
            ke_row = KE[a]
            inner = 0.0
            for b in range(8):
                inner += ke_row[b] * ue[b]
            quad += ua * inner
        energy[e] = quad
        compliance += _stiffness_scale(density[e], penal, Emin) * quad
    return compliance, energy


# ==========================================================================
# Optimality criteria update
# ==========================================================================

def _oc_step(x: Sequence[float], dc: Sequence[float], volfrac: float,
             move: float = 0.2) -> List[float]:
    """One optimality-criteria step at the volume-feasible multiplier.

    For a compliance objective the stationarity condition gives the scaling
    ``x <- x * sqrt(-dc / lam)``.  The delivered volume decreases monotonically
    in ``lam``, so a plain bisection on a bracket that is wide enough to be
    problem independent finds the multiplier that meets the budget.
    """
    n = len(x)
    target = volfrac * n
    lo, hi = 1e-12, 1e12
    proposal = list(x)

    def propose(lam: float) -> float:
        total = 0.0
        for i in range(n):
            ratio = -dc[i] / lam
            growth = math.sqrt(ratio) if ratio > 0.0 else 0.0
            floor = max(_X_MIN, x[i] - move)
            ceiling = min(_X_MAX, x[i] + move)
            value = min(ceiling, max(floor, x[i] * growth))
            proposal[i] = value
            total += value
        return total

    while (hi - lo) / (lo + hi) > 1e-9:
        mid = 0.5 * (lo + hi)
        if propose(mid) > target:
            lo = mid   # too much material: the multiplier must grow
        else:
            hi = mid
    return proposal


# ==========================================================================
# Benchmark problem
# ==========================================================================

def mbb_problem(nelx: int, nely: int) -> Dict[str, Any]:
    """Restraints and load of the half-beam benchmark.

    Only half the beam is modelled: the left edge carries the symmetry
    condition (x fixed, y free), a single roller under the bottom-right corner
    takes the vertical reaction, and a downward unit force is applied at the
    top-left node, i.e. on the symmetry plane where the real beam is loaded at
    mid-span.
    """
    mesh = Mesh2D(nelx, nely)
    restrained = [2 * mesh.node(0, j) for j in range(nely + 1)]
    restrained.append(2 * mesh.node(nelx, 0) + 1)
    load = [0.0] * mesh.ndof
    load[2 * mesh.node(0, nely) + 1] = -1.0
    return {"fixed": restrained, "F": load, "nelx": nelx, "nely": nely}


# ==========================================================================
# Driver
# ==========================================================================

def optimize(
    nelx: int,
    nely: int,
    volfrac: float,
    *,
    load_cases: Optional[Sequence[Dict[str, Any]]] = None,
    load_weights: Optional[Sequence[float]] = None,
    fixed: Optional[Sequence[int]] = None,
    penal: float = 3.0,
    rmin: float = 1.5,
    max_iter: int = 60,
    tol: float = 1e-3,
    nu: float = 0.3,
    Emin: float = 1e-9,
    symmetry: bool = False,
    overhang_angle: Optional[float] = None,
    draw_direction: bool = False,
    x_init: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Minimise weighted compliance at a fixed volume fraction.  Never raises.

    ``load_cases`` is a sequence of ``{"F": [...], "fixed": [...]}`` mappings;
    the objective is their ``load_weights``-weighted sum of compliances, the
    weights defaulting to a uniform average.  With no load cases the benchmark
    problem above is used.

    Returns ``{"ok": True, "compliance", "volume_fraction", "iterations",
    "converged", "history", "density", "nelx", "nely", "n_load_cases"}``, or
    ``{"ok": False, "reason": ...}`` if the request is malformed.
    """
    try:
        if not (0.0 < volfrac < 1.0):
            return {"ok": False, "reason": "volfrac must be in (0, 1)"}
        if nelx < 1 or nely < 1:
            return {"ok": False, "reason": "nelx and nely must be >= 1"}

        mesh = Mesh2D(nelx, nely)

        cases = list(load_cases) if load_cases else []
        if not cases:
            bench = mbb_problem(nelx, nely)
            cases = [{"F": bench["F"], "fixed": bench["fixed"]}]
            if fixed is None:
                fixed = bench["fixed"]
        ncase = len(cases)

        if load_weights is None:
            weights_per_case = [1.0 / ncase] * ncase
        else:
            weights_per_case = list(load_weights)
            if len(weights_per_case) != ncase:
                return {"ok": False, "reason": "load_weights length != load_cases"}

        for case in cases:
            forces = case.get("F")
            if forces is None or len(forces) != mesh.ndof:
                return {"ok": False,
                        "reason": "load case F missing or wrong length"}

        if x_init is None:
            x = [float(volfrac)] * mesh.nel
        else:
            x = [min(_X_MAX, max(_X_MIN, float(v))) for v in x_init]
            if len(x) != mesh.nel:
                return {"ok": False, "reason": "x_init length != element count"}

        KE = ke_q4(1.0, nu)
        kernel = build_filter(mesh, rmin)
        pairs = _mirror_pairs(mesh) if symmetry else []
        default_fixed = list(fixed) if fixed is not None else []

        def physical(raw: Sequence[float]) -> List[float]:
            """Filtered densities, symmetrised if the constraint is active."""
            field = apply_filter(raw, kernel)
            if pairs:
                _symmetrise(field, pairs)
            return field

        history: List[Dict[str, float]] = []
        previous: Optional[float] = None
        change = 1.0
        iteration = 0

        for iteration in range(1, int(max_iter) + 1):
            xphys = physical(x)

            # -- objective and sensitivity, summed over the load cases
            objective = 0.0
            dc = [0.0] * mesh.nel
            for weight, case in zip(weights_per_case, cases):
                restrained = case.get("fixed", default_fixed)
                compliance, energy = _analyse(mesh, xphys, KE, penal, Emin,
                                              restrained, case["F"])
                objective += weight * compliance
                for e in range(mesh.nel):
                    slope = penal * (xphys[e] ** (penal - 1.0)) * (1.0 - Emin)
                    dc[e] -= weight * slope * energy[e]

            # The update sees the same smoothing the physics saw.
            dc = apply_filter(dc, kernel)
            if pairs:
                _symmetrise(dc, pairs)

            candidate = _oc_step(x, dc, volfrac)
            if _repair(mesh, candidate, pairs, draw_direction, overhang_angle):
                # Repair only carves material away, so put the budget back.
                candidate = _rescale_to_volume(candidate, volfrac)

            change = max(abs(new - old) for new, old in zip(candidate, x))
            x = candidate

            history.append({
                "iter": float(iteration),
                "compliance": objective,
                "volume": sum(apply_filter(x, kernel)) / mesh.nel,
                "change": change,
            })

            # Stop once neither the objective nor the design is still moving.
            if previous is not None and previous > 0.0:
                if abs(objective - previous) / previous < tol and change < 0.02:
                    previous = objective
                    break
            previous = objective

        final = physical(x)
        _repair(mesh, final, pairs, draw_direction, overhang_angle)

        return {
            "ok": True,
            "compliance": history[-1]["compliance"] if history else 0.0,
            "volume_fraction": sum(final) / mesh.nel,
            "iterations": iteration,
            "converged": previous is not None and len(history) >= 2
                         and change < 0.05,
            "history": history,
            "density": final,
            "nelx": nelx,
            "nely": nely,
            "n_load_cases": ncase,
        }
    except Exception as exc:  # keep the entry point total
        return {"ok": False, "reason": f"optimize failed: {exc}"}


def pareto_sweep(nelx: int, nely: int, volfracs: Sequence[float],
                 **kwargs: Any) -> Dict[str, Any]:
    """Trace the stiffness/material trade-off by re-solving at each budget.

    Returns ``{"ok": True, "front": [{"volume_fraction", "compliance"}, ...]}``
    in the order requested.  Never raises; the first failing run aborts the
    sweep and its reason is passed through.
    """
    try:
        if not volfracs:
            return {"ok": False, "reason": "volfracs must be non-empty"}
        front: List[Dict[str, float]] = []
        for vf in volfracs:
            run = optimize(nelx, nely, vf, **kwargs)
            if not run.get("ok"):
                return {"ok": False,
                        "reason": f"sweep failed at volfrac={vf}: "
                                  f"{run.get('reason')}"}
            front.append({"volume_fraction": run["volume_fraction"],
                          "compliance": run["compliance"]})
        return {"ok": True, "front": front}
    except Exception as exc:  # keep the entry point total
        return {"ok": False, "reason": f"pareto_sweep failed: {exc}"}


# ==========================================================================
# Independent verification oracles
# ==========================================================================

def count_overhang_violations(nelx: int, nely: int, density: Sequence[float],
                              angle_deg: float, threshold: float = 0.5) -> int:
    """Solid cells that would print into thin air at the given build angle."""
    return _unsupported_cells(Mesh2D(nelx, nely), density, angle_deg, threshold)


def _solid_components(mesh: Mesh2D,
                      solid: Sequence[bool]) -> Iterator[List[int]]:
    """Yield the edge-connected components of the solid set, smallest index first.

    Four-connectivity: cells touching only at a corner are separate bodies,
    which is the conservative reading for a printable part.
    """
    visited = [False] * mesh.nel
    for seed, _sx, _sy in mesh.cells():
        if visited[seed] or not solid[seed]:
            continue
        visited[seed] = True
        frontier = [seed]
        members: List[int] = []
        while frontier:
            here = frontier.pop()
            members.append(here)
            hx, hy = mesh.cell(here)
            for kx, ky in ((hx + 1, hy), (hx - 1, hy),
                           (hx, hy + 1), (hx, hy - 1)):
                if 0 <= kx < mesh.nelx and 0 <= ky < mesh.nely:
                    nb = mesh.elem(kx, ky)
                    if solid[nb] and not visited[nb]:
                        visited[nb] = True
                        frontier.append(nb)
        yield members


def isolated_island_count(nelx: int, nely: int, density: Sequence[float],
                          max_cells: int, threshold: float = 0.5) -> int:
    """How many connected solid bodies are no larger than ``max_cells``.

    Small disconnected bodies are the usual symptom of a filter radius that is
    too small for the mesh; widening the radius washes them out, so this count
    is a practical monotone probe of minimum-member-size behaviour.
    """
    mesh = Mesh2D(nelx, nely)
    solid = [density[e] > threshold for e in range(mesh.nel)]
    return sum(1 for members in _solid_components(mesh, solid)
               if len(members) <= max_cells)


def min_member_ok(nelx: int, nely: int, density: Sequence[float], rmin: float,
                  threshold: float = 0.5) -> bool:
    """True when no solid feature is thinner than ``rmin`` cells.

    A feature at least ``rmin`` wide contains at least one cell whose whole
    disc of radius ``rmin / 2`` is solid -- a point on its medial axis --
    whereas a narrower feature has none, because every one of its cells is
    within half a radius of the void.  So the test is: erode by that disc, then
    require each connected body to retain at least one surviving core cell.

    Working per body rather than comparing the erosion's dilation against the
    original avoids relying on morphological opening being idempotent, which a
    digital disc on a coarse grid never quite is.
    """
    mesh = Mesh2D(nelx, nely)
    radius = max(1.0, float(rmin)) / 2.0
    span = int(math.ceil(radius))
    disc = [(dx, dy)
            for dx in range(-span, span + 1)
            for dy in range(-span, span + 1)
            if dx * dx + dy * dy <= radius * radius]

    solid = [density[e] > threshold for e in range(mesh.nel)]

    def is_core(ex: int, ey: int) -> bool:
        for dx, dy in disc:
            kx, ky = ex + dx, ey + dy
            if not (0 <= kx < mesh.nelx and 0 <= ky < mesh.nely):
                return False   # the disc left the domain: too close to an edge
            if not solid[mesh.elem(kx, ky)]:
                return False
        return True

    core = [False] * mesh.nel
    for e, ex, ey in mesh.cells():
        if solid[e]:
            core[e] = is_core(ex, ey)

    for members in _solid_components(mesh, solid):
        if not any(core[e] for e in members):
            return False
    return True


# ==========================================================================
# selfcheck
# ==========================================================================

def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, label: str) -> None:
        if not cond:
            failures.append(label)

    # -- Q4 element stiffness: symmetry and the rigid-body null space.
    KE = ke_q4(1.0, 0.3)
    check(len(KE) == 8 and all(len(r) == 8 for r in KE), "KE is 8x8")
    check(all(abs(KE[i][j] - KE[j][i]) < 1e-12 for i in range(8)
              for j in range(8)), "KE is symmetric")
    for name, rb in (("x-translation", [1.0, 0.0] * 4),
                     ("y-translation", [0.0, 1.0] * 4)):
        force = [sum(KE[a][b] * rb[b] for b in range(8)) for a in range(8)]
        check(max(abs(f) for f in force) < 1e-10,
              f"KE has a zero-energy {name} (rigid-body null space)")
    # Positive semi-definite: no deformation may release energy.
    probes = [[1.0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0, 0, 0],
              [1.0, -1.0, 0.5, 0.25, 0, 0, -0.5, 1.0], [0.1] * 8,
              [1.0, 2.0, -3.0, 0.5, 0.5, -1.0, 2.0, 0.0]]
    for p in probes:
        energy = sum(p[a] * KE[a][b] * p[b] for a in range(8) for b in range(8))
        check(energy >= -1e-12, "KE is positive semi-definite")
    # The quadrature must reproduce the analytic plane-stress entries.
    nu = 0.3
    f = 1.0 / (1.0 - nu * nu)
    check(abs(KE[0][0] - f * (0.5 - nu / 6.0)) < 1e-12,
          "KE diagonal matches the analytic plane-stress value")
    check(abs(KE[0][1] - f * (0.125 + nu / 8.0)) < 1e-12,
          "KE coupling term matches the analytic plane-stress value")

    # -- Banded LDL^T agrees with dense Gauss on an SPD system.
    dense_A = [[4.0, 1.0, 0.0, 0.0], [1.0, 3.0, 1.0, 0.0],
               [0.0, 1.0, 5.0, 2.0], [0.0, 0.0, 2.0, 6.0]]
    rhs = [1.0, 2.0, 3.0, 4.0]
    rows = [{j: v for j, v in enumerate(r) if v != 0.0} for r in dense_A]
    x_band = solve_spd_banded(rows, rhs, 2)
    x_dense = solve_dense(dense_A, rhs)
    check(max(abs(a - b) for a, b in zip(x_band, x_dense)) < 1e-9,
          "banded LDL^T matches dense Gauss")
    resid = [sum(dense_A[i][j] * x_band[j] for j in range(4)) - rhs[i]
             for i in range(4)]
    check(max(abs(r) for r in resid) < 1e-9, "banded solve residual is zero")
    try:
        solve_spd_banded([{0: -1.0}], [1.0], 1)
        check(False, "non-SPD pivot must raise")
    except ValueError:
        check(True, "non-SPD pivot raises ValueError for the dense fallback")
    try:
        solve_dense([[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0])
        check(False, "a singular dense system must raise")
    except ValueError:
        check(True, "a singular dense system raises ValueError")

    # -- Mesh numbering.
    m = Mesh2D(4, 3)
    check(m.nel == 12 and m.nnodes == 20 and m.ndof == 40, "mesh counts")
    check(m.node(0, 0) == 0 and m.node(1, 0) == 4, "column-major node numbering")
    check(all(m.cell(m.elem(ex, ey)) == (ex, ey)
              for ex in range(4) for ey in range(3)),
          "elem and cell are mutual inverses")
    check(len(m.edof(0)) == 8, "each Q4 element has 8 DOFs")
    check(len(set(m.edof(0))) == 8, "element DOFs are distinct")
    # Adjacent elements in a column share exactly one edge (2 nodes = 4 DOFs).
    check(len(set(m.edof(m.elem(0, 0))) & set(m.edof(m.elem(0, 1)))) == 4,
          "vertically adjacent elements share an edge")
    check(len(set(m.edof(m.elem(0, 0))) & set(m.edof(m.elem(2, 0)))) == 0,
          "non-adjacent elements share no DOFs")

    # -- Filter is a partition of unity: a constant field is its own filtrate.
    w = build_filter(m, 1.5)
    const = [0.37] * m.nel
    filt = apply_filter(const, w)
    check(max(abs(v - 0.37) for v in filt) < 1e-12,
          "filtering a constant field preserves it (partition of unity)")
    check(all(nb for nb in w), "every element has at least one filter neighbour")
    check(all(any(j == e for j, _ in w[e]) for e in range(m.nel)),
          "the filter kernel always includes the element itself")
    # A larger radius reaches at least as far.
    w_big = build_filter(m, 2.5)
    check(all(len(w_big[e]) >= len(w[e]) for e in range(m.nel)),
          "a larger filter radius never shrinks the kernel")

    # -- MBB problem wiring.
    mbb = mbb_problem(6, 4)
    check(len(mbb["fixed"]) == (4 + 1) + 1,
          "MBB fixes the left edge in x plus one roller")
    check(len(set(mbb["fixed"])) == len(mbb["fixed"]), "MBB fixed DOFs distinct")
    check(sum(1 for f in mbb["F"] if f != 0.0) == 1, "MBB has a single point load")
    check(min(mbb["F"]) == -1.0, "MBB load is a downward unit force")

    # -- The core regression: SIMP on the MBB beam.
    res = optimize(12, 6, 0.5, rmin=1.5, max_iter=14)
    check(res.get("ok") is True, "MBB optimize succeeds")
    if res.get("ok"):
        check(res["compliance"] > 0.0 and math.isfinite(res["compliance"]),
              "compliance is finite and positive")
        check(abs(res["volume_fraction"] - 0.5) < 0.05,
              "volume constraint is met at convergence")
        check(len(res["density"]) == 12 * 6, "density field covers every element")
        check(all(_X_MIN - 1e-9 <= d <= _X_MAX + 1e-9 for d in res["density"]),
              "densities stay inside the SIMP box [1e-3, 1]")
        hist = res["history"]
        check(len(hist) >= 2, "history records each iteration")
        check(hist[-1]["compliance"] < hist[0]["compliance"],
              "the optimizer actually reduces compliance")
        check(all(math.isfinite(h["compliance"]) for h in hist),
              "no iteration produced a non-finite compliance")
        # Optimizing must beat the uniform starting design it began from.
        uniform = optimize(12, 6, 0.5, rmin=1.5, max_iter=1)
        check(uniform.get("ok") and res["compliance"] < uniform["compliance"],
              "the converged design beats the uniform initial design")

    # -- Determinism: identical inputs give bit-identical output.
    r1 = optimize(8, 4, 0.5, rmin=1.5, max_iter=6)
    r2 = optimize(8, 4, 0.5, rmin=1.5, max_iter=6)
    check(r1["density"] == r2["density"] and r1["compliance"] == r2["compliance"],
          "optimize is deterministic")

    # -- Symmetry constraint couples mirrored elements exactly.
    rs = optimize(8, 4, 0.5, rmin=1.5, max_iter=6, symmetry=True)
    check(rs.get("ok") is True, "symmetric optimize succeeds")
    if rs.get("ok"):
        ms = Mesh2D(8, 4)
        d = rs["density"]
        check(all(abs(d[a] - d[b]) < 1e-12 for a, b in _mirror_pairs(ms)),
              "symmetry=True yields an exactly mirror-symmetric density field")
        # Not vacuous: the unconstrained MBB design is NOT mirror-symmetric
        # (its load sits at one corner), so the constraint really did bind.
        du = r1["density"]
        check(not all(abs(du[a] - du[b]) < 1e-12 for a, b in _mirror_pairs(ms)),
              "the unconstrained design is asymmetric, so symmetry=True binds")

    # -- Draw direction: density is monotone non-decreasing downward.
    rd = optimize(8, 4, 0.5, rmin=1.5, max_iter=6, draw_direction=True)
    check(rd.get("ok") is True, "draw-direction optimize succeeds")
    if rd.get("ok"):
        md = Mesh2D(8, 4)
        d = rd["density"]
        mono = all(d[md.elem(ex, ey)] >= d[md.elem(ex, ey + 1)] - 1e-12
                   for ex in range(8) for ey in range(3))
        check(mono, "draw_direction leaves no undercut (monotone along -y)")

    # -- Overhang repair leaves no unsupported solid.
    ro = optimize(8, 4, 0.5, rmin=1.5, max_iter=6, overhang_angle=45.0)
    check(ro.get("ok") is True, "overhang-constrained optimize succeeds")
    if ro.get("ok"):
        check(count_overhang_violations(8, 4, ro["density"], 45.0) == 0,
              "overhang repair leaves zero violations at the build angle")

    # -- The overhang oracle is not vacuous: a floating blob does violate.
    floating = [0.0] * (4 * 4)
    mf = Mesh2D(4, 4)
    floating[mf.elem(1, 2)] = 1.0
    floating[mf.elem(2, 2)] = 1.0
    check(count_overhang_violations(4, 4, floating, 89.0) == 2,
          "an unsupported floating blob is reported as violating")
    grounded = [0.0] * (4 * 4)
    for ey in range(4):
        grounded[mf.elem(1, ey)] = 1.0
    check(count_overhang_violations(4, 4, grounded, 89.0) == 0,
          "a column resting on the base plate is fully self-supporting")
    # A shallow angle bridges sideways, so the same blob becomes legal once it
    # sits one cell diagonally above solid material.
    stair = [0.0] * (4 * 4)
    stair[mf.elem(1, 0)] = 1.0
    stair[mf.elem(2, 1)] = 1.0
    check(count_overhang_violations(4, 4, stair, 45.0) == 0,
          "a 45-degree staircase is self-supporting at 45 degrees")
    check(count_overhang_violations(4, 4, stair, 89.0) == 1,
          "the same staircase is unsupported at a steep angle")

    # -- Island oracle counts small components, not large ones.
    blobs = [0.0] * (6 * 6)
    mb = Mesh2D(6, 6)
    blobs[mb.elem(0, 0)] = 1.0                    # 1-cell island
    blobs[mb.elem(5, 5)] = 1.0                    # 1-cell island
    for ex in range(2, 5):
        for ey in range(2, 5):
            blobs[mb.elem(ex, ey)] = 1.0          # 9-cell blob
    check(isolated_island_count(6, 6, blobs, max_cells=1) == 2,
          "both single-cell islands are counted")
    check(isolated_island_count(6, 6, blobs, max_cells=20) == 3,
          "raising max_cells also counts the large blob")
    check(isolated_island_count(6, 6, [0.0] * 36, max_cells=5) == 0,
          "an empty field has no islands")

    # -- min_member_ok: a hairline fails, a thick block passes.
    thin = [0.0] * (8 * 8)
    mt = Mesh2D(8, 8)
    for ey in range(8):
        thin[mt.elem(3, ey)] = 1.0                # 1-cell-wide wall
    check(not min_member_ok(8, 8, thin, rmin=4.0),
          "a 1-cell wall violates a 4-cell minimum member size")
    thick = [0.0] * (8 * 8)
    for ex in range(8):
        for ey in range(8):
            thick[mt.elem(ex, ey)] = 1.0
    check(min_member_ok(8, 8, thick, rmin=2.0),
          "a fully solid block satisfies minimum member size")
    check(min_member_ok(8, 8, [0.0] * 64, rmin=2.0),
          "an empty field vacuously satisfies minimum member size")
    # A single thin island anywhere is enough to fail the whole field.
    mixed = list(thick)
    check(min_member_ok(8, 8, mixed, rmin=2.0), "control: the block passes")

    # -- Multi-load-case: duplicating a case must not change the objective.
    mbb8 = mbb_problem(8, 4)
    one = optimize(8, 4, 0.5, rmin=1.5, max_iter=5,
                   load_cases=[{"F": mbb8["F"], "fixed": mbb8["fixed"]}])
    two = optimize(8, 4, 0.5, rmin=1.5, max_iter=5,
                   load_cases=[{"F": mbb8["F"], "fixed": mbb8["fixed"]},
                               {"F": mbb8["F"], "fixed": mbb8["fixed"]}],
                   load_weights=[0.5, 0.5])
    check(one.get("ok") and two.get("ok"), "multi-load-case runs succeed")
    if one.get("ok") and two.get("ok"):
        check(abs(one["compliance"] - two["compliance"]) < 1e-9,
              "a duplicated load case at half weight each is the single case")
        check(two["n_load_cases"] == 2, "load case count is reported")

    # -- Pareto sweep: more material can never cost more compliance.
    sweep = pareto_sweep(8, 4, [0.3, 0.5, 0.7], rmin=1.5, max_iter=6)
    check(sweep.get("ok") is True, "pareto sweep succeeds")
    if sweep.get("ok"):
        front = sweep["front"]
        check(len(front) == 3, "one front point per requested volume fraction")
        check(front[0]["compliance"] > front[-1]["compliance"],
              "compliance falls monotonically as the volume budget grows")

    # -- Never-raise contract: every malformed input returns ok=False.
    bad_cases = [
        ("volfrac=0", lambda: optimize(4, 4, 0.0)),
        ("volfrac=1", lambda: optimize(4, 4, 1.0)),
        ("volfrac>1", lambda: optimize(4, 4, 1.5)),
        ("volfrac<0", lambda: optimize(4, 4, -0.2)),
        ("nelx=0", lambda: optimize(0, 4, 0.5)),
        ("nely=0", lambda: optimize(4, 0, 0.5)),
        ("x_init wrong length",
         lambda: optimize(4, 4, 0.5, x_init=[0.5] * 3, max_iter=1)),
        ("load_weights mismatch",
         lambda: optimize(4, 4, 0.5, max_iter=1,
                          load_cases=[{"F": [0.0] * 50, "fixed": []}],
                          load_weights=[0.5, 0.5])),
        ("F wrong length",
         lambda: optimize(4, 4, 0.5, max_iter=1,
                          load_cases=[{"F": [0.0] * 3, "fixed": []}])),
        ("empty volfracs", lambda: pareto_sweep(4, 4, [])),
        ("sweep propagates failure", lambda: pareto_sweep(4, 4, [0.5, 2.0])),
    ]
    for label, fn in bad_cases:
        try:
            out = fn()
        except Exception as exc:
            failures.append(f"{label} raised instead of returning: {exc}")
            continue
        check(out.get("ok") is False, f"{label} returns ok=False")
        check(isinstance(out.get("reason"), str) and out["reason"],
              f"{label} explains itself")

    if failures:
        for f in failures:
            print(f"selfcheck FAIL: {f}")
        return 1
    print("topology_optimize selfcheck: OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manufacturing-constrained multi-load-case SIMP "
                    "topology optimization")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
