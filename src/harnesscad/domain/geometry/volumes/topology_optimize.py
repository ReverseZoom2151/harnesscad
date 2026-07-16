"""Manufacturing-constrained, multi-load-case SIMP topology optimization.

Ported from kerf-main ``packages/kerf-topo/src/kerf_topo/advanced.py``
(kerf, MIT License, Copyright (c) 2026 Imran Paruk -- verified at the repo root
``LICENSE``; the sibling ``LICENSE-CLOUD`` is proprietary but scopes only to
``packages/kerf-cloud/**``, ``packages/kerf-billing/**`` and ``src/cloud/**``,
none of which this port touches).

kerf's ``advanced.py`` is a hand-rolled SIMP (Solid Isotropic Material with
Penalization) topology optimizer that is pure Python end to end: a bilinear-quad
(Q4) plane-stress finite element solver with a banded LDL^T factorisation, a
density filter, and a set of manufacturing constraints. It needs no numpy,
scipy, dolfinx or OCC, which makes it portable into this harness verbatim in
spirit. Nothing else in the harness does structural optimization: the existing
``domain/geometry/mesh`` modules repair meshes and ``domain/geometry/volumes``
handles density/occupancy fields, but neither has an FE solve or a compliance
objective.

Ported
------
* :class:`Mesh2D` -- structured ``nelx x nely`` grid of unit Q4 elements with
  the classic column-major node numbering ``node(i, j) = i*(nely+1) + j``.
* :func:`ke_q4` -- the closed-form 8x8 plane-stress element stiffness used by
  Sigmund's 99-line code.
* :func:`solve_spd_banded` / :func:`solve_dense` -- banded LDL^T for the
  restrained (SPD) free-DOF stiffness, with a partial-pivot Gaussian
  elimination fallback. The band solve turns the FE step from O(n^3) into
  O(n*band^2), which is what keeps a hermetic selfcheck affordable.
* :func:`optimize` -- the SIMP loop: filter -> per-load-case FE solve ->
  weighted sensitivity accumulation -> filtered sensitivities -> OC update ->
  geometric repair -> volume renormalisation -> convergence test.
* :func:`mbb_problem` -- the textbook half-MBB beam boundary conditions, a
  free regression target.
* :func:`count_overhang_violations`, :func:`isolated_island_count`,
  :func:`min_member_ok` -- independent geometric oracles that verify the
  manufacturing constraints on a finished density field.
* :func:`pareto_sweep` -- epsilon-constraint sweep over volume fractions.

The never-raise contract is preserved exactly as the source states it: every
public entry point is total and reports failure as ``{"ok": False, "reason":
...}``. The internal linear solvers are the deliberate exception -- they raise
``ValueError``, which :func:`_fe_compliance` catches at the boundary to trigger
the dense fallback.

Deliberately NOT ported
-----------------------
* ``_mma_step`` -- presented as an MMA alternative to OC, but it is the OC
  update with an extra ``be ** 0.5`` damping term: no moving asymptotes, no
  subproblem, no dual update. Porting it under the name "MMA" would carry the
  overclaim across, so ``optimize`` here offers only ``update="oc"``.
* ``lattice_infill`` -- its portable path is a one-line Gibson-Ashby estimate
  (``0.5 * period * relative_density``); the real TPMS wall-thickness mapping
  lives in a sibling kerf package that is not vendored here.
* The FEniCSx/gmsh/OCC route layer (``kerf-topo/routes.py``) -- not stdlib.

Deterministic (no RNG; fixed iteration counts; bisection to a fixed tolerance),
stdlib-only, ASCII-only. Run ``python -m
harnesscad.domain.geometry.volumes.topology_optimize --selfcheck``.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# --------------------------------------------------------------------------
# Dense / banded linear algebra (pure Python)
# --------------------------------------------------------------------------

def solve_dense(A: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting.

    Works on copies, so the caller's matrices are untouched. Raises
    ``ValueError`` on a singular system; callers wrap this.
    """
    n = len(b)
    M = [list(row) for row in A]
    x = list(b)
    for col in range(n):
        piv = col
        best = abs(M[col][col])
        for r in range(col + 1, n):
            v = abs(M[r][col])
            if v > best:
                best = v
                piv = r
        if best < 1e-300:
            raise ValueError("singular matrix")
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            x[col], x[piv] = x[piv], x[col]
        inv = 1.0 / M[col][col]
        for r in range(col + 1, n):
            factor = M[r][col] * inv
            if factor == 0.0:
                continue
            Mr = M[r]
            Mc = M[col]
            for c in range(col, n):
                Mr[c] -= factor * Mc[c]
            x[r] -= factor * x[col]
    for col in range(n - 1, -1, -1):
        s = x[col]
        Mc = M[col]
        for c in range(col + 1, n):
            s -= Mc[c] * x[c]
        x[col] = s / Mc[col]
    return x


def solve_spd_banded(rows: Sequence[Dict[int, float]], b: Sequence[float],
                     bandwidth: int) -> List[float]:
    """Solve an SPD system stored as sparse rows via banded LDL^T.

    No pivoting: valid because the free-DOF stiffness of a properly restrained
    elastic body is symmetric positive definite. Only the lower band within
    ``bandwidth`` of the diagonal is factorised. Raises ``ValueError`` on a
    non-positive pivot, which is the signal that the restraint was insufficient
    (or the system is not SPD) and the dense path should take over.
    """
    n = len(b)
    L: List[Dict[int, float]] = [dict() for _ in range(n)]
    D = [0.0] * n
    for i in range(n):
        ri = rows[i]
        lo = max(0, i - bandwidth)
        Li = L[i]
        for j in range(lo, i):
            s = ri.get(j, 0.0)
            Lj = L[j]
            kmin = max(lo, j - bandwidth)
            for k in range(kmin, j):
                lik = Li.get(k)
                if lik is None:
                    continue
                ljk = Lj.get(k)
                if ljk is not None:
                    s -= lik * ljk * D[k]
            if D[j] == 0.0:
                continue
            v = s / D[j]
            if v != 0.0:
                Li[j] = v
        d = ri.get(i, 0.0)
        for k, lik in Li.items():
            d -= lik * lik * D[k]
        if d <= 1e-300:
            raise ValueError("non-SPD pivot in banded factorisation")
        D[i] = d
    y = list(b)
    for i in range(n):
        for k, lik in L[i].items():
            y[i] -= lik * y[k]
    for i in range(n):
        y[i] /= D[i]
    xv = list(y)
    for i in range(n - 1, -1, -1):
        xi = xv[i]
        for k, lik in L[i].items():
            xv[k] -= lik * xi
    return xv


# --------------------------------------------------------------------------
# Q4 plane-stress element stiffness
# --------------------------------------------------------------------------

def ke_q4(E: float, nu: float) -> List[List[float]]:
    """8x8 stiffness of a unit-square bilinear quad in plane stress.

    The standard closed form from Sigmund's 99-line code. For a unit square it
    is independent of element size, which is what keeps the MBB regression
    mesh-independent and deterministic.
    """
    k = [
        1.0 / 2.0 - nu / 6.0,
        1.0 / 8.0 + nu / 8.0,
        -1.0 / 4.0 - nu / 12.0,
        -1.0 / 8.0 + 3.0 * nu / 8.0,
        -1.0 / 4.0 + nu / 12.0,
        -1.0 / 8.0 - nu / 8.0,
        nu / 6.0,
        1.0 / 8.0 - 3.0 * nu / 8.0,
    ]
    f = E / (1.0 - nu * nu)
    idx = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 7, 6, 5, 4, 3, 2],
        [2, 7, 0, 5, 6, 3, 4, 1],
        [3, 6, 5, 0, 7, 2, 1, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 3, 2, 1, 0, 7, 6],
        [6, 3, 4, 1, 2, 7, 0, 5],
        [7, 2, 1, 4, 3, 6, 5, 0],
    ]
    return [[f * k[idx[i][j]] for j in range(8)] for i in range(8)]


# --------------------------------------------------------------------------
# Structured mesh
# --------------------------------------------------------------------------

class Mesh2D:
    """A structured ``nelx x nely`` grid of unit Q4 elements.

    Column-major node numbering ``node(i, j) = i * (nely + 1) + j`` with ``i``
    the column (x) and ``j`` the row (y). DOF ``2*n`` is x, ``2*n+1`` is y.
    """

    def __init__(self, nelx: int, nely: int) -> None:
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.nnodes = (self.nelx + 1) * (self.nely + 1)
        self.ndof = 2 * self.nnodes
        self.nel = self.nelx * self.nely
        self._edofs = [self._edof(e) for e in range(self.nel)]

    def node(self, i: int, j: int) -> int:
        return i * (self.nely + 1) + j

    def elem(self, ex: int, ey: int) -> int:
        return ex * self.nely + ey

    def elem_centroid(self, e: int) -> Tuple[float, float]:
        return (e // self.nely + 0.5, e % self.nely + 0.5)

    def _edof(self, e: int) -> List[int]:
        ex = e // self.nely
        ey = e % self.nely
        n1 = self.node(ex, ey)
        n2 = self.node(ex + 1, ey)
        n3 = self.node(ex + 1, ey + 1)
        n4 = self.node(ex, ey + 1)
        d: List[int] = []
        for n in (n1, n2, n3, n4):
            d.append(2 * n)
            d.append(2 * n + 1)
        return d

    def edof(self, e: int) -> List[int]:
        return self._edofs[e]


# --------------------------------------------------------------------------
# Density filter (minimum-member-size constraint)
# --------------------------------------------------------------------------

def build_filter(mesh: Mesh2D, rmin: float) -> List[List[Tuple[int, float]]]:
    """Linear-hat density filter neighbour weights of radius ``rmin``.

    Returns per element a list of ``(neighbour_index, weight)`` with weight
    ``rmin - dist`` over the disc. This is the standard density-filter
    realisation of minimum-member-size: features thinner than the kernel are
    washed out.
    """
    nelx, nely = mesh.nelx, mesh.nely
    R = max(1.0, float(rmin))
    span = int(math.ceil(R)) - 1
    weights: List[List[Tuple[int, float]]] = [[] for _ in range(mesh.nel)]
    for ex in range(nelx):
        for ey in range(nely):
            acc: List[Tuple[int, float]] = []
            for kx in range(max(0, ex - span - 1), min(nelx, ex + span + 2)):
                for ky in range(max(0, ey - span - 1), min(nely, ey + span + 2)):
                    w = R - math.hypot(ex - kx, ey - ky)
                    if w > 0.0:
                        acc.append((mesh.elem(kx, ky), w))
            weights[mesh.elem(ex, ey)] = acc
    return weights


def apply_filter(vec: Sequence[float],
                 weights: Sequence[Sequence[Tuple[int, float]]]) -> List[float]:
    """Apply a weight table as a normalised convex average (partition of unity)."""
    out = [0.0] * len(vec)
    for e, nb in enumerate(weights):
        s = 0.0
        wsum = 0.0
        for j, w in nb:
            s += w * vec[j]
            wsum += w
        out[e] = s / wsum if wsum > 0.0 else vec[e]
    return out


# --------------------------------------------------------------------------
# Manufacturing constraints
# --------------------------------------------------------------------------

def _mirror_pairs(mesh: Mesh2D) -> List[Tuple[int, int]]:
    """Element index pairs mirrored about the vertical mid-plane."""
    pairs: List[Tuple[int, int]] = []
    for ex in range(mesh.nelx // 2):
        mx = mesh.nelx - 1 - ex
        for ey in range(mesh.nely):
            pairs.append((mesh.elem(ex, ey), mesh.elem(mx, ey)))
    return pairs


def _enforce_symmetry(x: List[float], pairs: Sequence[Tuple[int, int]]) -> None:
    for a, b in pairs:
        avg = 0.5 * (x[a] + x[b])
        x[a] = avg
        x[b] = avg


def _reach_for_angle(angle_deg: float) -> int:
    """Lateral cell reach the self-support cone bridges at ``angle_deg``."""
    ang = max(1e-6, min(89.999, float(angle_deg)))
    return max(0, int(round(1.0 / math.tan(math.radians(ang)))))


def _overhang_violations(mesh: Mesh2D, x: Sequence[float], angle_deg: float,
                         threshold: float = 0.5) -> int:
    """Count solid elements with no support within the cone below (+y is build)."""
    reach = _reach_for_angle(angle_deg)
    viol = 0
    for ex in range(mesh.nelx):
        for ey in range(1, mesh.nely):
            if x[mesh.elem(ex, ey)] <= threshold:
                continue
            supported = False
            for dx in range(-reach, reach + 1):
                kx = ex + dx
                if 0 <= kx < mesh.nelx and x[mesh.elem(kx, ey - 1)] > threshold:
                    supported = True
                    break
            if not supported:
                viol += 1
    return viol


def _repair_overhang(mesh: Mesh2D, x: List[float], angle_deg: float) -> None:
    """Bottom-up support projection: cap each density at its cone max below."""
    reach = _reach_for_angle(angle_deg)
    for ey in range(1, mesh.nely):
        for ex in range(mesh.nelx):
            below_max = 0.0
            for dx in range(-reach, reach + 1):
                kx = ex + dx
                if 0 <= kx < mesh.nelx:
                    below_max = max(below_max, x[mesh.elem(kx, ey - 1)])
            e = mesh.elem(ex, ey)
            if x[e] > below_max:
                x[e] = below_max


def _apply_draw_direction(mesh: Mesh2D, x: List[float]) -> None:
    """Make the part mould-extractable along -y (monotone max carry downward)."""
    for ex in range(mesh.nelx):
        carry = 0.0
        for ey in range(mesh.nely - 1, -1, -1):
            e = mesh.elem(ex, ey)
            carry = max(carry, x[e])
            x[e] = carry


# --------------------------------------------------------------------------
# FE solve + compliance + sensitivity
# --------------------------------------------------------------------------

def _fe_compliance(mesh: Mesh2D, xphys: Sequence[float],
                   KE: Sequence[Sequence[float]], penal: float, Emin: float,
                   fixed: Sequence[int],
                   F: Sequence[float]) -> Tuple[float, List[float]]:
    """Assemble and solve ``K u = F``; return ``(compliance, per-element ce)``.

    SIMP modulus interpolation ``E(x) = Emin + x^p (1 - Emin)`` with ``E0 = 1``.
    """
    fixed_set = set(fixed)
    free = [d for d in range(mesh.ndof) if d not in fixed_set]
    nf = len(free)
    fidx = {d: i for i, d in enumerate(free)}

    Krows: List[Dict[int, float]] = [dict() for _ in range(nf)]
    for e in range(mesh.nel):
        scale = Emin + (xphys[e] ** penal) * (1.0 - Emin)
        ed = mesh.edof(e)
        for a in range(8):
            ia = fidx.get(ed[a])
            if ia is None:
                continue
            row = Krows[ia]
            KEa = KE[a]
            for b in range(8):
                jb = fidx.get(ed[b])
                if jb is None:
                    continue
                v = scale * KEa[b]
                if v != 0.0:
                    row[jb] = row.get(jb, 0.0) + v
    bf = [F[d] for d in free]

    # Half-bandwidth: two DOFs per node, columns offset by (nely+1) nodes.
    band = 2 * (mesh.nely + 1) + 4
    try:
        uf = solve_spd_banded(Krows, bf, band)
    except ValueError:
        dense = [[0.0] * nf for _ in range(nf)]
        for i, r in enumerate(Krows):
            di = dense[i]
            for j, v in r.items():
                di[j] = v
        uf = solve_dense(dense, bf)

    u = [0.0] * mesh.ndof
    for d, i in fidx.items():
        u[d] = uf[i]

    ce = [0.0] * mesh.nel
    comp = 0.0
    for e in range(mesh.nel):
        ue = [u[d] for d in mesh.edof(e)]
        s = 0.0
        for a in range(8):
            KEa = KE[a]
            row = 0.0
            for b in range(8):
                row += KEa[b] * ue[b]
            s += ue[a] * row
        ce[e] = s
        comp += (Emin + (xphys[e] ** penal) * (1.0 - Emin)) * s
    return comp, ce


# --------------------------------------------------------------------------
# Optimality Criteria update
# --------------------------------------------------------------------------

def _oc_step(x: Sequence[float], dc: Sequence[float], volfrac: float,
             move: float = 0.2) -> List[float]:
    """Bisection-on-Lagrange-multiplier OC update.

    ``x_new = x * sqrt(-dc / lambda)`` clamped by a move limit and the box
    ``[1e-3, 1]``; ``lambda`` is bisected until the volume target is met.
    """
    n = len(x)
    l1, l2 = 1e-12, 1e12
    xnew = list(x)
    target = volfrac * n
    while (l2 - l1) / (l1 + l2) > 1e-9:
        lmid = 0.5 * (l1 + l2)
        tot = 0.0
        for i in range(n):
            ratio = -dc[i] / lmid
            be = math.sqrt(ratio) if ratio > 0.0 else 0.0
            lo = max(1e-3, x[i] - move)
            hi = min(1.0, x[i] + move)
            v = min(hi, max(lo, x[i] * be))
            xnew[i] = v
            tot += v
        if tot > target:
            l1 = lmid
        else:
            l2 = lmid
    return xnew


# --------------------------------------------------------------------------
# Built-in benchmark problem
# --------------------------------------------------------------------------

def mbb_problem(nelx: int, nely: int) -> Dict[str, Any]:
    """Boundary conditions and unit load for the half-MBB beam.

    Left-edge x-symmetry, a y-roller at the bottom-right corner, and a
    downward unit point load at the top-left node. The textbook regression
    target for SIMP implementations.
    """
    mesh = Mesh2D(nelx, nely)
    fixed: List[int] = [2 * mesh.node(0, j) for j in range(nely + 1)]
    fixed.append(2 * mesh.node(nelx, 0) + 1)
    F = [0.0] * mesh.ndof
    F[2 * mesh.node(0, nely) + 1] = -1.0
    return {"fixed": fixed, "F": F, "nelx": nelx, "nely": nely}


# --------------------------------------------------------------------------
# Core solver
# --------------------------------------------------------------------------

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
    """Run manufacturing-constrained, multi-load-case SIMP. Never raises.

    ``load_cases`` is a sequence of ``{"F": [...], "fixed": [...]}`` dicts; the
    objective is the ``load_weights``-weighted sum of per-case compliance. When
    omitted a single MBB load case is used.

    Returns ``{"ok": True, "compliance", "volume_fraction", "iterations",
    "converged", "history", "density", "nelx", "nely", "n_load_cases"}`` or
    ``{"ok": False, "reason": ...}``.
    """
    try:
        if not (0.0 < volfrac < 1.0):
            return {"ok": False, "reason": "volfrac must be in (0, 1)"}
        if nelx < 1 or nely < 1:
            return {"ok": False, "reason": "nelx and nely must be >= 1"}

        mesh = Mesh2D(nelx, nely)

        cases: List[Dict[str, Any]] = list(load_cases) if load_cases else []
        if not cases:
            mbb = mbb_problem(nelx, nely)
            cases = [{"F": mbb["F"], "fixed": mbb["fixed"]}]
            if fixed is None:
                fixed = mbb["fixed"]

        ncase = len(cases)
        w_list = ([1.0 / ncase] * ncase if load_weights is None
                  else list(load_weights))
        if len(w_list) != ncase:
            return {"ok": False, "reason": "load_weights length != load_cases"}

        KE = ke_q4(1.0, nu)
        weights = build_filter(mesh, rmin)
        sym_pairs = _mirror_pairs(mesh) if symmetry else []

        if x_init is None:
            x = [float(volfrac)] * mesh.nel
        else:
            x = [min(1.0, max(1e-3, float(v))) for v in x_init]
        if len(x) != mesh.nel:
            return {"ok": False, "reason": "x_init length != element count"}

        history: List[Dict[str, float]] = []
        prev_c: Optional[float] = None
        last_change = 1.0
        it = 0
        for it in range(1, int(max_iter) + 1):
            xphys = apply_filter(x, weights)
            if symmetry:
                _enforce_symmetry(xphys, sym_pairs)

            total_c = 0.0
            dc = [0.0] * mesh.nel
            for w, case in zip(w_list, cases):
                cF = case.get("F")
                cfix = case.get("fixed", fixed if fixed is not None else [])
                if cF is None or len(cF) != mesh.ndof:
                    return {"ok": False,
                            "reason": "load case F missing or wrong length"}
                c, ce = _fe_compliance(mesh, xphys, KE, penal, Emin, cfix, cF)
                total_c += w * c
                for e in range(mesh.nel):
                    dscale = penal * (xphys[e] ** (penal - 1.0)) * (1.0 - Emin)
                    dc[e] += -w * dscale * ce[e]

            dcf = apply_filter(dc, weights)
            if symmetry:
                _enforce_symmetry(dcf, sym_pairs)

            xnew = _oc_step(x, dcf, volfrac)

            if symmetry:
                _enforce_symmetry(xnew, sym_pairs)
            if draw_direction:
                _apply_draw_direction(mesh, xnew)
            if overhang_angle is not None:
                _repair_overhang(mesh, xnew, overhang_angle)

            # Renormalise volume after any geometric repair so the constraint
            # still holds at convergence.
            if draw_direction or overhang_angle is not None:
                cur = sum(xnew) / mesh.nel
                if cur > 1e-12:
                    s = volfrac / cur
                    xnew = [min(1.0, max(1e-3, v * s)) for v in xnew]

            last_change = max(abs(a - b) for a, b in zip(xnew, x))
            x = xnew

            vol = sum(apply_filter(x, weights)) / mesh.nel
            history.append({"iter": float(it), "compliance": total_c,
                            "volume": vol, "change": last_change})

            if prev_c is not None and prev_c > 0.0:
                rel = abs(total_c - prev_c) / prev_c
                if rel < tol and last_change < 0.02:
                    prev_c = total_c
                    break
            prev_c = total_c

        xphys = apply_filter(x, weights)
        if symmetry:
            _enforce_symmetry(xphys, sym_pairs)
        if draw_direction:
            _apply_draw_direction(mesh, xphys)
        if overhang_angle is not None:
            _repair_overhang(mesh, xphys, overhang_angle)

        return {
            "ok": True,
            "compliance": history[-1]["compliance"] if history else 0.0,
            "volume_fraction": sum(xphys) / mesh.nel,
            "iterations": it,
            "converged": (prev_c is not None and len(history) >= 2
                          and last_change < 0.05),
            "history": history,
            "density": xphys,
            "nelx": nelx,
            "nely": nely,
            "n_load_cases": ncase,
        }
    except Exception as exc:  # defensive: preserve the total contract
        return {"ok": False, "reason": f"optimize failed: {exc}"}


def pareto_sweep(nelx: int, nely: int, volfracs: Sequence[float],
                 **kwargs: Any) -> Dict[str, Any]:
    """Epsilon-constraint sweep: run :func:`optimize` at each volume fraction.

    Returns ``{"ok": True, "front": [{"volume_fraction", "compliance"}, ...]}``
    ordered as given. Never raises. A failure in any single run aborts with that
    run's reason.
    """
    try:
        if not volfracs:
            return {"ok": False, "reason": "volfracs must be non-empty"}
        front: List[Dict[str, float]] = []
        for vf in volfracs:
            res = optimize(nelx, nely, vf, **kwargs)
            if not res.get("ok"):
                return {"ok": False,
                        "reason": f"sweep failed at volfrac={vf}: "
                                  f"{res.get('reason')}"}
            front.append({"volume_fraction": res["volume_fraction"],
                          "compliance": res["compliance"]})
        return {"ok": True, "front": front}
    except Exception as exc:  # defensive: preserve the total contract
        return {"ok": False, "reason": f"pareto_sweep failed: {exc}"}


# --------------------------------------------------------------------------
# Geometric verification oracles
# --------------------------------------------------------------------------

def count_overhang_violations(nelx: int, nely: int, density: Sequence[float],
                              angle_deg: float, threshold: float = 0.5) -> int:
    """Solid elements lacking support within the self-support cone below."""
    return _overhang_violations(Mesh2D(nelx, nely), density, angle_deg, threshold)


def isolated_island_count(nelx: int, nely: int, density: Sequence[float],
                          max_cells: int, threshold: float = 0.5) -> int:
    """Number of connected solid islands of at most ``max_cells`` elements.

    A density filter of radius ``rmin`` attenuates any solid feature narrower
    than its kernel, so this count is monotone non-increasing as ``rmin`` grows
    -- the property the minimum-member-size selfcheck asserts.
    """
    mesh = Mesh2D(nelx, nely)
    solid = [density[e] > threshold for e in range(mesh.nel)]
    seen = [False] * mesh.nel
    small = 0
    for ex in range(nelx):
        for ey in range(nely):
            e = mesh.elem(ex, ey)
            if seen[e] or not solid[e]:
                continue
            stack = [e]
            seen[e] = True
            comp = 0
            while stack:
                ce = stack.pop()
                comp += 1
                cex, cey = ce // nely, ce % nely
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    kx, ky = cex + dx, cey + dy
                    if 0 <= kx < nelx and 0 <= ky < nely:
                        ne = mesh.elem(kx, ky)
                        if not seen[ne] and solid[ne]:
                            seen[ne] = True
                            stack.append(ne)
            if comp <= max_cells:
                small += 1
    return small


def min_member_ok(nelx: int, nely: int, density: Sequence[float], rmin: float,
                  threshold: float = 0.5) -> bool:
    """True iff no solid feature is thinner than ``rmin``.

    Erode the thresholded solid set by a digital disc of radius ``rmin/2``: a
    feature of width >= rmin keeps at least one fully-eroded medial cell, a
    narrower one is annihilated. So minimum-member-size holds iff every
    connected solid component retains an eroded core cell. This formulation is
    grid-robust -- it does not assume morphological-opening idempotence, which
    a digital disc never gives on a coarse mesh.
    """
    mesh = Mesh2D(nelx, nely)
    r = max(1.0, float(rmin)) / 2.0
    rr = r * r
    solid = [density[e] > threshold for e in range(mesh.nel)]

    s = int(math.ceil(r))
    offsets = [(dx, dy) for dx in range(-s, s + 1) for dy in range(-s, s + 1)
               if dx * dx + dy * dy <= rr]

    eroded = [False] * mesh.nel
    for ex in range(nelx):
        for ey in range(nely):
            if not solid[mesh.elem(ex, ey)]:
                continue
            keep = True
            for dx, dy in offsets:
                kx, ky = ex + dx, ey + dy
                if not (0 <= kx < nelx and 0 <= ky < nely
                        and solid[mesh.elem(kx, ky)]):
                    keep = False
                    break
            eroded[mesh.elem(ex, ey)] = keep

    seen = [False] * mesh.nel
    for ex in range(nelx):
        for ey in range(nely):
            e = mesh.elem(ex, ey)
            if seen[e] or not solid[e]:
                continue
            stack = [e]
            seen[e] = True
            has_core = False
            while stack:
                ce = stack.pop()
                if eroded[ce]:
                    has_core = True
                cex, cey = ce // nely, ce % nely
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    kx, ky = cex + dx, cey + dy
                    if 0 <= kx < nelx and 0 <= ky < nely:
                        ne = mesh.elem(kx, ky)
                        if not seen[ne] and solid[ne]:
                            seen[ne] = True
                            stack.append(ne)
            if not has_core:
                return False
    return True


# --------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------

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

    # -- Mesh numbering.
    m = Mesh2D(4, 3)
    check(m.nel == 12 and m.nnodes == 20 and m.ndof == 40, "mesh counts")
    check(m.node(0, 0) == 0 and m.node(1, 0) == 4, "column-major node numbering")
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
        check(all(1e-3 - 1e-9 <= d <= 1.0 + 1e-9 for d in res["density"]),
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
                    "topology optimization (ported from kerf kerf-topo)")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
