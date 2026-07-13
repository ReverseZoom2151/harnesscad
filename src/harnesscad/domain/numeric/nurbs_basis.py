"""Cox-de Boor B-spline basis functions and knot-vector machinery (NURBGen).

Usama, Khan, Stricker & Afzal, *NURBGen: High-Fidelity Text-to-CAD Generation
through LLM-Driven NURBS Modeling* (AAAI 2026), "Background" section.

NURBGen serialises every CAD face as an untrimmed NURBS surface -- control
points, knot vectors, degrees and rational weights -- and converts the JSON back
to a B-rep.  The deterministic heart of that conversion is the **B-spline basis
function** ``N_{i,p}(u)``, defined by the Cox-de Boor recursion (paper Eq. 2):

    N_{i,0}(u) = 1  if  u_i <= u < u_{i+1},  else 0
    N_{i,p}(u) = (u - u_i)/(u_{i+p} - u_i)     * N_{i,p-1}(u)
               + (u_{i+p+1} - u)/(u_{i+p+1} - u_{i+1}) * N_{i+1,p-1}(u)

(with the convention that a 0/0 term contributes 0).  These basis functions are
the shared machinery behind both the NURBS *curve* (Eq. 1) and the tensor-product
NURBS *surface* (Eq. 3).  Existing package code (``geometry.dreamcad_rational_
bezier``) only covers the *Bernstein/Bezier* uniform case with no knot vector;
this module adds the genuinely non-uniform knot-vector machinery.

Implemented here (all pure-Python stdlib, deterministic):

  * :func:`cox_de_boor` -- a single basis function via the literal recursion.
  * :func:`find_span`   -- the knot span containing ``u`` (NURBS Book A2.1).
  * :func:`basis_functions` -- the ``p+1`` non-zero basis functions at ``u``
    (NURBS Book A2.2), the efficient form used by evaluation.
  * :func:`all_basis` -- the full length-``(n+1)`` basis vector.
  * :func:`basis_derivatives` -- basis functions *and* their derivatives up to
    order ``d`` (NURBS Book A2.3).
  * knot-vector helpers: :func:`uniform_clamped_knots`,
    :func:`validate_knot_vector`, :func:`knot_multiplicities`.

Conventions: a knot vector ``U`` has ``len(U) == n + p + 2`` for ``n + 1``
control points of degree ``p``.  Knots are non-decreasing floats.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

# Two knot values within this tolerance are treated as equal (zero interval).
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Knot-vector helpers
# ---------------------------------------------------------------------------

def validate_knot_vector(knots: Sequence[float], n: int, p: int) -> None:
    """Validate a knot vector for ``n + 1`` control points of degree ``p``.

    Requires ``len(knots) == n + p + 2`` and non-decreasing values.
    """
    if p < 0:
        raise ValueError("degree p must be >= 0")
    if n < 0:
        raise ValueError("n (last control-point index) must be >= 0")
    expected = n + p + 2
    if len(knots) != expected:
        raise ValueError(
            "knot vector length %d != n + p + 2 = %d" % (len(knots), expected))
    for a, b in zip(knots, knots[1:]):
        if b < a - _EPS:
            raise ValueError("knot vector must be non-decreasing")


def uniform_clamped_knots(n: int, p: int) -> List[float]:
    """Return a clamped (open) uniform knot vector on ``[0, 1]``.

    The first and last knots have multiplicity ``p + 1`` so the curve
    interpolates its first and last control points -- the standard CAD/OCCT
    convention that NURBGen's ``Geom_BSplineSurface`` uses.
    """
    if p < 0:
        raise ValueError("degree p must be >= 0")
    if n < p:
        raise ValueError("need at least p + 1 control points (n >= p)")
    interior = n - p  # number of interior knots
    knots = [0.0] * (p + 1)
    for j in range(1, interior + 1):
        knots.append(j / (interior + 1))
    knots.extend([1.0] * (p + 1))
    return knots


def knot_multiplicities(knots: Sequence[float]) -> List[Tuple[float, int]]:
    """Collapse a knot vector into ``(value, multiplicity)`` pairs.

    Mirrors NURBGen's stored ``mults`` field (paper Appendix, "knots"/"mults").
    """
    out: List[Tuple[float, int]] = []
    for u in knots:
        if out and abs(u - out[-1][0]) <= _EPS:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((float(u), 1))
    return out


def expand_multiplicities(values: Sequence[float],
                          mults: Sequence[int]) -> List[float]:
    """Inverse of :func:`knot_multiplicities`: expand ``(values, mults)``.

    NURBGen stores knots as distinct values plus multiplicities; expansion
    yields the flat knot vector the basis functions consume.
    """
    if len(values) != len(mults):
        raise ValueError("values and mults must have equal length")
    out: List[float] = []
    for v, m in zip(values, mults):
        if m < 1:
            raise ValueError("multiplicity must be >= 1")
        out.extend([float(v)] * int(m))
    return out


# ---------------------------------------------------------------------------
# Cox-de Boor recursion (literal form, paper Eq. 2)
# ---------------------------------------------------------------------------

def cox_de_boor(i: int, p: int, u: float, knots: Sequence[float]) -> float:
    """Single basis function ``N_{i,p}(u)`` via the literal Cox-de Boor recursion.

    Uses the half-open support convention ``u_i <= u < u_{i+1}`` for ``p == 0``,
    with a special case so the right endpoint ``u == knots[-1]`` still evaluates
    (assigned to the last non-empty span) -- matching the domain ``[u_p,
    u_{n+1}]`` NURBGen evaluates over.
    """
    m = len(knots) - 1
    if i < 0 or i + p + 1 > m:
        raise ValueError("basis index i out of range for this knot vector")
    if p == 0:
        if knots[i] <= u < knots[i + 1]:
            return 1.0
        # Right-endpoint: fold u == last knot into the last non-empty span.
        if abs(u - knots[m]) <= _EPS and knots[i] < knots[i + 1] \
                and abs(knots[i + 1] - knots[m]) <= _EPS:
            return 1.0
        return 0.0
    left_den = knots[i + p] - knots[i]
    right_den = knots[i + p + 1] - knots[i + 1]
    left = 0.0
    if left_den > _EPS:
        left = (u - knots[i]) / left_den * cox_de_boor(i, p - 1, u, knots)
    right = 0.0
    if right_den > _EPS:
        right = (knots[i + p + 1] - u) / right_den \
            * cox_de_boor(i + 1, p - 1, u, knots)
    return left + right


# ---------------------------------------------------------------------------
# Span location (NURBS Book A2.1)
# ---------------------------------------------------------------------------

def find_span(n: int, p: int, u: float, knots: Sequence[float]) -> int:
    """Index of the knot span containing ``u`` (returns ``i`` with knots[i] <= u).

    ``n`` is the index of the last control point.  Clamps ``u`` to the valid
    parameter domain ``[knots[p], knots[n+1]]``.  Implements A2.1 from *The
    NURBS Book* (Piegl & Tiller).
    """
    if u >= knots[n + 1] - _EPS:
        return n
    if u <= knots[p] + _EPS:
        return p
    low, high = p, n + 1
    mid = (low + high) // 2
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


# ---------------------------------------------------------------------------
# Non-zero basis functions at u (NURBS Book A2.2)
# ---------------------------------------------------------------------------

def basis_functions(span: int, u: float, p: int,
                    knots: Sequence[float]) -> List[float]:
    """The ``p + 1`` non-zero basis functions ``N_{span-p..span, p}(u)``.

    ``span`` is the value returned by :func:`find_span`.  This is the efficient
    triangular computation (A2.2) that avoids the exponential recursion of
    :func:`cox_de_boor`; the two agree to floating precision.
    """
    N = [0.0] * (p + 1)
    left = [0.0] * (p + 1)
    right = [0.0] * (p + 1)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            temp = N[r] / denom if abs(denom) > _EPS else 0.0
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N


def all_basis(n: int, p: int, u: float, knots: Sequence[float]) -> List[float]:
    """Full length-``(n + 1)`` basis vector at ``u`` (zeros outside the span).

    Convenient for tests (partition of unity) and for the straightforward
    (non-optimised) curve/surface evaluation forms.
    """
    result = [0.0] * (n + 1)
    span = find_span(n, p, u, knots)
    nz = basis_functions(span, u, p, knots)
    for k in range(p + 1):
        result[span - p + k] = nz[k]
    return result


# ---------------------------------------------------------------------------
# Basis functions and derivatives (NURBS Book A2.3)
# ---------------------------------------------------------------------------

def basis_derivatives(span: int, u: float, p: int, knots: Sequence[float],
                      d: int) -> List[List[float]]:
    """Basis functions and derivatives up to order ``d`` at ``u``.

    Returns ``ders`` with ``ders[k][j]`` = the ``k``-th derivative of the
    ``j``-th non-zero basis function (``j = 0..p`` maps to control-point index
    ``span - p + j``).  Derivatives above degree ``p`` are zero.  Implements
    A2.3 from *The NURBS Book*.
    """
    if d < 0:
        raise ValueError("derivative order d must be >= 0")
    ndu = [[0.0] * (p + 1) for _ in range(p + 1)]
    left = [0.0] * (p + 1)
    right = [0.0] * (p + 1)
    ndu[0][0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j][r] = right[r + 1] + left[j - r]  # lower triangle: denom
            temp = ndu[r][j - 1] / ndu[j][r] if abs(ndu[j][r]) > _EPS else 0.0
            ndu[r][j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j][j] = saved

    ders = [[0.0] * (p + 1) for _ in range(d + 1)]
    for j in range(p + 1):
        ders[0][j] = ndu[j][p]

    for r in range(p + 1):
        s1, s2 = 0, 1
        a = [[0.0] * (p + 1) for _ in range(2)]
        a[0][0] = 1.0
        for k in range(1, d + 1):
            der = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk] \
                    if abs(ndu[pk + 1][rk]) > _EPS else 0.0
                der += a[s2][0] * ndu[rk][pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            for j in range(j1, j2 + 1):
                denom = ndu[pk + 1][rk + j]
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / denom \
                    if abs(denom) > _EPS else 0.0
                der += a[s2][j] * ndu[rk + j][pk]
            if r <= pk:
                denom = ndu[pk + 1][r]
                a[s2][k] = -a[s1][k - 1] / denom if abs(denom) > _EPS else 0.0
                der += a[s2][k] * ndu[r][pk]
            ders[k][r] = der
            s1, s2 = s2, s1

    # Multiply through by the falling-factorial factors p!/(p-k)!.
    fac = p
    for k in range(1, d + 1):
        for j in range(p + 1):
            ders[k][j] *= fac
        fac *= (p - k)
    return ders
