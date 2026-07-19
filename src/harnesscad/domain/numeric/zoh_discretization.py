"""Zero-Order-Hold (ZOH) discretization of a continuous diagonal SSM.

the state-space model (Li et al., AAAI 2025) reviews the State Space Model (SSM) that
underlies Mamba: a continuous system that maps a 1-D signal ``X(t)`` to ``Y(t)``
through a hidden state ``H(t)`` via the ODE (paper "Preliminary")::

    H'(t) = A . H(t) + B . X(t)
    Y(t)  = C . H(t)

To run this on a discrete parametric-CAD token sequence the continuous
parameters ``(A, B)`` are discretized with the **Zero-Order-Hold (ZOH)** method
using a per-step time-scale ``delta`` (paper Eq. 1-4)::

    Abar = exp(delta A)                                  (Eq. 1)
    Bbar = (delta A)^-1 (exp(delta A) - I) . (delta B)   (Eq. 2)
    H_k  = Abar . H_{k-1} + Bbar . X_k                   (Eq. 3)
    Y_k  = C . H_k                                       (Eq. 4)

Mamba's ``A`` is *diagonal*, so every operator above is element-wise and this
module works channel-by-channel on plain ``tuple[float, ...]`` vectors.

Provided here (all deterministic, stdlib-only):

* :func:`zoh_abar` -- ``Abar = exp(delta A)`` (element-wise);
* :func:`zoh_bbar` -- exact ``Bbar`` (Eq. 2), with the numerically-stable
  ``a -> 0`` limit ``Bbar -> delta B`` handled per channel;
* :func:`zoh_bbar_simplified` -- the common ``Bbar ~= delta B`` approximation
  that Mamba's reference implementation uses in place of Eq. 2;
* :func:`discretize` -- returns ``(Abar, Bbar)`` for a diagonal system;
* :func:`discrete_scan` -- runs ``H_k = Abar H_{k-1} + Bbar X_k`` over a
  sequence (the ZOH-discretized recurrence, Eq. 3);
* :func:`analytic_state` -- the exact continuous-time state of a scalar
  constant-input system, used to verify the discretization is exact for a
  piecewise-constant hold.

The delta cancels in Eq. 2: ``(delta a)^-1 (exp(delta a) - 1) . delta b =
(exp(delta a) - 1)/a . b``.  As ``delta -> 0`` both ``Abar -> I`` and
``Bbar -> delta B``, recovering the continuous system to first order (forward
Euler), which the tests check.
"""

from __future__ import annotations

import math

Vec = tuple[float, ...]
Seq = tuple[Vec, ...]

# Below this magnitude of ``delta * a`` we use the analytic limit of Eq. 2 to
# avoid catastrophic cancellation / division by zero.
_SMALL = 1e-8


def _check(a_diag: Vec, b_diag: Vec) -> None:
    if len(a_diag) != len(b_diag):
        raise ValueError(f"dimension mismatch {len(a_diag)} vs {len(b_diag)}")


def zoh_abar(a_diag: Vec, delta: float) -> Vec:
    """``Abar = exp(delta A)`` for a diagonal ``A`` (Eq. 1), element-wise."""
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    return tuple(math.exp(delta * a) for a in a_diag)


def zoh_bbar(a_diag: Vec, b_diag: Vec, delta: float) -> Vec:
    """Exact ZOH input matrix ``Bbar`` (Eq. 2) for a diagonal system.

    Per channel ``Bbar_i = (exp(delta a_i) - 1) / a_i * b_i``; when
    ``|delta a_i|`` is tiny this reduces to the stable limit ``delta b_i``.
    """
    _check(a_diag, b_diag)
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    out = []
    for a, b in zip(a_diag, b_diag):
        da = delta * a
        if abs(da) < _SMALL:
            # (exp(da) - 1) / a -> delta as a -> 0 (l'Hopital / series).
            out.append(delta * b)
        else:
            out.append((math.expm1(da) / a) * b)
    return tuple(out)


def zoh_bbar_simplified(b_diag: Vec, delta: float) -> Vec:
    """Simplified input matrix ``Bbar ~= delta B`` (the approximation used by
    the reference Mamba implementation in place of Eq. 2)."""
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    return tuple(delta * b for b in b_diag)


def discretize(a_diag: Vec, b_diag: Vec, delta: float,
               simplified: bool = False) -> tuple[Vec, Vec]:
    """Return ``(Abar, Bbar)`` for a diagonal continuous SSM under ZOH.

    ``simplified=True`` uses the ``Bbar ~= delta B`` approximation for ``Bbar``
    while still using the exact ``Abar = exp(delta A)``.
    """
    _check(a_diag, b_diag)
    abar = zoh_abar(a_diag, delta)
    if simplified:
        bbar = zoh_bbar_simplified(b_diag, delta)
    else:
        bbar = zoh_bbar(a_diag, b_diag, delta)
    return abar, bbar


def discrete_scan(x_seq: Seq, abar: Vec, bbar: Vec,
                  h0: Vec | None = None) -> tuple[Seq, Vec]:
    """Run the ZOH-discretized recurrence ``H_k = Abar H_{k-1} + Bbar X_k``.

    ``abar, bbar`` are the (time-invariant) discretized diagonal matrices.
    Returns ``(states, h_final)`` where ``states[k]`` is ``H`` *after* absorbing
    ``X_k`` (i.e. ``H_0 = Abar h0 + Bbar X_0``).
    """
    length = len(x_seq)
    if length == 0:
        d = len(abar)
        return (), (h0 if h0 is not None else tuple(0.0 for _ in range(d)))
    d = len(x_seq[0])
    if not (len(abar) == len(bbar) == d):
        raise ValueError("Abar/Bbar width must match input feature width")
    h = tuple(0.0 for _ in range(d)) if h0 is None else h0
    if len(h) != d:
        raise ValueError("initial state width does not match feature width")
    states = []
    for k in range(length):
        xk = x_seq[k]
        if len(xk) != d:
            raise ValueError("ragged input sequence")
        h = tuple(abar[i] * h[i] + bbar[i] * xk[i] for i in range(d))
        states.append(h)
    return tuple(states), h


def analytic_state(a: float, b: float, u: float, t: float,
                   h0: float = 0.0) -> float:
    """Exact solution of the scalar constant-input ODE ``H' = a H + b u``.

    ``H(t) = exp(a t) h0 + (exp(a t) - 1)/a * b u`` (with the ``a -> 0`` limit
    ``H(t) = h0 + b u t``).  Because ZOH is exact for a piecewise-constant input,
    :func:`discrete_scan` sampled at ``t = n*delta`` reproduces this exactly.
    """
    at = a * t
    if abs(at) < _SMALL:
        return h0 + b * u * t
    ex = math.exp(at)
    return ex * h0 + (math.expm1(at) / a) * b * u
