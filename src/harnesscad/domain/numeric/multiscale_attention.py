"""Multi-scale adaptive fusion and hierarchical window attention .

The denoiser is a **Multi-Scale Transformer (MST)** whose every
layer runs three parallel attention branches with window sizes 64 / 128 / 256 to
capture "local geometric constraints, medium-range topological dependencies, and
global semantic coherence" (Sec. 3.3), then combines them with an **Adaptive
Fusion** gate (Eq. 5 / Eq. 12)::

    H = MLP( sigma(W_g [H_l || H_m || H_g]) (*) [H_l || H_m || H_g] )

The learned projections (``W_g``, the fusion MLP, and the Q/K/V weights) are
external, but three ingredients are **purely deterministic** and are implemented
here, distinct from the single-scale gated mixer in
``numeric.geofusion_state_space``:

* :func:`window_mask` -- the banded attention mask ``M_w`` (Eq. 11): 0 inside a
  window of half-width ``w`` and ``-inf`` outside; ``w=None`` gives the global
  (all-ones) mask;
* :func:`masked_attention` -- deterministic scaled-dot-product attention given
  explicit ``Q, K, V`` and a mask, so a single scale branch ``H_l/H_m/H_g`` can
  be computed exactly;
* :func:`softmax` -- numerically stable row softmax used above;
* :func:`adaptive_fusion` -- the gated concat-fusion of Eq. 5: concatenate the
  per-scale token features, form a sigmoid gate from them (with a supplied
  ``W_g`` or a default identity gate), multiply, and project;
* :func:`sequence_aware_pe` -- the scalable sinusoidal positional encoding
  ``PE(Z) = Z + eta * PE(pos, D)`` with a trainable scalar ``eta`` (Eq. 6 /
  Eq. 13), which reduces to a deterministic additive residual.

Sequences are ``tuple`` of equal-width feature vectors. stdlib-only,
deterministic.
"""

from __future__ import annotations

import math

Vec = tuple[float, ...]
Seq = tuple[Vec, ...]
Matrix = tuple[Vec, ...]

_NEG_INF = float("-inf")


def window_mask(length: int, half_width: int | None) -> Matrix:
    """Banded attention mask ``M_w`` (Eq. 11).

    Returns an ``length x length`` matrix whose ``(i, j)`` entry is ``0.0`` when
    position ``j`` lies within the window ``|i - j| <= half_width`` of query
    ``i`` and ``-inf`` otherwise. ``half_width=None`` yields the *global* branch
    ``W_g`` (all zeros / full attention).
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if half_width is not None and half_width < 0:
        raise ValueError("half_width must be >= 0 or None")
    rows: list[Vec] = []
    for i in range(length):
        row: list[float] = []
        for j in range(length):
            if half_width is None or abs(i - j) <= half_width:
                row.append(0.0)
            else:
                row.append(_NEG_INF)
        rows.append(tuple(row))
    return tuple(rows)


def softmax(scores: Vec) -> Vec:
    """Numerically stable softmax over a vector; rows that are entirely
    ``-inf`` (a query attends to nothing) return a uniform distribution.
    """
    finite = [s for s in scores if s != _NEG_INF]
    if not finite:
        n = len(scores)
        return tuple(1.0 / n for _ in range(n))
    m = max(finite)
    exps = [0.0 if s == _NEG_INF else math.exp(s - m) for s in scores]
    total = sum(exps)
    return tuple(e / total for e in exps)


def _matmul_row(q: Vec, k: Vec) -> float:
    return sum(a * b for a, b in zip(q, k))


def masked_attention(q: Matrix, k: Matrix, v: Matrix,
                     mask: Matrix | None = None) -> Seq:
    """Deterministic scaled-dot-product attention for one scale branch.

    ``q, k, v`` are ``L x d`` sequences (rows are per-token vectors). Scores are
    ``Q K^T / sqrt(d)`` plus the additive ``mask`` (from :func:`window_mask`),
    row-softmaxed and applied to ``V``. Returns the ``L x d_v`` output sequence.
    """
    lq = len(q)
    lk = len(k)
    if lk != len(v):
        raise ValueError("K and V must have equal length")
    if lq == 0:
        return ()
    d = len(q[0])
    scale = 1.0 / math.sqrt(d) if d > 0 else 1.0
    dv = len(v[0])
    out: list[Vec] = []
    for i in range(lq):
        scores = [_matmul_row(q[i], k[j]) * scale for j in range(lk)]
        if mask is not None:
            mrow = mask[i]
            scores = [scores[j] + mrow[j] for j in range(lk)]
        weights = softmax(tuple(scores))
        acc = [0.0] * dv
        for j in range(lk):
            w = weights[j]
            if w == 0.0:
                continue
            vj = v[j]
            for ch in range(dv):
                acc[ch] += w * vj[ch]
        out.append(tuple(acc))
    return tuple(out)


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _concat(*vecs: Vec) -> Vec:
    out: list[float] = []
    for v in vecs:
        out.extend(v)
    return tuple(out)


def _matvec(mat: Matrix, vec: Vec) -> Vec:
    return tuple(sum(row[i] * vec[i] for i in range(len(vec))) for row in mat)


def adaptive_fusion(h_local: Seq, h_mid: Seq, h_global: Seq,
                    w_gate: Matrix | None = None,
                    w_out: Matrix | None = None) -> Seq:
    """Adaptive Fusion of three scale branches (Eq. 5).

    For each token, concatenate ``[H_l || H_m || H_g]`` (call it ``c``), form the
    gate ``g = sigmoid(W_g c)`` (identity gate ``W_g = I`` if ``w_gate`` is
    ``None``, so ``g = sigmoid(c)``), take the gated feature ``g (*) c``, and
    optionally project it with ``W_out`` (the fusion MLP's linear map). When
    ``w_out`` is ``None`` the gated concatenation is returned directly.

    All three branches must share length and per-token width.
    """
    n = len(h_local)
    if not (len(h_mid) == len(h_global) == n):
        raise ValueError("all three branches must have equal length")
    out: list[Vec] = []
    for i in range(n):
        c = _concat(h_local[i], h_mid[i], h_global[i])
        gate_in = _matvec(w_gate, c) if w_gate is not None else c
        gate = tuple(sigmoid(x) for x in gate_in)
        if len(gate) != len(c):
            raise ValueError("w_gate must be square in the concat dimension")
        gated = tuple(g * x for g, x in zip(gate, c))
        out.append(_matvec(w_out, gated) if w_out is not None else gated)
    return tuple(out)


def sequence_aware_pe(seq: Seq, eta: float = 1.0) -> Seq:
    """Scalable sinusoidal positional encoding ``PE(Z) = Z + eta * PE(pos, D)``
    with trainable scalar ``eta`` (Eq. 6 / Eq. 13).

    ``PE(pos, D)`` is the standard transformer sinusoid: even channel ``2i`` uses
    ``sin(pos / 10000^(2i/D))`` and odd channel ``2i+1`` uses the matching
    cosine. The result is added to ``seq`` scaled by ``eta`` (adaptive positional
    weighting). Deterministic for any fixed ``eta``.
    """
    if not seq:
        return ()
    d = len(seq[0])
    out: list[Vec] = []
    for pos, token in enumerate(seq):
        if len(token) != d:
            raise ValueError("all tokens must share a common width")
        enc: list[float] = []
        for j in range(d):
            i = j // 2
            freq = 1.0 / (10000.0 ** (2.0 * i / d)) if d > 0 else 0.0
            angle = pos * freq
            pe = math.sin(angle) if j % 2 == 0 else math.cos(angle)
            enc.append(token[j] + eta * pe)
        out.append(tuple(enc))
    return tuple(out)


def multiscale_attention_flops(length: int, dim: int,
                               half_widths: tuple[int | None, ...]) -> int:
    """Analytic op count for one MST layer's three window branches.

    A branch with half-width ``w`` touches, per query, at most ``2w+1`` keys
    (``length`` for the global branch ``w=None``), each costing ``O(dim)``.
    Summing over branches shows the *local* branches are ``O(w L d)`` -- far
    cheaper than the ``O(L^2 d)`` global branch, the point of the multi-scale
    design.
    """
    if length <= 0 or dim <= 0:
        raise ValueError("length and dim must be positive")
    total = 0
    for w in half_widths:
        span = length if w is None else min(length, 2 * w + 1)
        total += length * span * dim
    return total
