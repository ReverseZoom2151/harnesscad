"""Geometric state-space forward scan for GeoFusion-CAD's G-Mamba block.

GeoFusion-CAD replaces quadratic self-attention with a *linear-time* geometric
state-space model (the "G-Mamba" / GSM-SSD block, Sec. 4.3, Alg. 1, App. B). The
*learned* kernel generator ``f_geom`` (a 2-3 layer MLP) is external, but every
numerical operation that consumes its output is a **deterministic linear
recurrence** and is implemented here.

Given per-token *diagonal* kernels ``A_k, B_k, C_k, G_k`` (each a length-``d``
vector -- "all operators are diagonal and thus element-wise", Eq. 24), the
discrete-time selective-scan update (Eq. 15 / Eq. 4) is::

    h_{k+1}  = A_k (*) h_k + B_k (*) Z_k
    Zout_k   = C_k (*) h_k + G_k (*) Z_k

where ``(*)`` is the Hadamard (element-wise) product. This module provides:

* :func:`selective_scan` -- the core O(Ld) forward recurrence above;
* :func:`curvature_descriptor` -- ``r_k`` (0 for a line, ``1/R`` for a circular
  arc of radius ``R``, discrete angular deviation for a general curve, Sec. B.2);
* :func:`conditioning_vector` -- ``Delta_k = [s_k, d_k, r_k]``, the geometric
  conditioning input to ``f_geom`` (Eq. 12);
* :func:`hierarchical_pe` -- ``Pi_k = PE(p_k, sigma_k, tau_k)``, the sinusoidal
  hierarchical positional embedding (Eq. 13), which is deterministic;
* :func:`depthwise_conv1d` -- the DWConv local-smoothness operator (Eq. 20/29);
* :func:`geometric_state_mixer` -- the GSM gated fusion
  ``hhat = W2 (h (*) sigma(z))`` with ``[h, z] = W1 h_in`` (Eq. 16-19), where the
  linear weights are supplied explicitly (the learned part is a plain matmul);
* :func:`film_modulate` -- the optional diffusion-time FiLM kernel modulation
  ``kernels <- psi_t (*) kernels`` (Eq. 21);
* :func:`gmamba_flops` -- the analytic ``O(Ld)`` operation count (App. B.6),
  used to verify linear (not quadratic) scaling.

Vectors are plain ``tuple[float, ...]``; a sequence is a ``tuple`` of such
vectors. Everything is stdlib-only and deterministic.
"""

from __future__ import annotations

import math

Vec = tuple[float, ...]
Seq = tuple[Vec, ...]


def _hadamard(a: Vec, b: Vec) -> Vec:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch {len(a)} vs {len(b)}")
    return tuple(x * y for x, y in zip(a, b))


def _add(a: Vec, b: Vec) -> Vec:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch {len(a)} vs {len(b)}")
    return tuple(x + y for x, y in zip(a, b))


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# --- core selective scan ----------------------------------------------------

def selective_scan(z_seq: Seq, a_seq: Seq, b_seq: Seq, c_seq: Seq, g_seq: Seq,
                   h0: Vec | None = None) -> tuple[Seq, Vec]:
    """Deterministic geometry-conditioned selective scan (Eq. 4 / Eq. 15).

    Parameters mirror the paper: ``z_seq`` are the input token features ``Z_k``;
    ``a_seq, b_seq, c_seq, g_seq`` are the per-token diagonal kernels
    ``A_k, B_k, C_k, G_k``. All must be sequences of equal length ``L`` with a
    common hidden width ``d``. ``h0`` is the initial state (defaults to zeros).

    Returns ``(outputs, h_final)`` where ``outputs[k] = C_k (*) h_k + G_k (*)
    Z_k`` and the state advances as ``h_{k+1} = A_k (*) h_k + B_k (*) Z_k``.
    """
    length = len(z_seq)
    if not (len(a_seq) == len(b_seq) == len(c_seq) == len(g_seq) == length):
        raise ValueError("all kernel sequences must match the input length")
    if length == 0:
        return (), (h0 or ())
    d = len(z_seq[0])
    h = tuple(0.0 for _ in range(d)) if h0 is None else h0
    if len(h) != d:
        raise ValueError("initial state width does not match feature width")
    outputs: list[Vec] = []
    for k in range(length):
        zk = z_seq[k]
        out = _add(_hadamard(c_seq[k], h), _hadamard(g_seq[k], zk))
        outputs.append(out)
        h = _add(_hadamard(a_seq[k], h), _hadamard(b_seq[k], zk))
    return tuple(outputs), h


# --- geometric conditioning (Sec. B.2) --------------------------------------

def curvature_descriptor(kind: str, radius: float | None = None,
                         angular_deviation: float | None = None) -> float:
    """Local curvature descriptor ``r_k`` (Sec. B.2).

    * ``line``   -> ``0`` (straight segment has zero curvature);
    * ``arc`` / ``circle`` -> ``1 / R`` for radius ``R`` (> 0);
    * general curve -> the supplied discrete ``angular_deviation`` approximation.
    """
    if kind == "line":
        return 0.0
    if kind in ("arc", "circle"):
        if radius is None or radius <= 0.0:
            raise ValueError("arc/circle curvature needs a positive radius")
        return 1.0 / radius
    if angular_deviation is None:
        raise ValueError("general curve needs an angular_deviation")
    return angular_deviation


def conditioning_vector(scale: float, depth: int, curvature: float) -> Vec:
    """Geometric conditioning vector ``Delta_k = g(s_k, d_k, r_k)`` (Eq. 12).

    ``g`` is a learned MLP in the paper; the deterministic *input packing*
    ``[scale, depth, curvature]`` fed to it is returned here so callers can drive
    an external ``f_geom`` or the identity default.
    """
    return (float(scale), float(depth), float(curvature))


def hierarchical_pe(parent_type: int, sibling_index: int, role: int,
                    dim: int) -> Vec:
    """Sinusoidal hierarchical positional embedding ``Pi_k = PE(p, sigma, tau)``
    (Eq. 13).

    Deterministic transformer-style sinusoidal encoding of the integer position
    ``pos = parent_type * P^2 + sibling_index * P + role`` (with a large base
    ``P`` so the three fields do not alias), producing a ``dim``-vector.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError("dim must be a positive even number")
    base = 1024
    pos = parent_type * base * base + sibling_index * base + role
    enc: list[float] = []
    half = dim // 2
    for i in range(half):
        freq = 1.0 / (10000.0 ** (2.0 * i / dim))
        enc.append(math.sin(pos * freq))
        enc.append(math.cos(pos * freq))
    return tuple(enc)


# --- DWConv (Eq. 20 / 29) ---------------------------------------------------

def depthwise_conv1d(z_seq: Seq, kernel: Vec, bias: Vec | None = None) -> Seq:
    """Depthwise (per-channel) 1D convolution with a shared temporal ``kernel``.

    Each of the ``d`` feature channels is convolved independently along the
    sequence with the same length-``K`` kernel (causal, left-padded with zeros),
    "preserving local geometric smoothness while reducing computational
    overhead" (Sec. 4.3.2). Complexity ``O(K L d)`` = ``O(Ld)`` for constant K.
    """
    length = len(z_seq)
    if length == 0:
        return ()
    d = len(z_seq[0])
    ksize = len(kernel)
    if ksize == 0:
        raise ValueError("kernel must be non-empty")
    if bias is not None and len(bias) != d:
        raise ValueError("bias width does not match feature width")
    out: list[Vec] = []
    for k in range(length):
        acc = [0.0] * d
        for j in range(ksize):
            src = k - (ksize - 1 - j)
            if src < 0:
                continue
            w = kernel[j]
            zsrc = z_seq[src]
            for ch in range(d):
                acc[ch] += w * zsrc[ch]
        if bias is not None:
            for ch in range(d):
                acc[ch] += bias[ch]
        out.append(tuple(acc))
    return tuple(out)


# --- Geometric State Mixer (Eq. 16-19) --------------------------------------

def geometric_state_mixer(a_k: Vec, b_k: Vec, z_k: Vec,
                          w1: tuple[Vec, ...], w2: tuple[Vec, ...]) -> Vec:
    """Geometric State Mixer gated fusion for one token (Eq. 16-19).

    ``h_in = (A_k (*) B_k) (*) Z_k`` (element-wise; the diagonal form of the
    ``(A (*) B)^T Z`` mixing), then ``[h, z] = W1 @ h_in`` split into two halves,
    fused as ``hhat = W2 @ (h (*) sigma(z))``. The linear maps ``w1`` (shape
    ``2m x d``) and ``w2`` (shape ``d x m``) are supplied explicitly -- they are
    the only learned pieces and here reduce to deterministic matmuls.
    """
    h_in = _hadamard(_hadamard(a_k, b_k), z_k)
    hz = _matvec(w1, h_in)
    if len(hz) % 2 != 0:
        raise ValueError("w1 must produce an even-length [h, z] vector")
    m = len(hz) // 2
    h = hz[:m]
    z = hz[m:]
    gated = tuple(hi * sigmoid(zi) for hi, zi in zip(h, z))
    return _matvec(w2, gated)


def _matvec(mat: tuple[Vec, ...], vec: Vec) -> Vec:
    return tuple(sum(row[i] * vec[i] for i in range(len(vec))) for row in mat)


# --- diffusion-time FiLM (Eq. 21) -------------------------------------------

def film_modulate(kernels: tuple[Vec, ...], psi: Vec) -> tuple[Vec, ...]:
    """Modulate a tuple of diagonal kernels by a diffusion-time gate ``psi_t``
    (Eq. 21): ``kernel <- psi_t (*) kernel`` for every kernel."""
    return tuple(_hadamard(k, psi) for k in kernels)


# --- complexity (App. B.6) --------------------------------------------------

def gmamba_flops(length: int, dim: int, dwc_kernel: int = 3) -> int:
    """Analytic scalar-op count of one G-Mamba block: ``O(Ld)`` (App. B.6).

    Returns the exact number of multiply/adds so a test can assert it grows
    *linearly* (not quadratically) in ``length`` -- the defining advantage over a
    Transformer's ``O(L^2 d)`` attention.
    """
    scan = length * dim * 4          # A,B,C,G element-wise ops per token
    dwc = length * dim * dwc_kernel  # depthwise conv
    gsm = length * dim * 3           # gated fusion element-wise ops
    return scan + dwc + gsm
