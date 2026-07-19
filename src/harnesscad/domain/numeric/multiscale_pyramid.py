"""Multi-scale (pyramid) latent encoding of a parametric CAD command sequence.

The denoiser generates *long* CAD command sequences (60-256
commands) by learning a **multi-scale latent representation**: the sequence is
processed at three different scales (window sizes 64/128/256) and a
Compress Block progressively pools the length-``N`` sequence into a compact
latent. The *learned* state-space/Transformer weights are external, but
the **multi-resolution pyramid that carries features across scales is a purely
deterministic pooling / interpolation construction** -- that is what this module
implements, distinct from the single-scale SSM scan in
``numeric.geofusion_state_space`` and the discretization/bidirectional scan of
the state-space modules.

A *sequence* is a ``tuple`` of equal-width feature vectors (``tuple[float,...]``).
This module provides:

* :func:`downsample` -- pool a sequence to half (or ``1/factor``) length by
  averaging (or max-pooling) each non-overlapping window of ``factor`` tokens;
* :func:`upsample` -- expand a coarse sequence back to a target length by
  nearest-neighbour or piecewise-linear interpolation;
* :func:`build_pyramid` -- the full Gaussian-style pyramid: level 0 is the input,
  each finer index is a coarser (down-pooled) copy, for a requested number of
  levels or a set of scale lengths;
* :func:`laplacian_pyramid` -- residual (band-pass) encoding: at each level the
  detail lost by down-then-up sampling, so the sequence can be reconstructed
  exactly by :func:`reconstruct_laplacian`;
* :func:`reconstruct_laplacian` -- invert a Laplacian pyramid (lossless);
* :func:`scale_lengths` -- the window/scale schedule (e.g. 256->128->64) used by
  the multi-scale branches;
* :func:`pyramid_pool_flops` -- analytic op count, ``O(N d)`` total across a
  pyramid (geometric series), verifying the cheap multi-scale overhead.

Everything is stdlib-only and deterministic (no randomness, no wall clock).
"""

from __future__ import annotations

Vec = tuple[float, ...]
Seq = tuple[Vec, ...]


def _width(seq: Seq) -> int:
    if not seq:
        raise ValueError("sequence must be non-empty")
    d = len(seq[0])
    for v in seq:
        if len(v) != d:
            raise ValueError("all tokens must share a common width")
    return d


def scale_lengths(base: int, levels: int, factor: int = 2) -> tuple[int, ...]:
    """Length schedule ``base, base//factor, base//factor^2, ...`` (``levels``
    entries), the multi-scale window schedule (e.g. 256, 128, 64 for
    ``factor=2``). Each length is floored at 1.
    """
    if base <= 0:
        raise ValueError("base length must be positive")
    if levels <= 0:
        raise ValueError("levels must be positive")
    if factor < 2:
        raise ValueError("factor must be >= 2")
    out: list[int] = []
    cur = base
    for _ in range(levels):
        out.append(max(1, cur))
        cur = cur // factor
    return tuple(out)


def downsample(seq: Seq, factor: int = 2, mode: str = "avg") -> Seq:
    """Pool ``seq`` down by ``factor`` (non-overlapping windows).

    ``mode="avg"`` averages each window (Gaussian-pyramid smoothing); ``"max"``
    takes the per-channel maximum. A trailing partial window (when ``len(seq)``
    is not a multiple of ``factor``) is pooled over its available tokens, so the
    output length is ``ceil(len(seq) / factor)``.
    """
    if factor < 2:
        raise ValueError("factor must be >= 2")
    d = _width(seq)
    n = len(seq)
    out: list[Vec] = []
    for start in range(0, n, factor):
        window = seq[start:start + factor]
        if mode == "avg":
            acc = [0.0] * d
            for v in window:
                for ch in range(d):
                    acc[ch] += v[ch]
            inv = 1.0 / len(window)
            out.append(tuple(x * inv for x in acc))
        elif mode == "max":
            acc = list(window[0])
            for v in window[1:]:
                for ch in range(d):
                    if v[ch] > acc[ch]:
                        acc[ch] = v[ch]
            out.append(tuple(acc))
        else:
            raise ValueError("mode must be 'avg' or 'max'")
    return tuple(out)


def upsample(seq: Seq, target_len: int, mode: str = "nearest") -> Seq:
    """Expand ``seq`` to ``target_len`` tokens.

    ``mode="nearest"`` repeats the closest source token; ``"linear"`` does
    piecewise-linear interpolation between neighbouring source tokens. Both are
    deterministic. ``target_len`` must be >= current length.
    """
    d = _width(seq)
    n = len(seq)
    if target_len < n:
        raise ValueError("target_len must be >= current length")
    if target_len == n:
        return seq
    if n == 1:
        return tuple(seq[0] for _ in range(target_len))
    out: list[Vec] = []
    for i in range(target_len):
        # map output index i to a fractional source position in [0, n-1]
        pos = i * (n - 1) / (target_len - 1)
        if mode == "nearest":
            out.append(seq[int(pos + 0.5)])
        elif mode == "linear":
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            frac = pos - lo
            a, b = seq[lo], seq[hi]
            out.append(tuple(a[ch] * (1.0 - frac) + b[ch] * frac
                             for ch in range(d)))
        else:
            raise ValueError("mode must be 'nearest' or 'linear'")
    return tuple(out)


def build_pyramid(seq: Seq, levels: int, factor: int = 2,
                  mode: str = "avg") -> tuple[Seq, ...]:
    """Gaussian-style pyramid: ``levels`` progressively coarser copies.

    ``result[0]`` is the input sequence; ``result[k]`` is ``result[k-1]``
    down-pooled by ``factor``. Pooling stops early (repeating a length-1 level)
    once a level reaches a single token.
    """
    if levels <= 0:
        raise ValueError("levels must be positive")
    _width(seq)
    pyr: list[Seq] = [seq]
    for _ in range(1, levels):
        prev = pyr[-1]
        if len(prev) <= 1:
            pyr.append(prev)
        else:
            pyr.append(downsample(prev, factor, mode))
    return tuple(pyr)


def _sub(a: Vec, b: Vec) -> Vec:
    return tuple(x - y for x, y in zip(a, b))


def _add(a: Vec, b: Vec) -> Vec:
    return tuple(x + y for x, y in zip(a, b))


def laplacian_pyramid(seq: Seq, levels: int, factor: int = 2,
                      up_mode: str = "linear") -> tuple[tuple[Seq, ...], Seq]:
    """Laplacian (band-pass residual) pyramid for lossless multi-scale coding.

    Returns ``(details, coarsest)`` where ``details[k]`` is the high-frequency
    residual at level ``k`` -- the difference between level ``k`` of the Gaussian
    pyramid and the up-sampled level ``k+1`` -- and ``coarsest`` is the smallest
    (top) Gaussian level. Together with :func:`reconstruct_laplacian` this gives
    an exact multi-scale decomposition of the CAD feature sequence.
    """
    gauss = build_pyramid(seq, levels, factor, mode="avg")
    details: list[Seq] = []
    for k in range(len(gauss) - 1):
        fine = gauss[k]
        coarse = gauss[k + 1]
        up = upsample(coarse, len(fine), up_mode)
        details.append(tuple(_sub(f, u) for f, u in zip(fine, up)))
    return tuple(details), gauss[-1]


def reconstruct_laplacian(details: tuple[Seq, ...], coarsest: Seq,
                          up_mode: str = "linear") -> Seq:
    """Invert :func:`laplacian_pyramid`, recovering the original sequence
    exactly (up to floating point): rebuild finer levels top-down by adding each
    stored detail band to the up-sampled coarser level.
    """
    cur = coarsest
    for detail in reversed(details):
        up = upsample(cur, len(detail), up_mode)
        cur = tuple(_add(u, dv) for u, dv in zip(up, detail))
    return cur


def pyramid_pool_flops(length: int, dim: int, levels: int,
                       factor: int = 2) -> int:
    """Total scalar-op count to build a Gaussian pyramid: sum over levels of
    ``len_k * dim``. Because lengths shrink geometrically this is bounded by
    ``length*dim*factor/(factor-1)`` = ``O(N d)`` -- the multi-scale encoding
    adds only constant-factor overhead over a single pass.
    """
    if length <= 0 or dim <= 0 or levels <= 0:
        raise ValueError("length, dim, levels must be positive")
    total = 0
    cur = length
    for lvl in range(levels):
        if lvl == 0:
            total += cur * dim
        else:
            nxt = max(1, (cur + factor - 1) // factor)
            total += nxt * dim
            cur = nxt
    return total
