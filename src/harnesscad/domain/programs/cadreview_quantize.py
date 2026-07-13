"""CADReview spatial-value quantization (Appendix B.4 + Eq. 1).

The CADReview code editor cannot reliably emit continuous spatial parameters
(axis coordinates, translation offsets, angles) as free-form decimals: tiny
regression errors violate geometric constraints, and long decimals are
token-inefficient. The paper's fix is to *quantize* every spatial value to 8
bits before it is learned or emitted — a fixed grid of 256 levels over the
value's range — and then to *re-weight* the loss so numeric tokens count double
(Eq. 1), forcing the model to attend to these quantized values.

Both steps are deterministic arithmetic with no learning, so this module
reproduces them exactly:

* :func:`quantize` / :func:`dequantize` map a real value to/from one of 256
  levels over ``[lo, hi]`` (round-half-to-even, clamped).
  :func:`quantize_program` rewrites every numeric literal in an OpenSCAD program
  to its dequantized grid value, so a program can be snapped onto the same grid
  the editor operates on (useful for building training targets and for comparing
  two programs modulo sub-grid noise).
* :func:`sgo_token_weights` reproduces the SGO re-weighting rule (Eq. 1):
  weight 2 for numeric tokens, 1 otherwise.

Pure stdlib; deterministic.
"""

from __future__ import annotations

import re
from typing import List, Sequence

_LEVELS = 256          # 8-bit
_MAX_LEVEL = _LEVELS - 1
_NUM_RE = re.compile(r"-?\d+\.\d+|-?\d+")


def quantize(value: float, lo: float = 0.0, hi: float = 256.0) -> int:
    """Map ``value`` in ``[lo, hi]`` to an 8-bit level in ``[0, 255]``.

    Values outside the range are clamped. Uses Python's round-half-to-even for
    a deterministic, bias-free grid assignment."""
    if hi <= lo:
        raise ValueError("hi must be greater than lo")
    frac = (value - lo) / (hi - lo)
    frac = min(1.0, max(0.0, frac))
    return int(round(frac * _MAX_LEVEL))


def dequantize(level: int, lo: float = 0.0, hi: float = 256.0) -> float:
    """Map an 8-bit level back to the representative real value in ``[lo, hi]``."""
    level = min(_MAX_LEVEL, max(0, int(level)))
    return lo + (level / _MAX_LEVEL) * (hi - lo)


def snap(value: float, lo: float = 0.0, hi: float = 256.0) -> float:
    """Round ``value`` to the nearest point on the 8-bit grid over ``[lo, hi]``."""
    return dequantize(quantize(value, lo, hi), lo, hi)


def quantize_program(src: str, lo: float = 0.0, hi: float = 256.0,
                     decimals: int = 4) -> str:
    """Rewrite every numeric literal in ``src`` to its 8-bit grid value."""
    def _repl(m: "re.Match") -> str:
        v = float(m.group())
        snapped = snap(v, lo, hi)
        if snapped == int(snapped):
            return str(int(snapped))
        return str(round(snapped, decimals))
    return _NUM_RE.sub(_repl, src)


def is_numeric_token(token: str) -> bool:
    """True when ``token`` is (or begins with) a real-number literal."""
    return bool(re.match(r"^-?\d+\.?\d*$", token.strip()))


def sgo_token_weights(tokens: Sequence[str],
                      numeric_weight: float = 2.0,
                      default_weight: float = 1.0) -> List[float]:
    """The SGO re-weighting weights (Eq. 1): numeric tokens weigh double."""
    return [numeric_weight if is_numeric_token(t) else default_weight
            for t in tokens]
