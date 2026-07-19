"""Parameter quantization ranges and codecs for two CAD sequence tokenizers.

Both DeepCAD and SkexGen turn a continuous CAD construction sequence into a
sequence of *discrete tokens* so a transformer can model it. The mapping from a
float parameter to a token id is the single most consequential piece of glue in
those pipelines: it fixes the vocabulary size, the achievable precision, and the
failure mode when a value leaves the modelled range. This module reproduces both
codecs as constants plus exact reimplementations, so the harness can reason
about token-space precision without importing either project (or numpy/torch).

The constants and numeric behaviour are transcribed as *facts*; no code is
copied, and the interesting content is the *behaviour*, documented below.

Five behaviours here are easy to get wrong and are what the selfcheck pins down:

1. **DeepCAD's grid is asymmetric.** ``numericalize`` maps ``x -> round((x+1)/2*n)``
   then clips to ``[0, n-1]``, while ``denumericalize`` inverts with ``q/n*2-1``.
   The forward map's natural top code is ``n`` but the vocabulary stops at
   ``n-1``, so ``x = -1`` round-trips exactly while ``x = +1`` comes back as
   ``0.9921875`` (for ``n = 256``). The low end is a fencepost; the high end is a
   clip.

2. **DeepCAD clamps silently, and the guard is looser than the range.**
   ``Extrude.numericalize`` asserts ``-2.0 <= extent <= 2.0`` but the affine map
   only covers ``[-1, 1]``; an extent of ``1.5`` passes the assert and is then
   clipped to the top bin, losing the value with no error raised.

3. **SkexGen truncates, it does not round.** Its ``quantize`` computes the scaled
   value, clips, then calls ``.astype('int32')`` -- truncation toward zero, which
   for the clipped (non-negative) value is a floor. So SkexGen's worst-case
   round-trip error is a *full* bin, not half a bin, and the error is one-sided
   (the dequantized value never exceeds the input). Its divisor is ``2**b - 1``
   rather than DeepCAD's ``n``, so the token grid spans the full range and both
   endpoints are exactly representable -- though truncation can still keep the
   forward map from reaching the top token, see 5.

4. **The unclipped variant emits out-of-vocabulary tokens.**
   ``geom_utils.quantize_verts`` has no clip step at all, so an input outside
   ``[-0.5, 0.5]`` yields a token id outside ``[0, 2**b - 1]``.

5. **Truncation makes the top endpoint floating-point fragile.** Because SkexGen
   floors instead of rounding, the exactness of its top token depends on
   ``(x-min)*R/(max-min)`` landing on ``R`` and not a hair below it. It does for
   the binary-exact ranges (SKETCH_R, RADIUS_R, EXTRUDE_R, OFFSET_R), but *not*
   for ``SCALE_R = 1.4``: ``1.4*63/1.4`` evaluates to ``62.99999999999999``, so
   the largest scale quantizes to token **62** and token 63 is unreachable for
   that field. This was verified against a numpy replay of the original
   ``quantize``; it is a property of that codec, not of this reimplementation. A
   rounding codec would not have this failure.

Note also a docstring/code discrepancy present in both codecs: the quantizer
docstrings say the output range is ``[0, n_bits**2 - 1]`` while every code path
uses ``2**n_bits - 1``. The code is authoritative (and is what this module
reproduces); at ``n_bits = 6`` the two readings differ (35 vs 63).

Deterministic, stdlib-only, ASCII-only. Run ``python -m
harnesscad.domain.programs.quantization_ranges --selfcheck``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = [
    "ARGS_DIM",
    "NORM_FACTOR",
    "PAD_VAL",
    "N_ARGS",
    "N_ARGS_SKETCH",
    "N_ARGS_PLANE",
    "N_ARGS_TRANS",
    "N_ARGS_EXT_PARAM",
    "N_ARGS_EXT",
    "MAX_N_EXT",
    "MAX_N_LOOPS",
    "MAX_N_CURVES",
    "MAX_TOTAL_LEN",
    "ALL_COMMANDS",
    "EXTRUDE_OPERATIONS",
    "EXTENT_TYPE",
    "FieldSpec",
    "DEEPCAD_FIELDS",
    "SKEXGEN_BIT",
    "SKEXGEN_RANGES",
    "deepcad_quantize",
    "deepcad_dequantize",
    "deepcad_quantize_plane_angle",
    "deepcad_dequantize_plane_angle",
    "deepcad_quantize_arc_angle",
    "deepcad_quantize_sketch_coord",
    "deepcad_quantize_radius",
    "deepcad_quantize_sketch_size",
    "deepcad_dequantize_sketch_size",
    "deepcad_sketch_normalize_scale",
    "deepcad_sketch_occupied_span",
    "deepcad_max_roundtrip_error",
    "skexgen_quantize",
    "skexgen_dequantize",
    "skexgen_quantize_unclipped",
    "skexgen_top_token_reachable",
    "skexgen_bin_width",
    "skexgen_max_roundtrip_error",
]


# --------------------------------------------------------------------------
# DeepCAD constants
# --------------------------------------------------------------------------

#: Argument vocabulary size: every quantized argument is one of 256 tokens.
ARGS_DIM = 256

#: Scale factor applied during normalization to prevent overflow during
#: augmentation. Shapes fill only 75% of the available extent.
NORM_FACTOR = 0.75

#: Fill value for arguments a command does not use.
PAD_VAL = -1

N_ARGS_SKETCH = 5      # x, y, alpha, f, r
N_ARGS_PLANE = 3       # theta, phi, gamma
N_ARGS_TRANS = 4       # p_x, p_y, p_z, s
N_ARGS_EXT_PARAM = 4   # e1, e2, b, u
N_ARGS_EXT = N_ARGS_PLANE + N_ARGS_TRANS + N_ARGS_EXT_PARAM   # 11
N_ARGS = N_ARGS_SKETCH + N_ARGS_EXT                           # 16

MAX_N_EXT = 10         # maximum number of extrusions
MAX_N_LOOPS = 6        # maximum number of loops per sketch
MAX_N_CURVES = 15      # maximum number of curves per loop
MAX_TOTAL_LEN = 60     # maximum CAD sequence length

ALL_COMMANDS = ("Line", "Arc", "Circle", "EOS", "SOL", "Ext")

EXTRUDE_OPERATIONS = (
    "NewBodyFeatureOperation",
    "JoinFeatureOperation",
    "CutFeatureOperation",
    "IntersectFeatureOperation",
)

EXTENT_TYPE = (
    "OneSideFeatureExtentType",
    "SymmetricFeatureExtentType",
    "TwoSidesFeatureExtentType",
)


@dataclass(frozen=True)
class FieldSpec:
    """One slot of DeepCAD's 16-dim argument vector.

    ``codec`` names the mapping used to reach token space:

    * ``"affine"``    -- ``round((x+1)/2*n)`` clipped to ``[0, n-1]``; source
                         range ``[-1, 1]``.
    * ``"affine_pi"`` -- as ``affine`` after dividing by ``pi``; source range
                         ``[-pi, pi]``.
    * ``"grid"``      -- already in sketch pixel space; ``round(x)`` clipped only.
    * ``"grid_r"``    -- as ``grid`` but clipped to ``[1, n-1]``: a radius may
                         never quantize to zero.
    * ``"halfscale"`` -- ``round(x/2*n)`` clipped to ``[0, n-1]``; source range
                         ``[0, 2]``.
    * ``"angle_2pi"`` -- ``round(x/(2*pi)*n)`` clipped to ``[0, n-1]``.
    * ``"index"``     -- a categorical index, not quantized at all.
    """

    name: str
    codec: str
    lo: float
    hi: float
    source: str
    note: str = ""


#: The 16 argument slots, in vector order, with the codec each one actually uses.
DEEPCAD_FIELDS: Tuple[FieldSpec, ...] = (
    FieldSpec("x", "grid", 0.0, 255.0, "cadlib/curves.py Line/Arc/Circle.numericalize",
              "sketch-plane pixel coordinate, already normalized by Profile.normalize"),
    FieldSpec("y", "grid", 0.0, 255.0, "cadlib/curves.py Line/Arc/Circle.numericalize",
              "sketch-plane pixel coordinate"),
    FieldSpec("alpha", "angle_2pi", 0.0, 255.0, "cadlib/curves.py Arc.to_vector",
              "arc sweep angle in quantized 2*pi/256 units; floored at 1 by "
              "max(abs(start-end), 1) so a zero-sweep arc is impossible"),
    FieldSpec("f", "index", 0.0, 1.0, "cadlib/curves.py Arc.to_vector",
              "arc clock sign flag, int(clock_sign); not quantized"),
    FieldSpec("r", "grid_r", 1.0, 255.0, "cadlib/curves.py Circle.numericalize",
              "circle radius; clipped to min=1, never 0"),
    FieldSpec("theta", "affine_pi", -math.pi, math.pi,
              "cadlib/extrude.py CoordSystem.numericalize", "sketch plane orientation"),
    FieldSpec("phi", "affine_pi", -math.pi, math.pi,
              "cadlib/extrude.py CoordSystem.numericalize", "sketch plane orientation"),
    FieldSpec("gamma", "affine_pi", -math.pi, math.pi,
              "cadlib/extrude.py CoordSystem.numericalize", "sketch plane orientation"),
    FieldSpec("p_x", "affine", -1.0, 1.0, "cadlib/extrude.py Extrude.numericalize",
              "sketch plane origin; the source comments that origin can be out of bounds"),
    FieldSpec("p_y", "affine", -1.0, 1.0, "cadlib/extrude.py Extrude.numericalize",
              "sketch plane origin"),
    FieldSpec("p_z", "affine", -1.0, 1.0, "cadlib/extrude.py Extrude.numericalize",
              "sketch plane origin"),
    FieldSpec("s", "halfscale", 0.0, 2.0, "cadlib/extrude.py Extrude.numericalize",
              "sketch bbox size; non-negative, so it uses a half-range map, not the "
              "signed affine one"),
    FieldSpec("e1", "affine", -1.0, 1.0, "cadlib/extrude.py Extrude.numericalize",
              "extrude extent one; guarded by assert -2<=e<=2, which is LOOSER than "
              "the [-1,1] the codec covers -- values in (1,2] clip silently"),
    FieldSpec("e2", "affine", -1.0, 1.0, "cadlib/extrude.py Extrude.numericalize",
              "extrude extent two; same loose-assert caveat as e1"),
    FieldSpec("b", "index", 0.0, 3.0, "cadlib/extrude.py Extrude.numericalize",
              "boolean operation, index into EXTRUDE_OPERATIONS; not quantized"),
    FieldSpec("u", "index", 0.0, 2.0, "cadlib/extrude.py Extrude.numericalize",
              "extent type, index into EXTENT_TYPE; not quantized"),
)


# --------------------------------------------------------------------------
# SkexGen constants
# --------------------------------------------------------------------------

#: Bit width the SkexGen pipeline uses at every stage (``--bit 6``),
#: i.e. 64 tokens per quantized field -- a quarter of DeepCAD's ARGS_DIM.
SKEXGEN_BIT = 6

#: Symmetric/positive half-ranges each SkexGen field is quantized against.
#: SKETCH_R/RADIUS_R/EXTRUDE_R are +-R; SCALE_R is [0, R]; OFFSET_R is +-R.
SKEXGEN_RANGES: Dict[str, Tuple[float, float]] = {
    "sketch": (-1.0, 1.0),      # SKETCH_R = 1   -- curve points, plane origin
    "radius": (-1.0, 1.0),      # RADIUS_R = 1
    "extrude": (-1.0, 1.0),     # EXTRUDE_R = 1.0 -- extrude values
    "scale": (0.0, 1.4),        # SCALE_R = 1.4  -- one-sided
    "offset": (-0.9, 0.9),      # OFFSET_R = 0.9 -- profile centre offset
}

#: geom_utils.quantize_verts hardcodes this narrower range and omits the clip.
SKEXGEN_GEOM_UTILS_RANGE: Tuple[float, float] = (-0.5, 0.5)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _clip(value: int, lo: int, hi: int) -> int:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _check_n(n: int) -> None:
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")


# --------------------------------------------------------------------------
# DeepCAD codecs
# --------------------------------------------------------------------------

def deepcad_quantize(x: float, n: int = ARGS_DIM) -> int:
    """Quantize a value in ``[-1, 1]`` to a token in ``[0, n-1]`` (DeepCAD affine).

    Reproduces ``round((x + 1) / 2 * n)`` clipped to ``[0, n-1]``. Rounding is
    half-to-even (numpy's ``.round()`` and Python's ``round()`` agree here).
    Out-of-range input clamps rather than raising -- see the module docstring.
    """
    _check_n(n)
    return _clip(int(round((float(x) + 1.0) / 2.0 * n)), 0, n - 1)


def deepcad_dequantize(q: int, n: int = ARGS_DIM) -> float:
    """Inverse of :func:`deepcad_quantize`: ``q / n * 2 - 1``.

    Note the divisor is ``n``, not ``n - 1``: this is what makes the grid
    asymmetric (``q = 0`` gives exactly ``-1.0``; ``q = n-1`` gives ``1 - 2/n``,
    never ``+1.0``).
    """
    _check_n(n)
    return int(q) / n * 2.0 - 1.0


def deepcad_quantize_plane_angle(theta: float, n: int = ARGS_DIM) -> int:
    """Quantize a sketch-plane orientation angle in ``[-pi, pi]`` to ``[0, n-1]``."""
    _check_n(n)
    return _clip(int(round((float(theta) / math.pi + 1.0) / 2.0 * n)), 0, n - 1)


def deepcad_dequantize_plane_angle(q: int, n: int = ARGS_DIM) -> float:
    """Inverse of :func:`deepcad_quantize_plane_angle`: ``(q / n * 2 - 1) * pi``."""
    _check_n(n)
    return (int(q) / n * 2.0 - 1.0) * math.pi


def deepcad_quantize_arc_angle(theta: float, n: int = ARGS_DIM) -> int:
    """Quantize an arc endpoint angle in ``[0, 2*pi)`` to ``[0, n-1]``.

    DeepCAD quantizes ``start_angle``/``end_angle`` this way and then emits only
    their *difference* (``Arc.to_vector``), so the repo defines no inverse for
    this codec; none is offered here either.
    """
    _check_n(n)
    return _clip(int(round(float(theta) / (2.0 * math.pi) * n)), 0, n - 1)


def deepcad_quantize_sketch_coord(x: float, n: int = ARGS_DIM) -> int:
    """Quantize a sketch pixel coordinate: round and clip, no affine rescale.

    Curve control points are already carried in ``[0, n]`` sketch space by
    ``Profile.normalize``, so ``Line/Arc/Circle.numericalize`` only rounds.
    """
    _check_n(n)
    return _clip(int(round(float(x))), 0, n - 1)


def deepcad_quantize_radius(r: float, n: int = ARGS_DIM) -> int:
    """Quantize a circle radius: round and clip to ``[1, n-1]``.

    The floor of 1 (not 0) is deliberate in ``Circle.numericalize``: it makes a
    degenerate zero-radius circle unrepresentable in token space.
    """
    _check_n(n)
    return _clip(int(round(float(r))), 1, n - 1)


def deepcad_quantize_sketch_size(s: float, n: int = ARGS_DIM) -> int:
    """Quantize a sketch bbox size in ``[0, 2]`` to ``[0, n-1]`` (``round(s/2*n)``).

    Distinct from :func:`deepcad_quantize`: sketch size is non-negative, so there
    is no ``+1`` shift.
    """
    _check_n(n)
    return _clip(int(round(float(s) / 2.0 * n)), 0, n - 1)


def deepcad_dequantize_sketch_size(q: int, n: int = ARGS_DIM) -> float:
    """Inverse of :func:`deepcad_quantize_sketch_size`: ``q / n * 2``."""
    _check_n(n)
    return int(q) / n * 2.0


def deepcad_sketch_normalize_scale(bbox_size: float, size: int = ARGS_DIM) -> float:
    """Scale factor ``Profile.normalize`` applies: ``(size/2*NORM_FACTOR - 1)/bbox``.

    The profile is translated so its start point sits at the grid centre
    ``size/2``, then scaled so its bbox half-extent becomes
    ``size/2*NORM_FACTOR - 1``.
    """
    if bbox_size <= 0.0:
        raise ValueError(f"bbox_size must be positive, got {bbox_size}")
    return (size / 2.0 * NORM_FACTOR - 1.0) / float(bbox_size)


def deepcad_sketch_occupied_span(size: int = ARGS_DIM) -> Tuple[float, float]:
    """Grid span a normalized profile can occupy: ``(centre - h, centre + h)``.

    With ``h = size/2*NORM_FACTOR - 1`` and centre ``size/2``. For the default
    ``size = 256`` this is ``(33.0, 223.0)`` -- the normalized sketch never
    reaches the grid edges, which is exactly the augmentation headroom
    ``NORM_FACTOR`` is there to buy.
    """
    half = size / 2.0 * NORM_FACTOR - 1.0
    centre = size / 2.0
    return (centre - half, centre + half)


def deepcad_max_roundtrip_error(n: int = ARGS_DIM) -> float:
    """Worst-case ``|dequantize(quantize(x)) - x|`` for ``x`` in ``[-1, 1-2/n]``.

    Half a bin, i.e. ``1/n``. Inputs above ``1 - 2/n`` are clipped and do worse;
    at ``x = +1`` the error is ``2/n``.
    """
    _check_n(n)
    return 1.0 / n


# --------------------------------------------------------------------------
# SkexGen codecs
# --------------------------------------------------------------------------

def skexgen_quantize(x: float, n_bits: int = SKEXGEN_BIT,
                     min_range: float = -1.0, max_range: float = 1.0) -> int:
    """Quantize ``x`` to ``[0, 2**n_bits - 1]`` the way SkexGen's ``quantize`` does.

    Scale, clip, then **truncate** (the source's ``.astype('int32')``). The clip
    happens first, so the truncated value is non-negative and truncation is a
    floor. Consequences: the error is one-sided (never overshoots) and the
    worst case is a full bin, not half.
    """
    if n_bits < 1:
        raise ValueError(f"n_bits must be at least 1, got {n_bits}")
    if not max_range > min_range:
        raise ValueError(f"empty range [{min_range}, {max_range}]")
    span = 2 ** n_bits - 1
    scaled = (float(x) - min_range) * span / (max_range - min_range)
    if scaled < 0.0:
        scaled = 0.0
    elif scaled > span:
        scaled = float(span)
    return int(scaled)   # truncation toward zero, matching astype('int32')


def skexgen_dequantize(q: int, n_bits: int = SKEXGEN_BIT,
                       min_range: float = -1.0, max_range: float = 1.0) -> float:
    """Inverse of :func:`skexgen_quantize`: ``q * (max-min) / (2**b - 1) + min``.

    The divisor is ``2**b - 1``, so the token grid *spans* the full range and
    both endpoints are exactly representable (``q = 0 -> min_range``,
    ``q = 2**b - 1 -> max_range``) -- unlike DeepCAD. Whether the forward map
    actually reaches the top token is a separate question: see note 5 in the
    module docstring and :func:`skexgen_top_token_reachable`.

    The source's optional ``add_noise`` dither is omitted: it is nondeterministic
    and is a training-time augmentation, not part of the codec.
    """
    if n_bits < 1:
        raise ValueError(f"n_bits must be at least 1, got {n_bits}")
    if not max_range > min_range:
        raise ValueError(f"empty range [{min_range}, {max_range}]")
    span = 2 ** n_bits - 1
    return int(q) * (max_range - min_range) / span + min_range


def skexgen_quantize_unclipped(x: float, n_bits: int = 8,
                               min_range: float = -0.5,
                               max_range: float = 0.5) -> int:
    """``geom_utils.quantize_verts``: same scale-and-truncate, but **no clip**.

    Kept distinct from :func:`skexgen_quantize` because the missing clip is a
    real hazard: an input outside ``[min_range, max_range]`` produces a token id
    outside the vocabulary, which an embedding lookup will not catch as an error
    in the way a range check would.
    """
    if n_bits < 1:
        raise ValueError(f"n_bits must be at least 1, got {n_bits}")
    if not max_range > min_range:
        raise ValueError(f"empty range [{min_range}, {max_range}]")
    span = 2 ** n_bits - 1
    return int((float(x) - min_range) * span / (max_range - min_range))


def skexgen_top_token_reachable(n_bits: int = SKEXGEN_BIT,
                                min_range: float = -1.0,
                                max_range: float = 1.0) -> bool:
    """True iff ``max_range`` itself quantizes to the top token ``2**n_bits - 1``.

    False when floating-point evaluation of the scale lands just below the top
    code and truncation drops it a bin -- which is the case for SkexGen's own
    ``SCALE_R = 1.4``. See note 5 in the module docstring.
    """
    return skexgen_quantize(max_range, n_bits, min_range, max_range) == 2 ** n_bits - 1


def skexgen_bin_width(n_bits: int = SKEXGEN_BIT,
                      min_range: float = -1.0, max_range: float = 1.0) -> float:
    """Width of one quantization bin: ``(max - min) / (2**n_bits - 1)``."""
    if n_bits < 1:
        raise ValueError(f"n_bits must be at least 1, got {n_bits}")
    return (max_range - min_range) / (2 ** n_bits - 1)


def skexgen_max_roundtrip_error(n_bits: int = SKEXGEN_BIT,
                                min_range: float = -1.0,
                                max_range: float = 1.0) -> float:
    """Worst-case round-trip error in range: one full bin, because it truncates."""
    return skexgen_bin_width(n_bits, min_range, max_range)


# --------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------

def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, label: str) -> None:
        if not cond:
            failures.append(label)

    n = ARGS_DIM

    # -- Constants are internally consistent with their own arithmetic.
    check(N_ARGS_EXT == N_ARGS_PLANE + N_ARGS_TRANS + N_ARGS_EXT_PARAM == 11,
          "N_ARGS_EXT derives from its parts")
    check(N_ARGS == 16, "argument vector is 16-dim")
    check(len(DEEPCAD_FIELDS) == N_ARGS, "field table covers every argument slot")
    check([f.name for f in DEEPCAD_FIELDS][:5]
          == ["x", "y", "alpha", "f", "r"], "sketch args come first, in order")
    check(len(ALL_COMMANDS) == 6 and ALL_COMMANDS.index("Ext") == 5,
          "command vocabulary matches the constant table")
    check(len(EXTRUDE_OPERATIONS) == 4 and len(EXTENT_TYPE) == 3,
          "categorical arg cardinalities")

    # -- DeepCAD bin edges: exact at the low end, clipped at the high end.
    check(deepcad_quantize(-1.0) == 0, "x=-1 is bin 0")
    check(deepcad_dequantize(0) == -1.0, "bin 0 round-trips to exactly -1.0")
    check(deepcad_quantize(1.0) == n - 1, "x=+1 clips to the top bin")
    check(deepcad_dequantize(n - 1) == 1.0 - 2.0 / n == 0.9921875,
          "top bin dequantizes to 1-2/n, NOT +1.0 -- the grid is asymmetric")
    check(round((1.0 + 1.0) / 2.0 * n) == n,
          "the unclipped forward map really does want bin n; the clip is load-bearing")
    check(deepcad_quantize(0.0) == n // 2, "x=0 lands on the midpoint bin")
    check(deepcad_dequantize(n // 2) == 0.0, "midpoint bin round-trips to exactly 0")

    # -- Bin boundaries fall at (2k+1)/n - 1, with half-to-even ties.
    edge = 1.0 / n - 1.0              # the k=0 -> k=1 tie point
    check(deepcad_quantize(edge) == 0, "tie at bin 0/1 boundary rounds to even (0)")
    check(deepcad_quantize(edge + 1e-9) == 1, "just above the tie moves to bin 1")
    check(deepcad_quantize(edge - 1e-9) == 0, "just below the tie stays in bin 0")

    # -- Clamping never raises, in either direction.
    for bad in (5.0, -5.0, 1e9, -1e9):
        q = deepcad_quantize(bad)
        check(0 <= q <= n - 1, f"out-of-range {bad} clamps into vocabulary")
    check(deepcad_quantize(5.0) == n - 1 and deepcad_quantize(-5.0) == 0,
          "clamping saturates to the correct end")

    # -- The extent guard is looser than the codec's range (silent loss).
    check(-2.0 <= 1.5 <= 2.0 and deepcad_quantize(1.5) == deepcad_quantize(1.0),
          "extent 1.5 passes DeepCAD's assert yet is indistinguishable from 1.0")

    # -- Round-trip error bound and monotonicity over the representable range.
    bound = deepcad_max_roundtrip_error(n)
    check(bound == 1.0 / n, "documented bound is half a bin")
    steps = 2000
    hi = 1.0 - 2.0 / n
    worst = 0.0
    prev_q = -1
    for i in range(steps + 1):
        x = -1.0 + (hi + 1.0) * i / steps
        q = deepcad_quantize(x)
        check(q >= prev_q, "quantizer is monotonic non-decreasing")
        prev_q = q
        worst = max(worst, abs(deepcad_dequantize(q) - x))
    check(worst <= bound + 1e-12, f"round-trip error {worst} within 1/n over [-1, 1-2/n]")
    check(worst > bound / 2, "the bound is tight, not merely an over-estimate")
    check(abs(deepcad_dequantize(deepcad_quantize(1.0)) - 1.0) == 2.0 / n,
          "clipping doubles the error exactly at x=+1")

    # -- Plane angles: same asymmetry, carried through the pi rescale.
    check(deepcad_quantize_plane_angle(-math.pi) == 0, "-pi is bin 0")
    check(deepcad_dequantize_plane_angle(0) == -math.pi, "bin 0 gives back -pi")
    check(deepcad_quantize_plane_angle(math.pi) == n - 1, "+pi clips to top bin")
    check(deepcad_dequantize_plane_angle(n - 1) < math.pi, "+pi does not round-trip")
    check(deepcad_quantize_plane_angle(0.0) == n // 2, "0 rad is the midpoint bin")

    # -- Arc angles use a 2*pi grid.
    check(deepcad_quantize_arc_angle(0.0) == 0, "arc angle 0 is bin 0")
    check(deepcad_quantize_arc_angle(math.pi) == n // 2, "pi is half of the 2pi grid")
    check(deepcad_quantize_arc_angle(2.0 * math.pi) == n - 1, "2pi clips to top bin")

    # -- Radius floor: a circle can never quantize away to nothing.
    check(deepcad_quantize_radius(0.0) == 1, "radius 0 floors at bin 1")
    check(deepcad_quantize_radius(0.4) == 1, "sub-pixel radius floors at bin 1")
    check(all(deepcad_quantize_radius(r / 10.0) >= 1 for r in range(0, 40)),
          "radius is never bin 0 for any small input")
    check(deepcad_quantize_sketch_coord(0.4) == 0,
          "plain sketch coords, unlike radius, DO reach bin 0")

    # -- Sketch size uses the unsigned half-range map, not the signed affine one.
    check(deepcad_quantize_sketch_size(0.0) == 0, "size 0 is bin 0")
    check(deepcad_quantize_sketch_size(2.0) == n - 1, "size 2 clips to top bin")
    check(deepcad_quantize_sketch_size(1.0) == n // 2, "size 1 is the midpoint bin")
    check(deepcad_dequantize_sketch_size(n // 2) == 1.0, "size midpoint round-trips")
    check(deepcad_quantize_sketch_size(1.0) != deepcad_quantize(1.0),
          "size and affine codecs are genuinely different maps")

    # -- NORM_FACTOR headroom: normalized sketches keep clear of the grid edges.
    lo_s, hi_s = deepcad_sketch_occupied_span(ARGS_DIM)
    check((lo_s, hi_s) == (33.0, 223.0), "default occupied span is [33, 223]")
    check(lo_s > 0.0 and hi_s < ARGS_DIM - 1, "span leaves margin at both edges")
    check(abs((lo_s + hi_s) / 2.0 - ARGS_DIM / 2.0) < 1e-12,
          "the span is centred on the grid centre")
    check(deepcad_sketch_normalize_scale(1.0, ARGS_DIM) == 95.0,
          "unit bbox scales to the 95px half-extent")
    check(deepcad_sketch_normalize_scale(2.0, ARGS_DIM) == 47.5,
          "scale is inversely proportional to bbox size")
    scaled = deepcad_sketch_normalize_scale(4.0) * 4.0
    check(abs(scaled - 95.0) < 1e-9, "any bbox normalizes to the same half-extent")

    # -- SkexGen: both endpoints exact, because the divisor is 2**b - 1.
    span = 2 ** SKEXGEN_BIT - 1
    check(span == 63, "6-bit vocabulary is 64 tokens (ids 0..63)")
    check(skexgen_quantize(-1.0) == 0, "min_range is token 0")
    check(skexgen_quantize(1.0) == span, "max_range is the top token")
    check(skexgen_dequantize(0) == -1.0, "token 0 dequantizes exactly to min_range")
    check(skexgen_dequantize(span) == 1.0,
          "top token dequantizes exactly to max_range -- unlike DeepCAD")
    check(span != SKEXGEN_BIT ** 2 - 1,
          "code uses 2**b-1 (63), not the docstring's b**2-1 (35)")

    # -- SkexGen truncates: one-sided error, up to a full bin.
    width = skexgen_bin_width()
    check(abs(width - 2.0 / 63.0) < 1e-12, "bin width is (max-min)/(2**b-1)")
    check(skexgen_max_roundtrip_error() == width, "worst case is a full bin")
    worst_sk = 0.0
    saw_over_half = False
    for i in range(steps + 1):
        x = -1.0 + 2.0 * i / steps
        q = skexgen_quantize(x)
        check(0 <= q <= span, "clipped quantizer stays in vocabulary")
        back = skexgen_dequantize(q)
        check(back <= x + 1e-12, "truncation never overshoots the input")
        err = abs(back - x)
        worst_sk = max(worst_sk, err)
        if err > width / 2.0 + 1e-12:
            saw_over_half = True
    check(worst_sk <= width + 1e-12, f"SkexGen error {worst_sk} within one bin")
    check(saw_over_half,
          "some inputs exceed half a bin -- proof this floors rather than rounds")

    # -- Bias direction is what distinguishes the two codecs.
    check(skexgen_dequantize(skexgen_quantize(0.5)) <= 0.5,
          "SkexGen is biased toward min_range")
    check(skexgen_quantize(0.0) == 31 and skexgen_dequantize(31) < 0.0,
          "x=0 has no exact token at even vocabulary size; it floors below zero")

    # -- SkexGen clamps without raising, like DeepCAD.
    for bad in (5.0, -5.0):
        q = skexgen_quantize(bad)
        check(0 <= q <= span, f"SkexGen clamps {bad} into vocabulary")

    # -- The geom_utils variant does NOT clamp: out-of-vocabulary tokens escape.
    g_lo, g_hi = SKEXGEN_GEOM_UTILS_RANGE
    g_span = 2 ** 8 - 1
    check(skexgen_quantize_unclipped(g_lo) == 0, "in-range low end is token 0")
    check(skexgen_quantize_unclipped(g_hi) == g_span, "in-range high end is top token")
    check(skexgen_quantize_unclipped(0.6) > g_span,
          "0.6 exceeds the [-0.5,0.5] range and yields an out-of-vocabulary token")
    check(skexgen_quantize_unclipped(-0.6) < 0,
          "below-range input yields a negative token id")
    check(skexgen_quantize(0.6, 8, g_lo, g_hi) == g_span,
          "the clipped codec saturates on the same input the unclipped one leaks")

    # -- Ranges table is well-formed and matches the source's asymmetric SCALE_R.
    for key, (lo_r, hi_r) in SKEXGEN_RANGES.items():
        check(hi_r > lo_r, f"range {key} is non-empty")
        check(skexgen_quantize(lo_r, SKEXGEN_BIT, lo_r, hi_r) == 0,
              f"range {key} low endpoint is token 0")
        check(skexgen_dequantize(span, SKEXGEN_BIT, lo_r, hi_r) == hi_r,
              f"range {key} top token dequantizes exactly to its high endpoint")

    # -- Truncation costs SCALE_R its top token (note 5): a real upstream quirk.
    for key in ("sketch", "radius", "extrude", "offset"):
        lo_r, hi_r = SKEXGEN_RANGES[key]
        check(skexgen_top_token_reachable(SKEXGEN_BIT, lo_r, hi_r),
              f"binary-exact range {key} reaches its top token")
    s_lo, s_hi = SKEXGEN_RANGES["scale"]
    check(not skexgen_top_token_reachable(SKEXGEN_BIT, s_lo, s_hi),
          "SCALE_R=1.4 does NOT reach its top token -- fp truncation drops a bin")
    check(skexgen_quantize(s_hi, SKEXGEN_BIT, s_lo, s_hi) == span - 1,
          "the largest scale quantizes to token 62, leaving 63 unreachable")
    check(round((s_hi - s_lo) * span / (s_hi - s_lo)) == span,
          "a rounding codec would have reached token 63; only truncation loses it")

    check(SKEXGEN_RANGES["scale"][0] == 0.0,
          "SCALE_R is one-sided [0, 1.4]: a scale is never negative")
    check(SKEXGEN_RANGES["offset"] == (-0.9, 0.9),
          "OFFSET_R is 0.9, deliberately tighter than SKETCH_R")

    # -- DeepCAD is 4x finer than SkexGen's shipped setting.
    check(ARGS_DIM // (2 ** SKEXGEN_BIT) == 4,
          "DeepCAD's 256 bins are 4x SkexGen's 64")
    check(deepcad_max_roundtrip_error() < skexgen_max_roundtrip_error(),
          "DeepCAD resolves finer than SkexGen at their shipped settings")

    if failures:
        for f in failures:
            print(f"selfcheck FAIL: {f}")
        return 1
    print("quantization_ranges selfcheck: OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="DeepCAD / SkexGen parameter quantization ranges and codecs")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
