"""Pointer-based CAD token vocabulary and ID scheme.

This encoding represents a CAD model as a command sequence in which every token
belongs to exactly one of three families:

  * **Label Token** -- carries semantic structure (start of sketch, extrusion,
    chamfer, fillet, ...), a direction, a boolean type, or an orientation flag;
  * **Value Token** -- a *quantized* continuous parameter (a normalised
    coordinate/length ``<nv>`` or an angle ``<ag>``);
  * **Pointer** -- a reference to a B-rep entity (edge or face). ``<pe>`` marks an
    *enabled* pointer (it references an entity); ``<pd>`` marks an *empty/disabled*
    pointer (the parameter is placed freely, not snapped).

To let a single classification head decode label and value tokens the scheme packs
them into *non-overlapping integer ranges*. This module implements that ID scheme
and the value-token quantisation (normalise to ``[0, 1]`` -- or an angle to
``[0, 360)`` -- then discretise into ``2**q`` bins, ``q = 8`` by default).
Everything here is pure stdlib and deterministic; the learned decoder is external.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- fixed (q-independent) token IDs -----------------------------------------
TOK_EM = 1   # end of model (final step)
TOK_ES = 2   # end of step (more steps follow)
TOK_SS = 3   # start of sketch
TOK_SE = 4   # start of extrusion
TOK_SC = 5   # start of chamfer
TOK_SF = 6   # start of fillet
TOK_SP = 7   # start of profile
TOK_SL = 8   # start of loop
TOK_SX = 9   # start of curve
TOK_PE = 10  # pointer: enabled (references an edge/face)
TOK_PD = 11  # pointer: empty / disabled

# Orientation {12, 13}: clockwise, counter-clockwise.
OR_BASE = 12
ORIENTATIONS: tuple[str, ...] = ("CW", "CCW")

# Direction [14, 19]: X+, X-, Y+, Y-, Z+, Z-.
DR_BASE = 14
DIRECTIONS: tuple[str, ...] = ("X+", "X-", "Y+", "Y-", "Z+", "Z-")

# Boolean [20, 23]: New, Join, Cut, Intersect.
BO_BASE = 20
BOOLEANS: tuple[str, ...] = ("New", "Join", "Cut", "Intersect")

# Value tokens start here; the range is [24, 24 + 2**q).
VALUE_BASE = 24

# Token families.
LABEL = "label"
VALUE = "value"
POINTER = "pointer"

# The single-integer label tokens (start-of-*, end-of-*) -> name.
_SINGLE_LABELS: dict[int, str] = {
    TOK_EM: "<em>", TOK_ES: "<es>", TOK_SS: "<ss>", TOK_SE: "<se>",
    TOK_SC: "<sc>", TOK_SF: "<sf>", TOK_SP: "<sp>", TOK_SL: "<sl>",
    TOK_SX: "<sx>",
}


class PointerTokenError(ValueError):
    """Raised for malformed token IDs or out-of-range quantiser input."""


def value_range(q: int = 8) -> tuple[int, int]:
    """Return ``(lo, hi)`` such that value-token IDs occupy ``[lo, hi)``."""
    if q < 1:
        raise PointerTokenError(f"q must be >= 1, got {q}")
    return VALUE_BASE, VALUE_BASE + (1 << q)


def vocab_size(q: int = 8) -> int:
    """Total number of distinct token IDs for a given quantisation width ``q``."""
    return VALUE_BASE + (1 << q)


def token_family(token_id: int, q: int = 8) -> str:
    """Classify ``token_id`` as ``LABEL``, ``VALUE`` or ``POINTER``."""
    if token_id in (TOK_PE, TOK_PD):
        return POINTER
    lo, hi = value_range(q)
    if lo <= token_id < hi:
        return VALUE
    if 1 <= token_id < TOK_PE or OR_BASE <= token_id < VALUE_BASE:
        return LABEL
    raise PointerTokenError(f"token id {token_id} is outside the vocabulary")


def is_pointer(token_id: int) -> bool:
    return token_id in (TOK_PE, TOK_PD)


def pointer_enabled(token_id: int) -> bool:
    """True for ``<pe>`` (enabled), False for ``<pd>`` (empty). Errors otherwise."""
    if token_id == TOK_PE:
        return True
    if token_id == TOK_PD:
        return False
    raise PointerTokenError(f"token id {token_id} is not a pointer token")


def orientation_id(name: str) -> int:
    if name not in ORIENTATIONS:
        raise PointerTokenError(f"unknown orientation {name!r}")
    return OR_BASE + ORIENTATIONS.index(name)


def orientation_name(token_id: int) -> str:
    idx = token_id - OR_BASE
    if not 0 <= idx < len(ORIENTATIONS):
        raise PointerTokenError(f"token id {token_id} is not an orientation")
    return ORIENTATIONS[idx]


def direction_id(name: str) -> int:
    if name not in DIRECTIONS:
        raise PointerTokenError(f"unknown direction {name!r}")
    return DR_BASE + DIRECTIONS.index(name)


def direction_name(token_id: int) -> str:
    idx = token_id - DR_BASE
    if not 0 <= idx < len(DIRECTIONS):
        raise PointerTokenError(f"token id {token_id} is not a direction")
    return DIRECTIONS[idx]


def boolean_id(name: str) -> int:
    if name not in BOOLEANS:
        raise PointerTokenError(f"unknown boolean {name!r}")
    return BO_BASE + BOOLEANS.index(name)


def boolean_name(token_id: int) -> str:
    idx = token_id - BO_BASE
    if not 0 <= idx < len(BOOLEANS):
        raise PointerTokenError(f"token id {token_id} is not a boolean")
    return BOOLEANS[idx]


def label_name(token_id: int) -> str:
    """Human-readable name for any Label Token."""
    if token_id in _SINGLE_LABELS:
        return _SINGLE_LABELS[token_id]
    if OR_BASE <= token_id < DR_BASE:
        return f"<or:{orientation_name(token_id)}>"
    if DR_BASE <= token_id < BO_BASE:
        return f"<dr:{direction_name(token_id)}>"
    if BO_BASE <= token_id < VALUE_BASE:
        return f"<bo:{boolean_name(token_id)}>"
    raise PointerTokenError(f"token id {token_id} is not a label token")


# --- value-token quantisation ------------------------------------------------
def quantize_nv(value: float, q: int = 8) -> int:
    """Quantise a value normalised to ``[0, 1]`` into a ``<nv>`` value-token ID.

    The continuous value is clamped to ``[0, 1]``, mapped to one of ``2**q``
    uniform bins, then offset by ``VALUE_BASE``.
    """
    if q < 1:
        raise PointerTokenError(f"q must be >= 1, got {q}")
    levels = 1 << q
    v = 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
    # Round to nearest bin; the top bin maps exactly to levels-1.
    bin_index = int(round(v * (levels - 1)))
    if bin_index >= levels:
        bin_index = levels - 1
    return VALUE_BASE + bin_index


def dequantize_nv(token_id: int, q: int = 8) -> float:
    """Inverse of :func:`quantize_nv`: value-token ID -> value in ``[0, 1]``."""
    lo, hi = value_range(q)
    if not lo <= token_id < hi:
        raise PointerTokenError(f"token id {token_id} is not an <nv>/<ag> value")
    levels = 1 << q
    return (token_id - VALUE_BASE) / (levels - 1)


def quantize_ag(angle_deg: float, q: int = 8) -> int:
    """Quantise an angle in degrees (wrapped to ``[0, 360)``) into an ``<ag>`` ID."""
    wrapped = angle_deg % 360.0
    return quantize_nv(wrapped / 360.0, q)


def dequantize_ag(token_id: int, q: int = 8) -> float:
    """Inverse of :func:`quantize_ag`: value-token ID -> angle in ``[0, 360)``."""
    return dequantize_nv(token_id, q) * 360.0


@dataclass(frozen=True)
class QuantizationReport:
    """Round-trip error of a value under a given quantiser width."""
    q: int
    levels: int
    max_abs_error: float  # worst-case reconstruction error for a value in [0, 1]


def quantization_report(q: int = 8) -> QuantizationReport:
    """Worst-case reconstruction error for uniform ``2**q``-level quantisation.

    The pointer mechanism is motivated partly by the *quantisation error* that
    discretising continuous parameters introduces. With ``2**q`` uniform bins over
    ``[0, 1]`` the worst-case round-trip error is half a bin,
    ``1 / (2 * (2**q - 1))``.
    """
    levels = 1 << q
    max_err = 1.0 / (2.0 * (levels - 1)) if levels > 1 else 1.0
    return QuantizationReport(q=q, levels=levels, max_abs_error=max_err)
