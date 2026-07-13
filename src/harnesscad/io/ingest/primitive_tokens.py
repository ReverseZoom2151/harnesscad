"""DAVINCI 8-token primitive parametrization (single-stage sketch inference).

DAVINCI (Karadeniz et al., 2024, Sec. 3 / Table 1) represents *every* CAD sketch
primitive with a **fixed-length sequence of 8 tokens** ``t1..t8`` regardless of
type, which is what lets a set-based transformer emit one uniform block per
primitive slot:

* ``t1`` -- the primitive **type** token in ``[1..5]``:
  ``arc, circle, line, point, none`` (``none`` == empty/no-primitive slot);
* ``t2..t7`` -- six **quantised parameter** tokens in ``[1..64]``. Each primitive
  type only uses a prefix of these six slots (Table 1):
  ``arc = (xs,ys,xm,ym,xe,ye)`` (6), ``circle = (xc,yc,r)`` (3),
  ``line = (xs,ys,xe,ye)`` (4), ``point = (xp,yp)`` (2); unused slots are padded;
* ``t8`` -- the **is-construction** flag in ``[0..1]``.

The *learned* MLP heads that emit these tokens are out of scope (research-heavy);
what is fully deterministic -- and implemented here -- is the token *schema* and the
quantise/dequantise codec that maps normalised primitive coordinates to and from
this 8-token layout. Nothing in the repo encodes this fixed 8-token block:
``ingest.cadvlm_codec`` uses variable per-type point counts and no construction
flag or ``none`` type. This module is a standalone, reversible codec.
"""

from __future__ import annotations

from dataclasses import dataclass

# t1: primitive type token in [1..5].
TYPE_TOKENS = {"arc": 1, "circle": 2, "line": 3, "point": 4, "none": 5}
TOKEN_TYPES = {v: k for k, v in TYPE_TOKENS.items()}

# Number of continuous parameters each type consumes from the 6 param slots.
PARAM_COUNT = {"arc": 6, "circle": 3, "line": 4, "point": 2, "none": 0}

N_TOKENS = 8            # t1 (type) + t2..t7 (params) + t8 (construction)
N_PARAM_SLOTS = 6       # t2..t7
PARAM_LOW, PARAM_HIGH = 1, 64
PAD_TOKEN = 1           # filler for unused parameter slots


def quantize(value: float, bins: int = PARAM_HIGH) -> int:
    """Map a normalised coordinate in ``[0, 1]`` to a token in ``[1..bins]``."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"coordinate must be numeric, got {type(value).__name__}")
    idx = int(value * bins) + 1
    if idx < PARAM_LOW:
        idx = PARAM_LOW
    elif idx > bins:
        idx = bins
    return idx


def dequantize(token: int, bins: int = PARAM_HIGH) -> float:
    """Inverse of :func:`quantize`; returns the bin-centre coordinate in ``[0, 1]``."""
    if not (isinstance(token, int) and PARAM_LOW <= token <= bins):
        raise ValueError(f"parameter token out of range: {token}")
    return (token - 0.5) / bins


@dataclass(frozen=True)
class Primitive:
    """A decoded DAVINCI primitive."""

    type: str                # one of TYPE_TOKENS keys
    params: tuple            # normalised coordinates, len == PARAM_COUNT[type]
    construction: int = 0    # is-construction flag (0/1)


def encode_primitive(ptype: str, params=(), construction: int = 0) -> tuple:
    """Encode a primitive into its 8-token block.

    ``params`` are normalised coordinates in ``[0, 1]`` (length must equal
    ``PARAM_COUNT[ptype]``). Unused parameter slots are padded with ``PAD_TOKEN``.
    """
    if ptype not in TYPE_TOKENS:
        raise KeyError(f"unknown primitive type: {ptype!r}")
    coords = tuple(params)
    if len(coords) != PARAM_COUNT[ptype]:
        raise ValueError(
            f"{ptype} expects {PARAM_COUNT[ptype]} params, got {len(coords)}")
    if construction not in (0, 1):
        raise ValueError(f"construction flag must be 0/1, got {construction}")
    param_tokens = [quantize(c) for c in coords]
    param_tokens += [PAD_TOKEN] * (N_PARAM_SLOTS - len(param_tokens))
    return (TYPE_TOKENS[ptype],) + tuple(param_tokens) + (construction,)


def decode_primitive(tokens) -> Primitive:
    """Inverse of :func:`encode_primitive`; recovers a :class:`Primitive`."""
    tok = tuple(tokens)
    if len(tok) != N_TOKENS:
        raise ValueError(f"expected {N_TOKENS} tokens, got {len(tok)}")
    type_token = tok[0]
    if type_token not in TOKEN_TYPES:
        raise ValueError(f"unknown type token: {type_token}")
    ptype = TOKEN_TYPES[type_token]
    if tok[7] not in (0, 1):
        raise ValueError(f"construction flag must be 0/1, got {tok[7]}")
    used = PARAM_COUNT[ptype]
    params = tuple(dequantize(t) for t in tok[1:1 + used])
    return Primitive(type=ptype, params=params, construction=tok[7])


def token_issues(tokens) -> tuple:
    """Return every schema violation in an 8-token block (empty tuple == valid)."""
    tok = tuple(tokens)
    problems = []
    if len(tok) != N_TOKENS:
        return (f"bad-token-count:{len(tok)}!={N_TOKENS}",)
    if tok[0] not in TOKEN_TYPES:
        problems.append(f"bad-type-token:{tok[0]}")
    for i, t in enumerate(tok[1:7], start=2):
        if not (isinstance(t, int) and PARAM_LOW <= t <= PARAM_HIGH):
            problems.append(f"t{i}:param-out-of-range:{t}")
    if tok[7] not in (0, 1):
        problems.append(f"t8:construction-not-binary:{tok[7]}")
    return tuple(problems)


def is_empty_slot(tokens) -> bool:
    """True if the block is a ``none`` (empty) primitive slot."""
    return tuple(tokens)[0] == TYPE_TOKENS["none"]
