"""Precision Token Loss -- gradient re-weighting for numeric tokens (CAD-RL, 2026).

GRPO computes its loss at the *sample* level by first averaging token-level
losses within each sequence and then aggregating across samples, so every token
contributes equally. In CAD code, however, different tokens play markedly
different roles: an error in a numeric *dimension* corrupts the resulting solid
far more than an error in a keyword or delimiter.

CAD-RL's **Precision Token Loss** (Eq. 8) amplifies the gradient on
semantically important tokens via per-token importance weights ``omega_t`` and
renormalises so that per-sample magnitude stays comparable::

    L_precision = (1/|B|) * sum_i  (1/Z_i) * sum_t  omega_t * l(s_t^i)

where ``l(s_t)`` is the token-level negative log-likelihood, ``Z_i = sum_t
omega_t`` is a per-sample normaliser, and ``omega_t > 1`` for tokens
corresponding to numerical parameters or geometry-affecting operations,
``omega_t = 1`` otherwise.

This module implements the weighted, per-sample-normalised loss and a
deterministic *token-type heuristic* that assigns ``omega_t`` (numbers and a
configurable set of geometry-affecting CADQuery ops get the heavy weight). The
per-token NLL values are injected (they come from the model); this module only
defines the deterministic weighting and aggregation. Pure stdlib.
"""

from __future__ import annotations

import re
from typing import Sequence

DEFAULT_HEAVY_WEIGHT = 2.0
DEFAULT_LIGHT_WEIGHT = 1.0

# CADQuery operations whose (mis)placement directly perturbs geometry; these
# are treated as numerically sensitive along with literal numbers.
GEOMETRY_AFFECTING_OPS = frozenset({
    "box", "sphere", "cylinder", "extrude", "revolve", "loft", "sweep",
    "chamfer", "fillet", "translate", "rotate", "moveTo", "lineTo", "center",
    "circle", "rect", "polygon", "hole", "cboreHole", "cskHole", "workplane",
    "polarLine", "radiusArc", "sagittaArc", "offset2D", "shell",
})

_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def is_numeric_token(token: str) -> bool:
    """True if ``token`` is a numeric literal (int / float / scientific)."""
    return bool(_NUMBER_RE.match(token.strip()))


def token_weight(token: str, heavy: float = DEFAULT_HEAVY_WEIGHT,
                 light: float = DEFAULT_LIGHT_WEIGHT) -> float:
    """Importance weight ``omega_t`` for a single token via the type heuristic.

    Numeric literals and geometry-affecting op names get ``heavy`` (>1); all
    other tokens get ``light`` (=1).
    """
    if heavy < light:
        raise ValueError("heavy weight must be >= light weight")
    tok = token.strip()
    if is_numeric_token(tok) or tok in GEOMETRY_AFFECTING_OPS:
        return heavy
    return light


def token_weights(tokens: Sequence[str], heavy: float = DEFAULT_HEAVY_WEIGHT,
                  light: float = DEFAULT_LIGHT_WEIGHT) -> list:
    """Vector of ``omega_t`` over a token sequence."""
    return [token_weight(t, heavy, light) for t in tokens]


def sample_precision_loss(nll: Sequence[float], weights: Sequence[float]) -> float:
    """Per-sample precision loss ``(1/Z) * sum_t omega_t * l(s_t)`` (inner sum).

    ``nll`` are the token-level negative log-likelihoods and ``weights`` the
    matching ``omega_t``. ``Z = sum_t omega_t`` normalises the magnitude.
    """
    losses = [float(x) for x in nll]
    ws = [float(w) for w in weights]
    if len(losses) != len(ws):
        raise ValueError("nll and weights must have equal length")
    if not losses:
        raise ValueError("cannot compute loss over an empty sequence")
    z = sum(ws)
    if z <= 0.0:
        raise ValueError("normaliser Z = sum(weights) must be positive")
    return sum(w * l for w, l in zip(ws, losses)) / z


def precision_token_loss(batch_nll: Sequence[Sequence[float]],
                         batch_weights: Sequence[Sequence[float]]) -> float:
    """Batch Precision Token Loss ``(1/|B|) sum_i sample_precision_loss_i`` (Eq. 8)."""
    nlls = list(batch_nll)
    ws = list(batch_weights)
    if len(nlls) != len(ws):
        raise ValueError("batch_nll and batch_weights must have equal length")
    if not nlls:
        raise ValueError("cannot compute loss over an empty batch")
    total = sum(sample_precision_loss(n, w) for n, w in zip(nlls, ws))
    return total / len(nlls)
