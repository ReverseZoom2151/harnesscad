"""Composite ("superposition") CAD-sketch primitive encoding (SketchDNN Sec. 2).

To resolve the *heterogeneity* of CAD primitive parameterisations, SketchDNN
represents every primitive as a single fixed-width vector that encodes ALL
primitive types at once -- a "superposition" that lets a diffusion model treat a
primitive as a probabilistic mixture over types. Each primitive is:

    (b, c, p_LINE, p_CIRCLE, p_ARC, p_POINT)

* ``b``  -- one-hot construction-aid flag (regular / construction).
* ``c``  -- one-hot class label over ``{LINE, CIRCLE, ARC, POINT, NONE}``.
* the four parameter blocks, one per primitive type, with the type-specific
  parameterisations from the paper:

      LINE   : (x1, y1, x2, y2)        start / end coords          (4)
      CIRCLE : (x, y, r)               centre coords, radius        (3)
      ARC    : (x1, y1, x2, y2, kappa) start/end coords, curvature  (5)
      POINT  : (x, y)                  xy-coordinates               (2)

The standard (decoded) primitive is recovered by taking the highest-confidence
class ``argmax(c)`` and reading its parameter block.

This module also implements two deterministic training/inference details:

* **Parameter masking** (Sec. 5.2): before the MSE loss the parameters of the
  irrelevant types are masked using the *ground-truth* class, so the network is
  supervised to predict every type's parameters but only scored on the true one.

* **Type-probability weighting** (Sec. 5.2): at inference the continuous
  parameters are weighted by *rescaled* class probabilities -- each probability
  vector is divided by its maximum element -- so relevant parameters do not
  decay to zero through the reverse process while irrelevant ones are suppressed.

Everything is stdlib-only and deterministic.
"""

from __future__ import annotations

from typing import Sequence

CLASS_NAMES = ["LINE", "CIRCLE", "ARC", "POINT", "NONE"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Parameter block widths per primitive type (order matters for the layout).
PARAM_TYPES = ["LINE", "CIRCLE", "ARC", "POINT"]
PARAM_DIMS = {"LINE": 4, "CIRCLE": 3, "ARC": 5, "POINT": 2}

CONSTRUCTION_DIM = 2          # one-hot: [regular, construction]
CLASS_DIM = len(CLASS_NAMES)  # 5, includes NONE

# Offsets into the flat feature vector.
_B_OFF = 0
_C_OFF = _B_OFF + CONSTRUCTION_DIM
_PARAM_OFF = {}
_acc = _C_OFF + CLASS_DIM
for _t in PARAM_TYPES:
    _PARAM_OFF[_t] = _acc
    _acc += PARAM_DIMS[_t]
FEATURE_DIM = _acc  # 2 + 5 + (4+3+5+2) = 21


def _one_hot(index: int, size: int) -> list[float]:
    v = [0.0] * size
    v[index] = 1.0
    return v


def encode_primitive(
    class_name: str,
    params: Sequence[float],
    construction: bool = False,
) -> list[float]:
    """Encode a typed primitive into the composite superposition vector.

    Only the active type's parameter block is filled; the others stay zero.
    """
    if class_name not in PARAM_TYPES:
        raise ValueError(f"unknown primitive class {class_name!r}")
    expected = PARAM_DIMS[class_name]
    if len(params) != expected:
        raise ValueError(
            f"{class_name} expects {expected} params, got {len(params)}"
        )
    vec = [0.0] * FEATURE_DIM
    # construction flag
    b = _one_hot(1 if construction else 0, CONSTRUCTION_DIM)
    vec[_B_OFF : _B_OFF + CONSTRUCTION_DIM] = b
    # class label
    vec[_C_OFF : _C_OFF + CLASS_DIM] = _one_hot(CLASS_INDEX[class_name], CLASS_DIM)
    # parameters
    off = _PARAM_OFF[class_name]
    vec[off : off + expected] = [float(p) for p in params]
    return vec


def class_logits(vec: Sequence[float]) -> list[float]:
    """Extract the class-confidence block ``c`` from a feature vector."""
    if len(vec) != FEATURE_DIM:
        raise ValueError(f"expected feature dim {FEATURE_DIM}, got {len(vec)}")
    return list(vec[_C_OFF : _C_OFF + CLASS_DIM])


def construction_logits(vec: Sequence[float]) -> list[float]:
    """Extract the construction-flag block ``b`` from a feature vector."""
    return list(vec[_B_OFF : _B_OFF + CONSTRUCTION_DIM])


def param_block(vec: Sequence[float], class_name: str) -> list[float]:
    """Read the parameter block for ``class_name`` from a feature vector."""
    if class_name not in PARAM_TYPES:
        raise ValueError(f"unknown primitive class {class_name!r}")
    off = _PARAM_OFF[class_name]
    return list(vec[off : off + PARAM_DIMS[class_name]])


def _argmax(values: Sequence[float]) -> int:
    best_i, best_v = 0, values[0]
    for i, v in enumerate(values):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def decode_primitive(vec: Sequence[float]) -> tuple[bool, str, list[float]]:
    """Recover ``(construction, class_name, params)`` via argmax of ``c``.

    If the winning class is ``NONE`` the primitive is treated as empty and no
    parameters are returned.
    """
    if len(vec) != FEATURE_DIM:
        raise ValueError(f"expected feature dim {FEATURE_DIM}, got {len(vec)}")
    construction = bool(_argmax(construction_logits(vec)) == 1)
    cls_idx = _argmax(class_logits(vec))
    cls = CLASS_NAMES[cls_idx]
    if cls == "NONE":
        return construction, cls, []
    return construction, cls, param_block(vec, cls)


def mask_irrelevant_params(
    vec: Sequence[float], true_class: str
) -> list[float]:
    """Zero out parameter blocks of every type except ``true_class`` (Sec. 5.2).

    Used before the MSE loss so the model is scored only on the ground-truth
    primitive type's parameters. ``b`` and ``c`` blocks are left untouched.
    """
    if true_class not in PARAM_TYPES:
        raise ValueError(f"unknown primitive class {true_class!r}")
    out = list(vec)
    for t in PARAM_TYPES:
        if t == true_class:
            continue
        off = _PARAM_OFF[t]
        for i in range(off, off + PARAM_DIMS[t]):
            out[i] = 0.0
    return out


def rescale_probs(probs: Sequence[float]) -> list[float]:
    """Divide a probability vector by its maximum element (Sec. 5.2 rescaling).

    The winning class becomes 1.0 and the rest are scaled proportionally,
    preventing the relevant type's parameters from decaying to zero during the
    reverse process.
    """
    m = max(probs)
    if m <= 0.0:
        raise ValueError("probabilities must have a positive maximum")
    return [p / m for p in probs]


def weight_params_by_type(
    vec: Sequence[float], class_probs: Sequence[float]
) -> list[float]:
    """Weight each type's parameter block by its rescaled class probability.

    ``class_probs`` is a distribution over ``PARAM_TYPES`` (length 4, in
    ``PARAM_TYPES`` order). Each block is multiplied by ``p_type / max(p)`` so
    the most-likely type keeps its parameters intact while unlikely types are
    attenuated (Sec. 5.2 confidence weighting).
    """
    if len(class_probs) != len(PARAM_TYPES):
        raise ValueError(f"class_probs must have length {len(PARAM_TYPES)}")
    weights = rescale_probs(class_probs)
    out = list(vec)
    for t, w in zip(PARAM_TYPES, weights):
        off = _PARAM_OFF[t]
        for i in range(off, off + PARAM_DIMS[t]):
            out[i] *= w
    return out


def reflect_arc(vec: Sequence[float]) -> list[float]:
    """Reflect an ARC across its terminal chord by negating curvature ``kappa``.

    The paper notes negating ``kappa`` reflects the arc across the line through
    its start/end points. Only the ARC block's last element is affected.
    """
    out = list(vec)
    kappa_idx = _PARAM_OFF["ARC"] + PARAM_DIMS["ARC"] - 1
    out[kappa_idx] = -out[kappa_idx]
    return out
