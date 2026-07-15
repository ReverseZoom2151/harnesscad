"""CAD edit-operation taxonomy and multimodal-request metrics (neuralCAD-Edit).

Mined from *neuralCAD-Edit: An Expert Benchmark for Multimodal-Instructed 3D CAD
Model Editing*. The dataset itself is human-collected, but two of its evaluation
structures are deterministic and portable:

*   the **edit-operation taxonomy** -- the vocabulary of B-Rep edit actions experts
    perform (``edit_sketch``, ``chamfer``, ``select``, ``mirror``, ``extrude`` ...),
    used to describe a ground-truth edit trajectory; and
*   the four **modality combinations** a request can use (text only; video+speech+
    interaction; +temporary drawing; +static drawing).

This module provides the taxonomy, an operation-sequence classifier, an
edit-operation-set F-score comparing a predicted edit trajectory to a reference
(a survivor-bias-free, feature-based edit metric that does not need geometry), and
the modality-combination enumeration. Deterministic and stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

__all__ = [
    "EDIT_OPERATIONS",
    "MODALITY_COMBINATIONS",
    "normalise_operation",
    "classify_sequence",
    "edit_operation_fscore",
    "modality_information_rate",
]

#: The canonical CAD edit-operation vocabulary (aliases fold into these).
EDIT_OPERATIONS: Tuple[str, ...] = (
    "select", "edit_sketch", "extrude", "cut", "revolve", "sweep", "loft",
    "fillet", "chamfer", "shell", "draft", "mirror", "pattern", "move",
    "scale", "rotate", "delete", "add_body", "boolean",
)

_ALIASES: Dict[str, str] = {
    "sketch": "edit_sketch",
    "editsketch": "edit_sketch",
    "extrusion": "extrude",
    "subtract": "boolean",
    "union": "boolean",
    "intersect": "boolean",
    "round": "fillet",
    "bevel": "chamfer",
    "translate": "move",
    "copy": "pattern",
    "reflect": "mirror",
    "remove": "delete",
}

#: The four request modality combinations (paper Fig. 1).
MODALITY_COMBINATIONS: Tuple[Tuple[str, ...], ...] = (
    ("text",),
    ("video", "speech", "interaction"),
    ("video", "speech", "interaction", "drawing_temporary"),
    ("video", "speech", "interaction", "drawing_static"),
)


def normalise_operation(op: str) -> str:
    """Fold an operation name (with aliases/casing) into the canonical vocabulary."""
    key = op.strip().lower().replace(" ", "_").replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in EDIT_OPERATIONS:
        raise ValueError(f"unknown edit operation {op!r}")
    return key


def classify_sequence(ops: Sequence[str]) -> List[str]:
    """Normalise a raw edit trajectory into canonical operation names, in order."""
    return [normalise_operation(o) for o in ops]


def edit_operation_fscore(
    predicted: Sequence[str], reference: Sequence[str]
) -> Dict[str, float]:
    """Set-level precision/recall/F1 over canonical edit operations.

    Compares which *operation types* appear, ignoring order and multiplicity -- a
    feature-based edit metric computable on any trajectory, including ones whose
    geometry never executed (no survivor bias). Two empty trajectories score 1.0.
    """
    pset: Set[str] = set(classify_sequence(predicted))
    rset: Set[str] = set(classify_sequence(reference))
    if not pset and not rset:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pset & rset)
    precision = tp / len(pset) if pset else 0.0
    recall = tp / len(rset) if rset else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def modality_information_rate(combination: Sequence[str]) -> float:
    """A simple #modalities-per-request score (richer requests carry more).

    The paper finds multimodal requests "convey more information in less time";
    this returns the modality count, a deterministic proxy for request richness.
    Raises if the combination is not one of :data:`MODALITY_COMBINATIONS`.
    """
    combo = tuple(combination)
    if combo not in MODALITY_COMBINATIONS:
        raise ValueError(f"unknown modality combination {combo!r}")
    return float(len(combo))
