"""CADReview error taxonomy — the eight CAD-program error scenarios.

CADReview ("Automatically Reviewing CAD Programs with Error Detection and
Correction", Chen et al.) defines a fixed set of *eight* error scenarios that a
CAD program (OpenSCAD in the paper) can exhibit relative to its reference
design, plus the ``No error`` case for a program that already matches. The
taxonomy is the backbone of the whole review task: a review must name the error
*type* AND the offending code block, and the accuracy metric ("Acc") is only
credited when both are right (see :mod:`cadreview_review`).

This module is the deterministic, self-contained registry of that taxonomy.
Unlike :class:`reliability.repair.RepairAdvisor` — which maps *B-rep kernel*
diagnostics (non-manifold, over-constrained, boolean-null) to CISP-op repairs —
this taxonomy is about *source-program* discrepancies against a reference
design: a cube that should be a cylinder, a rotation that drifted, a block that
went missing. The two are complementary: the kernel advisor heals a solid that
failed to build; this taxonomy classifies a solid that built fine but is *wrong*.

Each :class:`ErrorType` carries the paper's canonical label (as used in the
Table-10 review prompt), a stable snake-case id, a short description, the
geometric operation whose misuse produces it, and the corrective action a fixer
should take (consumed by :mod:`cadreview_correct`). Pure stdlib, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ErrorType:
    """One CADReview error scenario.

    ``id`` is the stable snake-case key; ``label`` is the paper's canonical name
    (matched case-insensitively by :func:`from_label`); ``operation`` names the
    CAD construct whose misuse causes it; ``fix_action`` is the abstract
    corrective move a fixer applies.
    """

    id: str
    label: str
    description: str
    operation: str
    fix_action: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "operation": self.operation,
            "fix_action": self.fix_action,
        }


# --------------------------------------------------------------------------- #
# The eight scenarios + the "no error" baseline (paper Fig. 3 / Table 10)
# --------------------------------------------------------------------------- #
NO_ERROR = ErrorType(
    "no_error", "No error",
    "The program is consistent with the reference design; nothing to correct.",
    "none", "none")

PRIMITIVE_ERROR = ErrorType(
    "primitive_error", "Primitive error",
    "An incorrect geometric primitive is used, e.g. a cube substituted by a "
    "cylinder or sphere, misrepresenting the intended shape.",
    "primitive", "replace_primitive")

ROTATION_ERROR = ErrorType(
    "rotation_error", "Rotation error",
    "A rotation transform has a wrong angle, or a rotation is wrongly added or "
    "missing, mis-orienting a component.",
    "rotate", "fix_rotation")

POSITION_ERROR = ErrorType(
    "position_error", "Position error",
    "A component deviates from its intended 3D coordinates: a wrong translation "
    "or a wrongly added / missing translation.",
    "translate", "fix_translation")

SIZE_ERROR = ErrorType(
    "size_error", "Size error",
    "The scale of a geometric component is wrong, e.g. a sphere radius or a cube "
    "side length, making the part look broken.",
    "primitive_arg", "fix_dimension")

CONSTANT_ERROR = ErrorType(
    "constant_error", "Constant error",
    "A global variable / initial macro or constant holds a wrong value or an "
    "invalid assignment, disrupting generation.",
    "assignment", "fix_constant")

LOGIC_ERROR = ErrorType(
    "logic_error", "Logic error",
    "A control-flow statement (an if condition or a for range) is wrong, causing "
    "an unintended execution path.",
    "control_flow", "fix_condition")

MISSING_BLOCK = ErrorType(
    "missing_block", "Missing block",
    "A code block was removed, yielding an incomplete object with a missing "
    "component.",
    "block", "insert_block")

REDUNDANT_BLOCK = ErrorType(
    "redundant_block", "Redundant block",
    "An unnecessary code block that does not contribute to the intended object.",
    "block", "remove_block")


#: The eight error scenarios, in the paper's presentation order (Fig. 3).
ERROR_TYPES: Tuple[ErrorType, ...] = (
    PRIMITIVE_ERROR,
    ROTATION_ERROR,
    POSITION_ERROR,
    SIZE_ERROR,
    CONSTANT_ERROR,
    LOGIC_ERROR,
    MISSING_BLOCK,
    REDUNDANT_BLOCK,
)

#: All types including the ``No error`` baseline.
ALL_TYPES: Tuple[ErrorType, ...] = (NO_ERROR,) + ERROR_TYPES

_BY_ID: Dict[str, ErrorType] = {t.id: t for t in ALL_TYPES}
_BY_LABEL: Dict[str, ErrorType] = {t.label.lower(): t for t in ALL_TYPES}


def by_id(type_id: str) -> Optional[ErrorType]:
    """The :class:`ErrorType` with this snake-case id, or None."""
    return _BY_ID.get(type_id)


def from_label(label: str) -> Optional[ErrorType]:
    """Resolve a free-text error label to an :class:`ErrorType` (case /
    whitespace insensitive), or None if it does not name a known scenario."""
    if not label:
        return None
    return _BY_LABEL.get(" ".join(str(label).split()).lower())


def labels() -> List[str]:
    """The canonical labels of the eight error scenarios (no baseline)."""
    return [t.label for t in ERROR_TYPES]


def ids() -> List[str]:
    """The stable snake-case ids of the eight error scenarios (no baseline)."""
    return [t.id for t in ERROR_TYPES]
