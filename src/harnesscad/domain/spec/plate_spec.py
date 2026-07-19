"""Plate-stack building DSL schema and validator.

A text-to-architecture backend describes a building as an ordered list of
*plates* -- flat 2D profiles that are extruded by a thickness and stacked
vertically to form a tower.  Each plate dictionary carries a ``category`` that
selects how its 2D outline is defined:

  * ``vertex``     -- an explicit polygon given as ``(x, y)`` vertices;
  * ``parametric`` -- a curve sampled from string formulas ``x(t)``/``y(t)`` over
                      a parameter ``range`` in ``steps`` samples;
  * ``mixed``      -- a closed Bezier outline given by control ``vertices`` and
                      optional per-point ``handle_types``.

Such a backend typically feeds these dictionaries straight into a mesh-building
runtime with no validation, so a malformed dictionary surfaces only as an opaque
runtime traceback that the LLM repair loop must guess at.  This module
implements the *schema* deterministically and up front: it
enumerates the required and optional keys per category, checks their types and
value ranges, and reports every problem in one pass with a precise field path --
turning "some Blender error" into an actionable list.

The validator is the deterministic, transferable core; the mesh building,
the LLM calls and the web-service plumbing are out of scope.

Public API
----------
``CATEGORIES``                              -- the three legal categories.
``HANDLE_TYPES``                            -- legal Bezier handle types.
``validate_plate(plate) -> list[str]``      -- per-plate issues (empty == valid).
``validate_building(plates) -> list[str]``  -- whole-list issues (names, order).
``normalize_plate(plate) -> dict``          -- fill defaults; raises on invalid.
``PlateSpecError``                          -- raised by ``normalize_plate``.
``is_valid_plate(plate) -> bool``
``is_valid_building(plates) -> bool``

Deterministic: pure structural checks, no clock, no randomness, no I/O.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

CATEGORIES = ("vertex", "parametric", "mixed")
HANDLE_TYPES = ("AUTO", "VECTOR", "FREE", "ALIGNED")

_MIN_POLYGON_VERTICES = 3
_MIN_PARAMETRIC_STEPS = 3


class PlateSpecError(ValueError):
    """Raised when a plate dictionary cannot be normalized."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_point(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and _is_number(value[0])
        and _is_number(value[1])
    )


def _check_vertices(plate: Dict[str, Any], key: str, issues: List[str]) -> None:
    verts = plate.get(key)
    if not isinstance(verts, (list, tuple)):
        issues.append("{0}: must be a list of (x, y) points".format(key))
        return
    if len(verts) < _MIN_POLYGON_VERTICES:
        issues.append(
            "{0}: need at least {1} points, got {2}".format(
                key, _MIN_POLYGON_VERTICES, len(verts)
            )
        )
    for i, pt in enumerate(verts):
        if not _is_point(pt):
            issues.append("{0}[{1}]: not a numeric (x, y) pair".format(key, i))


def _check_formula(plate: Dict[str, Any], issues: List[str]) -> None:
    formula = plate.get("formula")
    if not isinstance(formula, dict):
        issues.append("formula: must be a dict with 'x' and 'y' keys")
        return
    for axis in ("x", "y"):
        expr = formula.get(axis)
        if not isinstance(expr, str) or not expr.strip():
            issues.append("formula.{0}: must be a non-empty string".format(axis))


def _check_range(plate: Dict[str, Any], issues: List[str]) -> None:
    rng = plate.get("range")
    if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
        issues.append("range: must be a (start, end) pair")
        return
    if not (_is_number(rng[0]) and _is_number(rng[1])):
        issues.append("range: start and end must be numbers")
        return
    if rng[0] == rng[1]:
        issues.append("range: start and end must differ")


def _check_steps(plate: Dict[str, Any], issues: List[str]) -> None:
    steps = plate.get("steps")
    if not (isinstance(steps, int) and not isinstance(steps, bool)):
        issues.append("steps: must be an integer")
        return
    if steps < _MIN_PARAMETRIC_STEPS:
        issues.append(
            "steps: need at least {0}, got {1}".format(_MIN_PARAMETRIC_STEPS, steps)
        )


def _check_handle_types(plate: Dict[str, Any], issues: List[str]) -> None:
    if "handle_types" not in plate:
        return
    handles = plate.get("handle_types")
    verts = plate.get("vertices")
    if not isinstance(handles, (list, tuple)):
        issues.append("handle_types: must be a list of (left, right) pairs")
        return
    if isinstance(verts, (list, tuple)) and len(handles) != len(verts):
        issues.append(
            "handle_types: length {0} must match vertices length {1}".format(
                len(handles), len(verts)
            )
        )
    for i, pair in enumerate(handles):
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            issues.append("handle_types[{0}]: not a (left, right) pair".format(i))
            continue
        for side, ht in zip(("left", "right"), pair):
            if ht not in HANDLE_TYPES:
                issues.append(
                    "handle_types[{0}].{1}: '{2}' not in {3}".format(
                        i, side, ht, HANDLE_TYPES
                    )
                )


def _check_rotation(plate: Dict[str, Any], issues: List[str]) -> None:
    if "rotation" in plate and not _is_number(plate["rotation"]):
        issues.append("rotation: must be a number (degrees)")


def _check_position(plate: Dict[str, Any], issues: List[str]) -> None:
    if "position" not in plate:
        return
    pos = plate["position"]
    if not isinstance(pos, dict):
        issues.append("position: must be a dict with optional x/y/z")
        return
    for axis in ("x", "y", "z"):
        if axis in pos and not _is_number(pos[axis]):
            issues.append("position.{0}: must be a number".format(axis))


def validate_plate(plate: Any) -> List[str]:
    """Return a sorted-by-appearance list of problems with one plate dict."""
    issues: List[str] = []
    if not isinstance(plate, dict):
        return ["plate: must be a dict"]

    name = plate.get("name")
    if not isinstance(name, str) or not name.strip():
        issues.append("name: must be a non-empty string")

    thickness = plate.get("thickness")
    if not _is_number(thickness):
        issues.append("thickness: must be a number")
    elif thickness <= 0:
        issues.append("thickness: must be positive")

    category = plate.get("category")
    if category not in CATEGORIES:
        issues.append("category: must be one of {0}".format(CATEGORIES))
    elif category == "vertex":
        _check_vertices(plate, "vertices", issues)
    elif category == "parametric":
        _check_formula(plate, issues)
        _check_range(plate, issues)
        _check_steps(plate, issues)
    elif category == "mixed":
        _check_vertices(plate, "vertices", issues)
        _check_handle_types(plate, issues)

    _check_rotation(plate, issues)
    _check_position(plate, issues)
    return issues


def validate_building(plates: Any) -> List[str]:
    """Return problems with a whole building (a list/tuple of plate dicts)."""
    issues: List[str] = []
    if not isinstance(plates, (list, tuple)):
        return ["building: must be a list of plate dicts"]
    if len(plates) == 0:
        issues.append("building: must contain at least one plate")

    seen: Dict[str, int] = {}
    for idx, plate in enumerate(plates):
        for problem in validate_plate(plate):
            issues.append("plate[{0}] {1}".format(idx, problem))
        if isinstance(plate, dict):
            name = plate.get("name")
            if isinstance(name, str) and name.strip():
                if name in seen:
                    issues.append(
                        "plate[{0}]: duplicate name '{1}' (first at plate[{2}])".format(
                            idx, name, seen[name]
                        )
                    )
                else:
                    seen[name] = idx
    return issues


def normalize_plate(plate: Any) -> Dict[str, Any]:
    """Validate and return a copy with defaults filled (rotation, position)."""
    issues = validate_plate(plate)
    if issues:
        raise PlateSpecError("; ".join(issues))
    out = dict(plate)
    out.setdefault("rotation", 0.0)
    out.setdefault("position", {})
    pos = dict(out["position"])
    for axis in ("x", "y", "z"):
        pos.setdefault(axis, 0.0)
    out["position"] = pos
    return out


def is_valid_plate(plate: Any) -> bool:
    return not validate_plate(plate)


def is_valid_building(plates: Any) -> bool:
    return not validate_building(plates)
