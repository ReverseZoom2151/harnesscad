"""Relative / proportional dimension resolver ("min + 2mm", "center", "50%").

CodeToCAD's ``Axis`` enum can be *concatenated* into strings such as ``"min + 2mm"``
and its landmark machinery checks ``Axis.is_axis_name_in_string`` to decide whether
a dimension is measured from the entity's bounding box rather than from the world
origin.  The upstream code only detects the keyword; the actual resolution is left
to each backend.

This module implements the resolver itself, deterministically:

* :func:`resolve_axis_value` -- evaluate an expression against one
  :class:`~geometry.codetocad_cardinal_landmark.BoundaryAxis`:

  ==========================  =========================================
  ``"min"`` / ``"max"``       the axis bound
  ``"center"``                the axis midpoint
  ``"min + 2mm"``             bound plus a unit expression
  ``"max - 10%"``             bound minus a *fraction of the axis length*
  ``"50%"``                   proportional: 50% of the way from min to max
  ``"5mm"`` / ``0.005``       absolute world coordinate (metres)
  ==========================  =========================================

* :func:`resolve_point` -- the same, applied per axis against a bounding box.
* :func:`resolve_relative_size` -- a *size* expression resolved against a base
  length ("50%" of the parent's width, "parent - 2mm" style offsets).
* :func:`is_relative` -- the ``is_axis_name_in_string`` check, generalised to also
  recognise bare percentages.

All numbers are metres; percentages are always relative to the supplied base.
"""

from __future__ import annotations

import re

from geometry.codetocad_cardinal_landmark import BoundaryAxis, BoundaryBox
from numeric.codetocad_length_expression import (
    LENGTH,
    PERCENT,
    SCALAR,
    ExpressionError,
    parse_length,
    parse_quantity,
)

__all__ = [
    "AXIS_KEYWORDS",
    "is_relative",
    "resolve_axis_value",
    "resolve_point",
    "resolve_relative_size",
    "AxisExpressionError",
]

AXIS_KEYWORDS = ("min", "max", "center")

_KEYWORD_RE = re.compile(
    r"^\s*(?P<keyword>min|max|center)\s*(?P<rest>.*)$", re.IGNORECASE
)


class AxisExpressionError(ValueError):
    """Raised for a malformed axis expression."""


def is_relative(expr) -> bool:
    """True when ``expr`` is measured relative to an entity (keyword or percent)."""
    if not isinstance(expr, str):
        return False
    lowered = expr.lower()
    if any(keyword in lowered for keyword in AXIS_KEYWORDS):
        return True
    return "%" in lowered


def _keyword_value(axis: BoundaryAxis, keyword: str) -> float:
    return axis.select(keyword.lower())


def resolve_axis_value(axis: BoundaryAxis, expr) -> float:
    """Resolve ``expr`` to a world coordinate on ``axis`` (metres)."""
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return float(expr)
    if not isinstance(expr, str):
        raise AxisExpressionError("axis expression must be a string or number")

    text = expr.strip()
    if not text:
        raise AxisExpressionError("empty axis expression")

    match = _KEYWORD_RE.match(text)
    if match:
        base = _keyword_value(axis, match.group("keyword"))
        rest = match.group("rest").strip()
        if not rest:
            return base
        if rest[0] not in "+-":
            raise AxisExpressionError(
                "axis keyword must be followed by '+' or '-': " + expr
            )
        try:
            offset = parse_length(rest, base=axis.length)
        except ExpressionError as error:
            raise AxisExpressionError(str(error)) from error
        return base + offset

    # No keyword: either a bare percentage (proportional along the axis) or an
    # absolute coordinate.
    try:
        quantity = parse_quantity(text)
    except ExpressionError as error:
        raise AxisExpressionError(str(error)) from error
    if quantity.kind == PERCENT:
        return axis.min + quantity.value * axis.length
    if quantity.kind in (LENGTH, SCALAR):
        return quantity.value
    raise AxisExpressionError("expected a length, got " + quantity.kind)


def resolve_point(box: BoundaryBox, x=0.0, y=0.0, z=0.0) -> tuple[float, float, float]:
    """Resolve three axis expressions against ``box`` into a world point."""
    return (
        resolve_axis_value(box.x, x),
        resolve_axis_value(box.y, y),
        resolve_axis_value(box.z, z),
    )


def resolve_relative_size(expr, base: float | None = None) -> float:
    """Resolve a *size* expression, where percentages scale ``base`` (metres).

    >>> resolve_relative_size("50%", base=0.08)
    0.04
    >>> resolve_relative_size("2mm")
    0.002
    """
    try:
        value = parse_length(expr, base=base)
    except ExpressionError as error:
        raise AxisExpressionError(str(error)) from error
    return value
