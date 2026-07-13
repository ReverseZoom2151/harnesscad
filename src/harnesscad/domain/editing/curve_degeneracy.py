"""Degenerate-curve resolution for mrCAD edits (``editing_actions.py`` semantics).

The harness's :func:`editing.mrcad_refinement.apply_action` implements the typed
edit vocabulary and the shared-control-point rule, but it applies ``move_point``
via :meth:`editing.mrcad_schema.Curve.replace_point`, which keeps a curve's
*kind* fixed.  The mrCAD reference (``mrcad/editing_actions.py`` ``MovePoint``)
does more: after moving a shared point it **resolves degenerate curves**:

* a **line** whose two endpoints coincide is **dropped**;
* an **arc** whose start and end coincide becomes a **circle** on the remaining
  two points (start == end -> a closed loop);
* an **arc** whose start/mid or mid/end coincide is **dropped** (it can no longer
  define a circular arc);
* a **circle** whose two diameter points coincide is **dropped**.

Without this resolution the harness would retain zero-length lines and malformed
arcs.  This module supplies that resolution as pure geometry:
:func:`resolve_curve` (curve -> canonical curve or ``None``),
:func:`move_point_resolved` (shared-point move + resolution), and
:func:`canonicalize_design` (drop/collapse every degenerate curve).

Point equality follows the reference: exact equality on the (float) control
points, matching :meth:`editing.mrcad_schema.Curve.replace_point`.

Pure stdlib, deterministic.
"""
from __future__ import annotations

from typing import Optional

from harnesscad.domain.editing.sketch_edit_schema import Curve, Design, Point


def resolve_curve(curve: Curve) -> Optional[Curve]:
    """Return the canonical form of ``curve``, collapsing/dropping degeneracies.

    Returns ``None`` when the curve has degenerated to nothing.
    """
    pts = curve.points
    if curve.kind == "line":
        if pts[0] == pts[1]:
            return None
        return curve
    if curve.kind == "circle":
        if pts[0] == pts[1]:
            return None
        return curve
    # arc: (start, mid, end)
    start, mid, end = pts
    if start == end:
        # Closed loop -> circle on the diameter (start, mid), matching the
        # reference which builds Circle(control_points=new_control_points[:2]).
        if start == mid:
            return None
        return Curve("circle", (start, mid))
    if start == mid or mid == end:
        return None
    return curve


def move_point_resolved(design: Design, old: Point, new: Point) -> Design:
    """Move every control point equal to ``old`` to ``new`` and resolve results.

    Shared-point semantics (Sec. 2.2): all curves touching ``old`` move; each is
    then passed through :func:`resolve_curve`, so collapsed curves are dropped and
    degenerate arcs become circles.  Curves not touching ``old`` are unchanged.
    """
    o = (float(old[0]), float(old[1]))
    n = (float(new[0]), float(new[1]))
    out = []
    for c in design.curves:
        if o in c.points:
            moved = Curve(c.kind, tuple(n if q == o else q for q in c.points))
            resolved = resolve_curve(moved)
            if resolved is not None:
                out.append(resolved)
        else:
            out.append(c)
    return Design(tuple(out))


def canonicalize_design(design: Design) -> Design:
    """Drop/collapse every degenerate curve in ``design``."""
    out = []
    for c in design.curves:
        resolved = resolve_curve(c)
        if resolved is not None:
            out.append(resolved)
    return Design(tuple(out))
