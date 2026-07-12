"""ShapeGraMM ``lod`` variable and LOD-driven rule selection.

From "ShapeGraMM: On the fly procedural generation of massive models for
real-time visualization" (Santos, Brazil, Raposo, 2023), Section 3.2.2.

ShapeGraMM adds a built-in ``lod`` variable to production rules. Its value is
computed by estimating the size of the current scope (oriented bounding box)
*projected on the screen*. Alternative production rules are then selected
according to the LOD of the scope, e.g. (paper Section 3.2.2):

    object [lod = 0]  -> I("object_full_mesh")
    object [lod = 1]  -> I("object_coarse_mesh")
    object [lod >= 1] -> I("object_coarsest_mesh")

Unlike classic discrete LOD which only swaps a mesh, ShapeGraMM lets the LOD
choose *which production rule to expand*, so a coarse LOD can prune whole
sub-trees of a BVH (Section 5.2: ``lod = 0`` generates all children,
``lod >= 2`` derives only the most representative rule).

This module computes the projected size, maps it to a discrete LOD level, and
selects the matching production rule from a set of ``[lod <op> value]``
conditions.

Deterministic: pure geometry, no randomness.
"""

import math

_EPS = 1e-12


def _box_world_size(box_min, box_max):
    """Bounding-sphere diameter of the box (world-space extent)."""
    dx = box_max[0] - box_min[0]
    dy = box_max[1] - box_min[1]
    dz = box_max[2] - box_min[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _box_center(box_min, box_max):
    return tuple((box_min[i] + box_max[i]) * 0.5 for i in range(3))


def projected_size(box_min, box_max, camera_pos, focal_length):
    """Estimate the screen-space size (in pixels) of a scope box.

    Uses a pinhole approximation: ``size = world_size * focal / distance``,
    where ``world_size`` is the box's bounding-sphere diameter, ``distance`` is
    the camera-to-center distance, and ``focal_length`` is in pixels. When the
    camera is inside the box the projected size is treated as effectively
    infinite (returns ``inf``) so it always takes the highest detail.
    """
    if focal_length <= 0:
        raise ValueError("focal_length must be positive")
    world = _box_world_size(box_min, box_max)
    cx, cy, cz = _box_center(box_min, box_max)
    dist = math.sqrt(
        (cx - camera_pos[0]) ** 2
        + (cy - camera_pos[1]) ** 2
        + (cz - camera_pos[2]) ** 2
    )
    if dist <= world * 0.5:
        return float("inf")  # camera within the bounding sphere
    return world * focal_length / dist


def lod_level(size, thresholds):
    """Map a projected ``size`` to a discrete LOD level.

    ``thresholds`` is a descending list of pixel sizes. LOD 0 (highest detail)
    applies when ``size`` is at or above the first threshold; each threshold
    crossed downward increases the LOD by one. With ``thresholds=[100, 20]``:

        size >= 100        -> lod 0
        20 <= size < 100   -> lod 1
        size < 20          -> lod 2
    """
    for i, t in enumerate(thresholds):
        if size >= t:
            return i
    return len(thresholds)


def _match(op, lhs, rhs):
    if op == "=":
        return lhs == rhs
    if op == ">=":
        return lhs >= rhs
    if op == "<=":
        return lhs <= rhs
    if op == ">":
        return lhs > rhs
    if op == "<":
        return lhs < rhs
    if op == "!=":
        return lhs != rhs
    raise ValueError("unknown comparison operator: %r" % op)


def select_lod_rule(conditions, lod):
    """Pick the first production whose ``[lod <op> value]`` condition holds.

    ``conditions`` is an ordered list of ``(op, value, payload)`` tuples, in
    grammar declaration order (ShapeGraMM evaluates alternatives top-down and
    takes the first that passes). Returns the matching ``payload`` or ``None``
    if no condition holds.
    """
    for op, value, payload in conditions:
        if _match(op, lod, value):
            return payload
    return None


def resolve_lod(box_min, box_max, camera_pos, focal_length, thresholds, conditions):
    """Full pipeline: project the box, derive the LOD, select the rule.

    Returns ``(lod, payload)``.
    """
    size = projected_size(box_min, box_max, camera_pos, focal_length)
    lod = lod_level(size, thresholds)
    return lod, select_lod_rule(conditions, lod)
