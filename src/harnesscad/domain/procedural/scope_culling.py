"""ShapeGraMM ``scope`` visibility variable and spatial culling.

From "ShapeGraMM: On the fly procedural generation of massive models for
real-time visualization" (Santos, Brazil, Raposo, 2023), Section 3.2.1.

ShapeGraMM adds a built-in ``scope`` variable to production rules that stores
the visibility of the current scope (an oriented bounding box) with respect to
the camera view. Instead of a boolean, standard frustum culling is extended to
an enumeration with four values (paper's Section 3.2.1):

    OUTSIDE     = -1   not visible to the frustum (or detail-culled)
    SURROUNDING =  0   the camera position is within the scope
    INTERSECTS  =  1   the scope intersects the frustum
    WITHIN      =  2   the scope is entirely within the frustum

A production rule condition such as ``[scope >= SURROUNDING]`` continues the
generation only when the scope is somehow visible; ``[scope = OUTSIDE]`` prunes
the whole sub-tree, which is how ShapeGraMM performs on-the-fly frustum and
occlusion culling during derivation.

Key optimization (Section 3.2): if a scope is ``WITHIN`` the frustum then all
its descendants are also within it, so the culling of descendants can be
skipped -- ``classify_hierarchy`` implements this propagation.

This module is DISTINCT from ``procedural.lazy_scene`` (which takes an opaque
``visible(node)`` predicate): here the four-valued classification is computed
from box geometry and camera frustum planes, and the WITHIN-propagation rule
is applied explicitly.

Deterministic: pure geometry, no randomness.
"""

OUTSIDE = -1
SURROUNDING = 0
INTERSECTS = 1
WITHIN = 2

_EPS = 1e-9


def make_aabb_frustum(lo, hi):
    """Return 6 planes describing an axis-aligned box view volume.

    A convenience frustum for testing/orthographic-like views. Each plane is a
    tuple ``(nx, ny, nz, d)`` with inward normal; a point ``p`` is on the inner
    side when ``nx*px + ny*py + nz*pz + d >= 0``.
    """
    lx, ly, lz = lo
    hx, hy, hz = hi
    return (
        (1.0, 0.0, 0.0, -lx),   # x >= lx
        (-1.0, 0.0, 0.0, hx),   # x <= hx
        (0.0, 1.0, 0.0, -ly),   # y >= ly
        (0.0, -1.0, 0.0, hy),   # y <= hy
        (0.0, 0.0, 1.0, -lz),   # z >= lz
        (0.0, 0.0, -1.0, hz),   # z <= hz
    )


def _point_in_box(p, box_min, box_max):
    return all(box_min[i] - _EPS <= p[i] <= box_max[i] + _EPS for i in range(3))


def _plane_dist(plane, x, y, z):
    return plane[0] * x + plane[1] * y + plane[2] * z + plane[3]


def classify_scope(box_min, box_max, planes, camera_pos=None):
    """Classify an axis-aligned scope box against frustum ``planes``.

    Uses the p-vertex / n-vertex AABB test: for each plane the positive vertex
    (farthest along the inward normal) and negative vertex are evaluated.

    * If the positive vertex is outside any plane -> the whole box is OUTSIDE.
    * Else if the camera position lies inside the box -> SURROUNDING.
    * Else if every negative vertex is inside all planes -> WITHIN.
    * Otherwise the box straddles a plane -> INTERSECTS.
    """
    fully_inside = True
    for plane in planes:
        nx, ny, nz, _ = plane
        # positive vertex: pick the box corner maximizing the plane normal dot
        px = box_max[0] if nx >= 0 else box_min[0]
        py = box_max[1] if ny >= 0 else box_min[1]
        pz = box_max[2] if nz >= 0 else box_min[2]
        if _plane_dist(plane, px, py, pz) < -_EPS:
            return OUTSIDE
        # negative vertex: opposite corner
        qx = box_min[0] if nx >= 0 else box_max[0]
        qy = box_min[1] if ny >= 0 else box_max[1]
        qz = box_min[2] if nz >= 0 else box_max[2]
        if _plane_dist(plane, qx, qy, qz) < -_EPS:
            fully_inside = False
    if camera_pos is not None and _point_in_box(camera_pos, box_min, box_max):
        return SURROUNDING
    return WITHIN if fully_inside else INTERSECTS


def is_visible(scope_value):
    """True when a scope value is somehow visible (``>= SURROUNDING``)."""
    return scope_value >= SURROUNDING


def classify_hierarchy(root, box_of, children_of, planes, camera_pos=None):
    """Classify a scope tree, propagating WITHIN to descendants.

    ``box_of(node)`` returns ``(box_min, box_max)``; ``children_of(node)``
    returns an iterable of child nodes. Descendant boxes are assumed contained
    in their parent's box (the ShapeGraMM invariant), so a WITHIN parent lets
    us mark every descendant WITHIN without re-running the frustum test.

    Returns ``(values, stats)`` where ``values`` maps node -> enum and
    ``stats`` reports ``computed`` (full frustum tests run), ``propagated``
    (WITHIN results inherited without a test), and ``culled`` (OUTSIDE nodes
    whose sub-tree was pruned).
    """
    values = {}
    stats = {"computed": 0, "propagated": 0, "culled": 0}
    # stack of (node, inherited_within)
    stack = [(root, False)]
    while stack:
        node, inherited = stack.pop()
        if inherited:
            values[node] = WITHIN
            stats["propagated"] += 1
            for child in children_of(node):
                stack.append((child, True))
            continue
        box_min, box_max = box_of(node)
        value = classify_scope(box_min, box_max, planes, camera_pos)
        values[node] = value
        stats["computed"] += 1
        if value == OUTSIDE:
            stats["culled"] += 1
            continue  # prune sub-tree: descendants are also outside
        propagate = value == WITHIN
        for child in children_of(node):
            stack.append((child, propagate))
    return values, stats
