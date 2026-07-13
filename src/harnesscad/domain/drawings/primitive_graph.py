"""Vectorised-CAD primitive graph construction (CADTransformer, CVPR 2022).

CADTransformer turns a vectorised floor-plan (line / circle / ellipse
primitives) into a graph whose nodes are primitives and whose edges connect
each primitive to its ``k`` nearest neighbours.  The transferable, fully
deterministic pieces reimplemented here (stdlib only, no torch/svgpathtools):

* :func:`primitive_segment` -- reduce a line/circle/ellipse to the endpoint
  segment ``(x1, y1, x2, y2)`` CADTransformer uses as the primitive's spatial
  proxy (circle/ellipse -> horizontal diameter through the centre).
* :func:`primitive_length` -- arc length of a primitive; circle ``2*pi*r`` and
  the CADTransformer ellipse approximation ``2*pi*r_min + 4*(r_max-r_min)``.
* :func:`node_feature` -- the 6-D node feature ``[length/W, midx_norm,
  midy_norm, is_line, is_circle, is_ellipse]``.
* :func:`normalize_center` -- map a centre into ``[-1, 1]`` about the viewbox
  half-extent (CADTransformer ``ct_norm``).
* :func:`endpoint_knn` -- the primitive-adjacency scheme: neighbours ranked by
  the minimum over the four endpoint-to-endpoint squared distances.
* :func:`build_primitive_graph` -- the full ``svg2graph`` pipeline returning
  node features, centres, normalised centres and the adjacency lists.

Endpoint-distance adjacency is the key CADTransformer idea: two primitives are
"close" when *any* pair of their endpoints is close, so a wall segment and the
door arc that touches its end are neighbours even though their midpoints are
far apart -- ordinary centre-distance KNN would miss that.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Segment = Tuple[float, float, float, float]

LINE = "line"
CIRCLE = "circle"
ELLIPSE = "ellipse"
_TYPES = (LINE, CIRCLE, ELLIPSE)


class Primitive:
    """A vectorised CAD primitive.

    ``kind`` is one of ``"line"``, ``"circle"``, ``"ellipse"``.  For a line the
    geometry is ``(x1, y1, x2, y2)``; for a circle ``(cx, cy, r)``; for an
    ellipse ``(cx, cy, rx, ry)``.
    """

    __slots__ = ("kind", "geom", "semantic_id", "instance_id")

    def __init__(self, kind: str, geom: Sequence[float],
                 semantic_id: int = 0, instance_id: int = -1) -> None:
        if kind not in _TYPES:
            raise ValueError("unknown primitive kind: {0!r}".format(kind))
        self.kind = kind
        self.geom = tuple(float(v) for v in geom)
        self.semantic_id = int(semantic_id)
        self.instance_id = int(instance_id)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return "Primitive({0!r}, {1})".format(self.kind, self.geom)


def primitive_segment(prim: Primitive) -> Segment:
    """Endpoint segment ``(x1, y1, x2, y2)`` used as the spatial proxy.

    Circles and ellipses collapse to the horizontal diameter through their
    centre, exactly as ``preprocess_svg.svg2graph`` does.
    """
    if prim.kind == LINE:
        x1, y1, x2, y2 = prim.geom
        return (x1, y1, x2, y2)
    if prim.kind == CIRCLE:
        cx, cy, r = prim.geom
        return (cx - r, cy, cx + r, cy)
    cx, cy, rx, ry = prim.geom
    return (cx - rx, cy, cx + rx, cy)


def primitive_center(prim: Primitive) -> Tuple[float, float]:
    """Centre point of a primitive (segment midpoint / circle centre)."""
    if prim.kind == LINE:
        x1, y1, x2, y2 = prim.geom
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    cx, cy = prim.geom[0], prim.geom[1]
    return (cx, cy)


def primitive_length(prim: Primitive) -> float:
    """Arc length of a primitive.

    * line -> Euclidean endpoint distance,
    * circle -> ``2*pi*r``,
    * ellipse -> CADTransformer approximation
      ``2*pi*r_min + 4*(r_max - r_min)``.
    """
    if prim.kind == LINE:
        x1, y1, x2, y2 = prim.geom
        return math.hypot(x2 - x1, y2 - y1)
    if prim.kind == CIRCLE:
        r = prim.geom[2]
        return 2.0 * math.pi * r
    rx, ry = prim.geom[2], prim.geom[3]
    r_min, r_max = (rx, ry) if rx <= ry else (ry, rx)
    return 2.0 * math.pi * r_min + 4.0 * (r_max - r_min)


def _type_onehot(kind: str) -> Tuple[int, int, int]:
    return (1 if kind == LINE else 0,
            1 if kind == CIRCLE else 0,
            1 if kind == ELLIPSE else 0)


def node_feature(prim: Primitive, minx: float, miny: float,
                 width: float, height: float) -> List[float]:
    """The 6-D CADTransformer node feature.

    ``[length / width, (midx - minx) / width, (midy - miny) / height,
    is_line, is_circle, is_ellipse]``.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    mx, my = primitive_center(prim)
    length = primitive_length(prim)
    is_line, is_circle, is_ellipse = _type_onehot(prim.kind)
    return [length / width, (mx - minx) / width, (my - miny) / height,
            float(is_line), float(is_circle), float(is_ellipse)]


def normalize_center(center: Tuple[float, float], width: float,
                     height: float) -> Tuple[float, float]:
    """Map a centre into ``[-1, 1]`` about the viewbox half-extent."""
    half_w = width / 2.0
    half_h = height / 2.0
    return ((center[0] - half_w) / half_w, (center[1] - half_h) / half_h)


def _endpoint_min_sqdist(a: Segment, b: Segment) -> float:
    """Minimum of the four endpoint-to-endpoint squared distances."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    d1 = (ax1 - bx1) ** 2 + (ay1 - by1) ** 2
    d2 = (ax1 - bx2) ** 2 + (ay1 - by2) ** 2
    d3 = (ax2 - bx1) ** 2 + (ay2 - by1) ** 2
    d4 = (ax2 - bx2) ** 2 + (ay2 - by2) ** 2
    return min(d1, d2, d3, d4)


def endpoint_knn(segments: Sequence[Segment], max_degree: int = 4) -> List[List[int]]:
    """Primitive adjacency by minimum endpoint distance.

    For every segment, neighbours are the ``max_degree`` primitives (including
    itself) with the smallest :func:`_endpoint_min_sqdist`.  Ties break by
    index so the result is fully deterministic.  Mirrors
    ``preprocess_svg.get_nn`` but without torch.
    """
    if max_degree < 1:
        raise ValueError("max_degree must be >= 1")
    n = len(segments)
    result: List[List[int]] = []
    for i in range(n):
        keyed = [(_endpoint_min_sqdist(segments[i], segments[j]), j)
                 for j in range(n)]
        keyed.sort(key=lambda t: (t[0], t[1]))
        result.append([j for _, j in keyed[:max_degree]])
    return result


def build_primitive_graph(primitives: Sequence[Primitive], minx: float,
                          miny: float, width: float, height: float,
                          max_degree: int = 4) -> Dict[str, object]:
    """Full CADTransformer ``svg2graph`` graph for a set of primitives.

    Returns a dict with keys ``nd_ft`` (node features), ``ct`` (centres),
    ``ct_norm`` (normalised centres), ``nns`` (adjacency lists), ``cat``
    (per-node semantic id) and ``inst`` (per-node instance id).
    """
    nodes = [node_feature(p, minx, miny, width, height) for p in primitives]
    centers = [primitive_center(p) for p in primitives]
    centers_norm = [normalize_center(c, width, height) for c in centers]
    segments = [primitive_segment(p) for p in primitives]
    nns = endpoint_knn(segments, max_degree=max_degree)
    return {
        "nd_ft": nodes,
        "ct": centers,
        "ct_norm": centers_norm,
        "nns": nns,
        "cat": [[p.semantic_id] for p in primitives],
        "inst": [[p.instance_id] for p in primitives],
    }
