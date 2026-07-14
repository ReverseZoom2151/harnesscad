"""Orthographic engineering-drawing export -- the real drawing route.

Until now the only drawing this repo could produce was
``io.formats.svg.get_svg``: one isometric wireframe of *every* mesh edge, with no
views, no scale, no hidden-line treatment and no dimensions. Meanwhile the whole
drawing stack sat unreachable under ``domain.drawings``: the projection
convention (first/third angle), the viewport transform, the linetype dasher, the
dimension geometry, the drawing-command vocabulary and the SVG view metrics.

This module is the route that uses them. From a triangle mesh it produces a
standards-shaped multi-view drawing:

1.  **Feature edges.** A tessellation has thousands of edges, almost none of them
    drawable. Only edges whose two adjacent faces differ in orientation by more
    than ``angle`` (plus any boundary edge) survive -- that is the silhouette and
    crease set an engineering drawing actually shows.
2.  **Projection.** Each view is an axis-aligned orthographic projection
    (front/top/side), placed on the sheet by
    ``domain.drawings.projection_convention.view_placements`` -- so the sheet
    honours first- or third-angle convention rather than inventing a layout.
    ``views_sufficient`` is asked whether the chosen view set even pins the part
    down; an insufficient set is reported, not silently drawn.
3.  **Hidden lines.** An edge is hidden when the solid is between it and the
    viewer. That is a visibility ray query, so it runs against the mesh BVH
    (``domain.geometry.mesh.bvh``): the ray's box is queried, and only the
    candidate triangles it returns are intersected. Hidden edges are dashed with
    the ISO HIDDEN pattern by ``domain.drawings.linetypes``, not drawn as solid
    lines and not thrown away.
4.  **Scale + viewport.** One scale for the whole sheet (drawings are to scale),
    computed with ``domain.drawings.viewport.zoom_to_fit`` and applied through
    ``to_canvas_coordinates``.
5.  **Dimensions.** Overall width/height dimensions per view come from
    ``domain.drawings.dimensions.overall_dimensions`` -- real dimension lines,
    extension lines, arrows and a text anchor, not a decorative rectangle.
6.  **Entities, then SVG.** The drawing is assembled as
    ``domain.drawings.drawing_commands`` entities (a host-independent 2D CAD
    command list) and only then serialised to SVG here. The same entity list can
    therefore drive a different writer later.

Honest limits: visibility is decided per edge from its MIDPOINT, so an edge that
is half-occluded is classified by its middle. That is the classic cheap
hidden-line test, and it is stated rather than hidden.

stdlib-only, deterministic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.drawings import dimensions as dim
from harnesscad.domain.drawings import drawing_commands as cmd
from harnesscad.domain.drawings import linetypes
from harnesscad.domain.drawings import projection_convention as convention
from harnesscad.domain.drawings import svg_view_metrics
from harnesscad.domain.drawings import viewport as vp
from harnesscad.domain.geometry.mesh import bvh as mesh_bvh

__all__ = [
    "VIEWS",
    "DrawingError",
    "feature_edges",
    "orthographic_drawing",
    "drawing_metrics",
]

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]
Mesh = Tuple[Sequence[Vec3], Sequence[Sequence[int]]]

#: view name -> (in-plane u axis, in-plane v axis, direction FROM the part TO the
#: viewer). The three standard orthographic views; the names are the ones
#: ``projection_convention`` places on the sheet.
VIEWS: Dict[str, Tuple[int, int, Vec3]] = {
    "front": (0, 2, (0.0, -1.0, 0.0)),   # look along +Y: sees XZ
    "top": (0, 1, (0.0, 0.0, 1.0)),      # look along -Z: sees XY
    "side": (1, 2, (1.0, 0.0, 0.0)),     # look along -X: sees YZ
    # A pictorial view. The three orthographic views are the CAD deliverable,
    # but none of them shows the part as a solid, so a reader cannot see the
    # shape. `iso` is a true isometric projection (not an axis pick), viewed
    # from (1, 1, 1); the -1 index marks it for _project.
    "iso": (-1, -1, (1.0, 1.0, 1.0)),
}

# cos(30), sin(30): the isometric foreshortening.
_ISO_C = 0.8660254037844387
_ISO_S = 0.5

_EPS = 1e-9


class DrawingError(Exception):
    """The mesh or the view set cannot produce a drawing."""


# ---------------------------------------------------------------------------
# feature edges
# ---------------------------------------------------------------------------

def _face_normal(v: Sequence[Vec3], f: Sequence[int]) -> Vec3:
    a, b, c = v[f[0]], v[f[1]], v[f[2]]
    e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    n = (e1[1] * e2[2] - e1[2] * e2[1],
         e1[2] * e2[0] - e1[0] * e2[2],
         e1[0] * e2[1] - e1[1] * e2[0])
    m = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    if m == 0.0:
        return (0.0, 0.0, 0.0)
    return (n[0] / m, n[1] / m, n[2] / m)


def feature_edges(mesh: Mesh, angle: float = 25.0) -> List[Tuple[int, int]]:
    """Crease and boundary edges: the edges a drawing actually shows.

    An edge is kept when its two adjacent faces disagree in normal direction by
    more than ``angle`` degrees (a crease), or when it belongs to only one face
    (a boundary). Every other edge is interior tessellation noise.
    """
    verts, faces = mesh
    normals = [_face_normal(verts, f) for f in faces]
    adj: Dict[Tuple[int, int], List[int]] = {}
    for fi, f in enumerate(faces):
        ids = [int(i) for i in f]
        for k in range(len(ids)):
            a, b = ids[k], ids[(k + 1) % len(ids)]
            key = (a, b) if a < b else (b, a)
            adj.setdefault(key, []).append(fi)
    cos_limit = math.cos(math.radians(float(angle)))
    out: List[Tuple[int, int]] = []
    for key in sorted(adj):
        fs = adj[key]
        if len(fs) == 1:
            out.append(key)
            continue
        n0, n1 = normals[fs[0]], normals[fs[1]]
        dot = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]
        if dot < cos_limit - 1e-12:
            out.append(key)
    return out


# ---------------------------------------------------------------------------
# visibility (BVH-accelerated)
# ---------------------------------------------------------------------------

def _ray_triangle(origin: Vec3, direction: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Optional[float]:
    """Moller-Trumbore. Returns the ray parameter of the hit, or None."""
    e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    p = (direction[1] * e2[2] - direction[2] * e2[1],
         direction[2] * e2[0] - direction[0] * e2[2],
         direction[0] * e2[1] - direction[1] * e2[0])
    det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2]
    if abs(det) < 1e-12:
        return None
    inv = 1.0 / det
    t = (origin[0] - a[0], origin[1] - a[1], origin[2] - a[2])
    u = (t[0] * p[0] + t[1] * p[1] + t[2] * p[2]) * inv
    if u < -1e-9 or u > 1.0 + 1e-9:
        return None
    q = (t[1] * e1[2] - t[2] * e1[1],
         t[2] * e1[0] - t[0] * e1[2],
         t[0] * e1[1] - t[1] * e1[0])
    v = (direction[0] * q[0] + direction[1] * q[1] + direction[2] * q[2]) * inv
    if v < -1e-9 or u + v > 1.0 + 1e-9:
        return None
    dist = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv
    return dist if dist > 0.0 else None


class _Visibility:
    """Hidden-line oracle over a mesh, accelerated by its BVH."""

    def __init__(self, mesh: Mesh) -> None:
        self.verts = [tuple(float(c) for c in v) for v in mesh[0]]
        self.faces = [tuple(int(i) for i in f) for f in mesh[1]]
        boxes = mesh_bvh.boxes_of_triangles(self.verts, self.faces)
        self.bvh = mesh_bvh.BVH(boxes)
        lo = [min(v[i] for v in self.verts) for i in range(3)]
        hi = [max(v[i] for v in self.verts) for i in range(3)]
        self.diagonal = math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3))) or 1.0
        self.tested = 0          # candidate triangles the BVH actually handed back
        self.brute = 0           # what a brute-force test would have cost

    def hidden(self, point: Vec3, toward_viewer: Vec3) -> bool:
        """Is ``point`` occluded by the solid when seen from ``toward_viewer``?"""
        # start a hair off the surface so the edge's own faces are not hit
        eps = 1e-6 * self.diagonal
        origin = (point[0] + toward_viewer[0] * eps,
                  point[1] + toward_viewer[1] * eps,
                  point[2] + toward_viewer[2] * eps)
        far = (origin[0] + toward_viewer[0] * 2.0 * self.diagonal,
               origin[1] + toward_viewer[1] * 2.0 * self.diagonal,
               origin[2] + toward_viewer[2] * 2.0 * self.diagonal)
        box = mesh_bvh.AABB.of_points([origin, far])
        candidates = self.bvh.query(box)
        self.tested += len(candidates)
        self.brute += len(self.faces)
        for fi in candidates:
            a, b, c = (self.verts[i] for i in self.faces[fi])
            if _ray_triangle(origin, toward_viewer, a, b, c) is not None:
                return True
        return False


# ---------------------------------------------------------------------------
# the drawing
# ---------------------------------------------------------------------------

def _project(p: Vec3, view: str) -> Vec2:
    iu, iv, _ = VIEWS[view]
    if iu < 0:  # isometric: a real projection, not an axis pick
        x, y, z = p
        return ((x - y) * _ISO_C, (x + y) * _ISO_S + z)
    return (p[iu], p[iv])


def _bbox(points: Sequence[Vec2]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _svg_num(v: float) -> str:
    return ("%.3f" % float(v)).rstrip("0").rstrip(".")


def orthographic_drawing(mesh: Mesh,
                         views: Sequence[str] = ("front", "top", "side"),
                         angle_convention: str = "third_angle",
                         width: float = 900.0,
                         height: float = 640.0,
                         crease_angle: float = 25.0,
                         show_hidden: bool = True,
                         show_dimensions: bool = True,
                         hidden_scale: float = 16.0,
                         margin: float = 0.12,
                         title: str = "harnesscad") -> str:
    """A multi-view orthographic engineering drawing of ``mesh``, as SVG text."""
    verts, faces = mesh
    if not faces:
        raise DrawingError("nothing to draw: the mesh has no faces")
    for name in views:
        if name not in VIEWS:
            raise DrawingError("unknown view %r (known: %s)"
                               % (name, ", ".join(sorted(VIEWS))))
    placements = convention.view_placements(angle_convention)
    sufficiency = convention.views_sufficient(list(views))

    edges = feature_edges(mesh, angle=crease_angle)
    if not edges:
        raise DrawingError("nothing to draw: the mesh has no feature edges")
    oracle = _Visibility(mesh) if show_hidden else None

    # `iso` is a pictorial, not one of the projected views, so the first/third
    # angle convention has no cell for it. Put it in the free diagonal cell.
    if "iso" in views and "iso" not in placements:
        taken = {placements[v] for v in views if v in placements}
        placements = dict(placements)
        placements["iso"] = next(
            ((c, r) for r in range(3) for c in range(3) if (c, r) not in taken),
            (1, 1),
        )

    # one scale for the whole sheet: the largest view drives it
    cols = sorted({placements[v][0] for v in views})
    rows = sorted({placements[v][1] for v in views})
    cell = (width / max(len(cols), 1), height / max(len(rows), 1))
    boxes = {v: _bbox([_project(p, v) for p in verts]) for v in views}
    scale = min(vp.zoom_to_fit(boxes[v], cell, margin=margin).pixels_per_drawing_unit
                for v in views)

    builder = cmd.DrawingBuilder()
    for name in views:
        col, row = placements[name]
        ox = cols.index(col) * cell[0]
        oy = rows.index(row) * cell[1]
        box = boxes[name]
        centre = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
        port = vp.Viewport(centre=centre, pixels_per_drawing_unit=scale)

        def to_px(pt: Vec2, ox=ox, oy=oy, port=port) -> Vec2:
            cx, cy = vp.to_canvas_coordinates(pt, port, cell)
            return (ox + cx, oy + cy)

        _, _, toward = VIEWS[name]
        visible: List[Tuple[Vec2, Vec2]] = []
        hidden: List[Tuple[Vec2, Vec2]] = []
        for (ia, ib) in edges:
            a3, b3 = verts[ia], verts[ib]
            mid = (0.5 * (a3[0] + b3[0]), 0.5 * (a3[1] + b3[1]), 0.5 * (a3[2] + b3[2]))
            seg = (to_px(_project(a3, name)), to_px(_project(b3, name)))
            if oracle is not None and oracle.hidden(mid, toward):
                hidden.append(seg)
            else:
                visible.append(seg)

        for (a, b) in visible:
            builder.add(cmd.line((a[0], a[1], 0.0), (b[0], b[1], 0.0)))
        for (a, b) in hidden:
            # the ISO hidden-line pattern, applied along the edge in canvas units
            stroked = linetypes.apply_named([a, b], "HIDDEN",
                                            scale=float(hidden_scale))
            for (x0, y0, x1, y1) in stroked.segments:
                e = cmd.line((x0, y0, 0.0), (x1, y1, 0.0))
                builder.add(cmd.DrawingEntity(kind=e.kind, geometry=e.geometry,
                                              layer="hidden"))
        if show_dimensions:
            # Measure in MODEL space, place in PIXEL space. Measuring the
            # already-pixelised points made every dimension report screen
            # distance: a 40 mm plate was annotated "342".
            pts_model = [_project(p, name) for p in verts]
            offset_model = 18.0 / scale if scale else 18.0
            for geom in dim.overall_dimensions(pts_model, offset=offset_model):
                _add_dimension(builder, geom, to_px)
        builder.add(cmd.text((ox + 8.0, oy + 16.0, 0.0), name.upper(), height=11.0))

    return _to_svg(builder.to_list(), width, height, title, sufficiency)


def _add_dimension(builder: "cmd.DrawingBuilder", geom: "dim.DimensionGeometry",
                   to_px=None) -> None:
    """Emit a dimension's line, extension lines, arrow ticks and text.

    ``geom`` is measured in MODEL units, so ``geom.measured`` is the real
    dimension (mm). ``to_px`` maps its placement into sheet pixels. The value
    printed is never rescaled: a 40 mm edge is annotated "40", whatever the
    sheet scale happens to be.
    """
    px = to_px if to_px is not None else (lambda p: p)
    for seg in (geom.dimension_line, geom.extension_a, geom.extension_b):
        a = px((seg[0], seg[1]))
        b = px((seg[2], seg[3]))
        e = cmd.line((a[0], a[1], 0.0), (b[0], b[1], 0.0))
        builder.add(cmd.DrawingEntity(kind=e.kind, geometry=e.geometry, layer="dimension"))
    for arrow in (geom.arrow_a, geom.arrow_b):
        ax, ay = px(arrow)
        e = cmd.circle((ax, ay, 0.0), 1.5)
        builder.add(cmd.DrawingEntity(kind=e.kind, geometry=e.geometry, layer="dimension"))
    tx, ty = px(geom.text_anchor)
    e = cmd.text((tx, ty - 3.0, 0.0), _svg_num(geom.measured), height=10.0)
    builder.add(cmd.DrawingEntity(kind=e.kind, geometry=e.geometry, layer="dimension"))


_STYLE = {
    None: 'stroke="#111" stroke-width="1.2" fill="none"',
    "hidden": 'stroke="#888" stroke-width="0.9" fill="none"',
    "dimension": 'stroke="#2255aa" stroke-width="0.7" fill="none"',
}


def _to_svg(entities: Sequence[dict], width: float, height: float, title: str,
            sufficiency) -> str:
    """Serialise the drawing-command entity list to an SVG document."""
    out: List[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append('<svg xmlns="http://www.w3.org/2000/svg" width="%s" height="%s" '
               'viewBox="0 0 %s %s">'
               % (_svg_num(width), _svg_num(height), _svg_num(width), _svg_num(height)))
    out.append("<title>%s</title>" % title)
    out.append("<desc>views sufficient: %s; covered: %s</desc>"
               % (str(bool(sufficiency.sufficient)).lower(),
                  ",".join(sufficiency.covered)))
    out.append('<rect x="0" y="0" width="%s" height="%s" fill="#fff"/>'
               % (_svg_num(width), _svg_num(height)))
    for ent in entities:
        style = _STYLE.get(ent.get("layer"), _STYLE[None])
        g = ent["geometry"]
        kind = ent["kind"]
        if kind == "line":
            s, e = g["start"], g["end"]
            out.append('<path d="M %s %s L %s %s" %s/>'
                       % (_svg_num(s[0]), _svg_num(s[1]),
                          _svg_num(e[0]), _svg_num(e[1]), style))
        elif kind == "circle":
            c = g["center"]
            out.append('<circle cx="%s" cy="%s" r="%s" %s/>'
                       % (_svg_num(c[0]), _svg_num(c[1]), _svg_num(g["radius"]), style))
        elif kind == "text":
            p = g["position"]
            out.append('<text x="%s" y="%s" font-size="%s" fill="#2255aa">%s</text>'
                       % (_svg_num(p[0]), _svg_num(p[1]), _svg_num(g["height"]),
                          str(g["text"])))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def drawing_metrics(svg_text: str) -> dict:
    """Measure a drawing we just produced (``domain.drawings.svg_view_metrics``).

    The metrics module was written to grade *generated* engineering drawings; it
    grades ours too, which is the cheapest possible self-check that the export
    really put geometry on the sheet.
    """
    return svg_view_metrics.analyze_svg_text(svg_text)
