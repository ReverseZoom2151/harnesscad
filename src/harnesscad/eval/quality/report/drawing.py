"""2D engineering-drawing generation — a dimensioned orthographic drawing sheet.

This produces the standard mechanical deliverable a CAD product ships alongside
the 3D model: a multi-view drawing sheet with the solid projected to the
orthographic views (front / top / right) plus an isometric, laid out in a
third-angle (or first-angle) arrangement, annotated with the overall bounding
DIMENSIONS and key feature callouts (hole diameters), and finished with a
bordered TITLE BLOCK (part name, material, units, scale, date, drawing number).

Design goals (mirroring :mod:`surfaces.render` and :mod:`quality.describe`):

  * **Reuse the projection path.** Real orthographic/iso views are obtained from
    :func:`surfaces.render.render` (CadQuery SVG exporters). Each exported view is
    embedded as a nested ``<svg>`` scaled into its cell, so the sheet is a single
    self-contained SVG string.
  * **Degrade cleanly when headless.** When CadQuery/OCCT is absent, or the
    backend holds no renderable solid, we still emit a *usable* drawing: a
    schematic sheet of dimensioned silhouette boxes derived from the bounding box
    (from ``query('metrics')`` when available, otherwise inferred from the feature
    graph's sketch profiles + extrude depth). We attach a ``note`` and never crash.
  * **Grounded & deterministic.** Every number on the sheet comes from
    ``query('metrics')`` or the feature graph — never invented. No wall-clock is
    read: the date is injected (``date=``) and defaults to a placeholder.

Stdlib only for the SVG assembly; OCCT/CadQuery are touched only indirectly and
lazily through :mod:`surfaces.render`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

from harnesscad.eval.quality.graph.featuregraph import build_feature_graph


# --- small formatting helpers ----------------------------------------------
def _fmt_num(x: float) -> str:
    """Compact number: drop a trailing '.0', otherwise round to 2 dp."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "0"
    if abs(xf - round(xf)) < 1e-9:
        return str(int(round(xf)))
    return ("%.2f" % xf).rstrip("0").rstrip(".")


def _esc(text: Any) -> str:
    return escape("" if text is None else str(text))


# --- result container ------------------------------------------------------
@dataclass
class Drawing:
    """A rendered engineering-drawing sheet.

    ``svg`` is a single self-contained SVG document string. ``views`` lists the
    view names placed on the sheet. ``dimensions`` records the grounded bounding
    dimensions + feature callouts. ``title_block`` is the resolved title-block
    field map. ``note`` explains a schematic/headless fallback (``None`` when real
    views were embedded).
    """

    svg: str
    views: List[str] = field(default_factory=list)
    dimensions: Dict[str, Any] = field(default_factory=dict)
    title_block: Dict[str, str] = field(default_factory=dict)
    note: Optional[str] = None

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.svg)
        return path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "svg": self.svg,
            "views": list(self.views),
            "dimensions": dict(self.dimensions),
            "title_block": dict(self.title_block),
            "note": self.note,
        }


# --- bounding-box acquisition ----------------------------------------------
def _safe_query(backend: Any, q: str) -> Dict[str, Any]:
    try:
        result = backend.query(q)
        return result if isinstance(result, dict) else {}
    except Exception:  # noqa: BLE001 - a query must never break the drawing
        return {}


def _bbox_from_metrics(backend: Any) -> Optional[List[float]]:
    metrics = _safe_query(backend, "metrics") or _safe_query(backend, "measure")
    bbox = metrics.get("bbox")
    if bbox and len(bbox) >= 3 and any(float(v) > 0 for v in bbox[:3]):
        return [float(bbox[0]), float(bbox[1]), float(bbox[2])]
    return None


def _bbox_from_graph(graph) -> Optional[List[float]]:
    """Infer a bbox from the sketch profiles (planar extent) + extrude depth.

    A schematic approximation used only when no measured geometry is available
    (e.g. the dependency-free StubBackend): still grounded in the model's real
    sketch primitives and feature parameters, never invented.
    """
    xs: List[float] = []
    ys: List[float] = []
    for s in graph.find("sketch"):
        # The feature graph keeps only primitive *types* on sketch nodes; pull
        # the concrete primitive params back off the backend via the graph's
        # per-sketch primitive records when present.
        for prim in s.params.get("primitives", []):
            _accumulate_prim(prim, xs, ys)
    # Fall back to reading primitives straight off the graph's sketch params if
    # the richer 'primitives' list was not carried (backend-state path).
    depth = 0.0
    for f in graph.find("extrude"):
        d = f.params.get("distance")
        if d is not None:
            depth = max(depth, abs(float(d)))
    if not xs or not ys:
        return None
    length = max(xs) - min(xs)
    width = max(ys) - min(ys)
    if length <= 0 or width <= 0:
        return None
    if depth <= 0:
        depth = min(length, width)  # nominal thickness when no extrude depth
    return [float(length), float(width), float(depth)]


def _accumulate_prim(prim: Dict[str, Any], xs: List[float], ys: List[float]) -> None:
    t = prim.get("type")
    if t == "rectangle":
        x, y = float(prim.get("x", 0.0)), float(prim.get("y", 0.0))
        w, h = float(prim.get("w", 0.0)), float(prim.get("h", 0.0))
        xs.extend([x, x + w])
        ys.extend([y, y + h])
    elif t == "circle":
        cx, cy, r = float(prim.get("cx", 0.0)), float(prim.get("cy", 0.0)), float(prim.get("r", 0.0))
        xs.extend([cx - r, cx + r])
        ys.extend([cy - r, cy + r])
    elif t == "line":
        xs.extend([float(prim.get("x1", 0.0)), float(prim.get("x2", 0.0))])
        ys.extend([float(prim.get("y1", 0.0)), float(prim.get("y2", 0.0))])
    elif t == "point":
        xs.append(float(prim.get("x", 0.0)))
        ys.append(float(prim.get("y", 0.0)))


def _bbox_from_ops(ops: List[Any]) -> Optional[List[float]]:
    """Infer a bbox from an applied op stream (the richest schematic source).

    Both backends retain their accepted ops (``_oplog``); the pure StubBackend
    keeps no numeric params on its entities, so the op stream is the grounded
    source of the sketch profile extents + extrude depth.
    """
    from harnesscad.core.cisp.ops import (
        AddRectangle, AddCircle, AddLine, AddPoint, Extrude,
    )
    xs: List[float] = []
    ys: List[float] = []
    depth = 0.0
    for op in ops or []:
        if isinstance(op, AddRectangle):
            xs.extend([float(op.x), float(op.x) + float(op.w)])
            ys.extend([float(op.y), float(op.y) + float(op.h)])
        elif isinstance(op, AddCircle):
            xs.extend([float(op.cx) - float(op.r), float(op.cx) + float(op.r)])
            ys.extend([float(op.cy) - float(op.r), float(op.cy) + float(op.r)])
        elif isinstance(op, AddLine):
            xs.extend([float(op.x1), float(op.x2)])
            ys.extend([float(op.y1), float(op.y2)])
        elif isinstance(op, AddPoint):
            xs.append(float(op.x))
            ys.append(float(op.y))
        elif isinstance(op, Extrude):
            depth = max(depth, abs(float(op.distance)))
    if not xs or not ys:
        return None
    length = max(xs) - min(xs)
    width = max(ys) - min(ys)
    if length <= 0 or width <= 0:
        return None
    if depth <= 0:
        depth = min(length, width)
    return [float(length), float(width), float(depth)]


def _bbox_from_graph_backend(backend: Any) -> Optional[List[float]]:
    """Read primitives directly off the backend state (they carry full params).

    The feature graph normalises sketch nodes to primitive *types* only, so for
    the schematic inference we read the backend's own ``sketches``/``entities``
    which retain each primitive's numeric params (CadQuery/Stub both expose them,
    though the pure stub records only types — handled gracefully).
    """
    entities = getattr(backend, "entities", {}) or {}
    sketches = getattr(backend, "sketches", {}) or {}
    xs: List[float] = []
    ys: List[float] = []
    for s in sketches.values():
        for eid in s.get("entities", []):
            ent = entities.get(eid, {})
            params = ent.get("params") or {}
            prim = {"type": ent.get("type"), **params}
            _accumulate_prim(prim, xs, ys)
    depth = 0.0
    for f in getattr(backend, "features", []) or []:
        if f.get("type") in ("extrude",) and f.get("distance") is not None:
            depth = max(depth, abs(float(f["distance"])))
    if not xs or not ys:
        return None
    length = max(xs) - min(xs)
    width = max(ys) - min(ys)
    if length <= 0 or width <= 0:
        return None
    if depth <= 0:
        depth = min(length, width)
    return [float(length), float(width), float(depth)]


def _resolve_bbox(backend: Any, graph, _opdag_ops=None) -> Tuple[List[float], str]:
    """Return (bbox_dims, source) where source is 'metrics'|'derived'|'none'."""
    bbox = _bbox_from_metrics(backend)
    if bbox is not None:
        return bbox, "metrics"
    # The applied op stream is the grounded schematic source: the pure stub keeps
    # no numeric params on its entities, but its _oplog retains every op.
    ops = list(_opdag_ops) if _opdag_ops is not None else \
        list(getattr(backend, "_oplog", []) or [])
    bbox = _bbox_from_ops(ops)
    if bbox is not None:
        return bbox, "derived"
    # Fallback: read primitive params off a backend that retains them (CadQuery).
    bbox = _bbox_from_graph_backend(backend)
    if bbox is not None:
        return bbox, "derived"
    bbox = _bbox_from_graph(graph)
    if bbox is not None:
        return bbox, "derived"
    return [0.0, 0.0, 0.0], "none"


# --- feature callouts (hole diameters) -------------------------------------
def _hole_callouts(graph) -> List[Dict[str, Any]]:
    holes = graph.find("hole")
    counts: Dict[Tuple[Any, bool], int] = {}
    for h in holes:
        d = h.params.get("diameter")
        through = bool(h.params.get("through", True))
        counts[(d, through)] = counts.get((d, through), 0) + 1
    out: List[Dict[str, Any]] = []
    for (d, through), n in sorted(
        counts.items(), key=lambda kv: (kv[0][0] is None, kv[0][0] or 0, kv[0][1])
    ):
        out.append({"diameter": d, "through": through, "count": n})
    return out


# --- SVG primitives (dimensions with arrowheads) ---------------------------
_ARROW = 7.0  # arrowhead length in px


def _line(x1: float, y1: float, x2: float, y2: float,
          width: float = 1.0, color: str = "#111", dash: str = "") -> str:
    d = ' stroke-dasharray="%s"' % dash if dash else ""
    return ('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" '
            'stroke="%s" stroke-width="%.2f"%s/>' % (x1, y1, x2, y2, color, width, d))


def _text(x: float, y: float, s: str, size: float = 11.0,
          anchor: str = "middle", rotate: float = 0.0,
          weight: str = "normal", color: str = "#111") -> str:
    tr = ' transform="rotate(%.2f %.2f %.2f)"' % (rotate, x, y) if rotate else ""
    return ('<text x="%.2f" y="%.2f" font-family="Helvetica,Arial,sans-serif" '
            'font-size="%.2f" text-anchor="%s" font-weight="%s" fill="%s"%s>%s</text>'
            % (x, y, size, anchor, weight, color, tr, _esc(s)))


def _dim_horizontal(x1: float, x2: float, y: float, sil_bottom: float,
                    label: str) -> str:
    """A horizontal overall dimension below a silhouette (extension lines +
    dimension line with inward arrowheads + centred value)."""
    parts: List[str] = []
    # extension lines from the silhouette edge down past the dimension line
    parts.append(_line(x1, sil_bottom, x1, y + 5, 0.6, "#555"))
    parts.append(_line(x2, sil_bottom, x2, y + 5, 0.6, "#555"))
    # dimension line
    parts.append(_line(x1, y, x2, y, 1.0, "#111"))
    # arrowheads (filled triangles) pointing outward
    parts.append('<polygon points="%.2f,%.2f %.2f,%.2f %.2f,%.2f" fill="#111"/>'
                 % (x1, y, x1 + _ARROW, y - 3, x1 + _ARROW, y + 3))
    parts.append('<polygon points="%.2f,%.2f %.2f,%.2f %.2f,%.2f" fill="#111"/>'
                 % (x2, y, x2 - _ARROW, y - 3, x2 - _ARROW, y + 3))
    parts.append(_text((x1 + x2) / 2.0, y - 4, label, 11.0, "middle"))
    return "".join(parts)


def _dim_vertical(y1: float, y2: float, x: float, sil_left: float,
                  label: str) -> str:
    """A vertical overall dimension to the left of a silhouette."""
    parts: List[str] = []
    parts.append(_line(sil_left, y1, x - 5, y1, 0.6, "#555"))
    parts.append(_line(sil_left, y2, x - 5, y2, 0.6, "#555"))
    parts.append(_line(x, y1, x, y2, 1.0, "#111"))
    parts.append('<polygon points="%.2f,%.2f %.2f,%.2f %.2f,%.2f" fill="#111"/>'
                 % (x, y1, x - 3, y1 + _ARROW, x + 3, y1 + _ARROW))
    parts.append('<polygon points="%.2f,%.2f %.2f,%.2f %.2f,%.2f" fill="#111"/>'
                 % (x, y2, x - 3, y2 - _ARROW, x + 3, y2 - _ARROW))
    parts.append(_text(x - 4, (y1 + y2) / 2.0, label, 11.0, "middle", rotate=-90.0))
    return "".join(parts)


# --- embedding a rendered view SVG -----------------------------------------
_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>\s*", re.IGNORECASE)
_DOCTYPE = re.compile(r"^\s*<!DOCTYPE[^>]*>\s*", re.IGNORECASE)
_SVG_OPEN = re.compile(r"<svg\b([^>]*)>", re.IGNORECASE)
_ATTR = re.compile(r'([\w:-]+)\s*=\s*"([^"]*)"')


def _embed_svg(svg_text: str, x: float, y: float, w: float, h: float) -> Optional[str]:
    """Re-wrap an exported view SVG as a nested ``<svg>`` fitted into (x,y,w,h)."""
    if not svg_text:
        return None
    body = _XML_DECL.sub("", svg_text, count=1)
    body = _DOCTYPE.sub("", body, count=1)
    m = _SVG_OPEN.search(body)
    if not m:
        return None
    attrs = dict(_ATTR.findall(m.group(1)))
    view_box = attrs.get("viewBox") or attrs.get("viewbox")
    if not view_box:
        vw = _strip_unit(attrs.get("width"))
        vh = _strip_unit(attrs.get("height"))
        if vw and vh:
            view_box = "0 0 %s %s" % (vw, vh)
    vb_attr = ' viewBox="%s"' % view_box if view_box else ""
    new_open = ('<svg x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
                'preserveAspectRatio="xMidYMid meet"%s>'
                % (x, y, w, h, vb_attr))
    return new_open + body[m.end():]


def _strip_unit(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    m = re.match(r"\s*([0-9.]+)", value)
    return m.group(1) if m else None


# --- layout ----------------------------------------------------------------
# Preferred (col, row) slot per view for each projection convention, in a 2-col
# grid where row 0 is the top band and row 1 the bottom band. Collisions are
# resolved by falling through to the next free slot.
_SLOTS_THIRD = {
    "top": (0, 0), "iso": (1, 0),
    "front": (0, 1), "right": (1, 1),
    "left": (1, 1), "bottom": (0, 0), "back": (1, 1),
}
_SLOTS_FIRST = {
    "bottom": (0, 0), "iso": (1, 0),
    "front": (0, 1), "left": (1, 1),
    "right": (1, 1), "top": (0, 0), "back": (1, 1),
}

# (horizontal-dim-key, vertical-dim-key) each view can carry.
_VIEW_AXES = {
    "front": ("L", "H"), "back": ("L", "H"),
    "top": ("L", "W"), "bottom": ("L", "W"),
    "right": ("W", "H"), "left": ("W", "H"),
}


def _assign_slots(views: List[str], angle: str,
                  cols: int, rows: int) -> Dict[str, Tuple[int, int]]:
    preferred = _SLOTS_FIRST if angle == "first" else _SLOTS_THIRD
    free = [(c, r) for r in range(rows) for c in range(cols)]
    taken: Dict[Tuple[int, int], str] = {}
    placement: Dict[str, Tuple[int, int]] = {}
    for v in views:
        slot = preferred.get(v)
        if slot is None or slot not in free or slot in taken:
            slot = next((s for s in free if s not in taken), None)
        if slot is None:
            continue
        taken[slot] = v
        placement[v] = slot
    return placement


def _silhouette_mm(view: str, dims: List[float]) -> Tuple[float, float]:
    """(width_mm, height_mm) of a view's projected silhouette."""
    L, W, H = dims
    if view in ("front", "back"):
        return L, H
    if view in ("top", "bottom"):
        return L, W
    if view in ("right", "left"):
        return W, H
    return L, H  # iso / unknown -> use the front footprint as a placeholder


# --- title block -----------------------------------------------------------
def _default_title_block(backend: Any, dims: List[float], scale: str,
                         angle: str, date: Optional[str]) -> Dict[str, str]:
    try:
        digest = backend.state_digest()
    except Exception:  # noqa: BLE001
        digest = ""
    dwg_no = "HC-" + (digest[:8].upper() if digest else "00000000")
    return {
        "part": "PART",
        "material": "N/A",
        "units": "mm",
        "scale": scale,
        "date": date if date else "YYYY-MM-DD",
        "drawing_number": dwg_no,
        "projection": "FIRST ANGLE" if angle == "first" else "THIRD ANGLE",
        "sheet": "1 / 1",
    }


def _cell(x: float, y: float, w: float, h: float, label: str, value: str,
          value_size: float = 12.0) -> str:
    parts = ['<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
             'fill="none" stroke="#111" stroke-width="1"/>' % (x, y, w, h)]
    parts.append(_text(x + 4, y + 11, label, 7.5, "start", color="#555"))
    parts.append(_text(x + w / 2.0, y + h - 7, value, value_size, "middle",
                       weight="bold"))
    return "".join(parts)


def _render_title_block(tx: float, ty: float, tw: float, th: float,
                        tb: Dict[str, str]) -> str:
    parts: List[str] = []
    parts.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
                 'fill="none" stroke="#111" stroke-width="1.5"/>'
                 % (tx, ty, tw, th))
    lw = tw * 0.5
    # left column: title (top 60%) + drawing number (bottom 40%)
    parts.append(_cell(tx, ty, lw, th * 0.6, "TITLE", tb.get("part", ""), 15.0))
    parts.append(_cell(tx, ty + th * 0.6, lw, th * 0.4, "DRAWING NO.",
                       tb.get("drawing_number", ""), 12.0))
    # right block: 3 rows x 2 cols
    rx, rw = tx + lw, tw - lw
    cw, rh = rw / 2.0, th / 3.0
    grid = [
        [("MATERIAL", tb.get("material", "")), ("UNITS", tb.get("units", ""))],
        [("SCALE", tb.get("scale", "")), ("DATE", tb.get("date", ""))],
        [("PROJECTION", tb.get("projection", "")), ("SHEET", tb.get("sheet", ""))],
    ]
    for r, row in enumerate(grid):
        for c, (label, value) in enumerate(row):
            parts.append(_cell(rx + c * cw, ty + r * rh, cw, rh, label, value, 11.0))
    return "".join(parts)


# --- the sheet -------------------------------------------------------------
_STD_SCALES = [(5, 1), (2, 1), (1, 1), (1, 2), (1, 5),
               (1, 10), (1, 20), (1, 50), (1, 100)]
_PX_PER_MM = 96.0 / 25.4  # CSS reference: 96 dpi


def _nice_scale(px_per_mm: float) -> str:
    if px_per_mm <= 0:
        return "NTS"
    ratio = px_per_mm / _PX_PER_MM  # drawing : real
    import math
    best = min(_STD_SCALES, key=lambda ab: abs(math.log((ab[0] / ab[1]) / ratio)))
    return "%d:%d" % best


def make_drawing(backend: Any,
                 views: Tuple[str, ...] = ("front", "top", "right", "iso"),
                 size: Tuple[int, int] = (1120, 792),
                 title_block: Optional[Dict[str, str]] = None,
                 angle: str = "third",
                 date: Optional[str] = None,
                 opdag: Any = None) -> Drawing:
    """Generate a dimensioned multi-view engineering-drawing sheet.

    Projects the backend's solid to the requested orthographic/iso ``views``
    (via :func:`surfaces.render.render`), lays them out in a third-angle (default)
    or first-angle arrangement, annotates the overall bounding dimensions and hole
    callouts, and adds a bordered title block. Returns a :class:`Drawing` whose
    ``svg`` is a single self-contained document.

    Never raises: when CadQuery/OCCT is unavailable or the backend holds no solid,
    a schematic sheet (dimensioned silhouette boxes from the bbox) is produced with
    an explanatory ``note``. ``date`` is injected (no wall-clock); when ``None`` a
    placeholder is used so output stays deterministic.
    """
    view_list = [str(v).lower() for v in views]
    graph = build_feature_graph(opdag if opdag is not None else backend,
                                backend=backend if opdag is not None else None)
    opdag_ops = None
    if opdag is not None and hasattr(opdag, "ops") and callable(opdag.ops):
        try:
            opdag_ops = list(opdag.ops())
        except Exception:  # noqa: BLE001
            opdag_ops = None
    dims, source = _resolve_bbox(backend, graph, opdag_ops)
    holes = _hole_callouts(graph)

    # Try to obtain real projected views; degrade to schematic on any failure.
    rendered: Dict[str, Optional[bytes]] = {}
    render_note: Optional[str] = None
    try:
        from harnesscad.io.surfaces.render import render as _render
        result = _render(backend, views=view_list, size=(460, 460), fmt="svg")
        rendered = result.images
        if not result.any_rendered:
            render_note = result.note
    except Exception as exc:  # noqa: BLE001 - drawing must never crash
        render_note = "view rendering unavailable: %s" % exc

    any_real = any(v is not None for v in rendered.values())

    W, H = int(size[0]), int(size[1])
    margin = 14.0
    tb_h = 120.0
    # drawing area (inside the border, above the title block)
    da_x, da_y = margin, margin
    da_w = W - 2 * margin
    da_h = H - 2 * margin - tb_h

    cols = 2
    rows = max(1, (len(view_list) + cols - 1) // cols)
    placement = _assign_slots(view_list, angle, cols, rows)
    cell_w = da_w / cols
    cell_h = da_h / rows

    # Global scale (px per mm) so silhouettes fit their cells with room for dims.
    pad = 52.0
    L, Wd, Ht = dims
    scale = float("inf")
    for v in view_list:
        sw, sh = _silhouette_mm(v, dims)
        sw = max(sw, 1e-6)
        sh = max(sh, 1e-6)
        avail_w = max(cell_w - 2 * pad, 20.0)
        avail_h = max(cell_h - 2 * pad, 20.0)
        scale = min(scale, avail_w / sw, avail_h / sh)
    if scale == float("inf") or scale <= 0:
        scale = 1.0

    parts: List[str] = []
    parts.append('<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' % (W, H))
    # outer + inner border
    parts.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
                 'fill="none" stroke="#111" stroke-width="2"/>'
                 % (margin / 2, margin / 2, W - margin, H - margin))
    parts.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
                 'fill="none" stroke="#111" stroke-width="1"/>'
                 % (da_x, da_y, da_w, da_h))

    drawn_dims: set = set()
    for v in view_list:
        slot = placement.get(v)
        if slot is None:
            continue
        col, row = slot
        cx = da_x + col * cell_w
        cy = da_y + row * cell_h
        sw_mm, sh_mm = _silhouette_mm(v, dims)
        bw = max(sw_mm * scale, 8.0)
        bh = max(sh_mm * scale, 8.0)
        bx = cx + (cell_w - bw) / 2.0
        by = cy + (cell_h - bh) / 2.0

        group: List[str] = ['<g>']
        embedded = None
        if any_real and rendered.get(v):
            try:
                embedded = _embed_svg(rendered[v].decode("utf-8", "replace"),
                                      bx, by, bw, bh)
            except Exception:  # noqa: BLE001
                embedded = None
        if embedded is not None:
            group.append(embedded)
        else:
            # schematic silhouette rectangle
            group.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
                         'fill="#f4f6f8" stroke="#111" stroke-width="1.2"/>'
                         % (bx, by, bw, bh))
            if v == "iso":
                # a light offset box to read as 3D
                dx, dy = min(bw, bh) * 0.22, -min(bw, bh) * 0.22
                group.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
                             'fill="none" stroke="#999" stroke-width="0.7"/>'
                             % (bx + dx, by + dy, bw, bh))
        # view label
        group.append(_text(cx + cell_w / 2.0, cy + cell_h - 6,
                           v.upper(), 10.0, "middle", color="#555"))

        # overall dimensions (each drawn once, on the first view that carries it)
        axes = _VIEW_AXES.get(v)
        if axes and source != "none":
            h_key, w_key = axes
            h_val = {"L": L, "W": Wd}[h_key]
            if h_key not in drawn_dims:
                group.append(_dim_horizontal(bx, bx + bw, by + bh + 26, by + bh,
                                             _fmt_num(h_val)))
                drawn_dims.add(h_key)
            v_val = {"H": Ht, "W": Wd}[w_key]
            if w_key not in drawn_dims:
                group.append(_dim_vertical(by, by + bh, bx - 26, bx,
                                           _fmt_num(v_val)))
                drawn_dims.add(w_key)
        group.append('</g>')
        parts.append("".join(group))

    # hole callouts (notes block, top-left inside the drawing area)
    if holes:
        note_x, note_y = da_x + 8, da_y + 16
        parts.append(_text(note_x, note_y, "NOTES", 9.0, "start",
                           weight="bold", color="#555"))
        for i, hc in enumerate(holes):
            d = hc["diameter"]
            dtxt = ("Ø" + _fmt_num(d)) if d is not None else "Ø?"
            thru = " THRU" if hc["through"] else ""
            line = "%d× %s%s" % (hc["count"], dtxt, thru)
            parts.append(_text(note_x, note_y + 14 * (i + 1), line, 10.0, "start"))

    # scale + title block
    scale_str = _nice_scale(scale)
    tb_defaults = _default_title_block(backend, dims, scale_str, angle, date)
    if title_block:
        tb_defaults.update({k: str(v) for k, v in title_block.items()})
    tx = margin
    ty = H - margin - tb_h
    parts.append(_render_title_block(tx, ty, W - 2 * margin, tb_h, tb_defaults))

    svg = ('<svg xmlns="http://www.w3.org/2000/svg" '
           'width="%d" height="%d" viewBox="0 0 %d %d">%s</svg>'
           % (W, H, W, H, "".join(parts)))

    dimensions = {
        "bbox": dims,
        "length": L,
        "width": Wd,
        "height": Ht,
        "units": "mm",
        "holes": holes,
        "source": source,
    }
    note: Optional[str] = None
    if not any_real:
        base = render_note or "no renderable solid"
        note = ("schematic drawing from bounding-box dimensions (%s); "
                "real orthographic views require CadQuery/OCCT and a solid." % base)
        if source == "none":
            note += " No dimensions available from this backend."

    return Drawing(svg=svg, views=view_list, dimensions=dimensions,
                   title_block=tb_defaults, note=note)
