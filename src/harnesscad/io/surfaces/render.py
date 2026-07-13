"""Multi-view rendering — turn the backend's current solid into per-view images.

This is the *observation* half of the SpatialHero render->judge->reward loop
(docs/blueprint.md sec.21): a CadQuery model is rendered from an isometric
camera plus the orthographic views (front/top/right/...), and those images feed
a vision model (see :mod:`checks_vision`). It also supplies the "rendered
viewport" half of the hybrid observation from sec.5 (geometry summary + image).

Design goals:
  * **Non-fatal & optional.** Rendering libraries (cadquery/OCP) are imported
    lazily. If they are absent, or the backend holds no solid (e.g. the stub
    backend, or a headless CI box), we return ``None`` per view together with a
    human-readable ``note`` — we never crash. A real renderer drops straight in
    behind :func:`render_views` without changing any caller.
  * **Backend-agnostic.** We duck-type the backend: any backend exposing a
    ``_combined()`` OCCT shape (the CadQueryBackend) can be rendered; everything
    else takes the skip path. The kernel is never mutated.
  * **Deterministic where possible.** SVG export of a fixed shape from a fixed
    camera direction is stable across runs, so image bytes are reproducible.

Default output format is SVG (vector, dependency-light, deterministic); a PNG
raster path is attempted when a raster exporter is available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


# --- camera / projection description ---------------------------------------
@dataclass(frozen=True)
class ViewSpec:
    """A named camera: a projection direction (from the object toward the eye),
    an up hint, and whether it reads as an isometric or an orthographic view.

    ``direction`` is the SVG/OCCT ``projectionDir`` — the vector pointing from
    the model toward the viewer. ``up`` is advisory (kept for a future raster
    renderer that needs an explicit up vector).
    """

    name: str
    direction: Tuple[float, float, float]
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    projection: str = "ortho"  # 'iso' | 'ortho'


# Isometric + the six orthographic faces. "under" is an alias for "bottom"
# (the SpatialHero deck names its ortho set under/top/front/back).
STANDARD_VIEWS: Dict[str, ViewSpec] = {
    "iso":    ViewSpec("iso",    (1.0, 1.0, 1.0),   (0.0, 0.0, 1.0), "iso"),
    "front":  ViewSpec("front",  (0.0, -1.0, 0.0),  (0.0, 0.0, 1.0), "ortho"),
    "back":   ViewSpec("back",   (0.0, 1.0, 0.0),   (0.0, 0.0, 1.0), "ortho"),
    "left":   ViewSpec("left",   (-1.0, 0.0, 0.0),  (0.0, 0.0, 1.0), "ortho"),
    "right":  ViewSpec("right",  (1.0, 0.0, 0.0),   (0.0, 0.0, 1.0), "ortho"),
    "top":    ViewSpec("top",    (0.0, 0.0, 1.0),   (0.0, 1.0, 0.0), "ortho"),
    "bottom": ViewSpec("bottom", (0.0, 0.0, -1.0),  (0.0, 1.0, 0.0), "ortho"),
    "under":  ViewSpec("under",  (0.0, 0.0, -1.0),  (0.0, 1.0, 0.0), "ortho"),
}

DEFAULT_VIEWS: Tuple[str, ...] = ("iso", "front", "top", "right")


def resolve_views(views: Iterable) -> List[ViewSpec]:
    """Normalise a mix of view names and ViewSpec objects into ViewSpecs."""
    out: List[ViewSpec] = []
    for v in views:
        if isinstance(v, ViewSpec):
            out.append(v)
            continue
        key = str(v).lower()
        if key not in STANDARD_VIEWS:
            raise KeyError(
                f"unknown view '{v}'; known: {sorted(STANDARD_VIEWS)}")
        out.append(STANDARD_VIEWS[key])
    return out


# --- result container ------------------------------------------------------
@dataclass
class RenderResult:
    """Images keyed by view name (``bytes`` or ``None``) plus context.

    ``note`` explains a skip (no kernel / no solid); ``fmt`` is the encoding of
    any rendered bytes ('svg' | 'png').
    """

    images: Dict[str, Optional[bytes]] = field(default_factory=dict)
    fmt: str = "svg"
    note: Optional[str] = None

    @property
    def any_rendered(self) -> bool:
        return any(v is not None for v in self.images.values())

    @property
    def rendered(self) -> Dict[str, bytes]:
        return {k: v for k, v in self.images.items() if v is not None}


# --- geometry acquisition --------------------------------------------------
def _cadquery():
    import cadquery  # noqa: WPS433 (deliberately lazy)
    return cadquery


def _shape_from_backend(backend):
    """Best-effort OCCT shape for the backend's current model, or ``None``.

    Duck-typed and defensive: any exception (no kernel, no solid, foreign
    backend) yields ``None`` so callers take the graceful skip path. The backend
    is only read, never mutated.
    """
    combined = getattr(backend, "_combined", None)
    if not callable(combined):
        return None
    try:
        return combined()
    except Exception:  # noqa: BLE001 - rendering must never crash the loop
        return None


# --- per-view rendering ----------------------------------------------------
def _render_svg(cq, shape, spec: ViewSpec, size: Tuple[int, int]) -> Optional[bytes]:
    """Render one view to SVG bytes via cadquery's exporter, or ``None``."""
    from cadquery import exporters  # lazy

    wp = cq.Workplane("XY").add(shape)
    opt = {
        "width": int(size[0]),
        "height": int(size[1]),
        "projectionDir": tuple(float(c) for c in spec.direction),
        "showAxes": False,
        "showHidden": False,
    }
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".svg")
    os.close(fd)
    try:
        exporters.export(wp, path, exportType="SVG", opt=opt)
        with open(path, "rb") as fh:
            return fh.read()
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _render_png(cq, shape, spec: ViewSpec, size: Tuple[int, int]) -> Optional[bytes]:
    """Optional raster path. Attempts an OCCT offscreen render when available;
    returns ``None`` (so the caller can fall back to SVG) if it is not."""
    try:
        from cadquery import exporters  # lazy

        wp = cq.Workplane("XY").add(shape)
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            exporters.export(wp, path, exportType="PNG")  # not always available
            with open(path, "rb") as fh:
                return fh.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    except Exception:  # noqa: BLE001 - no raster path here; caller falls back
        return None


# --- public API ------------------------------------------------------------
def render(
    backend,
    views: Iterable = DEFAULT_VIEWS,
    size: Tuple[int, int] = (512, 512),
    fmt: str = "svg",
) -> RenderResult:
    """Render the backend's current model from each requested view.

    Returns a :class:`RenderResult`. On the skip path (no kernel / no solid)
    every image is ``None`` and ``note`` says why — this never raises.
    """
    specs = resolve_views(views)
    images: Dict[str, Optional[bytes]] = {s.name: None for s in specs}

    try:
        cq = _cadquery()
    except Exception:  # noqa: BLE001
        return RenderResult(images, fmt=fmt,
                            note="rendering unavailable: cadquery/OCP not installed")

    shape = _shape_from_backend(backend)
    if shape is None:
        return RenderResult(
            images, fmt=fmt,
            note="no renderable solid: backend holds no geometry (headless skip)")

    want_png = fmt.lower() == "png"
    used_fmt = "svg"
    for spec in specs:
        data = None
        if want_png:
            data = _render_png(cq, shape, spec, size)
            if data is not None:
                used_fmt = "png"
        if data is None:  # SVG default / PNG fallback
            data = _render_svg(cq, shape, spec, size)
        images[spec.name] = data

    if not any(v is not None for v in images.values()):
        return RenderResult(images, fmt=used_fmt,
                            note="renderer produced no images for the current shape")
    return RenderResult(images, fmt=used_fmt, note=None)


def render_views(
    backend,
    views: Iterable = DEFAULT_VIEWS,
    size: Tuple[int, int] = (512, 512),
    fmt: str = "svg",
) -> Dict[str, Optional[bytes]]:
    """Thin wrapper returning just the ``{view_name: bytes|None}`` mapping.

    This is the primary entry point named in the harness spec. Use
    :func:`render` when you also want the skip ``note`` / format.
    """
    return render(backend, views=views, size=size, fmt=fmt).images


def save_views(
    backend,
    directory: str,
    views: Iterable = DEFAULT_VIEWS,
    size: Tuple[int, int] = (512, 512),
    fmt: str = "svg",
) -> Dict[str, Optional[str]]:
    """Render and write each view to ``directory``; return ``{view: path|None}``.

    Skipped views (``None`` bytes) are written as ``None`` paths, not files.
    Creates ``directory`` if needed. Never raises for a missing renderer.
    """
    result = render(backend, views=views, size=size, fmt=fmt)
    os.makedirs(directory, exist_ok=True)
    ext = "png" if result.fmt == "png" else "svg"
    paths: Dict[str, Optional[str]] = {}
    for name, data in result.images.items():
        if data is None:
            paths[name] = None
            continue
        path = os.path.join(directory, f"{name}.{ext}")
        with open(path, "wb") as fh:
            fh.write(data)
        paths[name] = path
    return paths
