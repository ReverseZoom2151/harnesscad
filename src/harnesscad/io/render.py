"""Software rasteriser: a shaded-solid image codec for meshes (stdlib-only PNG).

The harness could already produce real geometry (an SDF backend, marching cubes,
a watertight manifold) and real vector drawings (feature edges, hidden lines,
dimensions) -- but every picture it could make was a *wireframe*. A wireframe
looks like a sketch; CAD looks like a shaded solid with its feature edges drawn
over the top. This module is the missing viewport.

It is a complete, deterministic, dependency-free 3D renderer:

*   **Camera** -- orthographic and perspective, :func:`look_at`, and named preset
    views (``front``, ``top``, ``side``, ``iso``, and a 3/4 ``hero`` view). The
    camera is fitted to the model's bounding box, so any part frames itself.
*   **Rasteriser** -- triangle scan-conversion with a z-buffer, optional backface
    culling, and perspective-correct depth/attribute interpolation when the
    projection is perspective (linear when it is orthographic, which is exact).
*   **Shading** -- Lambert diffuse plus ambient plus an optional Blinn-Phong
    specular, one or more directional lights (camera- or world-anchored), and
    either per-face flat shading or per-vertex Gouraud shading over
    area-weighted, CREASE-AWARE vertex normals: a corner only averages the faces
    on its own side of a crease, so a box stays flat-faced with crisp edges while
    the wall of a hole still shades smoothly. Lighting is two-sided, so a mesh
    with a flipped facet still shades.
*   **Edge overlay** -- the feature edges from :func:`harnesscad.io.drawing.
    feature_edges` (the drawing route's crease/boundary set, not a second
    implementation) drawn over the solid and depth-tested against the z-buffer,
    so only the visible ones appear. That is what a CAD viewport shows.
*   **Anti-aliasing** -- supersample at 2x/3x and box-downsample. Deterministic.
*   **PNG** -- an 8-bit RGB/RGBA writer built from ``zlib`` and ``struct``, plus
    :func:`png_size`, the header decoder the image-QC modules needed.

The renderer also un-blocks the modules that were stuck for want of one:
:func:`three_view` drives ``agents.generation.three_view``'s CADSmith camera
spec, :func:`visibility_audit` feeds ``eval.quality.perception.view_coverage``
with a real per-view visible-face set taken from the z-buffer, and :func:`qc`
gives ``data.dataengine.annotation.visual_qc`` an image it can actually decode.

Stdlib only (``math``, ``zlib``, ``struct``); no wall clock, no randomness --
the same inputs always produce byte-identical output.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.agents.generation import three_view as three_view_spec
from harnesscad.io import gate
from harnesscad.data.dataengine.annotation import visual_qc
from harnesscad.eval.quality.perception import view_coverage
from harnesscad.io import drawing as drawing_route

__all__ = [
    "RenderError",
    "Material",
    "Light",
    "Camera",
    "VIEW_PRESETS",
    "DEFAULT_MATERIAL",
    "DEFAULT_LIGHTS",
    "look_at",
    "preset_camera",
    "Framebuffer",
    "write_png",
    "png_size",
    "render",
    "render_session",
    "three_view",
    "visibility_audit",
    "qc",
]

Vec3 = Tuple[float, float, float]
RGB = Tuple[int, int, int]

_EPS = 1e-12


class RenderError(Exception):
    """The mesh or the render options cannot produce an image."""


# ---------------------------------------------------------------------------
# small vector helpers
# ---------------------------------------------------------------------------

def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a: Vec3) -> Vec3:
    m = math.sqrt(_dot(a, a))
    if m < _EPS:
        return (0.0, 0.0, 0.0)
    return (a[0] / m, a[1] / m, a[2] / m)


# ---------------------------------------------------------------------------
# materials + lights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Material:
    """A surface: base colour, ambient term, and an optional Blinn-Phong lobe."""

    base_color: RGB = (176, 188, 202)
    ambient: float = 0.30
    specular: float = 0.28
    shininess: float = 36.0


@dataclass(frozen=True)
class Light:
    """A directional light.

    ``direction`` points FROM the surface TOWARD the light. ``space`` is
    ``"camera"`` (the default: the light rides with the camera, so every preset
    view is lit the same way) or ``"world"`` (anchored to the model).
    """

    direction: Vec3 = (0.0, 0.0, 1.0)
    color: RGB = (255, 255, 255)
    intensity: float = 1.0
    space: str = "camera"


#: A key light over the viewer's shoulder plus a cool fill from the other side.
DEFAULT_LIGHTS: Tuple[Light, ...] = (
    Light(direction=(-0.35, 0.55, 0.76), color=(255, 253, 246), intensity=0.82),
    Light(direction=(0.62, -0.18, 0.42), color=(206, 220, 240), intensity=0.30),
)

DEFAULT_MATERIAL = Material()


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Camera:
    """An orthographic or perspective camera.

    ``eye``/``target``/``up`` define the view basis. ``ortho_height`` is the
    world-space height of the orthographic film; ``fov_deg`` the vertical field
    of view of the perspective one. :func:`fit` fills whichever is needed.
    """

    eye: Vec3
    target: Vec3
    up: Vec3 = (0.0, 0.0, 1.0)
    projection: str = "orthographic"
    ortho_height: float = 2.0
    fov_deg: float = 32.0

    def basis(self) -> Tuple[Vec3, Vec3, Vec3]:
        """(right, up, forward) -- an orthonormal, right-handed view basis."""
        forward = _norm(_sub(self.target, self.eye))
        if forward == (0.0, 0.0, 0.0):
            raise RenderError("camera eye and target coincide")
        up = _norm(self.up)
        right = _norm(_cross(forward, up))
        if right == (0.0, 0.0, 0.0):
            # up is parallel to the view direction: pick any perpendicular axis
            alt = (1.0, 0.0, 0.0) if abs(forward[2]) > 0.9 else (0.0, 0.0, 1.0)
            right = _norm(_cross(forward, alt))
        true_up = _cross(right, forward)
        return right, true_up, forward


def look_at(eye: Sequence[float], target: Sequence[float],
            up: Sequence[float] = (0.0, 0.0, 1.0),
            projection: str = "orthographic",
            ortho_height: float = 2.0,
            fov_deg: float = 32.0) -> Camera:
    """The camera looking from ``eye`` at ``target``."""
    if projection not in ("orthographic", "perspective"):
        raise RenderError("projection must be 'orthographic' or 'perspective', "
                          "got %r" % (projection,))
    return Camera(eye=tuple(float(c) for c in eye),
                  target=tuple(float(c) for c in target),
                  up=tuple(float(c) for c in up),
                  projection=projection,
                  ortho_height=float(ortho_height),
                  fov_deg=float(fov_deg))


#: Named preset view directions: the unit vector FROM the part TOWARD the eye,
#: in the repo's Z-up convention (the same one ``io.drawing`` projects with).
VIEW_PRESETS: Dict[str, Vec3] = {
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
    "side": (1.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    # the standard isometric eye
    "iso": (1.0, -1.0, 1.0),
    # a 3/4 "hero" view: lower elevation, off-axis azimuth -- the angle a CAD
    # package uses for a product shot, so the top face, one long side and one
    # short side are all legible at once.
    "hero": (0.78, -1.0, 0.52),
}


def preset_camera(view: str, center: Vec3, radius: float,
                  projection: str = "orthographic",
                  fov_deg: float = 32.0) -> Camera:
    """A camera for a named preset view, placed to see a sphere (center, radius)."""
    key = str(view).lower()
    if key not in VIEW_PRESETS:
        raise RenderError("unknown view %r (known: %s)"
                          % (view, ", ".join(sorted(VIEW_PRESETS))))
    direction = _norm(VIEW_PRESETS[key])
    radius = max(float(radius), _EPS)
    if projection == "perspective":
        distance = radius / max(math.sin(math.radians(fov_deg) * 0.5), 1e-3) * 1.35
    else:
        distance = radius * 4.0
    eye = _add(center, _scale(direction, distance))
    up = (0.0, 0.0, 1.0)
    if abs(direction[2]) > 0.999:            # looking straight down/up
        up = (0.0, 1.0, 0.0)
    return look_at(eye, center, up, projection=projection,
                   ortho_height=2.0 * radius, fov_deg=fov_deg)


# ---------------------------------------------------------------------------
# PNG (zlib + struct, nothing else)
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    return (struct.pack(">I", len(payload)) + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))


def encode_png(pixels: Sequence[int], width: int, height: int,
               alpha: bool = False, compress_level: int = 6) -> bytes:
    """8-bit RGB (or RGBA) PNG bytes from a flat channel-interleaved buffer."""
    channels = 4 if alpha else 3
    expected = width * height * channels
    if len(pixels) != expected:
        raise RenderError("pixel buffer is %d values, expected %d (%dx%d x%d)"
                          % (len(pixels), expected, width, height, channels))
    stride = width * channels
    raw = bytearray()
    for y in range(height):
        raw.append(0)                                    # filter type 0 (None)
        raw += bytes(pixels[y * stride:(y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6 if alpha else 2, 0, 0, 0)
    return (_PNG_MAGIC
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(bytes(raw), compress_level))
            + _chunk(b"IEND", b""))


def write_png(path: str, pixels: Sequence[int], width: int, height: int,
              alpha: bool = False) -> str:
    """Write the buffer to ``path`` as a PNG. Returns the path."""
    data = encode_png(pixels, width, height, alpha=alpha)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def png_size(payload: Any) -> Tuple[int, int]:
    """(width, height) from a PNG's IHDR -- bytes or a path. Raises on non-PNG.

    This is the whole PNG "decoder" the image-QC modules need: they check that
    an asset is a real image of a sane size, not that its pixels say anything.
    """
    if isinstance(payload, (bytes, bytearray)):
        data = bytes(payload[:33])
    else:
        with open(str(payload), "rb") as fh:
            data = fh.read(33)
    if len(data) < 24 or data[:8] != _PNG_MAGIC or data[12:16] != b"IHDR":
        raise RenderError("not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    return (int(width), int(height))


# ---------------------------------------------------------------------------
# framebuffer
# ---------------------------------------------------------------------------

class Framebuffer:
    """An RGB colour buffer plus a z-buffer, both flat lists.

    The z-buffer stores *closeness*: bigger is nearer, and the value is affine
    in screen space for BOTH projections (``-depth`` for an orthographic camera,
    ``1/depth`` for a perspective one), so a plain barycentric interpolation of
    it is exact in both cases.
    """

    __slots__ = ("width", "height", "color", "depth", "face")

    def __init__(self, width: int, height: int, background: RGB) -> None:
        self.width = int(width)
        self.height = int(height)
        n = self.width * self.height
        r, g, b = (int(c) & 0xFF for c in background)
        self.color = bytearray(bytes((r, g, b)) * n)
        self.depth = [-1e300] * n
        self.face = [-1] * n              # id buffer: which triangle won

    def downsample(self, factor: int) -> bytearray:
        """Box-filter the colour buffer down by ``factor`` (deterministic)."""
        factor = int(factor)
        if factor <= 1:
            return bytearray(self.color)
        w, h = self.width // factor, self.height // factor
        src = self.color
        out = bytearray(w * h * 3)
        area = factor * factor
        half = area // 2
        for y in range(h):
            for x in range(w):
                r = g = b = 0
                base_y = y * factor
                base_x = x * factor
                for sy in range(factor):
                    row = ((base_y + sy) * self.width + base_x) * 3
                    for sx in range(factor):
                        i = row + sx * 3
                        r += src[i]
                        g += src[i + 1]
                        b += src[i + 2]
                o = (y * w + x) * 3
                out[o] = (r + half) // area
                out[o + 1] = (g + half) // area
                out[o + 2] = (b + half) // area
        return out


# ---------------------------------------------------------------------------
# geometry preparation
# ---------------------------------------------------------------------------

def _as_indexed(mesh: Any) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    """Anything mesh-shaped -> welded (vertices, triangles)."""
    # Imported here (not at module scope) because the format registry imports
    # this module for the .png codec; the lazy import keeps that one-directional.
    from harnesscad.io.formats import registry as fmt

    if isinstance(mesh, (list, tuple)) and len(mesh) == 2 and mesh[0] and mesh[1] \
            and not isinstance(mesh[0], (int, float)):
        verts = [tuple(float(c) for c in v) for v in mesh[0]]
        faces: List[Tuple[int, int, int]] = []
        for f in mesh[1]:
            ids = [int(i) for i in f]
            for k in range(1, len(ids) - 1):
                faces.append((ids[0], ids[k], ids[k + 1]))
        return verts, faces
    try:
        neutral = fmt.to_mesh(mesh)
    except fmt.FormatError as exc:
        raise RenderError(str(exc)) from exc
    verts, tris = neutral.indexed()
    return [tuple(float(c) for c in v) for v in verts], [tuple(t) for t in tris]


def _bounds(verts: Sequence[Vec3]) -> Tuple[Vec3, float]:
    lo = [min(v[i] for v in verts) for i in range(3)]
    hi = [max(v[i] for v in verts) for i in range(3)]
    center = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
    radius = 0.5 * math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))
    return center, max(radius, _EPS)


def _face_normals(verts, faces) -> List[Vec3]:
    out = []
    for (i0, i1, i2) in faces:
        n = _cross(_sub(verts[i1], verts[i0]), _sub(verts[i2], verts[i0]))
        out.append(n)                     # unnormalised: its length is 2*area
    return out


def _corner_normals(verts, faces, raw_normals,
                    crease_angle: float = 25.0) -> List[Tuple[Vec3, Vec3, Vec3]]:
    """Area-weighted, CREASE-AWARE vertex normals -- one per face corner.

    Averaging every face around a vertex is what makes a naive Gouraud box look
    like a pillow: the 90-degree corners get a normal that points diagonally, so
    the flat faces shade as gradients and the sharp edges read as bevels. CAD
    does not look like that -- a planar face is flat and its edges are crisp.

    So a corner only averages the faces that are genuinely part of the same
    smooth surface as the face it belongs to: neighbours whose normal is within
    ``crease_angle`` of this face's normal. Across a crease the corner keeps its
    own face's normal. A box therefore stays perfectly flat-faced while the
    cylindrical wall of a hole still shades smoothly -- the same crease angle
    that :func:`drawing.feature_edges` draws is the one that shades here, so the
    edges the overlay draws are exactly the edges the shading breaks at.
    """
    cos_limit = math.cos(math.radians(max(0.0, float(crease_angle))))
    unit = [_norm(n) for n in raw_normals]

    around: List[List[int]] = [[] for _ in verts]
    for fi, (i0, i1, i2) in enumerate(faces):
        for i in (i0, i1, i2):
            around[i].append(fi)

    out: List[Tuple[Vec3, Vec3, Vec3]] = []
    for fi, tri in enumerate(faces):
        own = unit[fi]
        corners: List[Vec3] = []
        for i in tri:
            ax = ay = az = 0.0
            for fj in around[i]:
                # the raw normal is area-scaled, so this is an area weighting
                if _dot(unit[fj], own) >= cos_limit:
                    n = raw_normals[fj]
                    ax += n[0]
                    ay += n[1]
                    az += n[2]
            n = _norm((ax, ay, az))
            corners.append(n if n != (0.0, 0.0, 0.0) else own)
        out.append((corners[0], corners[1], corners[2]))
    return out


# ---------------------------------------------------------------------------
# shading
# ---------------------------------------------------------------------------

def _resolved_lights(lights: Sequence[Light], camera: Camera) -> List[Tuple[Vec3, Vec3, float]]:
    """(unit direction toward the light, colour in 0..1, intensity), in WORLD space."""
    right, up, forward = camera.basis()
    out = []
    for light in lights:
        d = _norm(light.direction)
        if light.space == "camera":
            # camera axes: +x right, +y up, +z toward the viewer (= -forward)
            d = _norm((right[0] * d[0] + up[0] * d[1] - forward[0] * d[2],
                       right[1] * d[0] + up[1] * d[1] - forward[1] * d[2],
                       right[2] * d[0] + up[2] * d[1] - forward[2] * d[2]))
        elif light.space != "world":
            raise RenderError("light space must be 'camera' or 'world', got %r"
                              % (light.space,))
        col = tuple(max(0.0, min(1.0, c / 255.0)) for c in light.color)
        out.append((d, col, float(light.intensity)))
    return out


def _shade(normal: Vec3, view_dir: Vec3, material: Material,
           lights: Sequence[Tuple[Vec3, Vec3, float]]) -> Tuple[float, float, float]:
    """Lambert + ambient (+ Blinn-Phong). Two-sided: the normal faces the viewer.

    ``view_dir`` points from the surface toward the eye. Returns linear 0..1 RGB.
    """
    n = normal
    if _dot(n, view_dir) < 0.0:
        n = (-n[0], -n[1], -n[2])
    base = tuple(c / 255.0 for c in material.base_color)
    a = material.ambient
    r, g, b = base[0] * a, base[1] * a, base[2] * a
    for (ldir, lcol, power) in lights:
        lambert = _dot(n, ldir)
        if lambert <= 0.0:
            continue
        k = lambert * power
        r += base[0] * lcol[0] * k
        g += base[1] * lcol[1] * k
        b += base[2] * lcol[2] * k
        if material.specular > 0.0:
            half = _norm(_add(ldir, view_dir))
            s = _dot(n, half)
            if s > 0.0:
                s = (s ** material.shininess) * material.specular * power
                r += lcol[0] * s
                g += lcol[1] * s
                b += lcol[2] * s
    return (min(1.0, r), min(1.0, g), min(1.0, b))


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------

class _Projector:
    """World -> screen, with the closeness value the z-buffer compares."""

    def __init__(self, camera: Camera, width: int, height: int,
                 verts: Sequence[Vec3], margin: float) -> None:
        self.camera = camera
        self.width = width
        self.height = height
        self.right, self.up, self.forward = camera.basis()
        self.eye = camera.eye
        self.perspective = camera.projection == "perspective"
        # view-space coordinates of every vertex
        self.view: List[Tuple[float, float, float]] = []
        for p in verts:
            d = _sub(p, self.eye)
            self.view.append((_dot(d, self.right), _dot(d, self.up),
                              _dot(d, self.forward)))
        if not self.view:
            raise RenderError("nothing to render: the mesh has no vertices")
        depths = [v[2] for v in self.view]
        if self.perspective:
            near = min(depths)
            if near <= _EPS:
                raise RenderError("the model is behind (or through) the camera")
            fpx = (0.5 * height) / math.tan(math.radians(camera.fov_deg) * 0.5)
            planar = [(v[0] * fpx / v[2], v[1] * fpx / v[2]) for v in self.view]
        else:
            planar = [(v[0], v[1]) for v in self.view]
        # fit: scale the projected extent into the frame with a margin
        xs = [p[0] for p in planar]
        ys = [p[1] for p in planar]
        ex = max(max(xs) - min(xs), _EPS)
        ey = max(max(ys) - min(ys), _EPS)
        usable = max(0.02, 1.0 - 2.0 * float(margin))
        scale = min(width * usable / ex, height * usable / ey)
        if self.perspective:
            scale = min(scale, 1.0)          # never magnify past the real fov
        cx = 0.5 * (max(xs) + min(xs))
        cy = 0.5 * (max(ys) + min(ys))
        self.screen: List[Tuple[float, float, float]] = []
        for (px, py), depth in zip(planar, depths):
            sx = 0.5 * width + (px - cx) * scale
            sy = 0.5 * height - (py - cy) * scale
            close = (1.0 / depth) if self.perspective else (-depth)
            self.screen.append((sx, sy, close))
        closes = [s[2] for s in self.screen]
        self.close_range = max(max(closes) - min(closes), _EPS)


# ---------------------------------------------------------------------------
# the rasteriser
# ---------------------------------------------------------------------------

def _raster(fb: Framebuffer, proj: _Projector, verts, faces, raw_normals,
            corner_normals, material: Material, lights, shading: str,
            cull: bool) -> None:
    W, H = fb.width, fb.height
    color, depth, facebuf = fb.color, fb.depth, fb.face
    screen = proj.screen
    eye = proj.eye
    perspective = proj.perspective
    smooth = shading == "smooth"
    view_axis = proj.forward

    for fi, (i0, i1, i2) in enumerate(faces):
        n_raw = raw_normals[fi]
        if n_raw == (0.0, 0.0, 0.0):
            continue
        centroid = _scale(_add(_add(verts[i0], verts[i1]), verts[i2]), 1.0 / 3.0)
        if perspective:
            to_eye = _norm(_sub(eye, centroid))
        else:
            to_eye = (-view_axis[0], -view_axis[1], -view_axis[2])
        if cull and _dot(n_raw, to_eye) <= 0.0:
            continue

        a, b, c = screen[i0], screen[i1], screen[i2]
        # order the corners so the doubled signed area is positive
        area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        ids = (i0, i1, i2)
        swapped = area < 0.0
        if swapped:
            b, c = c, b
            ids = (i0, i2, i1)
            area = -area
        if area < 1e-9:
            continue
        inv_area = 1.0 / area

        min_x = max(0, int(math.floor(min(a[0], b[0], c[0]))))
        max_x = min(W - 1, int(math.ceil(max(a[0], b[0], c[0]))))
        min_y = max(0, int(math.floor(min(a[1], b[1], c[1]))))
        max_y = min(H - 1, int(math.ceil(max(a[1], b[1], c[1]))))
        if min_x > max_x or min_y > max_y:
            continue

        if smooth:
            # the corner normals follow the same winding swap as the vertices
            cn = corner_normals[fi]
            if swapped:
                cn = (cn[0], cn[2], cn[1])
            cols = []
            for i, nv in zip(ids, cn):
                if nv == (0.0, 0.0, 0.0):
                    nv = _norm(n_raw)
                vd = _norm(_sub(eye, verts[i])) if perspective else to_eye
                cols.append(_shade(nv, vd, material, lights))
            (r0, g0, b0), (r1, g1, b1), (r2, g2, b2) = cols
        else:
            fr, fg, fb_ = _shade(_norm(n_raw), to_eye, material, lights)
            flat_rgb = (int(fr * 255.0 + 0.5), int(fg * 255.0 + 0.5),
                        int(fb_ * 255.0 + 0.5))

        # edge functions (a,b,c are in positive-area order)
        e0_dx, e0_dy = -(c[1] - b[1]), (c[0] - b[0])
        e1_dx, e1_dy = -(a[1] - c[1]), (a[0] - c[0])
        e2_dx, e2_dy = -(b[1] - a[1]), (b[0] - a[0])
        px0, py0 = min_x + 0.5, min_y + 0.5
        e0_row = (c[0] - b[0]) * (py0 - b[1]) - (c[1] - b[1]) * (px0 - b[0])
        e1_row = (a[0] - c[0]) * (py0 - c[1]) - (a[1] - c[1]) * (px0 - c[0])
        e2_row = (b[0] - a[0]) * (py0 - a[1]) - (b[1] - a[1]) * (px0 - a[0])

        za, zb, zc = a[2], b[2], c[2]
        for y in range(min_y, max_y + 1):
            e0, e1, e2 = e0_row, e1_row, e2_row
            row = y * W
            for x in range(min_x, max_x + 1):
                if e0 >= 0.0 and e1 >= 0.0 and e2 >= 0.0:
                    w0 = e0 * inv_area
                    w1 = e1 * inv_area
                    w2 = e2 * inv_area
                    z = w0 * za + w1 * zb + w2 * zc
                    idx = row + x
                    if z > depth[idx]:
                        depth[idx] = z
                        facebuf[idx] = fi
                        if smooth:
                            if perspective:
                                # perspective-correct: weight by 1/depth (= z)
                                d0, d1, d2 = w0 * za, w1 * zb, w2 * zc
                                s = d0 + d1 + d2
                                if s > _EPS:
                                    d0, d1, d2 = d0 / s, d1 / s, d2 / s
                                else:
                                    d0, d1, d2 = w0, w1, w2
                            else:
                                d0, d1, d2 = w0, w1, w2
                            rr = d0 * r0 + d1 * r1 + d2 * r2
                            gg = d0 * g0 + d1 * g1 + d2 * g2
                            bb = d0 * b0 + d1 * b1 + d2 * b2
                            o = idx * 3
                            color[o] = int(min(1.0, rr) * 255.0 + 0.5)
                            color[o + 1] = int(min(1.0, gg) * 255.0 + 0.5)
                            color[o + 2] = int(min(1.0, bb) * 255.0 + 0.5)
                        else:
                            o = idx * 3
                            color[o] = flat_rgb[0]
                            color[o + 1] = flat_rgb[1]
                            color[o + 2] = flat_rgb[2]
                e0 += e0_dx
                e1 += e1_dx
                e2 += e2_dx
            e0_row += e0_dy
            e1_row += e1_dy
            e2_row += e2_dy


def _draw_edges(fb: Framebuffer, proj: _Projector, edges, color: RGB,
                thickness: int, tolerance: float) -> None:
    """Depth-tested overlay of the feature edges (visible ones only)."""
    W, H = fb.width, fb.height
    buf, depth = fb.color, fb.depth
    r, g, b = (int(c) & 0xFF for c in color)
    screen = proj.screen
    reach = max(0, int(thickness) - 1)
    for (ia, ib) in edges:
        x0, y0, z0 = screen[ia]
        x1, y1, z1 = screen[ib]
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        dx = (x1 - x0) / steps
        dy = (y1 - y0) / steps
        dz = (z1 - z0) / steps
        for s in range(steps + 1):
            px = x0 + dx * s
            py = y0 + dy * s
            pz = z0 + dz * s
            xi = int(px)
            yi = int(py)
            for oy in range(yi - reach, yi + reach + 1):
                if oy < 0 or oy >= H:
                    continue
                base = oy * W
                for ox in range(xi - reach, xi + reach + 1):
                    if ox < 0 or ox >= W:
                        continue
                    idx = base + ox
                    if pz + tolerance < depth[idx]:
                        continue                    # the solid hides this edge
                    o = idx * 3
                    buf[o] = r
                    buf[o + 1] = g
                    buf[o + 2] = b


# ---------------------------------------------------------------------------
# the public render
# ---------------------------------------------------------------------------

def render(mesh: Any,
           path: Optional[str] = None,
           view: str = "iso",
           width: int = 1200,
           height: int = 900,
           shading: str = "smooth",
           edges: bool = True,
           background: RGB = (255, 255, 255),
           ssaa: int = 2,
           projection: str = "orthographic",
           camera: Optional[Camera] = None,
           material: Optional[Material] = None,
           lights: Optional[Sequence[Light]] = None,
           cull: bool = True,
           crease_angle: float = 25.0,
           edge_color: RGB = (38, 42, 50),
           edge_width: int = 1,
           margin: float = 0.10,
           fov_deg: float = 32.0,
           return_pixels: bool = False,
           force: bool = False,
           source: Any = None) -> Any:
    """Render ``mesh`` to a PNG at ``path``; returns the path.

    ``mesh`` is anything the format registry can coerce (a ``formats.Mesh``, a
    ``(vertices, faces)`` pair, a ``Polyhedron``, an STL triangle list, a
    HarnessSession or a raw backend). ``view`` names a preset (see
    :data:`VIEW_PRESETS`) and is ignored when an explicit ``camera`` is given.
    ``shading`` is ``"flat"`` or ``"smooth"``. With ``return_pixels`` the RGB
    buffer and its size are returned instead of (or as well as) being written.

    A render is an artifact: a picture of a part is how a wrong part gets shipped
    (the dilating shell rendered *beautifully*). So whenever a ``path`` is given
    this goes through :mod:`harnesscad.io.gate` first, and an invalid model
    raises :class:`~harnesscad.io.gate.InvalidArtifact` with no PNG written.
    ``force=True`` renders anyway and drops the ``.INVALID.json`` sidecar.
    With ``path=None`` nothing leaves the harness, so nothing is gated -- the
    pixels are returned in-process (this is how :func:`three_view` composes its
    panels, and it gates the composite it actually writes).

    ``source`` is the session/backend the mesh came from, when the caller has
    already tessellated it and only holds ``(verts, faces)``. Handing it over is
    what lets the gate check the DECLARED intent (a shell that grew the part, a
    cut that added volume) and not merely the measured geometry -- exactly the
    gap the dilating-shell render fell through.
    """
    if shading not in ("flat", "smooth"):
        raise RenderError("shading must be 'flat' or 'smooth', got %r" % (shading,))
    ssaa = int(ssaa)
    if ssaa < 1 or ssaa > 4:
        raise RenderError("ssaa must be 1..4, got %r" % (ssaa,))
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise RenderError("width and height must be positive")

    verts, faces = _as_indexed(mesh)
    if not faces:
        raise RenderError("nothing to render: the mesh has no triangles")

    # THE GATE. Nothing reaches the filesystem from here without passing it.
    report = (gate.guard((verts, faces), str(path),
                         source=source if source is not None else mesh, force=force)
              if path is not None else None)

    center, radius = _bounds(verts)
    cam = camera or preset_camera(view, center, radius, projection=projection,
                                  fov_deg=fov_deg)
    mat = material or DEFAULT_MATERIAL
    lit = _resolved_lights(list(lights) if lights is not None else list(DEFAULT_LIGHTS),
                           cam)

    sw, sh = width * ssaa, height * ssaa
    proj = _Projector(cam, sw, sh, verts, margin)
    fb = Framebuffer(sw, sh, background)

    raw_normals = _face_normals(verts, faces)
    corner_normals = (_corner_normals(verts, faces, raw_normals, crease_angle)
                      if shading == "smooth" else [])
    _raster(fb, proj, verts, faces, raw_normals, corner_normals, mat, lit, shading, cull)

    if edges:
        feature = drawing_route.feature_edges((verts, faces), angle=crease_angle)
        tol = 3e-3 * proj.close_range
        _draw_edges(fb, proj, feature, edge_color,
                    max(1, int(edge_width) * ssaa), tol)

    pixels = fb.downsample(ssaa)
    if path is not None:
        write_png(str(path), pixels, width, height)
        if report is not None and not report.ok:      # forced through the gate
            gate.write_sidecar(str(path), report)
    if return_pixels or path is None:
        return {"pixels": pixels, "width": width, "height": height,
                "path": str(path) if path else None}
    return str(path)


def render_session(session_or_backend: Any, path: str, **options: Any) -> str:
    """Render a HarnessSession (or a raw GeometryBackend) straight to a PNG."""
    return render(session_or_backend, path, **options)


# ---------------------------------------------------------------------------
# The modules that were blocked on a renderer.
# ---------------------------------------------------------------------------

def three_view(mesh: Any, path: str, panel: int = 800, force: bool = False,
               **options: Any) -> dict:
    """The CADSmith three-view judge image (``agents.generation.three_view``).

    That module fixes the exact cameras -- isometric (el 35, az 45), high-angle
    rear (el 65, az 220) and front profile (el 10, az 0) -- and the exact layout
    (three square panels side by side, 2400x800 at its default panel size). It
    could not run because nothing could render a panel. It can now: the spec's
    own ``ViewSpec.direction()`` / ``camera_position()`` place our cameras, and
    the three renders are composited into one PNG.
    """
    verts, faces = _as_indexed(mesh)
    if not faces:
        raise RenderError("nothing to render: the mesh has no triangles")
    # THE GATE: the composite PNG is the artifact; the panels never touch disk.
    report = gate.guard((verts, faces), str(path), source=mesh, force=force)
    center, radius = _bounds(verts)
    panel = int(panel)
    options.pop("camera", None)
    options.pop("view", None)
    options.pop("width", None)
    options.pop("height", None)
    panels: List[Sequence[int]] = []
    names: List[str] = []
    for spec in three_view_spec.THREE_VIEWS:
        eye = spec.camera_position(center, radius * 4.0)
        up = (0.0, 0.0, 1.0)
        if abs(_norm(spec.direction())[2]) > 0.999:
            up = (0.0, 1.0, 0.0)
        cam = look_at(eye, center, up, projection="orthographic")
        out = render((verts, faces), None, camera=cam, width=panel, height=panel,
                     **options)
        panels.append(out["pixels"])
        names.append(spec.name)

    total_w = panel * len(panels)
    composed = bytearray(total_w * panel * 3)
    for pi, buf in enumerate(panels):
        for y in range(panel):
            src = y * panel * 3
            dst = (y * total_w + pi * panel) * 3
            composed[dst:dst + panel * 3] = buf[src:src + panel * 3]
    write_png(str(path), composed, total_w, panel)
    if not report.ok:                                  # forced through the gate
        gate.write_sidecar(str(path), report)
    return {"path": str(path), "views": tuple(names),
            "width": total_w, "height": panel,
            "spec_resolution": three_view_spec.render_resolution()}


def visibility_audit(mesh: Any, views: Sequence[str] = ("front", "top", "side", "iso"),
                     width: int = 320, height: int = 320,
                     pixel_threshold: int = 8, **options: Any) -> dict:
    """Which faces does a view set actually show? (``eval.quality.perception.view_coverage``)

    ``view_coverage.audit`` asks, per view, which entities are *visible* and
    which are merely *potential* (front-facing but occluded). Both answers come
    straight out of this rasteriser: the id-buffer names the triangle that won
    each pixel, so a face is visible iff it owns at least one pixel, and
    potential iff it faces the camera at all. Faces that win fewer than
    ``pixel_threshold`` pixels in every view are reported as thin features.
    """
    verts, faces = _as_indexed(mesh)
    if not faces:
        raise RenderError("nothing to render: the mesh has no triangles")
    center, radius = _bounds(verts)
    raw_normals = _face_normals(verts, faces)

    view_records: List[dict] = []
    pixels_by_face: Dict[int, int] = {}
    for angle, name in enumerate(views):
        cam = preset_camera(name, center, radius)
        proj = _Projector(cam, int(width), int(height), verts, 0.08)
        fb = Framebuffer(int(width), int(height), (255, 255, 255))
        _raster(fb, proj, verts, faces, raw_normals, [], DEFAULT_MATERIAL,
                _resolved_lights(list(DEFAULT_LIGHTS), cam), "flat", True)
        counts: Dict[int, int] = {}
        for fi in fb.face:
            if fi >= 0:
                counts[fi] = counts.get(fi, 0) + 1
        for fi, n in counts.items():
            pixels_by_face[fi] = pixels_by_face.get(fi, 0) + n
        to_eye = tuple(-c for c in proj.forward)
        potential = [fi for fi, n in enumerate(raw_normals) if _dot(n, to_eye) > 0.0]
        view_records.append({
            "id": name,
            "visible": tuple(sorted(counts)),
            "potential": tuple(potential),
            "angle": angle,
            "pixels": dict(sorted(counts.items())),
        })
    thin = [{"id": fi, "pixels": pixels_by_face.get(fi, 0)}
            for fi in range(len(faces))
            if 0 < pixels_by_face.get(fi, 0) < int(pixel_threshold)]
    audit = view_coverage.audit(tuple(range(len(faces))), view_records,
                                thin_features=thin,
                                pixel_threshold=int(pixel_threshold))
    audit = dict(audit)
    audit["faces"] = len(faces)
    audit["views"] = tuple(str(v) for v in views)
    audit["visible"] = tuple(sorted(pixels_by_face))
    return audit


def qc(path_or_payload: Any, asset_id: str = "render",
       minimum_size: Tuple[int, int] = (64, 64),
       reviewers: Sequence[str] = ("renderer", "z_buffer"),
       ambiguous: bool = False, adjudicated: bool = False):
    """QC a rendered asset (``data.dataengine.annotation.visual_qc``).

    That module was written "independent of image-decoding libraries": it takes
    an injected ``decode(payload) -> (width, height)`` and had nobody to inject.
    :func:`png_size` is that decoder, so a render can now be QC'd as a real
    image -- a truncated or non-PNG payload fails with ``decode_failed``, an
    undersized one with ``undersized``.
    """
    if isinstance(path_or_payload, (bytes, bytearray)):
        payload: Any = bytes(path_or_payload)
    else:
        with open(str(path_or_payload), "rb") as fh:
            payload = fh.read()
    return visual_qc.inspect_visual(str(asset_id), payload, decode=png_size,
                                    minimum_size=tuple(minimum_size),
                                    ambiguous=bool(ambiguous),
                                    reviewers=tuple(reviewers),
                                    adjudicated=bool(adjudicated))
