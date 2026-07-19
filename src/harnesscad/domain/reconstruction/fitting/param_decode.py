"""Map -> operation-parameter decoding for sketch-driven modelling.

The regression networks never emit CAD parameters directly: they emit
*image-space maps* over the current viewport -- a stitching-face heat map, a
guiding-curve map, and (for the offsetting operations) offset distance /
direction / sign fields -- which the modelling client turns into a concrete CAD
operation against the current shape.  That lifting step is pure geometry and is
what this module implements.

Pipeline (all deterministic, stdlib-only):

  1. :func:`decode_stitching_face` -- threshold the face heat map, keep the
     largest 4-connected component (peak region), and read the context normal
     and depth maps inside it to get a supporting plane: the mean unit normal
     plus a plane point unprojected from the region's depth-weighted centroid.
     Robust to stray high pixels because only the dominant component survives.
  2. :func:`extract_curve_pixels` -- threshold the guiding-curve map, optionally
     intersected with the stroke mask (``1 - user_stroke``), which is the same
     masking the training losses apply.
  3. :func:`lift_curve_to_plane` -- cast a camera ray through each curve pixel
     and intersect it with the stitching-face plane.  This is how a 2D stroke
     drawn "in context" becomes a 3D base/profile/offset curve.
  4. :func:`decode_offset` -- aggregate the offset distance/direction/sign fields
     over the curve pixels (mean distance, mean normalised direction, majority
     sign) into a single signed offset vector.
  5. :func:`decode_operation` -- glue: given a routed operation
     (:mod:`reconstruction.s2cadsig_op_router`) and its maps, produce an
     :class:`OperationParameters` record ready to be applied to the shape.

The camera is a simple pinhole (:class:`PinholeCamera`) or an orthographic
camera (:class:`OrthoCamera`); both expose ``ray(u, v)`` and ``unproject``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]

EPS = 1e-9


class DecodeError(ValueError):
    """Raised when a map cannot be decoded into an operation parameter."""


# ---------------------------------------------------------------------------
# small vector helpers
# ---------------------------------------------------------------------------
def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def normalize(a: Vec3) -> Vec3:
    n = _norm(a)
    if n < EPS:
        raise DecodeError("cannot normalise a zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PinholeCamera:
    """Camera-space pinhole: +z points into the screen, depth is z."""
    fx: float
    fy: float
    cx: float
    cy: float

    def ray(self, u: float, v: float) -> Vec3:
        if abs(self.fx) < EPS or abs(self.fy) < EPS:
            raise DecodeError("degenerate focal length")
        return normalize(((u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0))

    def origin(self, u: float, v: float) -> Vec3:
        return (0.0, 0.0, 0.0)

    def unproject(self, u: float, v: float, depth: float) -> Vec3:
        return (
            (u - self.cx) / self.fx * depth,
            (v - self.cy) / self.fy * depth,
            depth,
        )


@dataclass(frozen=True)
class OrthoCamera:
    """Orthographic camera: rays are parallel to +z, pixel scale ``s``."""
    scale: float = 1.0
    cx: float = 0.0
    cy: float = 0.0

    def ray(self, u: float, v: float) -> Vec3:
        return (0.0, 0.0, 1.0)

    def origin(self, u: float, v: float) -> Vec3:
        return ((u - self.cx) * self.scale, (v - self.cy) * self.scale, 0.0)

    def unproject(self, u: float, v: float, depth: float) -> Vec3:
        return ((u - self.cx) * self.scale, (v - self.cy) * self.scale, depth)


# ---------------------------------------------------------------------------
# map utilities
# ---------------------------------------------------------------------------
def _check(map_: Sequence[float], height: int, width: int, what: str) -> None:
    if height <= 0 or width <= 0:
        raise DecodeError("map dimensions must be positive")
    if len(map_) != height * width:
        raise DecodeError(
            "{} has {} values, expected {}".format(what, len(map_), height * width)
        )


def threshold_pixels(
    map_: Sequence[float], height: int, width: int, threshold: float
) -> List[Tuple[int, int]]:
    """Row-major pixels whose value is >= ``threshold``, in scan order."""
    _check(map_, height, width, "map")
    out: List[Tuple[int, int]] = []
    for r in range(height):
        base = r * width
        for c in range(width):
            if map_[base + c] >= threshold:
                out.append((r, c))
    return out


def connected_components(
    pixels: Sequence[Tuple[int, int]]
) -> List[List[Tuple[int, int]]]:
    """4-connected components of a pixel set; each component is scan-ordered."""
    remaining = set(pixels)
    comps: List[List[Tuple[int, int]]] = []
    for seed in pixels:
        if seed not in remaining:
            continue
        stack = [seed]
        remaining.discard(seed)
        comp: List[Tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in remaining:
                    remaining.discard(nb)
                    stack.append(nb)
        comp.sort()
        comps.append(comp)
    # largest first, ties broken by first pixel for determinism
    comps.sort(key=lambda comp: (-len(comp), comp[0]))
    return comps


def largest_component(
    map_: Sequence[float], height: int, width: int, threshold: float
) -> List[Tuple[int, int]]:
    comps = connected_components(threshold_pixels(map_, height, width, threshold))
    if not comps:
        raise DecodeError("no pixel above threshold {}".format(threshold))
    return comps[0]


# ---------------------------------------------------------------------------
# 1. stitching face
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StitchingFace:
    """The face of the current model the operation attaches to."""
    point: Vec3
    normal: Vec3
    pixels: Tuple[Tuple[int, int], ...]
    centroid_uv: Tuple[float, float]
    mean_depth: float
    peak_value: float

    @property
    def area_px(self) -> int:
        return len(self.pixels)

    def signed_distance(self, p: Vec3) -> float:
        return _dot(_sub(p, self.point), self.normal)


def decode_stitching_face(
    face_heatmap: Sequence[float],
    context_normal: Sequence[Vec3],
    context_depth: Sequence[float],
    height: int,
    width: int,
    camera: object,
    threshold: float = 0.5,
) -> StitchingFace:
    """Heat map + context normal/depth -> the supporting plane of the face."""
    _check(face_heatmap, height, width, "face_heatmap")
    _check(context_depth, height, width, "context_depth")
    if len(context_normal) != height * width:
        raise DecodeError("context_normal size mismatch")
    comp = largest_component(face_heatmap, height, width, threshold)

    wsum = 0.0
    nx = ny = nz = 0.0
    du = dv = 0.0
    depth_sum = 0.0
    peak = 0.0
    for (r, c) in comp:
        i = r * width + c
        w = float(face_heatmap[i])
        wsum += w
        n = context_normal[i]
        nx += n[0] * w
        ny += n[1] * w
        nz += n[2] * w
        du += c * w
        dv += r * w
        depth_sum += float(context_depth[i]) * w
        peak = max(peak, w)
    if wsum < EPS:
        raise DecodeError("degenerate face heat map weights")

    normal = normalize((nx / wsum, ny / wsum, nz / wsum))
    cu = du / wsum
    cv = dv / wsum
    mean_depth = depth_sum / wsum
    point = camera.unproject(cu, cv, mean_depth)
    return StitchingFace(
        point=point,
        normal=normal,
        pixels=tuple(comp),
        centroid_uv=(cu, cv),
        mean_depth=mean_depth,
        peak_value=peak,
    )


# ---------------------------------------------------------------------------
# 2/3. guiding curve
# ---------------------------------------------------------------------------
def extract_curve_pixels(
    curve_map: Sequence[float],
    height: int,
    width: int,
    threshold: float = 0.5,
    user_stroke: Optional[Sequence[float]] = None,
) -> List[Tuple[int, int]]:
    """Curve-map pixels above threshold, masked by ``1 - user_stroke`` if given."""
    _check(curve_map, height, width, "curve_map")
    if user_stroke is not None:
        _check(user_stroke, height, width, "user_stroke")
    out: List[Tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            i = r * width + c
            if curve_map[i] < threshold:
                continue
            if user_stroke is not None and (1.0 - float(user_stroke[i])) <= 0.0:
                continue
            out.append((r, c))
    return out


def ray_plane_intersect(
    origin: Vec3, direction: Vec3, plane_point: Vec3, plane_normal: Vec3
) -> Vec3:
    """Intersect a ray with a plane; raises when the ray is parallel to it."""
    denom = _dot(direction, plane_normal)
    if abs(denom) < 1e-8:
        raise DecodeError("ray is parallel to the stitching face")
    t = _dot(_sub(plane_point, origin), plane_normal) / denom
    return _add(origin, _scale(direction, t))


def lift_curve_to_plane(
    pixels: Sequence[Tuple[int, int]], face: StitchingFace, camera: object
) -> List[Vec3]:
    """Project image-space curve pixels onto the stitching-face plane."""
    pts: List[Vec3] = []
    for (r, c) in pixels:
        u, v = float(c), float(r)
        pts.append(
            ray_plane_intersect(camera.origin(u, v), camera.ray(u, v), face.point, face.normal)
        )
    return pts


# ---------------------------------------------------------------------------
# 4. offset
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OffsetParameters:
    distance: float
    direction: Vec3
    sign: int

    @property
    def vector(self) -> Vec3:
        return _scale(self.direction, self.distance * self.sign)


def decode_offset(
    offset_distance: Sequence[float],
    offset_direction: Sequence[Vec3],
    offset_sign: Sequence[float],
    pixels: Sequence[Tuple[int, int]],
    width: int,
) -> OffsetParameters:
    """Aggregate the offset fields over ``pixels``.

    Distance is the mean; direction is the normalised mean direction; sign is the
    majority of ``sign(v)`` over non-zero entries (ties resolve to +1).
    """
    if not pixels:
        raise DecodeError("no pixels to aggregate the offset over")
    n = len(pixels)
    dist = 0.0
    dx = dy = dz = 0.0
    pos = neg = 0
    for (r, c) in pixels:
        i = r * width + c
        dist += float(offset_distance[i])
        d = offset_direction[i]
        dx += d[0]
        dy += d[1]
        dz += d[2]
        s = float(offset_sign[i])
        if s > 0:
            pos += 1
        elif s < 0:
            neg += 1
    direction = normalize((dx / n, dy / n, dz / n))
    sign = 1 if pos >= neg else -1
    return OffsetParameters(distance=dist / n, direction=direction, sign=sign)


# ---------------------------------------------------------------------------
# 5. full operation decode
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperationParameters:
    op_name: str
    face: StitchingFace
    curve_name: str
    curve_pixels: Tuple[Tuple[int, int], ...]
    curve_points: Tuple[Vec3, ...]
    offset: Optional[OffsetParameters]

    @property
    def offset_vector(self) -> Optional[Vec3]:
        return None if self.offset is None else self.offset.vector

    def summary(self) -> Dict[str, object]:
        return {
            "op": self.op_name,
            "face_normal": self.face.normal,
            "face_point": self.face.point,
            "curve": self.curve_name,
            "curve_len": len(self.curve_points),
            "offset_distance": None if self.offset is None else self.offset.distance,
            "offset_sign": None if self.offset is None else self.offset.sign,
        }


def decode_operation(
    op_spec: object,
    maps: Dict[str, object],
    height: int,
    width: int,
    camera: object,
    face_threshold: float = 0.5,
    curve_threshold: float = 0.5,
) -> OperationParameters:
    """Decode a routed operation's maps into geometric parameters.

    ``op_spec`` is an ``OperationSpec`` from
    :mod:`reconstruction.s2cadsig_op_router`.  ``maps`` must carry
    ``face_heatmap``, ``context_normal``, ``context_depth``, the operation's
    guiding-curve map under its own name, optionally ``user_stroke``, and — when
    the op needs an offset — ``offset_distance``/``offset_direction``/
    ``offset_sign``.
    """
    needed = ["face_heatmap", "context_normal", "context_depth", op_spec.guiding_curve]
    for key in needed:
        if key not in maps:
            raise DecodeError("missing map: {}".format(key))

    face = decode_stitching_face(
        maps["face_heatmap"],
        maps["context_normal"],
        maps["context_depth"],
        height,
        width,
        camera,
        threshold=face_threshold,
    )
    pixels = extract_curve_pixels(
        maps[op_spec.guiding_curve],
        height,
        width,
        threshold=curve_threshold,
        user_stroke=maps.get("user_stroke"),
    )
    if not pixels:
        raise DecodeError("guiding curve {} is empty".format(op_spec.guiding_curve))
    points = lift_curve_to_plane(pixels, face, camera)

    offset = None
    if op_spec.needs_offset:
        for key in ("offset_distance", "offset_direction", "offset_sign"):
            if key not in maps:
                raise DecodeError("missing map: {}".format(key))
        offset = decode_offset(
            maps["offset_distance"],
            maps["offset_direction"],
            maps["offset_sign"],
            pixels,
            width,
        )
    return OperationParameters(
        op_name=op_spec.name,
        face=face,
        curve_name=op_spec.guiding_curve,
        curve_pixels=tuple(pixels),
        curve_points=tuple(points),
        offset=offset,
    )
