"""Mesh colorization: per-vertex colours propagated across the surface.

A stylization stage may assign each mesh vertex a colour and then propagate
those vertex colours over the entire mesh surface using an interpolation-based
differentiable renderer.  The stage that *chooses* the colours (for example a
neural field driven by a text description) is learned and external, but the
colour-propagation math -- barycentric interpolation of per-vertex colours to
any point on a triangle -- is deterministic and classical.

This module implements:

* **barycentric coordinates** of a point inside (or projected onto) a triangle;
* **surface colour sampling**: interpolate per-vertex RGB colours at an
  arbitrary surface point via those barycentric weights, the exact operation an
  interpolation renderer performs per fragment;
* **face-centroid colours** and a whole-mesh **average colour**;
* a small deterministic **named-colour palette** (text-token -> RGB) so a colour
  word from a text prompt maps to a concrete RGB, plus nearest-palette naming.

Unlike ``fabrication.lego_coloring`` (which *snaps* averaged brick colours to a
fixed LEGO palette by squared distance), this module keeps colours continuous
and interpolates them across the surface -- the mesh-vertex colorization scheme.

Stdlib-only, deterministic.  Colours are RGB triples of floats in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]
RGB = Tuple[float, float, float]
Face = Tuple[int, int, int]

# A compact deterministic colour vocabulary for text-prompt colour words.
NAMED_COLORS: Dict[str, RGB] = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "grey": (0.5, 0.5, 0.5),
    "gray": (0.5, 0.5, 0.5),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "orange": (1.0, 0.5, 0.0),
    "purple": (0.5, 0.0, 0.5),
    "brown": (0.4, 0.26, 0.13),
}


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def barycentric(point: Vec3, tri: Sequence[Vec3]) -> Tuple[float, float, float]:
    """Barycentric coordinates ``(u, v, w)`` of ``point`` w.r.t. triangle ``tri``.

    Uses the standard projected-area (Cramer) solution; for a point in the
    triangle's plane the weights sum to 1 and are all >= 0 iff the point is
    inside.  A point off the plane yields the coordinates of its in-plane
    projection.  Raises ``ValueError`` for a degenerate (zero-area) triangle.
    """
    a, b, c = tri
    v0 = _sub(b, a)
    v1 = _sub(c, a)
    v2 = _sub(point, a)
    d00 = _dot(v0, v0)
    d01 = _dot(v0, v1)
    d11 = _dot(v1, v1)
    d20 = _dot(v2, v0)
    d21 = _dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-18:
        raise ValueError("degenerate triangle")
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u, v, w)


def interpolate_color(weights: Sequence[float], colors: Sequence[RGB]) -> RGB:
    """Blend vertex ``colors`` by barycentric ``weights`` (length 3 each)."""
    if len(weights) != 3 or len(colors) != 3:
        raise ValueError("need exactly 3 weights and 3 colors")
    r = g = b = 0.0
    for wgt, col in zip(weights, colors):
        r += wgt * col[0]
        g += wgt * col[1]
        b += wgt * col[2]
    return (r, g, b)


def sample_surface_color(
    point: Vec3, tri: Sequence[Vec3], vertex_colors: Sequence[RGB]
) -> RGB:
    """Interpolate the colour at ``point`` on triangle ``tri`` (vertex colours).

    This is the per-fragment operation of an interpolation renderer: compute the
    barycentric weights of the surface point and blend the three vertex colours.
    """
    weights = barycentric(point, tri)
    return interpolate_color(weights, vertex_colors)


def face_centroid_color(colors: Sequence[RGB]) -> RGB:
    """Colour at a triangle's centroid = mean of its three vertex colours."""
    if len(colors) != 3:
        raise ValueError("a triangle face needs exactly 3 vertex colours")
    return (
        sum(c[0] for c in colors) / 3.0,
        sum(c[1] for c in colors) / 3.0,
        sum(c[2] for c in colors) / 3.0,
    )


def mesh_average_color(
    faces: Sequence[Face], vertex_colors: Sequence[RGB]
) -> RGB:
    """Unweighted mean of per-face centroid colours over the whole mesh."""
    if not faces:
        raise ValueError("mesh has no faces")
    r = g = b = 0.0
    for a, bb, c in faces:
        col = face_centroid_color(
            (vertex_colors[a], vertex_colors[bb], vertex_colors[c])
        )
        r += col[0]
        g += col[1]
        b += col[2]
    n = len(faces)
    return (r / n, g / n, b / n)


def color_from_prompt(token: str, *, default: RGB = (0.5, 0.5, 0.5)) -> RGB:
    """Map a colour word (e.g. from a text prompt) to an RGB; ``default`` if unknown."""
    return NAMED_COLORS.get(token.strip().lower(), default)


def nearest_named_color(rgb: RGB) -> str:
    """Name of the palette colour closest to ``rgb`` by squared distance.

    Ties resolve to the alphabetically-first name for determinism.
    """
    best_name = None
    best_d = None
    for name in sorted(NAMED_COLORS):
        pr, pg, pb = NAMED_COLORS[name]
        d = (rgb[0] - pr) ** 2 + (rgb[1] - pg) ** 2 + (rgb[2] - pb) ** 2
        if best_d is None or d < best_d:
            best_d = d
            best_name = name
    return best_name
