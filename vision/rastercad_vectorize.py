"""rastercad_vectorize -- deterministic raster-sketch -> vector primitive extraction.

RECAD (Li et al., "Revisiting CAD Model Generation by Learning Raster Sketch")
generates a sketch as a **binary raster image** and then, to extrude it, must
convert that raster back to *vector contours / primitives* -- the paper does this
with Teh-Chin chain approximation followed by Douglas-Peucker simplification
(Sec. "Extrusion with Raster Sketch", "Contour extraction").  The learned model
produces the raster; the raster->vector conversion is a deterministic geometric
procedure.

This module implements that deterministic conversion as primitive fitting.  It
labels the ink of a binary sketch canvas into connected components (stroke
tracing), then fits each component to the best-matching CAD primitive -- a
straight **line**, a full **circle**, or a circular **arc** -- by comparing the
residual of a total-least-squares line fit against an algebraic (Kasa) circle
fit, and using the component's angular coverage to separate a closed circle from
an open arc.  Coordinates are emitted in a normalised ``[0, 1] x [0, 1]`` canvas
(matching :mod:`drawings.picasso_rasterizer`) so the extracted primitives can be
re-rasterised or measured directly.

Pure stdlib, fully deterministic (no randomness, no wall clock).  Input is a
row-major binary ``list[list[int]]`` (1 = ink).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


Grid = list[list[int]]
Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Fitted-primitive result types (normalised [0, 1] canvas coordinates).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineFit:
    """A straight-line primitive from ``start`` to ``end`` with RMS ``residual``."""

    start: Point
    end: Point
    residual: float
    size: int


@dataclass(frozen=True)
class CircleFit:
    """A full-circle primitive: ``center`` + ``radius``, RMS ``residual``."""

    center: Point
    radius: float
    residual: float
    size: int


@dataclass(frozen=True)
class ArcFit:
    """A circular arc primitive through ``start``, ``mid``, ``end`` (in order)."""

    start: Point
    mid: Point
    end: Point
    center: Point
    radius: float
    residual: float
    size: int


Primitive = LineFit | CircleFit | ArcFit


# ---------------------------------------------------------------------------
# Connected-component labelling (deterministic stroke tracing).
# ---------------------------------------------------------------------------


def connected_components(
    grid: Grid, connectivity: int = 8
) -> list[list[tuple[int, int]]]:
    """Label ink pixels into connected components.

    Returns a list of components, each a list of ``(row, col)`` pixels.  Pixels
    and components are visited in deterministic row-major scan order.
    ``connectivity`` is 4 or 8.
    """

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if not grid or not grid[0]:
        return []
    height = len(grid)
    width = len(grid[0])
    for row in grid:
        if len(row) != width:
            raise ValueError("grid rows must all have the same width")
    if connectivity == 4:
        neigh = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        neigh = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        )
    seen = [[False] * width for _ in range(height)]
    components: list[list[tuple[int, int]]] = []
    for r0 in range(height):
        for c0 in range(width):
            if grid[r0][c0] != 1 or seen[r0][c0]:
                continue
            stack = [(r0, c0)]
            seen[r0][c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in neigh:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if grid[nr][nc] == 1 and not seen[nr][nc]:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
            comp.sort()
            components.append(comp)
    return components


# ---------------------------------------------------------------------------
# Coordinate mapping (pixel centres <-> normalised canvas).
# ---------------------------------------------------------------------------


def _pixel_to_canvas(r: int, c: int, height: int, width: int) -> Point:
    x = c / (width - 1) if width > 1 else 0.0
    y = r / (height - 1) if height > 1 else 0.0
    return x, y


# ---------------------------------------------------------------------------
# Line fitting via total least squares (PCA of the point cloud).
# ---------------------------------------------------------------------------


def _principal_axis(pts: list[Point]) -> tuple[Point, Point, float, float]:
    """Return ``(centroid, unit_dir, lambda_major, lambda_minor)`` via 2x2 PCA."""

    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    sxx = syy = sxy = 0.0
    for x, y in pts:
        dx, dy = x - cx, y - cy
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= n
    syy /= n
    sxy /= n
    # Eigen-decomposition of the symmetric 2x2 covariance [[sxx, sxy],[sxy, syy]].
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = max(tr * tr / 4.0 - det, 0.0)
    root = math.sqrt(disc)
    lam_major = tr / 2.0 + root
    lam_minor = tr / 2.0 - root
    # Eigenvector for the major eigenvalue.
    if abs(sxy) > 1e-12:
        vx, vy = lam_major - syy, sxy
    elif sxx >= syy:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = 0.0, 1.0
    norm = math.hypot(vx, vy)
    if norm < 1e-12:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = vx / norm, vy / norm
    return (cx, cy), (vx, vy), lam_major, lam_minor


def fit_line(pts: list[Point]) -> LineFit:
    """Fit a straight segment to ``pts`` by total least squares."""

    if not pts:
        raise ValueError("cannot fit a line to zero points")
    if len(pts) == 1:
        p = pts[0]
        return LineFit(start=p, end=p, residual=0.0, size=1)
    (cx, cy), (vx, vy), _lam_major, lam_minor = _principal_axis(pts)
    # Project points onto the principal axis; endpoints are the extremes.
    tmin = math.inf
    tmax = -math.inf
    for x, y in pts:
        t = (x - cx) * vx + (y - cy) * vy
        if t < tmin:
            tmin = t
        if t > tmax:
            tmax = t
    start = (cx + tmin * vx, cy + tmin * vy)
    end = (cx + tmax * vx, cy + tmax * vy)
    # RMS perpendicular residual == sqrt(minor eigenvalue).
    residual = math.sqrt(max(lam_minor, 0.0))
    return LineFit(start=start, end=end, residual=residual, size=len(pts))


# ---------------------------------------------------------------------------
# Circle fitting via the algebraic (Kasa) method.
# ---------------------------------------------------------------------------


def _solve3(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve a 3x3 linear system by Gaussian elimination with partial pivoting."""

    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    n = 3
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-15:
            return None
        m[col], m[piv] = m[piv], m[col]
        inv = 1.0 / m[col][col]
        for j in range(col, n + 1):
            m[col][j] *= inv
        for r in range(n):
            if r != col and abs(m[r][col]) > 0.0:
                f = m[r][col]
                for j in range(col, n + 1):
                    m[r][j] -= f * m[col][j]
    return [m[i][n] for i in range(n)]


def fit_circle(pts: list[Point]) -> CircleFit | None:
    """Fit a circle to ``pts`` (Kasa algebraic fit); ``None`` if ill-conditioned.

    Solves ``x^2 + y^2 + A x + B y + C = 0`` in the least-squares sense, giving
    ``center = (-A/2, -B/2)`` and ``radius = sqrt(center^2 - C)``.
    """

    if len(pts) < 3:
        return None
    sx = sy = sxx = syy = sxy = sxz = syz = sz = 0.0
    n = len(pts)
    for x, y in pts:
        z = x * x + y * y
        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y
        sxz += x * z
        syz += y * z
        sz += z
    # Normal equations for [A, B, C].
    mat = [
        [sxx, sxy, sx],
        [sxy, syy, sy],
        [sx, sy, float(n)],
    ]
    rhs = [-sxz, -syz, -sz]
    sol = _solve3(mat, rhs)
    if sol is None:
        return None
    a_c, b_c, c_c = sol
    ux = -a_c / 2.0
    uy = -b_c / 2.0
    r2 = ux * ux + uy * uy - c_c
    if r2 <= 0.0:
        return None
    radius = math.sqrt(r2)
    # RMS radial residual.
    acc = 0.0
    for x, y in pts:
        acc += (math.hypot(x - ux, y - uy) - radius) ** 2
    residual = math.sqrt(acc / n)
    return CircleFit(center=(ux, uy), radius=radius, residual=residual, size=n)


def _angular_span(pts: list[Point], center: Point) -> tuple[float, float, float]:
    """Return ``(coverage, min_angle, max_angle)`` of ``pts`` about ``center``.

    ``coverage`` is the largest angular extent not spanned by the biggest gap,
    i.e. ``2*pi`` minus the maximum gap between consecutive sorted angles.
    """

    cx, cy = center
    angles = sorted(
        (math.atan2(y - cy, x - cx) % (2.0 * math.pi)) for x, y in pts
    )
    if len(angles) < 2:
        return 0.0, angles[0] if angles else 0.0, angles[0] if angles else 0.0
    max_gap = 0.0
    for i in range(len(angles)):
        nxt = angles[(i + 1) % len(angles)]
        gap = nxt - angles[i]
        if i == len(angles) - 1:
            gap = (angles[0] + 2.0 * math.pi) - angles[i]
        if gap > max_gap:
            max_gap = gap
    coverage = 2.0 * math.pi - max_gap
    return coverage, angles[0], angles[-1]


# ---------------------------------------------------------------------------
# Classification / vectorisation.
# ---------------------------------------------------------------------------


def classify_component(
    comp: list[tuple[int, int]],
    height: int,
    width: int,
    circle_closed_coverage: float = 5.0,
    circle_gain: float = 1.5,
) -> Primitive:
    """Fit and classify one component to a line, circle or arc primitive.

    A circle/arc is preferred over a line only when its radial residual is at
    least ``circle_gain`` times smaller than the line residual.  When curved, a
    component whose angular coverage exceeds ``circle_closed_coverage`` radians is
    a closed :class:`CircleFit`; otherwise an open :class:`ArcFit`.
    """

    if not comp:
        raise ValueError("cannot classify an empty component")
    pts = [_pixel_to_canvas(r, c, height, width) for r, c in comp]
    line = fit_line(pts)
    circle = fit_circle(pts)
    if circle is None:
        return line
    # Prefer the curved fit only if it is materially better than the line.
    curved_better = circle.residual * circle_gain < line.residual + 1e-12
    if not curved_better:
        return line
    coverage, _amin, _amax = _angular_span(pts, circle.center)
    if coverage >= circle_closed_coverage:
        return circle
    # Build an arc: pick start/end as the two points bounding the covered arc,
    # and a mid point near the middle of the traversal for orientation.
    cx, cy = circle.center
    ordered = sorted(
        pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx) % (2.0 * math.pi)
    )
    # Rotate ordering so the largest gap sits at the seam (arc endpoints).
    angles = [math.atan2(p[1] - cy, p[0] - cx) % (2.0 * math.pi) for p in ordered]
    seam = 0
    max_gap = -1.0
    for i in range(len(angles)):
        nxt = angles[(i + 1) % len(angles)]
        gap = (nxt - angles[i]) % (2.0 * math.pi)
        if gap > max_gap:
            max_gap = gap
            seam = (i + 1) % len(angles)
    seq = ordered[seam:] + ordered[:seam]
    start = seq[0]
    end = seq[-1]
    mid = seq[len(seq) // 2]
    return ArcFit(
        start=start,
        mid=mid,
        end=end,
        center=circle.center,
        radius=circle.radius,
        residual=circle.residual,
        size=len(pts),
    )


def vectorize(
    grid: Grid,
    connectivity: int = 8,
    min_size: int = 2,
    circle_closed_coverage: float = 5.0,
    circle_gain: float = 1.5,
) -> list[Primitive]:
    """Vectorise a binary sketch canvas into a list of fitted CAD primitives.

    Components smaller than ``min_size`` pixels are discarded as noise.
    Primitives are returned in deterministic scan order of their components.
    """

    if not grid or not grid[0]:
        return []
    height = len(grid)
    width = len(grid[0])
    comps = connected_components(grid, connectivity=connectivity)
    out: list[Primitive] = []
    for comp in comps:
        if len(comp) < min_size:
            continue
        out.append(
            classify_component(
                comp,
                height,
                width,
                circle_closed_coverage=circle_closed_coverage,
                circle_gain=circle_gain,
            )
        )
    return out
