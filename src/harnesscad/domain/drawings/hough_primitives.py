"""Hough-transform line and circle detection from 2D edge points.

"Advanced Knowledge Extraction of Physical Design Drawings ... conversion to CAD
formats" (Jesher Joshua, Ragav, Syed Ibrahim, VIT Chennai) digitises engineering
drawings by detecting geometric primitives from the edge image. Its deep-learning
stages (YOLO ROI/ornament detection, OCR) are out of scope, but the primitive-extraction
core is the classical, fully deterministic Hough transform (their Sec. 3.1.4): a voting
scheme in a discretised parameter space that recovers lines and circles from an edge
point set and is "robust against noise and partial occlusions."

* **Line detection** votes each edge point into the ``(rho, theta)`` polar-line
  accumulator (``rho = x cos(theta) + y sin(theta)``); accumulator peaks are the most
  supported lines. :func:`hough_lines`.
* **Circle detection** (their ``get_circles`` / HoughCircles) votes over
  ``(cx, cy, r)`` for a supplied radius range; peaks are circle centres.
  :func:`hough_circles`.

This is the deterministic feeder from a drawing's edge pixels to CAD sketch primitives,
complementing :mod:`harnesscad.domain.drawings.primitive_points` (which already carries
per-primitive point features) with the raw voting-based detector the paper describes.
Stdlib only; deterministic given a fixed discretisation and stable peak ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "HoughLine",
    "HoughCircle",
    "hough_lines",
    "hough_circles",
    "point_line_distance",
]


@dataclass(frozen=True)
class HoughLine:
    """A detected line in polar form ``rho = x cos(theta) + y sin(theta)``."""

    rho: float
    theta_rad: float
    votes: int


@dataclass(frozen=True)
class HoughCircle:
    """A detected circle centre and radius with its vote count."""

    cx: float
    cy: float
    radius: float
    votes: int


def point_line_distance(px: float, py: float, rho: float, theta_rad: float) -> float:
    """Perpendicular distance from a point to a polar line."""
    return abs(px * math.cos(theta_rad) + py * math.sin(theta_rad) - rho)


def hough_lines(
    points: Sequence[Sequence[float]],
    *,
    theta_steps: int = 180,
    rho_res: float = 1.0,
    threshold: int = 2,
    max_lines: int = 0,
) -> tuple[HoughLine, ...]:
    """Detect lines from 2D edge points by Hough voting.

    ``theta_steps`` discretises ``[0, pi)``; ``rho_res`` is the rho-bin width;
    ``threshold`` is the minimum votes to report a line; ``max_lines`` (>0) caps the
    returned peaks. Peaks are sorted by votes descending, then by ``(theta, rho)`` for
    deterministic ties.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    if not pts:
        return ()
    thetas = [math.pi * k / theta_steps for k in range(theta_steps)]
    cos_t = [math.cos(t) for t in thetas]
    sin_t = [math.sin(t) for t in thetas]

    acc: dict[tuple[int, int], int] = {}
    for (x, y) in pts:
        for ti in range(theta_steps):
            rho = x * cos_t[ti] + y * sin_t[ti]
            ri = int(round(rho / rho_res))
            key = (ti, ri)
            acc[key] = acc.get(key, 0) + 1

    peaks = [
        HoughLine(rho=ri * rho_res, theta_rad=thetas[ti], votes=v)
        for (ti, ri), v in acc.items()
        if v >= threshold
    ]
    peaks.sort(key=lambda h: (-h.votes, h.theta_rad, h.rho))
    if max_lines > 0:
        peaks = peaks[:max_lines]
    return tuple(peaks)


def hough_circles(
    points: Sequence[Sequence[float]],
    radii: Iterable[float],
    *,
    center_res: float = 1.0,
    threshold: int = 3,
    angle_steps: int = 60,
    max_circles: int = 0,
) -> tuple[HoughCircle, ...]:
    """Detect circles of the supplied radii from edge points by Hough voting.

    For each candidate radius, every edge point votes for the ring of possible centres
    at that radius (sampled at ``angle_steps`` angles); centres are binned at
    ``center_res``. Peaks with at least ``threshold`` votes are returned, sorted by
    votes descending then by ``(radius, cx, cy)``.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    radii = [float(r) for r in radii]
    if not pts or not radii:
        return ()
    angles = [2.0 * math.pi * k / angle_steps for k in range(angle_steps)]
    cos_a = [math.cos(a) for a in angles]
    sin_a = [math.sin(a) for a in angles]

    acc: dict[tuple[float, int, int], int] = {}
    for r in radii:
        for (x, y) in pts:
            for k in range(angle_steps):
                cx = x - r * cos_a[k]
                cy = y - r * sin_a[k]
                key = (r, int(round(cx / center_res)), int(round(cy / center_res)))
                acc[key] = acc.get(key, 0) + 1

    peaks = [
        HoughCircle(cx=ci * center_res, cy=cj * center_res, radius=r, votes=v)
        for (r, ci, cj), v in acc.items()
        if v >= threshold
    ]
    peaks.sort(key=lambda c: (-c.votes, c.radius, c.cx, c.cy))
    if max_circles > 0:
        peaks = peaks[:max_circles]
    return tuple(peaks)
