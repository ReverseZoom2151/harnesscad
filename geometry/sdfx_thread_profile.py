"""Standard screw-thread cross-section profiles (sdfx).

Reimplementation of the 2D thread-profile generators from deadsy/sdfx
(``sdf/screw.go``).  Each function returns the vertex polygon of one *pitch
period* of a thread tooth, in the (x, y) plane where **x** runs along the screw
axis (one pitch wide, centered on 0) and **y** is the radial direction (the
crest sits at the major radius).  Sweeping this profile helically produces a
screw; that helical sweep already exists in the harness
(:mod:`geometry.solidpy_screw_thread`).  What sdfx adds -- and what was missing
-- is the *standard profile geometry*:

* :func:`iso_thread` -- the ISO metric / Unified 60-degree V-thread with the
  standard 7/8 H external and 1/4 H internal truncations and rounded root/crest;
* :func:`acme_thread` -- the 29-degree Acme trapezoidal power-thread;
* :func:`ansi_buttress_thread` -- the ANSI 45/7 asymmetric buttress thread,
  authored across two pitch periods so it wraps continuously.

The returned vertex list is directly usable with
:func:`geometry.sdfx_polygon_sdf.polygon_sdf` (as a 2D field) or as the swept
section for a screw.  Fillet/chamfer tags from :mod:`geometry.sdfx_polygon_builder`
are resolved into explicit vertices before returning.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from geometry.sdfx_polygon_builder import Polygon

__all__ = [
    "iso_thread",
    "acme_thread",
    "ansi_buttress_thread",
]

Vec2 = Tuple[float, float]


def iso_thread(radius: float, pitch: float, external: bool = True) -> List[Vec2]:
    """2D profile of an ISO metric / UTS 60-degree thread.

    ``radius`` is the nominal major radius, ``pitch`` the axial thread spacing.
    ``external`` selects the shaft (external) vs bore (internal) truncation.
    """
    if radius <= 0 or pitch <= 0:
        raise ValueError("radius and pitch must be positive")
    theta = math.radians(30.0)
    h = pitch / (2.0 * math.tan(theta))  # height of the sharp V triangle
    r_major = radius
    r0 = r_major - (7.0 / 8.0) * h

    p = Polygon()
    if external:
        r_root = (pitch / 8.0) / math.cos(theta)
        x_ofs = (1.0 / 16.0) * pitch
        p.add(pitch, 0)
        p.add(pitch, r0 + h)
        p.add(pitch / 2.0, r0).smooth(r_root, 5)
        p.add(x_ofs, r_major)
        p.add(-x_ofs, r_major)
        p.add(-pitch / 2.0, r0).smooth(r_root, 5)
        p.add(-pitch, r0 + h)
        p.add(-pitch, 0)
    else:
        r_minor = r0 + (1.0 / 4.0) * h
        r_crest = (pitch / 16.0) / math.cos(theta)
        x_ofs = (1.0 / 8.0) * pitch
        p.add(pitch, 0)
        p.add(pitch, r_minor)
        p.add(pitch / 2 - x_ofs, r_minor)
        p.add(0, r0 + h).smooth(r_crest, 5)
        p.add(-pitch / 2 + x_ofs, r_minor)
        p.add(-pitch, r_minor)
        p.add(-pitch, 0)
    return p.vertices()


def acme_thread(radius: float, pitch: float) -> List[Vec2]:
    """2D profile of a 29-degree Acme trapezoidal power thread."""
    if radius <= 0 or pitch <= 0:
        raise ValueError("radius and pitch must be positive")
    h = radius - 0.5 * pitch
    theta = math.radians(29.0 / 2.0)
    delta = 0.25 * pitch * math.tan(theta)
    x_ofs0 = 0.25 * pitch - delta
    x_ofs1 = 0.25 * pitch + delta

    p = Polygon()
    p.add(radius, 0)
    p.add(radius, h)
    p.add(x_ofs1, h)
    p.add(x_ofs0, radius)
    p.add(-x_ofs0, radius)
    p.add(-x_ofs1, h)
    p.add(-radius, h)
    p.add(-radius, 0)
    return p.vertices()


def ansi_buttress_thread(radius: float, pitch: float) -> List[Vec2]:
    """2D profile of an ANSI 45/7 asymmetric buttress thread.

    The polygon spans x in [-pitch, +pitch] (two periods) so the field is
    continuous across the x = +/- pitch/2 wrap boundary when swept.
    """
    if radius <= 0 or pitch <= 0:
        raise ValueError("radius and pitch must be positive")
    t0 = math.tan(math.radians(45.0))
    t1 = math.tan(math.radians(7.0))
    b = 0.6  # thread engagement

    h0 = pitch / (t0 + t1)
    h1 = ((b / 2.0) * pitch) + (0.5 * h0)
    hp = pitch / 2.0

    x_v = t0 * h0 - hp        # valley root x
    x7 = hp - (h0 - h1) * t1  # 7-degree flank top x
    x45 = (h0 - h1) * t0 - hp  # 45-degree flank top x
    y_edge = radius + x45     # y on 45-degree flank at x = +/- pitch

    p = Polygon()
    p.add(pitch, 0)
    p.add(pitch, y_edge)
    p.add(x45 + pitch, radius)
    p.add(x7, radius)
    p.add(x_v, radius - h1).smooth(0.0714 * pitch, 5)
    p.add(x45, radius)
    p.add(x7 - pitch, radius)
    p.add(x_v - pitch, radius - h1).smooth(0.0714 * pitch, 5)
    p.add(-pitch, y_edge)
    p.add(-pitch, 0)
    return p.vertices()
