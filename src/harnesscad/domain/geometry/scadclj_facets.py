"""scadclj_facets -- OpenSCAD's deterministic facet-count resolution.

Every curved OpenSCAD primitive (``circle``, ``sphere``, ``cylinder``,
``rotate_extrude``) is tessellated into a finite number of fragments before it
is rendered.  The fragment count is *not* free -- it is fixed by the special
variables ``$fn`` / ``$fa`` / ``$fs`` and the radius, through one exact formula
that OpenSCAD applies everywhere (``Calc::get_fragments_from_r`` in
``src/geometry/Calc.cc``)::

    if r < GRID_FINE           -> 3
    if $fn > 0                 -> max($fn, 3)
    otherwise                  -> ceil(max(min(360/$fa, 2*pi*r/$fs), 5))

scad-clj emits the ``$fn`` / ``$fa`` / ``$fs`` variables into the source but
leaves the *resolution* to the OpenSCAD renderer.  The harness had no way to
predict how a curved primitive would tessellate: this module supplies that
missing deterministic step, so a generator can know the exact vertex count (and
the exact polygon) a given ``$fn/$fa/$fs`` will produce -- for meshing, for
chord-error budgeting, or for reproducing OpenSCAD geometry without the kernel.

``$fa`` is the minimum fragment *angle* in degrees (default 12), ``$fs`` the
minimum fragment *size* in model units (default 2), ``$fn`` a hard override
(default 0 = "use $fa/$fs").  These defaults match OpenSCAD.

Also provided:

* :func:`circle_fragment_points` -- the exact CCW polygon OpenSCAD builds for a
  circle / the cross-section of a cylinder, at the resolved fragment count;
* :func:`sphere_rings` -- OpenSCAD's ring count for a sphere (``fragments`` from
  the equatorial radius, split into ``(fragments+1)/2`` latitude rings);
* :func:`chord_error` and :func:`fragments_for_chord_error` -- the sagitta
  (max radial deviation of the inscribed polygon) and its inverse, the standard
  way to choose ``$fn`` from a tolerance.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "GRID_FINE",
    "DEFAULT_FA",
    "DEFAULT_FS",
    "DEFAULT_FN",
    "get_fragments_from_r",
    "fragments_for_node",
    "circle_fragment_points",
    "sphere_rings",
    "chord_error",
    "fragments_for_chord_error",
]

# OpenSCAD's GRID_FINE == 1/1024/1024 -- radii below this collapse to 3 facets.
GRID_FINE = 0.00000095367431640625

DEFAULT_FA = 12.0
DEFAULT_FS = 2.0
DEFAULT_FN = 0.0


def get_fragments_from_r(r: float, fn: float = DEFAULT_FN, fs: float = DEFAULT_FS,
                         fa: float = DEFAULT_FA) -> int:
    """Number of fragments a circle of radius *r* tessellates into.

    Exact port of OpenSCAD ``Calc::get_fragments_from_r`` (argument order
    ``r, fn, fs, fa`` as in the C++)."""
    if r < GRID_FINE or math.isinf(fn) or math.isnan(fn):
        return 3
    if fn > 0.0:
        return int(fn) if fn >= 3 else 3
    return int(math.ceil(max(min(360.0 / fa, r * 2.0 * math.pi / fs), 5.0)))


def fragments_for_node(node: Any, fn: float = DEFAULT_FN, fs: float = DEFAULT_FS,
                       fa: float = DEFAULT_FA) -> int:
    """Resolve fragments for a scadclj_data_ir circle/sphere/cylinder node.

    The node's own ``$fn/$fa/$fs`` (stored as ``fn/fa/fs`` keys, e.g. from a
    ``with_fn`` binding) override the passed-in ambient values, mirroring
    OpenSCAD's per-primitive special-variable precedence."""
    if not (isinstance(node, tuple) and len(node) >= 2 and isinstance(node[1], dict)):
        raise TypeError("fragments_for_node expects a primitive IR node")
    args = node[1]
    r = args.get("r")
    if r is None:
        r = max(args.get("r1", 0.0), args.get("r2", 0.0))
    local_fn = args.get("fn", fn)
    local_fs = args.get("fs", fs)
    local_fa = args.get("fa", fa)
    return get_fragments_from_r(r, local_fn, local_fs, local_fa)


def circle_fragment_points(r: float, fn: float = DEFAULT_FN, fs: float = DEFAULT_FS,
                           fa: float = DEFAULT_FA) -> List[Tuple[float, float]]:
    """The CCW polygon OpenSCAD inscribes for a circle of radius *r*.

    Vertex ``i`` is at angle ``360*i/n`` degrees, matching
    ``primitives.cc`` ``generate_circle``."""
    n = get_fragments_from_r(r, fn, fs, fa)
    points: List[Tuple[float, float]] = []
    for i in range(n):
        phi = (2.0 * math.pi * i) / n
        points.append((r * math.cos(phi), r * math.sin(phi)))
    return points


def sphere_rings(r: float, fn: float = DEFAULT_FN, fs: float = DEFAULT_FS,
                 fa: float = DEFAULT_FA) -> Tuple[int, int]:
    """(fragments, rings) OpenSCAD uses for a sphere of radius *r*.

    OpenSCAD builds ``fragments`` meridians and ``(fragments + 1) / 2``
    latitude rings (integer division)."""
    fragments = get_fragments_from_r(r, fn, fs, fa)
    rings = (fragments + 1) // 2
    return fragments, rings


def chord_error(r: float, fragments: int) -> float:
    """Sagitta: max radial gap between the circle and its inscribed polygon.

    ``r * (1 - cos(pi / n))`` -- how far the true arc bulges beyond the
    ``n``-gon OpenSCAD renders."""
    if fragments < 3:
        raise ValueError("a polygon needs at least 3 fragments")
    return r * (1.0 - math.cos(math.pi / fragments))


def fragments_for_chord_error(r: float, max_error: float) -> int:
    """Smallest fragment count whose sagitta is within *max_error*.

    The inverse of :func:`chord_error`; the standard way to pick ``$fn`` from a
    geometric tolerance."""
    if max_error <= 0:
        raise ValueError("max_error must be positive")
    if max_error >= r:
        return 3
    n = math.pi / math.acos(1.0 - max_error / r)
    return max(3, int(math.ceil(n)))
