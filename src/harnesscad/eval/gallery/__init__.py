"""The rendered parts gallery: a deterministic catalogue of buildable parts.

The harness can build gears, threads, TPMS lattices, SDF blends, cams, sweeps,
enclosures and patterned bodies -- but nothing in the repo *exercised* that
breadth end to end and produced a picture of it. This package does:

  * :mod:`harnesscad.eval.gallery.parts` -- the catalogue. Sixteen distinct
    parts, each naming the capability module it demonstrates, the geometry
    service (if any) it dispatches, the CISP ops it emits, and the backends
    that can genuinely build it (and the ones that provably cannot).
  * :mod:`harnesscad.eval.gallery.render_gallery` -- the driver. Builds every
    part on its preferred backend, rasterises it through
    :mod:`harnesscad.io.render`, writes multi-view drawings through
    :mod:`harnesscad.io.drawing`, and PROGRAMMATICALLY QCs every PNG it wrote
    (stdlib zlib decode, pixel variance, silhouette coverage, colour count) so
    a blank or degenerate render cannot be shipped.

Stdlib only, deterministic (no wall clock in any output, no randomness): the
same catalogue always produces byte-identical images.
"""

from __future__ import annotations

from harnesscad.eval.gallery.parts import CATALOGUE, Part, get, names

__all__ = ["CATALOGUE", "Part", "get", "names"]
