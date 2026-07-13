"""t2cadtd_iso_ortho_consistency — isometric <-> orthographic dimensional check.

Text2CAD (Yavartanoo et al., "Text to 3D CAD Generation via Technical Drawings")
generates an *isometric* image first, then derives the three orthographic views
(top/front/side) from it, and stresses repeatedly that the pipeline "ensures
geometric and physical consistency across views" and upholds "physical and
dimensional consistency essential for practical engineering applications". The
learned Zero-1-to-3 view generator can drift, so a deterministic check that the
isometric image and the orthographic views describe *the same size object* is the
natural verifier.

This is DISTINCT from :mod:`drawings.creft_view_consistency`, which only compares
the three orthographic views to each other (ortho <-> ortho shared extents).
Here we close the loop the paper actually builds: **isometric <-> orthographic**.

The isometric image foreshortens each axis by a known factor
(:func:`drawings.t2cadtd_isometric_projection.axis_foreshortening`). So an edge
drawn along axis ``k`` with pixel/2D length ``L`` implies a true 3D extent
``L / foreshorten[k]``. Given the three projected axis-edge lengths measured off
the isometric drawing, :func:`recover_extents_from_isometric` inverts the
foreshortening to recover ``(width=X, height=Z, depth=Y)``. Those are then
compared against the extents implied by the orthographic view set
(:func:`drawings.creft_view_consistency.implied_dimensions`).

The paper's orientation ablation ("the side view of the original aligning with
the front view of the mirrored images") is captured by :func:`mirror_extents`,
which mirrors the object across a vertical plane and reports the swapped view
extents.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from harnesscad.domain.drawings.orthographic_projection import View
from harnesscad.domain.drawings.view_consistency import implied_dimensions
from harnesscad.domain.drawings.isometric_projection import (
    PAPER_AZIMUTH_DEG, PAPER_ELEVATION_DEG, axis_foreshortening,
)


def recover_extents_from_isometric(edge_lengths: Dict[str, float],
                                   azimuth_deg: float = PAPER_AZIMUTH_DEG,
                                   elevation_deg: float = PAPER_ELEVATION_DEG
                                   ) -> Dict[str, float]:
    """Invert the isometric foreshortening to recover true 3D axis extents.

    ``edge_lengths`` are the measured 2D lengths of the box edges drawn along
    each axis in the isometric image, keyed ``"x"``/``"y"``/``"z"``. Returns the
    recovered true extents keyed the same way. A missing axis is treated as 0.
    """
    f = axis_foreshortening(azimuth_deg, elevation_deg)
    out: Dict[str, float] = {}
    for axis in ("x", "y", "z"):
        length = float(edge_lengths.get(axis, 0.0))
        if length < 0.0:
            raise ValueError("edge length must be non-negative")
        factor = f[axis]
        out[axis] = length / factor if factor > 0.0 else 0.0
    return out


def isometric_edge_lengths(dx: float, dy: float, dz: float,
                           azimuth_deg: float = PAPER_AZIMUTH_DEG,
                           elevation_deg: float = PAPER_ELEVATION_DEG
                           ) -> Dict[str, float]:
    """Forward map: true box extents -> the 2D edge lengths drawn in isometric.

    Exact inverse of :func:`recover_extents_from_isometric`, useful for building
    a consistent isometric measurement from a known box (e.g. in tests / synth).
    """
    f = axis_foreshortening(azimuth_deg, elevation_deg)
    return {"x": dx * f["x"], "y": dy * f["y"], "z": dz * f["z"]}


@dataclass(frozen=True)
class IsoOrthoResult:
    consistent: bool
    iso_dimensions: Dict[str, float]
    ortho_dimensions: Dict[str, float]
    mismatches: Tuple[Dict[str, object], ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, object]:
        return {
            "consistent": self.consistent,
            "iso_dimensions": dict(self.iso_dimensions),
            "ortho_dimensions": dict(self.ortho_dimensions),
            "mismatches": [dict(m) for m in self.mismatches],
        }


# Map (width/height/depth) <-> (x/z/y).
_DIM_TO_AXIS = {"width": "x", "height": "z", "depth": "y"}


def check_iso_ortho_consistency(iso_edge_lengths: Dict[str, float],
                                views: Dict[str, View],
                                azimuth_deg: float = PAPER_AZIMUTH_DEG,
                                elevation_deg: float = PAPER_ELEVATION_DEG,
                                tol: float = 1e-6) -> IsoOrthoResult:
    """Do the isometric drawing and the orthographic view set agree on size?

    Recovers (width, height, depth) from the isometric edge lengths and compares
    to the same quantities implied by the orthographic views. Any axis differing
    by more than ``tol`` is reported as a mismatch.
    """
    recovered = recover_extents_from_isometric(iso_edge_lengths, azimuth_deg,
                                               elevation_deg)
    iso_dims = {
        "width": recovered["x"],
        "height": recovered["z"],
        "depth": recovered["y"],
    }
    ortho_dims = implied_dimensions(views)

    mismatches: List[Dict[str, object]] = []
    for dim in ("width", "height", "depth"):
        a = iso_dims[dim]
        b = ortho_dims[dim]
        if abs(a - b) > tol:
            mismatches.append({"dimension": dim, "axis": _DIM_TO_AXIS[dim],
                               "iso": a, "ortho": b, "delta": abs(a - b)})
    return IsoOrthoResult(consistent=not mismatches,
                          iso_dimensions=iso_dims,
                          ortho_dimensions=ortho_dims,
                          mismatches=tuple(mismatches))


def mirror_extents(dims: Dict[str, float]) -> Dict[str, float]:
    """Mirror the object across a vertical plane: swap the two horizontal extents.

    The paper's orientation ablation observes that mirroring the isometric image
    maps the original's side view onto the mirrored front view. Under our
    conventions (front spans X, side spans Y), a horizontal mirror swaps the X
    (width) and Y (depth) extents while leaving Z (height) fixed. ``dims`` is a
    ``{"width","height","depth"}`` mapping; the swapped mapping is returned.
    """
    return {
        "width": dims["depth"],
        "height": dims["height"],
        "depth": dims["width"],
    }
