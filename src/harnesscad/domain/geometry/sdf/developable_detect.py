"""Developable-surface detection & segmentation from an SDF.

Deterministic classifiers built on the geometric observations of **NeurCADRecon**
(Dong et al., ACM TOG 2024).  The paper's central prior is that a CAD surface is
piecewise smooth with each patch *approximately developable*, and:

* developability is equivalent to **zero Gaussian curvature** ``K = k1 k2 = 0``,
  i.e. at least one principal curvature vanishes (a ruled, flattenable patch);
* equivalently the tangent-plane shape operator ``S`` has **rank at most 1**
  (Sec. 3.2, following Sellan et al. 2020) -- this module offers that rank test
  as well as the numerically-friendlier ``|K|`` test the paper actually adopts;
* a **tip / corner** point has significantly non-zero Gaussian curvature, close
  to ``pi/2`` (Sec. 3.3) -- detected here so it can be tolerated rather than
  flattened.

Given per-point ``(grad, hess)`` samples this classifies a patch into
planar / developable / synclastic (elliptic) / anticlastic (hyperbolic) regions
and reports the developable fraction -- exactly the segmentation NeurCADRecon
uses to decompose a reconstructed SDF into smooth patches.

Curvature quantities are *reused* from :mod:`geometry.flatcad_weingarten`.
stdlib-only, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from harnesscad.domain.geometry.sdf.curvature import (
    gaussian_curvature, principal_curvatures, shape_operator,
    orthonormal_tangent_frame,
)

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Sequence[float], Sequence[float], Sequence[float]]
Sample = Tuple[Sequence[float], Mat3]

# A tip point's Gaussian curvature is near pi/2 (NeurCADRecon Sec. 3.3).
TIP_GAUSSIAN = math.pi / 2.0


def is_developable(grad: Sequence[float], hess: Mat3, tol: float = 1e-6) -> bool:
    """True iff the point is (approximately) developable: ``|k_Gauss| <= tol``.

    This is NeurCADRecon's ``K -> 0`` developability test (numerically stabler
    than the rank test).  Cylinders, cones and planes pass; spheres do not.
    """
    return abs(gaussian_curvature(grad, hess)) <= tol


def shape_operator_rank(grad: Sequence[float], hess: Mat3,
                        tol: float = 1e-6) -> int:
    """Rank (0, 1 or 2) of the 2x2 tangent-plane shape operator.

    Developability <=> rank <= 1 (Sec. 3.2).  Counts non-zero principal
    curvatures: plane -> 0, cylinder/cone -> 1, sphere/saddle -> 2.
    """
    k1, k2 = principal_curvatures(grad, hess)
    return int(abs(k1) > tol) + int(abs(k2) > tol)


def is_tip_point(grad: Sequence[float], hess: Mat3,
                 tol: float = 1e-2) -> bool:
    """True iff the Gaussian curvature is close to ``pi/2`` -- a CAD tip/corner
    (Sec. 3.3), which the developability prior should tolerate, not flatten."""
    return abs(abs(gaussian_curvature(grad, hess)) - TIP_GAUSSIAN) <= tol


def classify_developability(grad: Sequence[float], hess: Mat3,
                            tol: float = 1e-6) -> str:
    """Classify a point by its developability regime:

    * ``"planar"``      -- both principal curvatures ~0 (flat),
    * ``"developable"`` -- exactly one ~0 (``K=0``: cylinder/cone/ruled),
    * ``"synclastic"``  -- ``K > 0`` (elliptic / doubly-curved, e.g. sphere),
    * ``"anticlastic"`` -- ``K < 0`` (hyperbolic / saddle).

    ``planar`` and ``developable`` are the ``K=0`` cases NeurCADRecon rewards;
    ``synclastic`` and ``anticlastic`` are the doubly-curved cases it penalises.
    """
    k1, k2 = principal_curvatures(grad, hess)
    z1 = abs(k1) <= tol
    z2 = abs(k2) <= tol
    if z1 and z2:
        return "planar"
    if z1 or z2:
        return "developable"
    return "synclastic" if k1 * k2 > 0.0 else "anticlastic"


def is_doubly_curved(grad: Sequence[float], hess: Mat3,
                     tol: float = 1e-6) -> bool:
    """True iff the point is doubly curved (``|K| > tol``): not developable."""
    return abs(gaussian_curvature(grad, hess)) > tol


def segment_developable(samples: Sequence[Sample],
                        tol: float = 1e-6) -> List[str]:
    """Per-point developability labels for a patch (see ``classify_developability``)."""
    return [classify_developability(g, H, tol) for g, H in samples]


def developable_fraction(samples: Sequence[Sample], tol: float = 1e-6) -> float:
    """Fraction of points that are developable (``planar`` or ``developable``).

    A value of 1.0 means the whole patch is developable, as NeurCADRecon drives
    every non-corner point toward.
    """
    if not samples:
        raise ValueError("need at least one sample")
    dev = sum(1 for g, H in samples if is_developable(g, H, tol))
    return dev / len(samples)


def shape_operator_from_grad_hess(grad: Sequence[float], hess: Mat3):
    """Convenience: the 2x2 tangent shape operator in a deterministic frame
    normal to ``grad`` -- exposes the matrix whose rank encodes developability."""
    u, v = orthonormal_tangent_frame(grad)
    return shape_operator(grad, hess, u, v)
