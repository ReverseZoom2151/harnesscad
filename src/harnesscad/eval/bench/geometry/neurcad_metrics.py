"""Developability & reconstruction metrics for NeurCADRecon.

Deterministic evaluation metrics derived from **NeurCADRecon** (Dong et al.,
ACM TOG 2024).  The paper's headline reconstruction metrics (Chamfer, F1,
Normal-Consistency) require a full mesh, but its *developability* claim -- that
CAD surfaces are piecewise developable, ``K -> 0`` -- has closed-form scalar
proxies that this module computes from per-point ``(grad, hess)`` samples:

* **Developability ratio** -- fraction of surface points with ``|K| <= tol``;
  a perfectly developable (CAD) surface scores 1.0.
* **Mean / max absolute Gaussian curvature** -- the aggregate ``L_Gauss`` the
  paper minimises, and its worst-case counterpart.
* **Gaussian-curvature RMSE / MAE** against an analytic reference (e.g. a
  sphere's ``1/r^2``), for validating a reconstruction's curvature field.
* **Gauss-Bonnet defect** -- for a closed surface, ``integral(K dA)`` must equal
  ``2 pi * chi`` (Sec. 5, "the k_Gauss distribution must conform to the
  Gauss-Bonnet theorem"); a sphere gives ``4 pi``.  The defect measures how far
  an estimated curvature field strays from that topological constraint.

Gaussian curvature is *reused* from :mod:`geometry.flatcad_weingarten`.
stdlib-only, deterministic.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from harnesscad.domain.geometry.sdf.flatcad_weingarten import gaussian_curvature

Mat3 = Tuple[Sequence[float], Sequence[float], Sequence[float]]
Sample = Tuple[Sequence[float], Mat3]


def developability_ratio(samples: Sequence[Sample], tol: float = 1e-6) -> float:
    """Fraction of points with ``|k_Gauss| <= tol`` (developable).

    1.0 on a fully developable (CAD-like) surface; lower as doubly-curved
    regions appear.  This is the natural developability score for NeurCADRecon.
    """
    if not samples:
        raise ValueError("need at least one sample")
    dev = sum(1 for g, H in samples if abs(gaussian_curvature(g, H)) <= tol)
    return dev / len(samples)


def mean_abs_gaussian_curvature(samples: Sequence[Sample]) -> float:
    """Mean ``|k_Gauss|`` over the samples (the ``L_Gauss`` energy value)."""
    if not samples:
        raise ValueError("need at least one sample")
    return sum(abs(gaussian_curvature(g, H)) for g, H in samples) / len(samples)


def max_abs_gaussian_curvature(samples: Sequence[Sample]) -> float:
    """Worst-case ``|k_Gauss|`` over the samples."""
    if not samples:
        raise ValueError("need at least one sample")
    return max(abs(gaussian_curvature(g, H)) for g, H in samples)


def gaussian_curvature_mae(samples: Sequence[Sample],
                           reference: Sequence[float]) -> float:
    """Mean absolute error between computed and reference Gaussian curvatures.

    ``reference[i]`` is the analytic ``K`` at sample ``i`` (e.g. ``1/r^2`` on a
    sphere, ``0`` on a developable patch).
    """
    if not samples:
        raise ValueError("need at least one sample")
    if len(samples) != len(reference):
        raise ValueError("samples and reference length mismatch")
    return sum(abs(gaussian_curvature(g, H) - ref)
               for (g, H), ref in zip(samples, reference)) / len(samples)


def gaussian_curvature_rmse(samples: Sequence[Sample],
                            reference: Sequence[float]) -> float:
    """Root-mean-square error between computed and reference Gaussian curvatures."""
    if not samples:
        raise ValueError("need at least one sample")
    if len(samples) != len(reference):
        raise ValueError("samples and reference length mismatch")
    acc = sum((gaussian_curvature(g, H) - ref) ** 2
              for (g, H), ref in zip(samples, reference))
    return math.sqrt(acc / len(samples))


def gauss_bonnet_integral(samples: Sequence[Sample],
                          areas: Sequence[float]) -> float:
    """Discrete ``integral(K dA) = sum_i K_i * area_i`` over a closed surface.

    ``areas[i]`` is the surface-area weight (Voronoi/patch area) for sample ``i``.
    For a closed surface Gauss-Bonnet gives ``2 pi * chi`` (a sphere: ``4 pi``).
    """
    if not samples:
        raise ValueError("need at least one sample")
    if len(samples) != len(areas):
        raise ValueError("samples and areas length mismatch")
    return sum(gaussian_curvature(g, H) * area
               for (g, H), area in zip(samples, areas))


def gauss_bonnet_defect(samples: Sequence[Sample],
                        areas: Sequence[float],
                        euler_characteristic: int = 2) -> float:
    """``|integral(K dA) - 2 pi * chi|`` -- deviation from the Gauss-Bonnet
    topological constraint (default ``chi = 2`` for a genus-0 closed surface).

    Zero when the estimated curvature field is globally consistent, as
    NeurCADRecon requires its ``k_Gauss`` distribution to be (Sec. 5).
    """
    total = gauss_bonnet_integral(samples, areas)
    return abs(total - 2.0 * math.pi * euler_characteristic)
