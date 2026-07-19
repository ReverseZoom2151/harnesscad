"""Scan-annotation primitives pairing scan-like meshes with B-Rep labels.

This representation pairs high-resolution scan-like meshes with B-Rep labels. The
learned foundation model is out of scope, but three of its data-processing rules
are fully deterministic and stdlib-only, and none of them is covered by the
existing chain-complex / coedge-walk modules:

* **Tiny-loop (hole) classification.** A loop is a candidate tiny hole when the
  dataset has more than two loops (``|L| > 2``) and the loop's perimeter,
  relative to the largest loop perimeter, falls below a threshold ``tau_h``:
  ``L_ell / L_max < tau_h``. These small openings are the ones a short-range
  scanner fails to capture. :func:`classify_tiny_loops`.

* **Proximity-aware soft-label membership.** Instead of a hard nearest-neighbour
  assignment, each scan point aggregates SPH-style weights over a multi-scale
  neighbourhood and takes the ``arg max`` of the resulting probability, so a
  nearby-but-not-nearest candidate is not silently dropped.
  :func:`soft_label_membership` reproduces the weighting
  ``omega_i = sum_k w_k * W_k`` and the normalised ``p_i`` / ``arg max`` rule.

* **Sketch-artist skill schedule.** A single scalar ``kappa`` in ``{1..5}`` sets
  the hand-drawn distortion amplitude via ``alpha(kappa) = (6 - kappa) / 5`` and
  a base magnitude ``A0 = alpha * c_L * L``; lower skill means larger deviation.
  :func:`skill_amplitude`.

All routines are deterministic. Inputs are plain numbers / tuples so the rules
stay independent of any particular mesh or B-Rep encoding.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

__all__ = [
    "classify_tiny_loops",
    "soft_label_membership",
    "skill_amplitude",
    "sph_poly6_weight",
]


def classify_tiny_loops(
    perimeters: Sequence[float], tau_h: float = 0.15
) -> List[int]:
    """Return indices of loops classified as tiny holes.

    With ``|L| > 2`` loops, a loop is a tiny hole when ``L_ell / L_max < tau_h``.
    With two or fewer loops the rule is inactive and no loop is flagged.
    ``tau_h`` must be in ``(0, 1]``.
    """
    if not 0.0 < tau_h <= 1.0:
        raise ValueError("tau_h must be in (0, 1]")
    vals = [float(p) for p in perimeters]
    if any(p < 0 for p in vals):
        raise ValueError("perimeters must be non-negative")
    if len(vals) <= 2:
        return []
    l_max = max(vals)
    if l_max == 0.0:
        return []
    return [i for i, p in enumerate(vals) if p / l_max < tau_h]


def sph_poly6_weight(distance: float, radius: float) -> float:
    """Normalised SPH poly6 kernel weight ``(1 - (d/h)^2)^3`` for ``d <= h``.

    A monotone-decreasing compactly-supported smoothing weight (zero beyond the
    support radius ``h``), the family used for the proximity-aware labels.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    d = abs(float(distance))
    if d >= radius:
        return 0.0
    q = 1.0 - (d / radius) ** 2
    return q ** 3


def soft_label_membership(
    candidates: Sequence[Tuple[int, Sequence[float]]],
    scale_weights: Sequence[float],
    radii: Sequence[float],
) -> Dict[str, object]:
    """Assign a scan point to a BRep entity via multi-scale SPH soft labels.

    ``candidates`` is a sequence of ``(entity_id, distances)`` where ``distances``
    holds the point's distance to that entity at each of ``K`` scales. ``radii``
    are the ``K`` support radii and ``scale_weights`` the ``K`` scale mixing
    weights ``w_k``. For each candidate the aggregated weight is
    ``omega_i = sum_k w_k * W_poly6(d_ik, h_k)``; probabilities are
    ``p_i = omega_i / sum_j omega_j`` and the label is ``arg max_i p_i``.

    Returns ``{"label": entity_id, "probabilities": {id: p}, "weights": {id: w}}``.
    A point with zero aggregated weight everywhere returns ``label = None``.
    """
    ws = [float(w) for w in scale_weights]
    hs = [float(r) for r in radii]
    if len(ws) != len(hs) or not ws:
        raise ValueError("scale_weights and radii must be non-empty and equal length")
    weights: Dict[int, float] = {}
    for ent_id, dists in candidates:
        ds = [float(d) for d in dists]
        if len(ds) != len(ws):
            raise ValueError("each candidate needs one distance per scale")
        omega = sum(w * sph_poly6_weight(d, h) for w, d, h in zip(ws, ds, hs))
        weights[ent_id] = weights.get(ent_id, 0.0) + omega
    total = sum(weights.values())
    if total == 0.0:
        return {"label": None, "probabilities": {}, "weights": weights}
    probs = {k: v / total for k, v in weights.items()}
    # deterministic arg-max: highest probability, ties broken by smallest id.
    label = max(probs, key=lambda k: (probs[k], -k))
    return {"label": label, "probabilities": probs, "weights": weights}


def skill_amplitude(kappa: int, length: float, c_l: float = 5e-3) -> Dict[str, float]:
    """Sketch-artist skill-to-amplitude schedule.

    ``kappa in {1..5}`` (1 = crude hand drawing, 5 = professional). Returns
    ``alpha = (6 - kappa) / 5`` and the base deviation magnitude
    ``A0 = alpha * c_l * length``. Lower skill -> larger ``alpha`` -> larger
    ``A0``. ``c_l`` is the per-length coefficient (typically ~1e-3..1e-2).
    """
    if kappa not in (1, 2, 3, 4, 5):
        raise ValueError("kappa must be an integer in {1..5}")
    if length < 0 or c_l < 0:
        raise ValueError("length and c_l must be non-negative")
    alpha = (6 - kappa) / 5.0
    return {"alpha": alpha, "base_amplitude": alpha * c_l * float(length)}
