"""Fast heuristic structural scores for voxel/brick models (AlphaCAD).

Source: ``AlphaCAD-main`` (``summit-demo/vote_server.py``). The BrickGPT paper's
*exact* physical stability (force/torque equilibrium via linear programming) is
already implemented in ``verifiers/brick_stability.py``. AlphaCAD's demo layer
instead computes several *cheap, geometry-only* confidence heuristics that need
no solver -- useful as a fast pre-filter or a UI-facing 0..100 score. This
module reimplements those heuristics deterministically, stdlib only.

Provided scores (each independent, no wall clock, no RNG):

* ``support_and_violations`` -- per-voxel "directly below" support check. A
  brick above the ground layer whose cell directly under it is empty is a
  *violation* (a floating / unsupported brick). Distinct from the connectivity
  grounding check in ``geometry/brick_connectivity.py``: it is a local,
  layer-adjacent occupancy test, not a stud-overlap graph traversal.
* ``confidence_score`` -- 0..100 stability heuristic from the base-support vs
  top-mass ratio and a height-to-footprint aspect penalty, with reason strings.
* ``materials_score`` -- 0..100 normalised Shannon entropy of ``part_type``
  diversity.
* ``aesthetics_score`` -- 0..100 footprint symmetry plus feature bonuses.

A "model" is the dict emitted by ``procedural.alphacad_brick_templates`` (or any
dict with ``width``, ``depth``, ``height`` and a ``bricks`` list of
``{'id','x','y','z','part_type'}``).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Confidence:
    score: int
    reasons: tuple[str, ...]
    base_support: int
    top_mass: int
    support_ratio: float
    aspect_penalty: float


def _bricks(model: dict) -> list:
    return list(model.get("bricks", []))


def support_and_violations(model: dict) -> tuple[dict, list[int]]:
    """Return ``(position->id map, violation ids)``.

    A violation is a brick with ``z > 0`` whose cell directly below
    ``(x, y, z-1)`` is unoccupied (a floating brick / overhang).
    """
    id_by_pos: dict[tuple[int, int, int], int] = {}
    for b in _bricks(model):
        id_by_pos[(b["x"], b["y"], b["z"])] = b["id"]
    violations: list[int] = []
    for b in _bricks(model):
        if b["z"] == 0:
            continue
        if (b["x"], b["y"], b["z"] - 1) not in id_by_pos:
            violations.append(b["id"])
    return id_by_pos, violations


def base_top_counts(model: dict) -> tuple[int, int]:
    """Number of bricks on the ground layer and on the topmost layer."""
    height = model["height"]
    base = sum(1 for b in _bricks(model) if b["z"] == 0)
    top = sum(1 for b in _bricks(model) if b["z"] == height - 1)
    return base, top


def _describe(model: dict) -> list[str]:
    reasons: list[str] = []
    width, depth, height = model["width"], model["depth"], model["height"]
    area = max(1, width * depth)
    base, top = base_top_counts(model)
    base_fill = base / area
    top_fill = top / area
    if base_fill >= 0.95:
        reasons.append("Full base coverage improves load distribution")
    elif base_fill <= 0.6:
        reasons.append("Light base coverage -- less stable")
    if top_fill <= 0.5:
        reasons.append("Lightweight top reduces center of mass")
    part_types = {b.get("part_type", "default") for b in _bricks(model)}
    if "wall" in part_types and base_fill >= 0.9 and top_fill <= 0.6:
        reasons.append("Hollow walls reduce upper mass while keeping a solid base")
    features = model.get("features") or {}
    if isinstance(features, dict):
        t = features.get("leg_thickness")
        if t is not None:
            reasons.append("Thicker legs increase structural rigidity" if t >= 2
                           else "Slim legs reduce material but lower stiffness")
        if features.get("center_support"):
            reasons.append("Center support improves mid-span stability")
        if features.get("surface_pattern") == "border":
            reasons.append("Border top pattern reduces weight with peripheral support")
        if features.get("surface_pattern") == "cross_pattern":
            reasons.append("Cross-pattern top distributes load across axes")
        if features.get("armrests"):
            reasons.append("Armrests add lateral stiffness to the chair frame")
        if features.get("back_style") == "full":
            reasons.append("Full-height back improves vertical bracing")
    if width == depth:
        reasons.append("Square footprint provides symmetric support")
    return reasons


def confidence_score(model: dict) -> Confidence:
    """0..100 stability heuristic from base/top mass ratio and aspect penalty."""
    width, depth, height = model["width"], model["depth"], model["height"]
    base, top = base_top_counts(model)
    aspect_penalty = max(0.0, height / max(1, min(width, depth)) - 3.0)
    support_ratio = base / max(1, top)
    raw = 0.6 * min(1.0, support_ratio / 2.0) + 0.4 * (1.0 / (1.0 + aspect_penalty))
    score = int(max(0.0, min(1.0, raw)) * 100)

    reasons: list[str] = []
    reasons.append("Wide base vs top mass improves stability" if support_ratio > 1.5
                   else "Base support comparable to top mass")
    reasons.append("Balanced height-to-footprint ratio" if aspect_penalty < 0.5
                   else "Tall relative to footprint; lower stability")
    reasons.extend(_describe(model))
    return Confidence(
        score=score,
        reasons=tuple(reasons),
        base_support=base,
        top_mass=top,
        support_ratio=round(support_ratio, 2),
        aspect_penalty=round(aspect_penalty, 2),
    )


def materials_score(model: dict) -> int:
    """0..100 normalised Shannon entropy of ``part_type`` diversity."""
    part_types = [b.get("part_type", "default") for b in _bricks(model)]
    if not part_types:
        return 50
    cnt = Counter(part_types)
    if len(cnt) <= 1:
        return 0
    total = sum(cnt.values())
    entropy = -sum((c / total) * math.log(c / total) for c in cnt.values())
    max_entropy = math.log(len(cnt))
    return int(100 * max(0.0, min(1.0, entropy / max_entropy)))


def aesthetics_score(model: dict) -> int:
    """0..100 footprint symmetry plus template-feature bonuses."""
    width, depth = model["width"], model["depth"]
    symmetry = 1.0 - min(1.0, abs(width - depth) / max(width, depth, 1))
    bonus = 0.0
    features = model.get("features") or {}
    if features.get("surface_pattern") in ("border", "cross_pattern"):
        bonus += 0.2
    if features.get("center_support"):
        bonus += 0.1
    return int(100 * min(1.0, 0.6 * symmetry + bonus))


def score_all(model: dict) -> dict:
    """Convenience: confidence + materials + aesthetics + violations at once."""
    conf = confidence_score(model)
    _, violations = support_and_violations(model)
    return {
        "stability": conf.score,
        "materials": materials_score(model),
        "aesthetics": aesthetics_score(model),
        "violations": violations,
        "reasons": list(conf.reasons),
    }
