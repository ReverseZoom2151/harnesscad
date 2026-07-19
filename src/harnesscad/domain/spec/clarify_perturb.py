"""clarify_perturb -- deterministic ambiguity synthesis for text-to-CAD.

The clarifier is trained on synthetic *ambiguous* prompts produced by
perturbing verified specifications. Two error types are covered:

  * **under-specified** -- omit exactly K key dimensions;
  * **conflicting**    -- assign conflicting values to exactly K features.

It records a full clarification *trajectory* ``(p_hat, q, a, p)`` where ``q``
are the questions that recover the omitted/over-written information and ``a``
are the correct answers drawn from the ground truth.

This module reproduces that generator deterministically (seeded, stdlib-only)
so a robustness benchmark can be built without an LLM. The curation
rules -- keep a perturbed sample only when it is *genuinely
harmful* and the degradation is *substantial* -- are provided as
:func:`keep_sample`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from harnesscad.domain.spec.clarify_ambiguity import (
    CADSpec,
    Feature,
    UNDER_SPECIFIED,
    CONFLICTING,
    _REQUIRED_PARAMS,
    _feature_key,
    question_for,
    Issue,
)

UNDER = UNDER_SPECIFIED
CONFLICT = CONFLICTING


@dataclass
class Trajectory:
    """A synthetic clarification trajectory ``(p_hat, q, a, p)``."""

    original: CADSpec         # p  -- verified ground truth
    ambiguous: CADSpec        # p_hat -- perturbed / misleading prompt
    questions: Tuple[str, ...]
    answers: Tuple[Tuple[str, object], ...]
    ambiguity_type: str       # "under_specified" | "conflicting"
    num_issues: int           # K -- the number of introduced issues
    keys: Tuple[str, ...]     # perturbed slot keys (ground-truth question keys)


# --------------------------------------------------------------------------- #
# enumerate perturbable slots
# --------------------------------------------------------------------------- #

def _slots(spec: CADSpec) -> List[str]:
    """Ordered list of slot keys that carry a value and may be perturbed."""
    keys: List[str] = []
    if spec.workplane is not None:
        keys.append("setup.workplane")
    if spec.origin is not None:
        keys.append("setup.origin")
    if spec.extrude_direction is not None:
        keys.append("build.extrude_direction")
    if spec.extrude_distance is not None:
        keys.append("build.extrude_distance")
    for feat in spec.features:
        for param in _REQUIRED_PARAMS.get(feat.kind, ()):
            if feat.params.get(param) is not None:
                keys.append(_feature_key(feat, param))
    return keys


def _get(spec: CADSpec, key: str) -> object:
    if key == "setup.workplane":
        return spec.workplane
    if key == "setup.origin":
        return spec.origin
    if key == "build.extrude_direction":
        return spec.extrude_direction
    if key == "build.extrude_distance":
        return spec.extrude_distance
    feat, param = key.split(".")[0], key.split(".")[-1]
    for f in spec.features:
        if (f.name or f.kind) == feat:
            return f.params.get(param)
    return None


def _omit(spec: CADSpec, key: str) -> None:
    if key == "setup.workplane":
        spec.workplane = None
    elif key == "setup.origin":
        spec.origin = None
    elif key == "build.extrude_direction":
        spec.extrude_direction = None
    elif key == "build.extrude_distance":
        spec.extrude_distance = None
    else:
        feat, param = key.split(".")[0], key.split(".")[-1]
        for f in spec.features:
            if (f.name or f.kind) == feat:
                f.params[param] = None


def _inject_conflict(spec: CADSpec, key: str, rng: random.Random) -> None:
    """Overwrite a numeric slot with two conflicting stated values."""
    if key == "build.extrude_distance":
        true = float(spec.extrude_distance)
        spec.extrude_distance = [true, _perturb_value(true, rng)]
        return
    feat, param = key.split(".")[0], key.split(".")[-1]
    for f in spec.features:
        if (f.name or f.kind) == feat:
            true = float(f.params[param])
            f.params[param] = [true, _perturb_value(true, rng)]


def _perturb_value(v: float, rng: random.Random) -> float:
    """Return a distinct conflicting value near ``v`` (deterministic)."""
    deltas = [2.0, -2.0, 5.0, -5.0, v * 0.5 if v else 1.0, v + 1.0]
    for d in deltas:
        cand = round(v + d, 6) if abs(d) in (2.0, 5.0) else round(d, 6)
        if cand != v and cand > 0:
            return cand
    return v + 1.0


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #

class AmbiguityGenerator:
    """Deterministic perturbation generator."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def perturb(self, spec: CADSpec, ambiguity_type: str, k: int
                ) -> Trajectory:
        """Introduce exactly ``k`` ambiguities of ``ambiguity_type``.

        Numeric conflicts require numeric slots; setup/direction slots are only
        eligible for the under-specified type.
        """
        if ambiguity_type not in (UNDER, CONFLICT):
            raise ValueError("unknown ambiguity type: " + ambiguity_type)
        original = spec.copy()
        candidates = _slots(spec)
        if ambiguity_type == CONFLICT:
            candidates = [c for c in candidates if _numeric(_get(spec, c))]
        if k < 1 or k > len(candidates):
            raise ValueError(
                "cannot introduce {0} ambiguities; {1} eligible slots"
                .format(k, len(candidates)))
        chosen = sorted(self.rng.sample(candidates, k))

        amb = spec.copy()
        answers: List[Tuple[str, object]] = []
        questions: List[str] = []
        for key in chosen:
            truth = _get(original, key)
            answers.append((key, truth))
            if ambiguity_type == UNDER:
                _omit(amb, key)
            else:
                _inject_conflict(amb, key, self.rng)
            issue = Issue(ambiguity_type, key, "", ())
            questions.append(question_for(issue).text)

        return Trajectory(
            original=original,
            ambiguous=amb,
            questions=tuple(questions),
            answers=tuple(answers),
            ambiguity_type=ambiguity_type,
            num_issues=k,
            keys=tuple(chosen),
        )


def _numeric(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --------------------------------------------------------------------------- #
# curation rule
# --------------------------------------------------------------------------- #

def keep_sample(orig_cd: float, perturbed_cd: float,
                *, cd_threshold: float = 2e-4, ratio_min: float = 10.0) -> bool:
    """Three selection rules for a synthetic ambiguous sample.

    Keep a sample only if:
      (i)   the verified spec is high quality: ``orig_cd < cd_threshold``;
      (ii)  the perturbed prompt is genuinely harmful: ``perturbed_cd > cd_threshold``;
      (iii) degradation is substantial: ``perturbed_cd / orig_cd >= ratio_min``.
    """
    if not (orig_cd < cd_threshold):
        return False
    if not (perturbed_cd > cd_threshold):
        return False
    if orig_cd <= 0:
        return perturbed_cd > cd_threshold
    return (perturbed_cd / orig_cd) >= ratio_min
