"""CAD-RL executability-gated reward with external-evaluation failure modes.

CAD-RL ("From Intent to Execution", AAAI 2026) post-trains a VLM for precise,
executable CADQuery code generation using a reward built from three
*complementary* components (Sec. 3.3):

  * **Executability reward** ``R_exec`` (Eq. 3) -- a binary 0/1 signal: 1 if the
    generated Python/CADQuery code parses and executes without exception, 0
    otherwise. It acts as a *multiplicative gate* so that a broken program earns
    zero total reward regardless of geometry ("preventing reward leakage from
    malformed samples").

  * **Geometric accuracy reward** ``R_geom`` (Eq. 4) -- the volumetric IoU
    between the generated solid ``M_gen`` and the ground truth ``M_gt``::

        R_geom = |M_gen ∩ M_gt| / |M_gen ∪ M_gt|

  * **External evaluation reward** ``R_eval`` (Eq. 5) -- a normalised soft score
    from an external evaluator (GPT-4o in the paper) assessing semantic fidelity
    to the design intent, ``R_eval = Norm(Score(c_gen, x))``. Crucially the
    paper imposes *stronger penalties on two harmful error categories* detected
    by the evaluator: **Reference Frame Misalignment** and **Parametric
    Misassignment**.

The total reward (Eq. 6) is::

    R = R_exec(c) * [ lambda_geom * R_geom + lambda_eval * R_eval ]

This composition is deliberately DISTINCT from the repository's existing
rewards:

  * ``dataengine.cadrille_reward`` -- additive ``IOU_SCALE*IoU + invalid_penalty``
    (no gate, no external term).
  * ``dataengine.cmecad_reward`` -- gated by BOTH format and exec and adds a
    *work-plane* term ``lambda_plane*R_plane`` (no external-evaluator term).

Here there is a single ``R_exec`` gate, no format gate, no plane term, and an
external soft-score term carrying structured failure-mode deductions.

The raw evaluator score is *injected* (the model / GPT-4o call is external);
this module only defines the deterministic composition, the score
normalisation, and the failure-mode deduction. Pure stdlib, deterministic.
"""

from __future__ import annotations

from typing import Iterable, Mapping

# Convex-combination weights for the geometric and external terms (Eq. 6).
DEFAULT_LAMBDA_GEOM = 1.0
DEFAULT_LAMBDA_EVAL = 1.0

# The two failure modes the external evaluator is prompted to catch, with the
# paper's "severe deduction" applied to the normalised score (Sec. 3.3).
REFERENCE_FRAME_MISALIGNMENT = "reference_frame_misalignment"
PARAMETRIC_MISASSIGNMENT = "parametric_misassignment"
DEFAULT_FAILURE_PENALTY = 0.5


def r_exec(executes: bool) -> float:
    """Binary executability reward (Eq. 3): 1.0 if ``executes`` else 0.0."""
    return 1.0 if executes else 0.0


def r_geom(intersection: float, union: float) -> float:
    """Volumetric IoU reward (Eq. 4) ``|inter| / |union|`` in ``[0, 1]``.

    ``union`` of zero (two empty solids) yields 0.0. Values are validated so a
    negative or intersection-exceeds-union input raises rather than silently
    producing an out-of-range reward.
    """
    inter = float(intersection)
    uni = float(union)
    if inter < 0.0 or uni < 0.0:
        raise ValueError("intersection and union volumes must be non-negative")
    if inter > uni + 1e-12:
        raise ValueError("intersection cannot exceed union")
    if uni == 0.0:
        return 0.0
    return inter / uni


def normalize_score(score: float, lo: float = 0.0, hi: float = 10.0) -> float:
    """``Norm(Score)`` -- min-max normalise a raw evaluator score to ``[0, 1]``.

    The external evaluator returns a score in ``[lo, hi]`` (e.g. a 0-10 rubric);
    this maps it linearly to ``[0, 1]`` and clamps out-of-range inputs.
    """
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        raise ValueError("hi must be greater than lo")
    frac = (float(score) - lo) / (hi - lo)
    if frac < 0.0:
        return 0.0
    if frac > 1.0:
        return 1.0
    return frac


def r_eval(
    score: float,
    failures: Iterable[str] = (),
    *,
    lo: float = 0.0,
    hi: float = 10.0,
    penalties: Mapping[str, float] | None = None,
) -> float:
    """External-evaluation reward (Eq. 5) with failure-mode deductions.

    ``score`` is the raw evaluator score (normalised via :func:`normalize_score`)
    and ``failures`` is the set of detected harmful error categories. Each
    recognised failure subtracts its penalty (defaulting to
    ``DEFAULT_FAILURE_PENALTY`` for the two paper categories) from the
    normalised score, clamped to ``[0, 1]``. Unknown failure labels raise.
    """
    base = normalize_score(score, lo=lo, hi=hi)
    table = {
        REFERENCE_FRAME_MISALIGNMENT: DEFAULT_FAILURE_PENALTY,
        PARAMETRIC_MISASSIGNMENT: DEFAULT_FAILURE_PENALTY,
    }
    if penalties is not None:
        table = dict(penalties)
    for label in failures:
        if label not in table:
            raise KeyError(f"unknown failure mode: {label!r}")
        base -= table[label]
    if base < 0.0:
        return 0.0
    if base > 1.0:
        return 1.0
    return base


def total_reward(
    executes: bool,
    geom: float,
    evaluation: float,
    *,
    lambda_geom: float = DEFAULT_LAMBDA_GEOM,
    lambda_eval: float = DEFAULT_LAMBDA_EVAL,
) -> float:
    """Total CAD-RL reward (Eq. 6).

    ``R = R_exec * (lambda_geom * R_geom + lambda_eval * R_eval)``. ``geom`` and
    ``evaluation`` are the already-computed ``R_geom`` and ``R_eval`` terms in
    ``[0, 1]``; when the program does not execute the whole reward is gated to 0.
    """
    if lambda_geom < 0.0 or lambda_eval < 0.0:
        raise ValueError("reward weights must be non-negative")
    gate = r_exec(executes)
    return gate * (lambda_geom * float(geom) + lambda_eval * float(evaluation))
