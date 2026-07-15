"""CAE credibility ladder -- an evidence-tier classifier for CAE results.

Mined from **cad-cae-copilot** (AIENG workbench, ``docs/cae-credibility-ladder.md``
plus its V&V-40 credibility stamp). AIENG's discipline is to *never* conflate
"a result file exists" with "a solver ran" with "an engineer can rely on this
for a reviewed claim". It grades every result-bearing output by an ordered
credibility ladder, and it "always returns ``certified: false``".

This is distinct from the harness's existing CAE pieces
(:mod:`harnesscad.eval.bench.protocols.cad_cae_closed_loop` -- the run protocol;
:mod:`harnesscad.eval.quality.physics.cae_workflow` -- the workflow). Those
answer "did the loop run". This answers a different question: **how much should
anyone trust the number that came out**, on a fixed honesty ladder, with two
disciplines that can *cap* the tier no matter what evidence is present.

The ladder (low -> high credibility):

    no_result_artifact < artifact_present < solver_completed
      < numerical_result_parsed < plausibility_checked
      < design_target_compared < benchmark_calibrated
      < human_review_supported

Two caps (AIENG's "Mesh and Calibration Discipline"):

*   **Mesh discipline.** Unknown mesh quality keeps a limitation on the result;
    a *failed / not-converged* mesh caps the result strictly below
    ``benchmark_calibrated``, even when a numerical metric exists.
*   **Benchmark discipline.** ``benchmark_calibrated`` is only reachable when an
    analytical/reference comparison passes within a documented tolerance.

The V&V-40 tier (a second, coarser axis AIENG stamps on outputs):

    unverified < critique_finding < surrogate_prediction
      < proxy_assembly_result < executed_solver_result

with the invariant that a tier is **never more credible than its evidence** --
an output that claims ``solver`` but whose ``solver_executed`` is not true is
downgraded to ``unverified``.

Everything is a pure function over a small evidence record. Nothing is ever
``certified``. stdlib-only, deterministic, absolute imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__all__ = [
    "LADDER",
    "ladder_rank",
    "MeshStatus",
    "CaeEvidence",
    "CredibilityAssessment",
    "assess_cae_credibility",
    "VNV40_TIERS",
    "vnv40_rank",
    "assess_vnv40_tier",
]

# Ordered credibility ladder, low -> high.
LADDER: Tuple[str, ...] = (
    "no_result_artifact",
    "artifact_present",
    "solver_completed",
    "numerical_result_parsed",
    "plausibility_checked",
    "design_target_compared",
    "benchmark_calibrated",
    "human_review_supported",
)

_LADDER_INDEX = {name: i for i, name in enumerate(LADDER)}


def ladder_rank(level: str) -> int:
    """Ordinal rank of a ladder ``level`` (0 = lowest). Unknown -> -1."""
    return _LADDER_INDEX.get(level, -1)


# Mesh-quality evidence states.
class MeshStatus:
    UNKNOWN = "unknown"
    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    FAILED = "failed"


@dataclass(frozen=True)
class CaeEvidence:
    """The evidence record a CAE output carries.

    Each flag is a checkable fact about the run, not a judgement:

    * ``artifact_present``   -- a result-like file exists.
    * ``solver_completed``   -- the solver run record indicates completion.
    * ``metrics_parsed``     -- numerical metrics were extracted.
    * ``plausibility_checked`` -- units/signs/magnitudes/field locations checked.
    * ``design_target_compared`` -- metrics compared to explicit design targets.
    * ``benchmark_passed``   -- a reference/analytical case agreed within tolerance.
    * ``human_review_supported`` -- a reviewer supports a claim on this chain.
    * ``mesh_status``        -- one of :class:`MeshStatus`.
    """

    artifact_present: bool = False
    solver_completed: bool = False
    metrics_parsed: bool = False
    plausibility_checked: bool = False
    design_target_compared: bool = False
    benchmark_passed: bool = False
    human_review_supported: bool = False
    mesh_status: str = MeshStatus.UNKNOWN


@dataclass(frozen=True)
class CredibilityAssessment:
    """Result of :func:`assess_cae_credibility`.

    ``level`` is the highest ladder level the evidence supports after caps;
    ``uncapped_level`` is what the evidence alone would reach; ``limitations``
    explains any gap. ``certified`` is *always* ``False``.
    """

    level: str
    rank: int
    uncapped_level: str
    limitations: Tuple[str, ...] = ()
    certified: bool = False


def _evidence_level(ev: CaeEvidence) -> str:
    """Highest ladder level supported by the raw evidence, ignoring caps.

    The ladder is cumulative: each higher level presupposes the ones below it,
    so we walk up while every prerequisite flag holds.
    """
    level = "no_result_artifact"
    if not ev.artifact_present:
        return level
    level = "artifact_present"
    if not ev.solver_completed:
        return level
    level = "solver_completed"
    if not ev.metrics_parsed:
        return level
    level = "numerical_result_parsed"
    if not ev.plausibility_checked:
        return level
    level = "plausibility_checked"
    if not ev.design_target_compared:
        return level
    level = "design_target_compared"
    if not ev.benchmark_passed:
        return level
    level = "benchmark_calibrated"
    if not ev.human_review_supported:
        return level
    return "human_review_supported"


def assess_cae_credibility(ev: CaeEvidence) -> CredibilityAssessment:
    """Grade a CAE result on the credibility ladder, applying mesh discipline.

    Never mutates, never certifies. A failed / not-converged mesh caps the
    result strictly below ``benchmark_calibrated``; an unknown mesh records a
    limitation but does not lower a sub-benchmark level.
    """
    uncapped = _evidence_level(ev)
    level = uncapped
    limitations: List[str] = []

    if ev.mesh_status in (MeshStatus.FAILED, MeshStatus.NOT_CONVERGED):
        # Cap strictly below benchmark_calibrated.
        cap = "design_target_compared"
        if ladder_rank(level) > ladder_rank(cap):
            level = cap
        limitations.append(
            f"mesh {ev.mesh_status}: capped below benchmark_calibrated"
        )
    elif ev.mesh_status == MeshStatus.UNKNOWN:
        limitations.append("mesh quality unknown: result carries a limitation")

    if ev.benchmark_passed and ev.mesh_status != MeshStatus.CONVERGED:
        limitations.append(
            "benchmark agreement is regression evidence, not production safety"
        )

    return CredibilityAssessment(
        level=level,
        rank=ladder_rank(level),
        uncapped_level=uncapped,
        limitations=tuple(limitations),
        certified=False,
    )


# ---------------------------------------------------------------------------
# V&V-40 credibility tier (the coarser second axis AIENG stamps on outputs).
# ---------------------------------------------------------------------------

VNV40_TIERS: Tuple[str, ...] = (
    "unverified",
    "critique_finding",
    "surrogate_prediction",
    "proxy_assembly_result",
    "executed_solver_result",
)

_VNV40_INDEX = {name: i for i, name in enumerate(VNV40_TIERS)}


def vnv40_rank(tier: str) -> int:
    """Ordinal rank of a V&V-40 ``tier`` (0 = unverified). Unknown -> -1."""
    return _VNV40_INDEX.get(tier, -1)


def assess_vnv40_tier(claimed_tier: str, *, solver_executed: bool) -> Tuple[str, Optional[str]]:
    """Enforce the honesty invariant on a claimed V&V-40 tier.

    Returns ``(effective_tier, downgrade_reason)``. A claim of
    ``executed_solver_result`` whose ``solver_executed`` is not true is
    downgraded to ``unverified`` -- a tier is never more credible than its
    evidence. Any unknown tier is treated as ``unverified``.
    """
    if claimed_tier not in _VNV40_INDEX:
        return "unverified", f"unknown tier {claimed_tier!r}"
    if claimed_tier == "executed_solver_result" and not solver_executed:
        return "unverified", "claims solver result but solver_executed is not true"
    return claimed_tier, None
