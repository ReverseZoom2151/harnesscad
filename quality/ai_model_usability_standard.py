"""AI-generated 3D model usability standard.

Deterministic implementation of the multi-step validation framework proposed by
Kung & Liang (2025), "Exploring the Usability and Future Development of
AI-Generated 3D Models in CAD Workflows and the Metaverse Based on 3D Model
Standards" (Computer-Aided Design & Applications 22(5), 782-804).

The paper is largely a qualitative usability study, but three of its evaluation
criteria are fully mechanical and reproducible from geometry alone. This module
implements those three as stdlib-only, deterministic scorers:

  1. Topology uniformity via closed-edge-loop size statistics -- the count of
     closed edge loops, the average loop size (paper eqn 2), the standard
     deviation of loop sizes (eqn 3), and the coefficient of variation
     CV = std / mean (eqn 4), with the paper's variability bands (Section
     4.3.3: CV < 0.10 low, CV > 0.50 high). Reproduces Table 9's methodology.

  2. Polygon-budget conformance against the VR/AR face-count bands quoted from
     3D-Ace in Section 4.3.1 (low/high-detail characters, simple/complex props).

  3. Basic mesh-defect readiness gate over the six Blender 3D Print Toolbox
     checks of Table 4 / Table 8 (non-manifold edges, bad contiguous edges,
     intersecting faces, zero-area faces, zero-length edges, non-flat faces).
     A model is "clean" for downstream CAD only when every count is zero.

These roll up into an overall usability verdict for AI-generated models entering
CAD / CAM / CAE workflows.

No trained models, no external tooling: every number below is derived purely
from counts already produced by a mesh inspector.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# 1. Closed-edge-loop size statistics (topology uniformity, eqns 1-4)
# ---------------------------------------------------------------------------

# Section 4.3.3: "A smaller CV (typically less than 10%) indicates low data
# variability, while a larger CV (typically greater than 50%) indicates high
# data variability."
CV_LOW_THRESHOLD = 0.10
CV_HIGH_THRESHOLD = 0.50


@dataclass(frozen=True)
class LoopSizeStats:
    """Distribution statistics over the sizes of closed edge loops."""

    closed_loop_count: int
    average_loop_size: float
    standard_deviation: float
    coefficient_of_variation: float
    variability: str  # "low", "moderate", or "high"


def classify_variability(cv: float) -> str:
    """Bucket a coefficient of variation per the paper's bands."""
    if cv < CV_LOW_THRESHOLD:
        return "low"
    if cv > CV_HIGH_THRESHOLD:
        return "high"
    return "moderate"


def loop_size_statistics(loop_sizes: Sequence[float]) -> LoopSizeStats:
    """Compute closed-edge-loop size statistics.

    ``loop_sizes`` is one entry per closed edge loop, each entry being that
    loop's size (its edge count, paper eqn 1). Returns the count, mean (eqn 2),
    population standard deviation (eqn 3), coefficient of variation (eqn 4) and
    the variability band. An empty input yields a zeroed, "low" result.
    """
    sizes = [float(s) for s in loop_sizes]
    n = len(sizes)
    if n == 0:
        return LoopSizeStats(0, 0.0, 0.0, 0.0, "low")
    mean = sum(sizes) / n
    variance = sum((s - mean) ** 2 for s in sizes) / n
    std = sqrt(variance)
    cv = std / mean if mean else 0.0
    return LoopSizeStats(n, mean, std, cv, classify_variability(cv))


def loop_sizes_from_loops(loops: Iterable[Sequence[object]]) -> list[int]:
    """Reduce explicit edge loops to their sizes (edge counts).

    Each loop is a sequence of edges; its size is ``len(loop)``. Convenience for
    callers that carry full loop membership rather than pre-counted sizes.
    """
    return [len(loop) for loop in loops]


# ---------------------------------------------------------------------------
# 2. Quad-based topology fraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuadTopology:
    quad_count: int
    total_faces: int
    quad_fraction: float
    quad_based: bool  # majority quads -> preferred per McCallum topology rules


def quad_topology(quad_count: int, total_faces: int, *, threshold: float = 0.5) -> QuadTopology:
    """Fraction of faces that are quads; quad-based when it clears ``threshold``.

    The paper (Table 5) recommends quad-based meshes over triangles/n-gons for
    deformation quality. Low-poly, real-time targets legitimately use triangles
    (Section 4.3), so this is reported, not gated.
    """
    if quad_count < 0 or total_faces < 0:
        raise ValueError("counts must be non-negative")
    if quad_count > total_faces:
        raise ValueError("quad_count cannot exceed total_faces")
    fraction = quad_count / total_faces if total_faces else 0.0
    return QuadTopology(quad_count, total_faces, fraction, fraction >= threshold)


# ---------------------------------------------------------------------------
# 3. Polygon-budget conformance (Section 4.3.1, VR/AR bands from 3D-Ace)
# ---------------------------------------------------------------------------

# (inclusive_low, inclusive_high) face-count bands quoted in Section 4.3.1.
POLYGON_BUDGETS: dict[str, tuple[int, int]] = {
    "low_detail_character": (2000, 10000),
    "high_detail_character": (10000, 20000),
    "simple_prop": (500, 1500),
    "complex_prop": (1500, 5000),
}


@dataclass(frozen=True)
class BudgetResult:
    category: str
    face_count: int
    low: int
    high: int
    status: str  # "under", "within", or "over"


def polygon_budget_check(face_count: int, category: str) -> BudgetResult:
    """Classify a model's face count against its VR/AR polygon budget band."""
    if category not in POLYGON_BUDGETS:
        raise KeyError(
            f"unknown category {category!r}; expected one of {sorted(POLYGON_BUDGETS)}"
        )
    if face_count < 0:
        raise ValueError("face_count must be non-negative")
    low, high = POLYGON_BUDGETS[category]
    if face_count < low:
        status = "under"
    elif face_count > high:
        status = "over"
    else:
        status = "within"
    return BudgetResult(category, face_count, low, high, status)


# ---------------------------------------------------------------------------
# 4. Basic mesh-defect readiness gate (Table 4 / Table 8)
# ---------------------------------------------------------------------------

# The six Blender 3D Print Toolbox checks of Table 4, in reporting order.
BASIC_DEFECT_CHECKS: tuple[str, ...] = (
    "non_manifold_edges",
    "bad_contiguous_edges",
    "intersecting_faces",
    "zero_area_faces",
    "zero_length_edges",
    "non_flat_faces",
)


@dataclass(frozen=True)
class DefectReadiness:
    counts: Mapping[str, int]
    total_defects: int
    failing_checks: tuple[str, ...]
    clean: bool  # every basic check is zero -> ready for CAD downstream


def mesh_defect_readiness(defects: Mapping[str, int]) -> DefectReadiness:
    """Roll the six basic mesh checks into a downstream-CAD readiness gate.

    Unknown keys are rejected so a caller cannot silently pass a defect under a
    misspelled name. Missing checks default to zero (absent == clean). A model
    is "clean" only when every one of the six counts is zero (Section 3.4:
    zero-area/overlapping/non-planar faces and non-manifold edges each break
    Boolean and feature operations).
    """
    unknown = set(defects) - set(BASIC_DEFECT_CHECKS)
    if unknown:
        raise KeyError(f"unknown defect checks: {sorted(unknown)}")
    counts: dict[str, int] = {}
    for check in BASIC_DEFECT_CHECKS:
        value = int(defects.get(check, 0))
        if value < 0:
            raise ValueError(f"defect count for {check!r} must be non-negative")
        counts[check] = value
    failing = tuple(c for c in BASIC_DEFECT_CHECKS if counts[c] > 0)
    total = sum(counts.values())
    return DefectReadiness(counts, total, failing, not failing)


# ---------------------------------------------------------------------------
# Combined multi-step usability verdict
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UsabilityReport:
    defects: DefectReadiness
    topology: LoopSizeStats
    budget: BudgetResult | None
    verdict: str  # "usable", "needs_refinement", or "not_ready"
    reasons: tuple[str, ...]


def evaluate_model_usability(
    defects: Mapping[str, int],
    loop_sizes: Sequence[float],
    *,
    face_count: int | None = None,
    category: str | None = None,
) -> UsabilityReport:
    """Combine the three deterministic checks into one usability verdict.

    Verdict logic (mirrors the paper's multi-step framework, Section 5):

      * "not_ready"        -- any basic mesh defect present (blocks CAD kernels);
      * "needs_refinement" -- clean mesh but non-uniform topology (CV high) or a
                              face count outside its VR/AR budget band;
      * "usable"           -- clean mesh, non-high topology variability, and (if
                              supplied) a within-budget face count.

    ``face_count``/``category`` are optional; when omitted the budget step is
    skipped and does not affect the verdict.
    """
    readiness = mesh_defect_readiness(defects)
    topology = loop_size_statistics(loop_sizes)
    budget = (
        polygon_budget_check(face_count, category)
        if face_count is not None and category is not None
        else None
    )

    reasons: list[str] = []
    if not readiness.clean:
        reasons.append(
            "basic mesh defects present: " + ", ".join(readiness.failing_checks)
        )
    if topology.variability == "high":
        reasons.append(
            f"uneven edge-loop distribution (CV={topology.coefficient_of_variation:.3f})"
        )
    if budget is not None and budget.status != "within":
        reasons.append(
            f"face count {budget.face_count} {budget.status} budget "
            f"[{budget.low}, {budget.high}] for {budget.category}"
        )

    if not readiness.clean:
        verdict = "not_ready"
    elif topology.variability == "high" or (budget is not None and budget.status != "within"):
        verdict = "needs_refinement"
    else:
        verdict = "usable"

    return UsabilityReport(readiness, topology, budget, verdict, tuple(reasons))
