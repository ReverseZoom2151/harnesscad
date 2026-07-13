"""Topological validity checker for CMT-generated B-Reps (the "Valid" metric).

CMT reports a *Valid ratio*: the fraction of generated models free of "broken
topology" -- the paper names "unbounded open regions" and "self-intersecting
edges" as the failure modes the cascade + topology predictor suppress. The
checker here operates on the edge-surface adjacency matrix ``R`` produced by
``cmt_topology_predictor`` (optionally with edge geometry) and flags:

  * **unbounded open region** -- a surface incident to no edges (open, not
    contoured), i.e. an empty column of ``R``;
  * **non-manifold edge** -- an edge that does not bound exactly two surfaces
    ("edges contour a surface"): a dangling edge (< 2) or an over-shared edge
    (> 2), i.e. a row of ``R`` whose sum != 2;
  * **degenerate / self-intersecting edge** (geometry, optional) -- a zero-length
    edge or two edges collapsing onto identical endpoints, e.g. after 4-bit
    quantization ("after 4 bit quantization, the Valid ratio ...").

``valid_ratio`` reproduces the batch-level metric over many models.
"""

from __future__ import annotations

from dataclasses import dataclass

from harnesscad.domain.reconstruction.cmt_tokenization import quantize

Point = tuple[float, float, float]


@dataclass(frozen=True)
class ValidityDiagnostic:
    code: str
    detail: str
    context: tuple = ()


def check_adjacency(adjacency: tuple[tuple[bool, ...], ...],
                    require_edges: int = 2) -> tuple[bool, tuple[ValidityDiagnostic, ...]]:
    """Validate an edge-surface adjacency matrix ``R`` (rows edges, cols surfaces)."""
    diagnostics: list[ValidityDiagnostic] = []
    if not adjacency:
        return True, ()
    n_surfaces = len(adjacency[0])
    if any(len(row) != n_surfaces for row in adjacency):
        raise ValueError("adjacency rows must have equal length")

    # Columns: each surface must be contoured by at least one edge.
    open_surfaces = tuple(
        s for s in range(n_surfaces)
        if not any(row[s] for row in adjacency)
    )
    for s in open_surfaces:
        diagnostics.append(ValidityDiagnostic(
            "unbounded-open-region",
            "surface is contoured by no edge", context=(s,)))

    # Rows: each edge must bound exactly ``require_edges`` surfaces (manifold).
    for e, row in enumerate(adjacency):
        count = sum(1 for v in row if v)
        if count < require_edges:
            diagnostics.append(ValidityDiagnostic(
                "dangling-edge",
                f"edge bounds {count} surfaces, expected {require_edges}",
                context=(e, count)))
        elif count > require_edges:
            diagnostics.append(ValidityDiagnostic(
                "over-shared-edge",
                f"edge bounds {count} surfaces, expected {require_edges}",
                context=(e, count)))
    return not diagnostics, tuple(diagnostics)


def check_edge_geometry(edges: tuple[tuple[Point, Point], ...],
                        tolerance: float = 0.0
                        ) -> tuple[bool, tuple[ValidityDiagnostic, ...]]:
    """Flag zero-length edges and duplicate (self-overlapping) edges."""
    diagnostics: list[ValidityDiagnostic] = []
    seen: dict = {}
    for i, (start, end) in enumerate(edges):
        length = sum((a - b) ** 2 for a, b in zip(start, end)) ** 0.5
        if length <= tolerance:
            diagnostics.append(ValidityDiagnostic(
                "degenerate-edge", "edge has zero length", context=(i,)))
        key = tuple(sorted((tuple(start), tuple(end))))
        if key in seen:
            diagnostics.append(ValidityDiagnostic(
                "self-intersecting-edge",
                "edge duplicates an existing edge", context=(seen[key], i)))
        else:
            seen[key] = i
    return not diagnostics, tuple(diagnostics)


def is_valid(adjacency: tuple[tuple[bool, ...], ...],
             edges: tuple[tuple[Point, Point], ...] | None = None,
             require_edges: int = 2) -> bool:
    """Whole-model validity: adjacency plus optional edge geometry."""
    ok, _ = check_adjacency(adjacency, require_edges)
    if edges is not None:
        geo_ok, _ = check_edge_geometry(edges)
        ok = ok and geo_ok
    return ok


def quantized_is_valid(adjacency: tuple[tuple[bool, ...], ...],
                       edges: tuple[tuple[Point, Point], ...],
                       bits: int = 4,
                       lo: float = 0.0, hi: float = 1.0) -> bool:
    """Validity after quantizing vertex coordinates to ``bits`` (paper: 4-bit).

    Quantization can collapse distinct vertices, turning valid edges degenerate;
    this reproduces the paper's stricter quantized Valid check.
    """
    def q(point: Point) -> Point:
        return tuple(quantize(v, bits, lo, hi) for v in point)
    q_edges = tuple((q(s), q(e)) for (s, e) in edges)
    return is_valid(adjacency, q_edges)


def valid_ratio(models: tuple[tuple[tuple[tuple[bool, ...], ...],
                                    tuple[tuple[Point, Point], ...] | None], ...]) -> float:
    """Fraction of ``(adjacency, edges)`` models that are valid (the metric)."""
    if not models:
        return 0.0
    good = sum(1 for adjacency, edges in models if is_valid(adjacency, edges))
    return good / len(models)
