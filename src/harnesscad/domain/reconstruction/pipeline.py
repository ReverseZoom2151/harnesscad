"""Composed SVG-to-topology reconstruction with auditable stage reports."""

from __future__ import annotations

from .edges import normalize_edges
from .model import ReconstructionResult, StageReport, StitchStatus
from .patterns import match_patterns
from .svg import parse_svg
from .topology import cluster_planar_loops, find_face_loops, manifold_gate


def reconstruct(svg: str, *, scale: float = 1.0, tolerance: float = 0.005,
                stitcher=None) -> ReconstructionResult:
    reports = []
    drawing, diagnostics = parse_svg(svg, scale=scale, tolerance=tolerance)
    reports.append(StageReport("parse", len(svg), 0 if drawing is None else
                               sum(len(v.edges) for v in drawing.views.values()), diagnostics))
    if drawing is None or any(item.severity == "error" for item in diagnostics):
        return ReconstructionResult(drawing, (), (), (), False, tuple(reports),
                                    diagnostics, StitchStatus("not_requested"))
    normalized = {}
    before = 0
    for name, view in drawing.views.items():
        before += len(view.edges)
        normalized[name] = normalize_edges(view.edges, tolerance)
    reports.append(StageReport("normalize", before, sum(map(len, normalized.values()))))
    edges, match_diags = match_patterns(normalized, tolerance)
    reports.append(StageReport("match_edges", sum(map(len, normalized.values())),
                               len(edges), match_diags))
    loops = find_face_loops(edges, tolerance)
    reports.append(StageReport("detect_loops", len(edges), len(loops)))
    faces = cluster_planar_loops(loops, tolerance)
    reports.append(StageReport("cluster_faces", len(loops), len(faces)))
    manifold, gate_diags = manifold_gate(faces, len(edges))
    reports.append(StageReport("manifold_gate", len(edges), int(manifold), gate_diags))
    all_diags = diagnostics + match_diags + gate_diags
    if stitcher is None:
        stitch = StitchStatus("unavailable", "no kernel stitch adapter was supplied")
    else:
        try:
            artifact = stitcher(edges, faces)
            stitch = StitchStatus("succeeded", artifact=artifact)
        except Exception as exc:  # adapter boundary: structured, never fatal
            stitch = StitchStatus("failed", f"{type(exc).__name__}: {exc}")
    reports.append(StageReport("stitch", len(faces), int(stitch.status == "succeeded")))
    return ReconstructionResult(drawing, edges, loops, faces, manifold,
                                tuple(reports), all_diags, stitch)
