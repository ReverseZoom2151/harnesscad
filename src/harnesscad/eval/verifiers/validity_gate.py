"""CAD validity as a hard scoring gate, plus non-gating advisory diagnostics.

CADGenBench runs this before every other metric, on the raw candidate. Any
failure sets ``is_valid = False`` and forces ``cad_score = 0``, with a
human-readable reason, so an invalid solid can never outrank a worse-but-valid
one. A candidate is valid when **all** of:

1. **Well-formed BREP** - the kernel's analyzer reports no per-face, per-edge or
   per-vertex errors (self-intersecting wires, edges off their surface, ...).
   The kernel is external, so its verdict is passed in as a list of errors.
2. **Watertight** - every shell is closed; no naked or free edges.
3. **Meshable as a closed orientable manifold** - tessellation yields a triangle
   mesh where every edge is shared by exactly two triangles, traversed in
   opposite directions. Checked here, deterministically, via
   :func:`bench.cgb_mesh_betti.mesh_gate_errors`.

The **advisory diagnostics** are the second half of the contract and the part
that is easy to get wrong: sliver faces, loose tolerances and near-degenerate
features are *flagged, never gated*. They identify geometry worth cleaning up;
they never move the score. Their thresholds are calibrated to sit far from
healthy geometry (healthy face areas bottom out near 0.05 mm^2, genuine defects
fall to 1e-19; healthy aspect ratios are single- to double-digit, slivers run
1e5-5e8), so a flag is a real signal rather than a nag.

The harness's existing ``verifiers.geometry.BRepValidityCheck`` reports a
backend's validity as a diagnostic; it neither gates a composite score nor
carries the advisory tier. ``bench.text2cad2_invalidity_ratio`` checks *command
sequences*, not solids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from harnesscad.eval.bench.geometry.betti_graded import MeshSurface, mesh_gate_errors

# Advisory (never gating) thresholds.
MIN_FACE_AREA_MM2 = 0.001
MAX_FACE_ASPECT_RATIO = 1000.0
MAX_BREP_TOLERANCE_MM = 0.1

STATUS_VALID = "valid"
STATUS_INVALID = "invalid"
STATUS_MISSING = "missing"


@dataclass(frozen=True)
class ValidityResult:
    """The gate's verdict plus the advisory tier that rides alongside it."""

    is_valid: bool
    reasons: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    is_watertight: bool = True
    diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return STATUS_VALID if self.is_valid else STATUS_INVALID

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "is_watertight": self.is_watertight,
            "reasons": list(self.reasons),
            "flags": list(self.flags),
            "diagnostics": dict(self.diagnostics),
        }


def candidate_status(*, candidate_exists: bool, is_valid: bool) -> str:
    """Map a candidate onto the three-way status the leaderboard reports.

    "The agent never produced an output" is a distinct outcome from "it produced
    an invalid one", even though both score zero.
    """
    if not candidate_exists:
        return STATUS_MISSING
    return STATUS_VALID if is_valid else STATUS_INVALID


def triangle_area(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    ab = [b[i] - a[i] for i in range(3)]
    ac = [c[i] - a[i] for i in range(3)]
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * math.sqrt(sum(x * x for x in cross))


def _aspect_ratio(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Face length / width: the longest edge over the height across from it.

    A sliver (a face far longer than it is wide) blows this up; the reciprocal
    ``2 * area / longest_edge`` is the face's width, so the ratio is
    ``longest_edge^2 / (2 * area)``. A fully degenerate face has zero area and
    reports infinity, which is exactly the flag it deserves.
    """
    edges = [
        math.dist(a, b),
        math.dist(b, c),
        math.dist(c, a),
    ]
    longest = max(edges)
    area = triangle_area(a, b, c)
    if area <= 0.0:
        return math.inf
    return (longest * longest) / (2.0 * area)


def mesh_face_diagnostics(mesh: MeshSurface) -> Dict[str, float]:
    """Minimum face area and maximum face aspect ratio over the tessellation."""
    if not mesh.triangles:
        return {}
    areas = []
    ratios = []
    for i, j, k in mesh.triangles:
        a, b, c = mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]
        areas.append(triangle_area(a, b, c))
        ratios.append(_aspect_ratio(a, b, c))
    return {
        "min_face_area": min(areas),
        "max_face_aspect_ratio": max(ratios),
    }


def advisory_flags(
    *,
    min_face_area: Optional[float] = None,
    max_face_aspect_ratio: Optional[float] = None,
    max_brep_tolerance: Optional[float] = None,
) -> List[str]:
    """Flag fragile-but-valid geometry. These NEVER affect the score."""
    flags: List[str] = []
    if min_face_area is not None and min_face_area < MIN_FACE_AREA_MM2:
        flags.append(
            f"min face area {min_face_area:.3e} mm^2 below {MIN_FACE_AREA_MM2} "
            "(near-degenerate face)"
        )
    if (
        max_face_aspect_ratio is not None
        and max_face_aspect_ratio > MAX_FACE_ASPECT_RATIO
    ):
        flags.append(
            f"max face aspect ratio {max_face_aspect_ratio:.3e} above "
            f"{MAX_FACE_ASPECT_RATIO} (sliver face)"
        )
    if max_brep_tolerance is not None and max_brep_tolerance > MAX_BREP_TOLERANCE_MM:
        flags.append(
            f"max BREP tolerance {max_brep_tolerance:.3e} mm above "
            f"{MAX_BREP_TOLERANCE_MM} mm (loose export)"
        )
    return flags


def validate_candidate(
    *,
    brep_errors: Optional[Sequence[str]] = None,
    is_watertight: bool = True,
    mesh: Optional[MeshSurface] = None,
    max_brep_tolerance: Optional[float] = None,
) -> ValidityResult:
    """Run the three gate stages, then attach the advisory tier.

    ``brep_errors`` is the external kernel's analyzer output (empty = clean);
    ``mesh`` is the tessellation, whose closed-orientable-manifold check runs
    here. A candidate that cannot be tessellated at all should be passed
    ``mesh=None``, which fails the gate.
    """
    reasons: List[str] = []
    for err in brep_errors or []:
        reasons.append(f"brep: {err}")
    if not is_watertight:
        reasons.append("not watertight: shell has naked or free edges")

    diagnostics: Dict[str, float] = {}
    if mesh is None:
        reasons.append("not meshable: tessellation produced no mesh")
    else:
        reasons.extend(f"mesh: {e}" for e in mesh_gate_errors(mesh))
        diagnostics.update(mesh_face_diagnostics(mesh))

    if max_brep_tolerance is not None:
        diagnostics["max_brep_tolerance"] = float(max_brep_tolerance)

    flags = advisory_flags(
        min_face_area=diagnostics.get("min_face_area"),
        max_face_aspect_ratio=diagnostics.get("max_face_aspect_ratio"),
        max_brep_tolerance=diagnostics.get("max_brep_tolerance"),
    )
    return ValidityResult(
        is_valid=not reasons,
        reasons=reasons,
        flags=flags,
        is_watertight=is_watertight,
        diagnostics=diagnostics,
    )
