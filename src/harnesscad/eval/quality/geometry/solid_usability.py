"""Solid-usability gate over *measured* geometry descriptors (cadsmith).

**cadsmith** is a self-correcting text-to-CAD agent: it asks a model for CadQuery
code, executes it, then measures the resulting B-Rep and -- crucially -- feeds
concrete, model-readable failure messages back so the model can self-correct.
Its ``validator.validate_shape`` (see ``src/cadsmith/validator.py``) treats the
solid as ground truth and rejects it for a fixed taxonomy of defects.

That validator is entangled with OpenCASCADE (it calls ``shape.Volume()``,
``BRepCheck`` etc.). This module lifts the *decision layer* out of OCCT: it
consumes an already-measured :class:`Measurement` (volume, bounding box, solid
count, free-edge count, validity flag -- whatever the caller's backend produced)
and applies cadsmith's checks as a deterministic gate that emits the exact
model-readable feedback string the self-correction loop needs.

The taxonomy (cadsmith's table, reframed as checkable metrics):

===============================  ==========================================
check                            catches
===============================  ==========================================
``no_solid``                     wires/faces left un-lofted / un-extruded
``below_min_volume``             a sketch that was never extruded
``degenerate_bbox``              a zero-thickness dimension
``units_mistake``                a model 1000x too large (bbox blows the max)
``malformed_brep``               self-intersecting / invalid geometry
``not_watertight``               open shell (free edges) that will not print
===============================  ==========================================

This is deliberately distinct from the differential oracle and the measured gate
in :mod:`harnesscad.eval`: those compare a candidate against a reference or a
target measurement; this is a *self-contained usability floor* -- "is this solid
even a usable part?" -- that needs no reference, exactly cadsmith's loop check.
It is also distinct from :mod:`harnesscad.domain.geometry.mesh.polyhedron`
(which computes topology from a raw mesh); here the topology is already measured.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__all__ = [
    "DEFAULT_MIN_VOLUME",
    "DEFAULT_MAX_DIM",
    "Measurement",
    "UsabilityFinding",
    "UsabilityReport",
    "assess",
]

DEFAULT_MIN_VOLUME = 1.0  # mm^3
DEFAULT_MAX_DIM = 10000.0  # mm


@dataclass(frozen=True)
class Measurement:
    """What a geometry backend measured about a candidate solid.

    ``free_edges`` is the number of edges bordering fewer than two faces (open
    shell markers); ``is_valid`` is the backend's B-Rep validity verdict. Any
    field left ``None`` is treated as "not measured" and its checks are skipped.
    """

    volume: Optional[float] = None
    bbox: Optional[Tuple[float, float, float]] = None
    n_solids: int = 0
    free_edges: Optional[int] = None
    is_valid: Optional[bool] = None


@dataclass(frozen=True)
class UsabilityFinding:
    """One usability defect."""

    code: str
    severity: str  # "fatal" (unusable) or "warning"
    message: str


@dataclass
class UsabilityReport:
    """The outcome of the usability gate."""

    ok: bool
    findings: List[UsabilityFinding] = field(default_factory=list)
    measurement: Optional[Measurement] = None

    @property
    def fatal(self) -> List[UsabilityFinding]:
        return [f for f in self.findings if f.severity == "fatal"]

    def codes(self) -> List[str]:
        return [f.code for f in self.findings]

    def describe(self) -> str:
        """One-line measurement summary (mirrors cadsmith's report)."""
        m = self.measurement
        if m is None:
            return "no measurement"
        parts: List[str] = []
        if m.volume is not None:
            parts.append(f"volume={m.volume:.3f} mm^3")
        if m.bbox is not None:
            x, y, z = m.bbox
            parts.append(f"bbox={x:.1f}x{y:.1f}x{z:.1f} mm")
        parts.append(f"solids={m.n_solids}")
        if m.is_valid is not None:
            parts.append(f"valid={m.is_valid}")
        if m.free_edges is not None:
            parts.append(f"free_edges={m.free_edges}")
        return ", ".join(parts)

    def failure_message(self) -> Optional[str]:
        """The concise, model-readable reason string to feed back, or None."""
        fatal = self.fatal
        if not fatal:
            return None
        return "invalid geometry: " + "; ".join(f.message for f in fatal)


def assess(
    measurement: Measurement,
    *,
    min_volume: float = DEFAULT_MIN_VOLUME,
    max_dim: float = DEFAULT_MAX_DIM,
) -> UsabilityReport:
    """Apply cadsmith's usability checks to a measured solid.

    A report is ``ok`` only when it carries no *fatal* finding. Warnings (e.g. a
    validity flag that was not measured) do not fail the gate.
    """
    findings: List[UsabilityFinding] = []

    # --- contains a solid ------------------------------------------------
    if measurement.n_solids <= 0:
        findings.append(
            UsabilityFinding(
                "no_solid",
                "fatal",
                "no solid present (wires/faces left un-lofted or un-extruded)",
            )
        )

    # --- volume ----------------------------------------------------------
    if measurement.volume is not None:
        if measurement.volume < min_volume:
            findings.append(
                UsabilityFinding(
                    "below_min_volume",
                    "fatal",
                    f"volume {measurement.volume:.3f} mm^3 below threshold {min_volume} "
                    "(sketch probably never extruded into a solid)",
                )
            )

    # --- bounding box ----------------------------------------------------
    if measurement.bbox is not None:
        x, y, z = measurement.bbox
        if min(x, y, z) <= 0:
            findings.append(
                UsabilityFinding(
                    "degenerate_bbox",
                    "fatal",
                    "degenerate bounding box (a dimension is zero)",
                )
            )
        if max(x, y, z) > max_dim:
            findings.append(
                UsabilityFinding(
                    "units_mistake",
                    "fatal",
                    f"bounding box {max(x, y, z):.1f} mm exceeds {max_dim} mm "
                    "(likely a units mistake -- CAD works in millimetres)",
                )
            )

    # --- OCCT validity ---------------------------------------------------
    if measurement.is_valid is False:
        findings.append(
            UsabilityFinding(
                "malformed_brep",
                "fatal",
                "malformed B-Rep (self-intersecting or invalid geometry)",
            )
        )

    # --- watertight ------------------------------------------------------
    if measurement.free_edges is not None and measurement.n_solids > 0:
        if measurement.free_edges > 0:
            findings.append(
                UsabilityFinding(
                    "not_watertight",
                    "fatal",
                    f"solid is not watertight ({measurement.free_edges} free edge(s) -- "
                    "open shell that will not 3D-print)",
                )
            )

    ok = not any(f.severity == "fatal" for f in findings)
    return UsabilityReport(ok=ok, findings=findings, measurement=measurement)
