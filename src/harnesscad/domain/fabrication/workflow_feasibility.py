"""Per-workflow feasibility analysis (CAMeleon's "learning from limits").

Feng et al. stress that a comparison tool must "show what is not working": each
workflow gets its *own* feasibility rule rather than one generic check
(Section 7.1.3, Figure 16, Appendix Figure 22). This module implements that
dispatch of workflow-specific, deterministic checks:

  * ``machine_fit``   -- part bounding box vs the machine work envelope; if it
                         overflows, how many pieces must it split into
                         (Figure 4a: "splitting into multiple parts for the
                         available printer bed size").
  * ``print_time``    -- deterministic FDM print-time estimate from volume,
                         infill and material (Figure 4b: "7,479 minutes ...
                         20% infill, generic PLA").
  * ``material_stock``-- snap a requested sheet thickness to the discrete stock
                         a shop actually carries (Figure 15c).
  * ``wire_form``     -- minimum feasible segment length and bend angle for a
                         wire-bent frame; flag bends a standard bender cannot
                         make (Figure 16b).
  * ``foam_load``     -- warn when a non-load-bearing material (foam, felt,
                         pulp) is used structurally (the "foam strength tip").
  * ``draft_angle``   -- warn when a mold face has insufficient / negative draft
                         to demold (Figure 16a).

These are *workflow-level* feasibility rules keyed on the chosen paradigm; they
complement, and never duplicate, the per-solid heuristics in ``verifiers/dfm.py``
(wall thickness / draft on a single OCCT solid) or the cost/BOM math in
``quality/estimate.py``. Findings are advisory, mirroring the DFM critic's
severity model: ``ok`` / ``warning`` / ``error`` (error only for a truly
impossible fabrication, e.g. a bend sharper than the bender can physically make).

Stdlib-only, deterministic. A "part" is described by a light dict/dataclass of
measurements a caller already has (bbox, volume, thickness, wire polyline) — no
geometry kernel is invoked here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.fabrication.workflow_taxonomy import MACHINES, MATERIALS, WORKFLOWS, get_workflow


# --------------------------------------------------------------------------- #
# Finding model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Finding:
    """One feasibility finding.

    ``severity`` is "ok" | "warning" | "error". ``data`` carries the computed
    numbers (split counts, estimated minutes, snapped thickness) so a UI can
    render them without re-deriving.
    """

    check: str
    severity: str
    message: str
    data: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.severity == "ok"


@dataclass(frozen=True)
class PartSpec:
    """The measurements a feasibility check may consult.

    Only the fields a given check needs must be supplied; the rest stay ``None``
    and dependent checks degrade to an INFO-style skip rather than crash.

      * ``bbox``            -- (x, y, z) extent in mm.
      * ``volume_mm3``      -- solid volume in mm^3 (for print time).
      * ``sheet_thickness`` -- requested plate thickness in mm.
      * ``wire_segments``   -- ordered (length_mm, bend_angle_deg) pairs; the
                               bend angle is the turn *after* that segment.
      * ``load_bearing``    -- whether the part must carry structural load.
      * ``min_draft_deg``   -- the smallest draft angle over the part's faces
                               (negative = undercut).
    """

    bbox: Optional[Tuple[float, float, float]] = None
    volume_mm3: Optional[float] = None
    sheet_thickness: Optional[float] = None
    wire_segments: Optional[Sequence[Tuple[float, float]]] = None
    load_bearing: bool = False
    min_draft_deg: Optional[float] = None


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def _pick_machine(workflow_id: str, machine_id: Optional[str]) -> Optional[str]:
    """Resolve which machine a workflow runs on (explicit override or first)."""
    wf = get_workflow(workflow_id)
    if machine_id is not None:
        if machine_id not in wf.machines:
            raise ValueError(
                f"machine {machine_id!r} not valid for workflow {workflow_id!r}"
            )
        return machine_id
    return wf.machines[0] if wf.machines else None


def check_machine_fit(
    workflow_id: str, part: PartSpec, machine_id: Optional[str] = None
) -> Finding:
    """Does the part fit the machine envelope? If not, how many splits?

    Splitting is per-axis ceil(extent / envelope); the product is the number of
    pieces the part must be divided into to fabricate on this machine.
    """
    if part.bbox is None:
        return Finding("machine_fit", "ok", "no bbox supplied; fit not evaluated")
    mid = _pick_machine(workflow_id, machine_id)
    if mid is None or MACHINES[mid].kind == "manual":
        return Finding("machine_fit", "ok", "no machine envelope constraint")
    env = MACHINES[mid].work_volume
    # Sort both so we test the tightest packing (largest part axis vs largest
    # envelope axis) — orientation is free within the bed.
    part_axes = sorted(part.bbox, reverse=True)
    env_axes = sorted(env, reverse=True)
    splits = []
    for p, e in zip(part_axes, env_axes):
        if e <= 0:
            splits.append(1)
        else:
            splits.append(max(1, math.ceil(p / e)))
    total = splits[0] * splits[1] * splits[2]
    if total <= 1:
        return Finding(
            "machine_fit", "ok",
            f"part fits {MACHINES[mid].name} envelope",
            {"machine": mid, "splits": 1},
        )
    return Finding(
        "machine_fit", "warning",
        f"part exceeds {MACHINES[mid].name} envelope; split into {total} pieces",
        {"machine": mid, "splits": total, "per_axis": tuple(splits)},
    )


def estimate_print_time(
    part: PartSpec, infill: float = 0.20, material: str = "pla"
) -> Finding:
    """Deterministic FDM print-time estimate in minutes.

    Model: the extruded solid volume is the shell plus an infill fraction of the
    interior. We approximate deposited volume as ``V * (shell + (1-shell)*infill)``
    with a fixed shell fraction, then divide by a material-specific volumetric
    deposition rate (mm^3/min). This is an order-of-magnitude planning estimate
    (matching the paper's "7,479 minutes" flavour), not a slicer.
    """
    if part.volume_mm3 is None:
        return Finding("print_time", "ok", "no volume supplied; time not estimated")
    infill = max(0.0, min(1.0, infill))
    shell_fraction = 0.35  # perimeters/top/bottom always solid
    deposited = part.volume_mm3 * (shell_fraction + (1.0 - shell_fraction) * infill)
    # Volumetric deposition rate, mm^3/min. Slower for higher-temp materials.
    rate = {"pla": 300.0, "petg": 240.0, "abs": 210.0}.get(material, 250.0)
    minutes = deposited / rate
    minutes_r = round(minutes)
    hours = minutes / 60.0
    sev = "warning" if minutes > 24 * 60 else "ok"
    msg = f"estimated print time {minutes_r} min ({hours:.1f} h) at {int(infill*100)}% infill, {material}"
    if sev == "warning":
        msg += " -- exceeds 24 h; consider a faster workflow or splitting"
    return Finding("print_time", sev, msg,
                   {"minutes": minutes_r, "hours": round(hours, 1),
                    "infill": infill, "material": material})


def check_material_stock(
    workflow_id: str, part: PartSpec, material: Optional[str] = None
) -> Finding:
    """Snap a requested sheet thickness to available stock gauges (Figure 15c).

    If the design asks for a thickness the shop does not stock, snap to the
    nearest available gauge and warn; if it matches, pass.
    """
    if part.sheet_thickness is None:
        return Finding("material_stock", "ok", "no sheet thickness requested")
    wf = get_workflow(workflow_id)
    mat_id = material if material is not None else (wf.materials[0] if wf.materials else None)
    if mat_id is None or mat_id not in MATERIALS:
        return Finding("material_stock", "ok", "no material with stock gauges")
    stock = MATERIALS[mat_id]
    if not stock.sheet_thicknesses:
        return Finding("material_stock", "ok",
                       f"{stock.name} is not a sheet stock")
    req = part.sheet_thickness
    if req in stock.sheet_thicknesses:
        return Finding("material_stock", "ok",
                       f"{req} mm matches {stock.name} stock",
                       {"requested": req, "snapped": req})
    nearest = min(stock.sheet_thicknesses, key=lambda t: (abs(t - req), t))
    return Finding("material_stock", "warning",
                   f"{req} mm not stocked for {stock.name}; snap to {nearest} mm",
                   {"requested": req, "snapped": nearest,
                    "available": stock.sheet_thicknesses})


def check_kerf(workflow_id: str, part: PartSpec, kerf_mm: float = 0.2) -> Finding:
    """Advisory: laser-cut joints need kerf compensation (Figure 17 checklist)."""
    return Finding("kerf", "ok",
                   f"apply ~{kerf_mm} mm kerf compensation to slot joints",
                   {"kerf_mm": kerf_mm})


def check_wire_form(
    part: PartSpec,
    min_segment_mm: float = 15.0,
    max_bend_deg: float = 135.0,
) -> Finding:
    """Wire-forming feasibility (Figure 16b).

    A standard bender needs a minimum straight length between bends to grip the
    wire, and cannot exceed a maximum turn angle in one bend. Segments shorter
    than ``min_segment_mm`` or bends sharper than ``max_bend_deg`` are flagged;
    an impossible bend is an *error* (physically unmakeable), a short segment is
    a *warning*.
    """
    if not part.wire_segments:
        return Finding("wire_form", "ok", "no wire polyline supplied")
    short: List[int] = []
    impossible: List[int] = []
    for i, (length, angle) in enumerate(part.wire_segments):
        if length < min_segment_mm:
            short.append(i)
        if abs(angle) > max_bend_deg:
            impossible.append(i)
    if impossible:
        return Finding("wire_form", "error",
                       f"bend(s) at segment {impossible} exceed {max_bend_deg} deg "
                       f"max bend; not achievable on a standard wire bender",
                       {"impossible_bends": impossible, "short_segments": short,
                        "min_segment_mm": min_segment_mm, "max_bend_deg": max_bend_deg})
    if short:
        return Finding("wire_form", "warning",
                       f"segment(s) {short} shorter than {min_segment_mm} mm; "
                       f"bender may not grip",
                       {"short_segments": short, "min_segment_mm": min_segment_mm})
    return Finding("wire_form", "ok", "all wire segments and bends feasible",
                   {"segments": len(part.wire_segments)})


def check_foam_load(workflow_id: str, part: PartSpec) -> Finding:
    """Warn when a non-load-bearing material is used structurally.

    Covers the paper's "Standard EPS/XPS foam has limited load-bearing capacity"
    tip, and generalizes to felt / papier-mache which also cannot carry load.
    """
    wf = get_workflow(workflow_id)
    non_structural = any(
        (m in MATERIALS and not MATERIALS[m].load_bearing) for m in wf.materials
    )
    if not non_structural:
        return Finding("foam_load", "ok", "material is load-bearing")
    if part.load_bearing:
        return Finding("foam_load", "warning",
                       f"{wf.name} uses a non-load-bearing material; limited "
                       f"load capacity for a structural/furniture part",
                       {"load_bearing_required": True})
    return Finding("foam_load", "ok",
                   "non-load-bearing material acceptable for a decorative part")


def check_draft_angle(
    part: PartSpec, min_draft_deg: float = 2.0
) -> Finding:
    """Mold-demold feasibility (Figure 16a).

    A cast/molded face needs at least ``min_draft_deg`` of draft to release; a
    negative angle is an undercut that traps the part (error).
    """
    if part.min_draft_deg is None:
        return Finding("draft_angle", "ok", "no draft data supplied")
    d = part.min_draft_deg
    if d < 0.0:
        return Finding("draft_angle", "error",
                       f"undercut face (draft {d:.1f} deg); part cannot demold "
                       f"without a side-action or split mold",
                       {"min_draft_deg": d, "required": min_draft_deg})
    if d < min_draft_deg:
        return Finding("draft_angle", "warning",
                       f"draft {d:.1f} deg below {min_draft_deg} deg minimum; "
                       f"add draft for clean release",
                       {"min_draft_deg": d, "required": min_draft_deg})
    return Finding("draft_angle", "ok",
                   f"draft {d:.1f} deg sufficient to demold",
                   {"min_draft_deg": d, "required": min_draft_deg})


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def analyze_workflow(
    workflow_id: str,
    part: PartSpec,
    machine_id: Optional[str] = None,
    material: Optional[str] = None,
    infill: float = 0.20,
) -> List[Finding]:
    """Run exactly the feasibility checks a workflow declares (per-workflow
    dispatch, not one generic rule).

    Returns findings in the workflow's declared check order. Unknown check names
    are skipped (forward-compatible).
    """
    wf = get_workflow(workflow_id)
    out: List[Finding] = []
    for name in wf.checks:
        if name == "machine_fit":
            out.append(check_machine_fit(workflow_id, part, machine_id))
        elif name == "print_time":
            mat = material or "pla"
            out.append(estimate_print_time(part, infill=infill, material=mat))
        elif name == "material_stock":
            out.append(check_material_stock(workflow_id, part, material))
        elif name == "kerf":
            out.append(check_kerf(workflow_id, part))
        elif name == "wire_form":
            out.append(check_wire_form(part))
        elif name == "foam_load":
            out.append(check_foam_load(workflow_id, part))
        elif name == "draft_angle":
            out.append(check_draft_angle(part))
        # unknown -> skip
    return out


def worst_severity(findings: Sequence[Finding]) -> str:
    """Roll up a list of findings to the worst severity present."""
    order = {"ok": 0, "warning": 1, "error": 2}
    worst = "ok"
    for f in findings:
        if order[f.severity] > order[worst]:
            worst = f.severity
    return worst
