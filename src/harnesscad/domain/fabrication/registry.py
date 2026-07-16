"""The MANUFACTURING surface -- can this actually be made, and how?

``domain/fabrication`` carried a workflow taxonomy, per-process feasibility
checks, a prototype-readiness gate, flat-pack panel decomposition + nesting,
brick legolization and colouring, a machining-feature taxonomy, and an OpenSCAD
export planner. None of it was reachable. This module is the dispatcher: the
route a model takes AFTER it verifies, on the way to a machine.

    workflows(part)                 -> which processes can make this at all
    analyze("fdm_3d_printing", ...) -> the findings for one process
    readiness(descriptor)           -> the go / no-go prototype gate
    panels(...) / nest(...)         -> a cabinet -> panels -> a sheet layout
    bricks(voxels)                  -> a voxel model -> a legal brick layout
    export_plan(source, "stl")      -> the OpenSCAD invocation, planned not run

NOTHING HERE SHELLS OUT. :func:`export_plan` returns the ExportPlan (argv,
artifact name, cache key) and :func:`classify_export` reads a result you got
from somewhere else -- planning an ``openscad`` invocation is deterministic;
running one is not this module's business and is not stdlib.

Adapters only: the fabrication modules are never modified. Deterministic,
stdlib-only, no network.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "FabricationError",
    "part_spec",
    "workflows",
    "machines",
    "analyze",
    "compare",
    "rank_workflows",
    "checklist",
    "readiness",
    "panels",
    "nest",
    "nest_parts",
    "bricks",
    "brick_colors",
    "brick_assembly",
    "overhangs",
    "planar_drc",
    "printability",
    "feature_minima",
    "printer_profile",
    "rule_packs",
    "difficulty",
    "feature_attributes",
    "export_plan",
    "classify_export",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_FAB = "harnesscad.domain.fabrication."


class FabricationError(ValueError):
    """Base class for every manufacturing-surface failure."""


# --------------------------------------------------------------------------- #
# Workflow selection + feasibility
# --------------------------------------------------------------------------- #
def part_spec(bbox: Sequence[float], volume_mm3: float = 0.0, **kw):
    """The manufacturing view of a part: its envelope, volume and process hints."""
    from harnesscad.domain.fabrication.workflow_feasibility import PartSpec

    return PartSpec(bbox=tuple(float(v) for v in bbox),
                    volume_mm3=float(volume_mm3), **kw)


def workflows(category: Optional[str] = None,
              machine_ids: Optional[Sequence[str]] = None) -> List[dict]:
    """The manufacturing workflows on offer, optionally filtered."""
    from harnesscad.domain.fabrication.workflow_taxonomy import (
        WORKFLOWS, available_workflows, workflows_in_category,
    )

    if machine_ids:
        found = available_workflows(list(machine_ids))
    elif category:
        found = workflows_in_category(category)
    else:
        found = [WORKFLOWS[k] for k in sorted(WORKFLOWS)]
    return [{"id": w.id, "name": w.name, "category": w.category,
             "machines": list(w.machines), "materials": list(w.materials),
             "cost": w.cost, "time": w.time, "precision": w.precision,
             "skill": w.skill} for w in found]


def machines() -> List[str]:
    from harnesscad.domain.fabrication.workflow_taxonomy import MACHINES

    return sorted(MACHINES)


def analyze(workflow_id: str, part, machine_id: Optional[str] = None,
            material: Optional[str] = None, infill: float = 0.2) -> List[dict]:
    """Every feasibility finding for ONE process: fit, stock, kerf, draft, load.

    A finding is ``{severity, code, message}``; ``severity == 'error'`` means the
    process cannot make this part as specified. Nothing is averaged across
    processes -- a part that FDM can make and a laser cannot is not "half
    manufacturable".
    """
    from harnesscad.domain.fabrication.workflow_feasibility import analyze_workflow

    findings = analyze_workflow(workflow_id, part, machine_id=machine_id,
                                material=material, infill=float(infill))
    return [{"check": f.check, "severity": f.severity, "message": f.message,
             "data": dict(f.data or {}), "ok": bool(f.ok)} for f in findings]


def compare(workflow_ids: Sequence[str],
            criteria: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, object]]:
    """Side-by-side criterion table for several workflows. NOT a single score."""
    from harnesscad.domain.fabrication.workflow_compare import compare_workflows

    return compare_workflows(list(workflow_ids), criteria=list(criteria) if criteria else None)


def rank_workflows(requirements: Dict[str, object],
                   candidate_ids: Optional[Sequence[str]] = None,
                   machine_ids: Optional[Sequence[str]] = None,
                   top_k: int = 3) -> List[dict]:
    """Match stated requirements against the workflow taxonomy. Ranked, with reasons."""
    from harnesscad.domain.fabrication.workflow_compare import rank_by_intent

    matches = rank_by_intent(dict(requirements),
                             candidate_ids=list(candidate_ids) if candidate_ids else None,
                             machine_ids=list(machine_ids) if machine_ids else None,
                             top_k=int(top_k))
    return [{"workflow_id": m.workflow_id, "score": m.score,
             "matched": list(m.matched), "missed": list(m.missed),
             "reasons": list(m.reasons)} for m in matches]


def checklist(workflow_id: str) -> dict:
    """The reflection checklist a process demands before you commit to it."""
    from harnesscad.domain.fabrication.workflow_compare import reflection_checklist

    c = reflection_checklist(workflow_id)
    return {"workflow_id": c.workflow_id, "general": list(c.general),
            "specific": list(c.specific)}


# --------------------------------------------------------------------------- #
# The prototype-readiness gate
# --------------------------------------------------------------------------- #
def readiness(descriptor: Mapping[str, Any], allow_warnings: bool = True) -> dict:
    """Go / no-go: is this model actually ready to send to a printer?

    ``descriptor`` carries what the checks read: ``hole_count`` (watertightness),
    ``component_count`` (fragmentation), ``volume_mm3``, the surface-smoothness
    signal, and the source prompt/image provenance flags. A key the descriptor
    does not carry is not checked -- never guessed.
    """
    from harnesscad.domain.fabrication.prototype_readiness import readiness_report

    return readiness_report(dict(descriptor), allow_warnings=bool(allow_warnings))


# --------------------------------------------------------------------------- #
# Flat-pack: a cabinet -> panels -> a sheet layout
# --------------------------------------------------------------------------- #
def panels(exterior_height: float, exterior_width: float, exterior_depth: float,
           thickness: float, num_shelves: int = 1,
           back_full_cover: bool = False) -> List[dict]:
    from harnesscad.domain.fabrication.flatpack_panels import (
        decompose_cabinet, total_material_area,
    )

    ps = decompose_cabinet(float(exterior_height), float(exterior_width),
                           float(exterior_depth), float(thickness),
                           num_shelves=int(num_shelves),
                           back_full_cover=bool(back_full_cover))
    return [{"name": p.name, "width": p.width, "height": p.height,
             "thickness": p.thickness, "area": p.area,
             "holes": list(p.holes),
             "total_material_area": total_material_area(ps)} for p in ps]


def nest(exterior_height: float, exterior_width: float, exterior_depth: float,
         thickness: float, bed_w: float, bed_h: float, kerf: float = 0.0,
         num_shelves: int = 1) -> dict:
    """Panels -> a sheet-bed nesting report (what fits, what must be split)."""
    from harnesscad.domain.fabrication.flatpack_panels import (
        decompose_cabinet, nest_report,
    )

    ps = decompose_cabinet(float(exterior_height), float(exterior_width),
                           float(exterior_depth), float(thickness),
                           num_shelves=int(num_shelves))
    return nest_report(ps, float(bed_w), float(bed_h), kerf=float(kerf))


# --------------------------------------------------------------------------- #
# Sheet nesting (an already-flattened list of rectangles -> a sheet layout)
# --------------------------------------------------------------------------- #
def nest_parts(parts: Sequence[Mapping[str, Any]], sheet_w: float, sheet_h: float,
               kerf: float = 0.0, margin: float = 0.0,
               allow_rotate: bool = True, material: Optional[str] = None) -> dict:
    """Pack rectangular blanks onto stock sheets (skyline bottom-left nesting).

    Distinct from :func:`nest`, which DERIVES a cabinet's panels first. This
    takes an already-flattened list of ``{name, w, h, qty}`` rectangles and
    answers the shop-floor questions: how many sheets, what utilisation, cut
    length. Parts that exceed the usable area are reported, never dropped.
    """
    from harnesscad.domain.fabrication.nesting import (
        Part, nest_parts as nest_parts_impl, nest_report,
    )

    blanks = [Part(name=str(p["name"]), w=float(p["w"]), h=float(p["h"]),
                   qty=int(p.get("qty", 1))) for p in parts]
    result = nest_parts_impl(blanks, float(sheet_w), float(sheet_h),
                             kerf=float(kerf), margin=float(margin),
                             allow_rotate=bool(allow_rotate))
    return {
        "ok": bool(result.ok),
        "error": result.error,
        "sheets_used": result.sheets_used,
        "utilization": result.utilization,
        "cut_length": result.cut_length,
        "placements": [{"name": p.name, "x": p.x, "y": p.y, "w": p.w, "h": p.h,
                        "rotated": bool(p.rotated), "sheet": p.sheet}
                       for p in result.placements],
        "report": nest_report(result, material=material, kerf=float(kerf)),
    }


# --------------------------------------------------------------------------- #
# Bricks
# --------------------------------------------------------------------------- #
def brick_assembly(text: str, world_dim: int = 20) -> dict:
    """Parse BrickGPT ``HxW (x,y,z)`` text and return the buildability verdict.

    A structure is buildable only when every brick is in bounds, non-overlapping,
    supported (not floating), and connected to the ground -- the deterministic,
    solver-free structural predicates a verifier-first harness re-prompts on.
    """
    from harnesscad.domain.fabrication.brick_assembly import parse_text, validate

    structure = parse_text(text, world_dim=int(world_dim))
    report = validate(structure)
    return {
        "buildable": bool(report.buildable),
        "out_of_bounds": bool(report.out_of_bounds),
        "collisions": bool(report.collisions),
        "floating": bool(report.floating),
        "disconnected": bool(report.disconnected),
        "reasons": list(report.reasons),
        "brick_count": len(structure),
    }


def bricks(voxels: Sequence[Tuple[int, int, int]], seed: Optional[int] = 0) -> List[dict]:
    """A voxel occupancy set -> a LEGAL brick layout that covers it exactly."""
    from harnesscad.domain.fabrication.legolization import covers_exactly, legolize

    cells = [tuple(int(c) for c in v) for v in voxels]
    laid = legolize(cells, seed=seed)
    exact = covers_exactly(laid, cells)
    return [{"x": b.x, "y": b.y, "z": b.z, "h": b.h, "w": b.w,
             "valid_part": bool(b.is_valid_part()), "covers_exactly": bool(exact)}
            for b in laid]


def brick_colors(voxels: Sequence[Tuple[int, int, int]],
                 voxel_face_colors: Mapping[Any, Sequence[Tuple[int, int, int]]],
                 seed: Optional[int] = 0) -> List[str]:
    """Per-brick LEGO palette colours, from the per-voxel face colours."""
    from harnesscad.domain.fabrication.brick_coloring import assign_brick_colors
    from harnesscad.domain.fabrication.legolization import legolize

    cells = [tuple(int(c) for c in v) for v in voxels]
    return assign_brick_colors(legolize(cells, seed=seed),
                               dict(voxel_face_colors))


# --------------------------------------------------------------------------- #
# Machining features
# --------------------------------------------------------------------------- #
def difficulty(counts: Mapping[str, int]) -> dict:
    """How hard is this part to machine, from its feature histogram."""
    from harnesscad.domain.fabrication.feature_difficulty import classify_difficulty

    r = classify_difficulty(dict(counts))
    return {"level": r.level, "total_quantity": r.total_quantity,
            "excluded_present": list(r.excluded_present),
            "hard_present": list(r.hard_present),
            "reasons": list(r.reasons)}


def feature_attributes(feature: str, raw: Mapping[str, Any]) -> dict:
    """Normalise a machining feature and extract only the attributes it DECLARES."""
    from harnesscad.domain.fabrication.feature_attributes import extract_attributes
    from harnesscad.domain.fabrication.feature_taxonomy import (
        attributes_of, category_of, normalize_feature,
    )

    name = normalize_feature(feature)
    return {
        "feature": name,
        "category": category_of(name),
        "declares": list(attributes_of(name)),
        "attributes": extract_attributes(name, dict(raw)),
    }


# --------------------------------------------------------------------------- #
# DFAM: overhang detection + build-orientation search
# --------------------------------------------------------------------------- #
def overhangs(faces: Sequence[Mapping[str, Any]],
              build_dir: Sequence[float] = (0.0, 0.0, 1.0),
              threshold_deg: float = 45.0) -> dict:
    """Flag unsupported FDM overhang faces and pick the best build orientation.

    Each face carries an outward ``normal`` (and optional ``area``/``id``). A
    downward-facing surface tilted past the threshold cannot print without
    support; :func:`best_orientation` minimises total overhang area over the six
    axis-aligned build directions. AgentsCAD's deterministic DFAM floor.
    """
    from harnesscad.domain.fabrication.overhang import (
        best_orientation, overhang_faces, total_overhang_area,
    )

    faces = list(faces)
    bd = tuple(float(c) for c in build_dir)
    flagged = overhang_faces(faces, bd, float(threshold_deg))
    best_dir, best_area = best_orientation(faces, threshold_deg=float(threshold_deg))
    return {
        "overhang_faces": [{"face_id": f.face_id, "angle_from_up_deg": f.angle_from_up_deg,
                            "overhang_deg": f.overhang_deg, "area": f.area}
                           for f in flagged],
        "total_overhang_area": total_overhang_area(faces, bd, float(threshold_deg)),
        "best_orientation": list(best_dir),
        "best_overhang_area": best_area,
    }


# --------------------------------------------------------------------------- #
# Planar (mask-style) layout DRC
# --------------------------------------------------------------------------- #
def planar_drc(boxes: Sequence[Mapping[str, Any]],
               layers: Optional[Mapping[str, Sequence[int]]] = None,
               dbu_um: float = 0.001, min_width_um: Optional[float] = None,
               min_spacing_um: Optional[float] = None,
               check_shorts: bool = True) -> dict:
    """DBU-quantised planar layout + design-rule checker.

    ``boxes`` are ``{layer, x1, y1, x2, y2}`` rectangles authored in micrometres
    and stored as exact integer database units. Checks positive closed area,
    minimum width, same-layer spacing, and same-layer overlaps (shorts). The
    2-D, integer-grid checking counterpart to :func:`nest_parts`.
    """
    from harnesscad.domain.fabrication.planar_layout import PlanarLayout, run_drc

    layout = PlanarLayout(dbu_um=float(dbu_um))
    for name, spec in dict(layers or {}).items():
        spec = list(spec)
        layout.ensure_layer(name, int(spec[0]),
                            int(spec[1]) if len(spec) > 1 else 0)
    for b in boxes:
        layer = str(b["layer"])
        if layer not in layout.layers:
            layout.ensure_layer(layer, len(layout.layers))
        layout.add_box_um(layer, float(b["x1"]), float(b["y1"]),
                          float(b["x2"]), float(b["y2"]))
    report = run_drc(layout, min_width_um=min_width_um,
                     min_spacing_um=min_spacing_um, check_shorts=bool(check_shorts))
    return {
        "passed": bool(report.passed),
        "rule_ids": report.rule_ids(),
        "findings": [{"severity": f.severity, "rule_id": f.rule_id,
                      "layer": f.layer, "message": f.message}
                     for f in report.findings],
    }


# --------------------------------------------------------------------------- #
# Printability verdict (judges already-measured metrics)
# --------------------------------------------------------------------------- #
def printability(size_mm: Sequence[float], is_valid_solid: Optional[bool] = None,
                 is_watertight: Optional[bool] = None,
                 min_wall_mm: Optional[float] = None,
                 overhang_area_ratio: Optional[float] = None,
                 short_edges: int = 0, tiny_faces: int = 0,
                 profile: Optional[Mapping[str, Any]] = None) -> dict:
    """Judge already-measured geometry metrics into forgent3d's printability contract.

    Turns numbers (bbox size, min wall, overhang ratio, small-feature counts)
    into issue codes, a 0-100 score and a printable verdict. Distinct from
    :func:`overhangs`, which DETECTS overhang; this JUDGES a measured bundle
    against a printer profile, with one deterministic scoring definition.
    """
    from harnesscad.domain.fabrication.printability_verdict import (
        Measurements, PrinterProfile, printability_verdict,
    )

    prof = PrinterProfile(**dict(profile)) if profile else PrinterProfile()
    m = Measurements(
        size_mm=tuple(float(v) for v in size_mm),
        is_valid_solid=is_valid_solid, is_watertight=is_watertight,
        min_wall_mm=min_wall_mm, overhang_area_ratio=overhang_area_ratio,
        short_edges=int(short_edges), tiny_faces=int(tiny_faces),
    )
    return printability_verdict(m, prof)


# --------------------------------------------------------------------------- #
# Feature-typed printability minima (judges named features, one at a time)
# --------------------------------------------------------------------------- #
def feature_minima(features: Sequence[Mapping[str, Any]],
                   measurements: Optional[Mapping[str, Any]] = None,
                   profile: Optional[Mapping[str, Any]] = None) -> dict:
    """Judge INDIVIDUAL named features against AgentSCAD's per-feature minima.

    Each entry is ``{feature, value, label}`` -- "this boss is 2.4 mm across" --
    checked against a floor (wall, rib, through/blind hole, boss, text,
    clearance gap, merge overlap, cut extension) or a ceiling (bridge span,
    overhang angle) that is far stricter than the nozzle-width floor. Distinct
    from :func:`printability`, which judges ONE whole-model metric bundle: this
    judges each feature by its TYPE. Pass ``measurements`` to merge the two --
    the feature findings become printability issues and share its
    ``{printable, score, issues}`` contract, one scoring definition throughout.
    """
    from harnesscad.domain.fabrication.feature_minima import (
        MeasuredFeature, check_features, feature_verdict,
    )
    from harnesscad.domain.fabrication.printability_verdict import (
        Measurements, PrinterProfile,
    )

    measured = [MeasuredFeature(feature=str(f["feature"]), value=float(f["value"]),
                                label=str(f.get("label", "")))
                for f in features]
    prof = PrinterProfile(**dict(profile)) if profile else PrinterProfile()
    m = None
    if measurements is not None:
        spec = dict(measurements)
        spec["size_mm"] = tuple(float(v) for v in spec["size_mm"])
        m = Measurements(**spec)
    verdict = feature_verdict(measured, measurements=m, profile=prof)
    verdict["findings"] = [{"feature": f.feature, "label": f.label, "value": f.value,
                            "severity": f.severity, "threshold": f.threshold,
                            "recommended": f.recommended, "units": f.units,
                            "message": f.message, "rule": f.rule}
                           for f in check_features(measured)]
    return verdict


# --------------------------------------------------------------------------- #
# Printer wrapper profiles (the machine the part has to survive)
# --------------------------------------------------------------------------- #
def printer_profile(profile: Mapping[str, Any],
                    gcode: Optional[str] = None) -> dict:
    """Validate a printer wrapper profile, and optionally G-code against it.

    The profile carries the ENVELOPE -- the machine's real motion bounds -- which
    is the bound :func:`printability`'s build-volume fit exists to check against.
    Pass ``gcode`` to additionally validate a body statically: absolute moves
    outside the envelope fail; relative positioning (``G91``) warns and suspends
    bounds checking until ``G90`` restores it, because a relative move's absolute
    position is not knowable from the line. Unknown commands warn and never fail:
    this reads G-code, it does not claim to be a firmware.
    """
    from harnesscad.domain.fabrication.printer_profiles import (
        load_profile, to_printability_profile, validate_gcode,
    )

    spec = load_profile(profile)
    out: dict = {
        "backend": spec.backend,
        "machine": spec.machine_name,
        "filament": spec.filament_type,
        "motion_bounds_mm": {
            axis: list(bounds) for axis, bounds in spec.motion_bounds_mm.items()
        },
        "build_volume_mm": list(to_printability_profile(spec).build_volume_mm),
    }
    if gcode is not None:
        report = validate_gcode(gcode, spec)
        out["gcode"] = report.to_dict() if hasattr(report, "to_dict") else {
            "ok": report.ok, "errors": list(report.errors),
            "warnings": list(report.warnings),
        }
    return out


# --------------------------------------------------------------------------- #
# Declarative rule packs (rules as data, not as code)
# --------------------------------------------------------------------------- #
def rule_packs(metrics: Mapping[str, Any], family: Optional[str] = None,
               pack_ids: Optional[Sequence[str]] = None,
               packs: Optional[Sequence[Mapping[str, Any]]] = None) -> dict:
    """Evaluate versioned condition-expression rule packs over design metrics.

    A rule is DATA: a boolean pass expression over named metrics
    (``"hole_edge_distance >= 1.5 * hole_diameter"``), a ``when`` gate, a
    severity, a confidence and reasoning links. Distinct from :func:`analyze`
    and :func:`readiness`, whose checks are hard-coded Python for a fixed
    process or a fixed descriptor: here the rules ship as packs you can filter,
    version and extend without touching this module. Defaults to the vendored
    IntentForge packs; pass ``packs`` (pack dicts) to bring your own. Expressions
    run on an interpreter, never ``eval``; a missing metric yields a typed
    ``not_evaluable`` finding, never a crash or a silent pass.
    """
    from harnesscad.domain.fabrication.rule_packs import (
        FabricationRulePack, evaluate_packs, vendored_packs,
    )

    loaded = ([FabricationRulePack.from_dict(dict(p)) for p in packs]
              if packs is not None else vendored_packs())
    if pack_ids:
        wanted = set(pack_ids)
        loaded = [p for p in loaded if p.pack_id in wanted]
    result = evaluate_packs(loaded, dict(metrics), family=family)
    out = result.to_dict()
    out["packs"] = [{"pack_id": p.pack_id, "pack_version": p.pack_version,
                     "category": p.category, "rules": len(p.rules)}
                    for p in loaded]
    return out


# --------------------------------------------------------------------------- #
# Export planning (planned, never executed)
# --------------------------------------------------------------------------- #
def export_plan(source: str, export_format: str = "stl", out_dir: str = ".",
                defines: Optional[Mapping[str, Any]] = None,
                executable: str = "openscad") -> dict:
    """Plan an OpenSCAD export. Returns the argv/artifact/cache key -- runs nothing."""
    from harnesscad.domain.fabrication.openscad_export import (
        plan_export, plan_cache_key, sorted_formats,
    )

    plan = plan_export(source, export_format=export_format, out_dir=out_dir,
                       defines=dict(defines or {}), executable=executable)
    return {
        "argv": list(plan.argv),
        "scad_path": plan.scad_path,
        "output_path": plan.output_path,
        "digest": plan.digest,
        "format": plan.export_format,
        "cache_key": plan_cache_key(source, export_format, dict(defines or {})),
        "formats_3d": list(sorted_formats("3d")),
    }


def classify_export(returncode: int, stderr: str = "") -> dict:
    """Read an OpenSCAD run's result (which somebody else executed)."""
    from harnesscad.domain.fabrication.openscad_export import (
        classify_result, is_success, warnings_only,
    )

    return {
        "classification": classify_result(int(returncode), stderr),
        "success": bool(is_success(int(returncode), stderr)),
        "warnings_only": bool(warnings_only(stderr)),
    }


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    return {e.dotted: e
            for e in capability_registry.find(package="fabrication")}


def _available(dotted: str) -> bool:
    return dotted in _index()


_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("workflow", "workflows", _FAB + "workflow_taxonomy",
     "the workflow / machine / material taxonomy"),
    ("workflow", "analyze", _FAB + "workflow_feasibility",
     "per-process feasibility: machine fit, stock, kerf, draft, load, print time"),
    ("workflow", "compare", _FAB + "workflow_compare",
     "side-by-side criterion table + intent ranking + reflection checklist"),
    ("gate", "readiness", _FAB + "prototype_readiness",
     "the go/no-go prototype gate (watertight, fragmentation, volume, provenance)"),
    ("flatpack", "panels", _FAB + "flatpack_panels",
     "a cabinet -> panels -> a sheet-bed nesting report"),
    ("flatpack", "nest_parts", _FAB + "nesting",
     "2-D rectangular part nesting onto stock sheets (skyline bin-packing)"),
    ("brick", "bricks", _FAB + "legolization",
     "a voxel model -> a legal brick layout that covers it exactly"),
    ("brick", "brick_colors", _FAB + "brick_coloring",
     "per-brick LEGO palette colours from per-voxel face colours"),
    ("brick", "bricks", _FAB + "brick_library",
     "the legal brick parts, their serialisation and raster ordering"),
    ("brick", "brick_assembly", _FAB + "brick_assembly",
     "BrickGPT brick structure + buildability checks (bounds, collision, floating, connectivity)"),
    ("dfam", "overhangs", _FAB + "overhang",
     "FDM overhang detection + build-orientation search (AgentsCAD DFAM floor)"),
    ("dfam", "printability", _FAB + "printability_verdict",
     "printability verdict, issue-code taxonomy and build-volume fit (forgent3d)"),
    ("dfam", "feature_minima", _FAB + "feature_minima",
     "per-feature FDM minima (wall, rib, hole, boss, text, gap, bridge, overhang); "
     "composes into the printability verdict (AgentSCAD)"),
    ("dfam", "printer_profile", _FAB + "printer_profiles",
     "printer wrapper profiles: validated machine envelope + static G-code bounds "
     "checking, the machine side of the printability fit (text-to-cad)"),
    ("rules", "rule_packs", _FAB + "rule_packs",
     "versioned declarative rule packs: condition expressions over design metrics, "
     "evaluated without eval, with dependency interactions (IntentForge)"),
    ("layout", "planar_drc", _FAB + "planar_layout",
     "DBU-quantised planar layout + mask-style design-rule checker"),
    ("feature", "difficulty", _FAB + "feature_difficulty",
     "machining difficulty from a feature histogram; dataset stratification"),
    ("feature", "feature_attributes", _FAB + "feature_attributes",
     "extract only the attributes a machining feature declares"),
    ("feature", "feature_attributes", _FAB + "feature_taxonomy",
     "the machining-feature taxonomy (normalise, categorise, leaves)"),
    ("export", "export_plan", _FAB + "openscad_export",
     "plan an OpenSCAD export (argv, artifact, cache key) -- never executes it"),
)


def routed_modules() -> Tuple[str, ...]:
    return tuple(sorted({m for _g, _n, m, _d in _ROUTES if _available(m)}))


def discover() -> List[dict]:
    return [{"group": g, "route": n, "module": m, "doc": d,
             "present": _available(m)}
            for (g, n, m, d) in _ROUTES]


def unadapted() -> List[Tuple[str, str]]:
    routed = set(routed_modules())
    return [(d, "no route yet") for d in sorted(_index())
            if d not in routed and not d.endswith(".registry")]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every manufacturing route")
    parser.add_argument("--workflows", action="store_true",
                        help="list the manufacturing workflows")
    parser.add_argument("--machines", action="store_true",
                        help="list the machines")
    parser.add_argument("--analyze", default=None, metavar="WORKFLOW",
                        help="feasibility-check --bbox against this workflow")
    parser.add_argument("--bbox", default=None, metavar="X,Y,Z",
                        help="the part envelope in mm")
    parser.add_argument("--volume", type=float, default=0.0,
                        help="the part volume in mm^3")
    parser.add_argument("--machine", default=None, help="the machine id")
    parser.add_argument("--material", default=None, help="the material id")
    parser.add_argument("--readiness", default=None, metavar="JSON",
                        help="a descriptor (JSON object or @file) for the readiness gate")
    parser.add_argument("--unadapted", action="store_true",
                        help="list fabrication modules with no route")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def _load(text: str) -> Any:
    if text.startswith("@"):
        with open(text[1:], "r", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(text)


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "machines", False):
        for m in machines():
            print(m)
        return 0

    if getattr(args, "workflows", False):
        rows = workflows()
        if getattr(args, "json", False):
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            for w in rows:
                print("%-26s %-16s %s" % (w["id"], w["category"],
                                          ",".join(w["machines"])))
        return 0

    if getattr(args, "readiness", None):
        report = readiness(_load(args.readiness))
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if report.get("ready") else 1

    if getattr(args, "analyze", None):
        if not getattr(args, "bbox", None):
            print("--analyze needs --bbox X,Y,Z", file=__import__("sys").stderr)
            return 2
        bbox = [float(v) for v in args.bbox.split(",")]
        part = part_spec(bbox, volume_mm3=getattr(args, "volume", 0.0) or 0.0)
        findings = analyze(args.analyze, part,
                           machine_id=getattr(args, "machine", None),
                           material=getattr(args, "material", None))
        if getattr(args, "json", False):
            print(json.dumps(findings, indent=2, sort_keys=True))
        else:
            for f in findings:
                print("[%s] %s: %s" % (f["severity"], f["code"], f["message"]))
        return 0 if all(f["ok"] for f in findings) else 1

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-9s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad fabricate",
        description="manufacturing surface: workflows, feasibility, readiness, "
                    "flat-pack, bricks, export planning")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
