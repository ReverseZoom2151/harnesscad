"""Fabrication-workflow taxonomy, machine registry and material-stock presets.

Distilled from Feng et al., *Comparing Fabrication Workflows in CAD to Support
Design Reasoning* (the CAMeleon system). The paper surveys 55 real fabrication
projects and clusters them into a small taxonomy of fabrication paradigms, then
implements 16 representative workflows, each carrying a standardized bundle of
metadata: the machines it needs, the material stock it consumes, its assembly
step count, capability keywords ("durable", "lightweight", ...), and a set of
per-workflow feasibility checks.

This module is the *data model* for that survey — the workflow-level layer that
sits **above** the per-part critics already in the repo (``verifiers/dfm.py``
does wall-thickness/draft heuristics on one solid; ``quality/estimate.py`` does
mass/cost/BOM). Nothing here re-implements those; instead it captures the
paper's contribution: a machine-aware, comparable *catalog of fabrication
paradigms* so a design can be reasoned about across processes, not locked into
one.

Everything is stdlib-only, deterministic (no wall clock, no randomness) and
free of any external geometry kernel: a workflow is described by declarative
metadata, and the machines/materials are plain frozen records.

Taxonomy (Figure 10 of the paper): five primary paradigms plus an "other"
bucket for the two projects that did not cluster (epoxy shapes, spar/frame
boats):

    mold_casting            -- pour/press into a negative (silicone, stack mold)
    stacked_slice           -- stack planar slices into a volume
    interlocking            -- slot planar pieces together without glue
    guide_structure         -- form material around/along a guide (wire, felt)
    wire_forming            -- bend a continuous wire into a frame
    other                   -- everything that did not cluster
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Fabrication paradigms (the five clusters + other)
# --------------------------------------------------------------------------- #
CATEGORIES: Dict[str, str] = {
    "mold_casting": "Pour or press material into a negative mold, then demold.",
    "stacked_slice": "Stack planar slices/contours to approximate a volume.",
    "interlocking": "Slot planar pieces together via notches (glue-free).",
    "guide_structure": "Build material up around or along a guide/armature.",
    "wire_forming": "Bend a continuous wire into a structural frame.",
    "other": "Workflows that do not cluster into a single paradigm.",
}


# --------------------------------------------------------------------------- #
# Machines
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Machine:
    """A fabrication machine with its working envelope.

    ``work_volume`` is the usable build/cut envelope in millimetres. For a
    machine that works a flat sheet (laser cutter) the Z entry is the maximum
    material thickness it can cut. ``kind`` is a coarse process family used by
    the feasibility layer to pick which checks apply.
    """

    id: str
    name: str
    kind: str  # "printer" | "laser" | "wire_bender" | "foam_cutter" | "mill" | "manual"
    work_volume: Tuple[float, float, float]  # mm (x, y, z)
    notes: str = ""


MACHINES: Dict[str, Machine] = {
    m.id: m
    for m in (
        # 3D printers (working dims pre-loaded, per Figure 15a: Prusa, Ender).
        Machine("prusa_mk3", "Prusa MK3", "printer", (250.0, 210.0, 210.0)),
        Machine("ender3", "Creality Ender 3", "printer", (220.0, 220.0, 250.0)),
        Machine("prusa_xl", "Prusa XL", "printer", (360.0, 360.0, 360.0)),
        # Laser cutters: bed X/Y, Z = max cut thickness.
        Machine("laser_generic", "Generic laser cutter", "laser", (600.0, 300.0, 10.0)),
        Machine("laser_large", "Large-format laser cutter", "laser", (900.0, 600.0, 12.0)),
        # Wire bender: max wire length it can feed, working reach.
        Machine("wire_bender", "CNC wire bender", "wire_bender", (2000.0, 300.0, 300.0)),
        # Hot-wire foam cutter: cutting frame envelope.
        Machine("foam_cutter", "Hot-wire foam cutter", "foam_cutter", (500.0, 400.0, 400.0)),
        # CNC mill (for mold making / subtractive guide work).
        Machine("cnc_mill", "3-axis CNC mill", "mill", (300.0, 200.0, 100.0)),
        # Manual / no machine (felt, papier-mache, paper folding).
        Machine("manual", "Manual (no machine)", "manual", (10000.0, 10000.0, 10000.0)),
    )
}


# --------------------------------------------------------------------------- #
# Material stock presets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MaterialStock:
    """A stock material with the discrete sizes actually available.

    ``sheet_thicknesses`` are the plate/sheet gauges a shop stocks (mm); a
    laser/interlocking workflow must snap to one of these (Figure 15c: "align
    generated geometries to realistic stock sizes"). ``wire_diameter`` is set
    for wire stock. ``load_bearing`` flags whether the material can carry
    structural load (foam cannot — the paper's "foam strength tip").
    """

    id: str
    name: str
    sheet_thicknesses: Tuple[float, ...] = ()  # mm
    wire_diameter: Optional[float] = None  # mm
    load_bearing: bool = True
    food_safe: bool = False
    transparent: bool = False
    heat_resistant: bool = False
    flexible: bool = False
    density: float = 0.0  # g/cm^3 (informational)


MATERIALS: Dict[str, MaterialStock] = {
    s.id: s
    for s in (
        MaterialStock("pla", "PLA filament", density=1.24),
        MaterialStock("abs", "ABS filament", heat_resistant=True, density=1.04),
        MaterialStock("petg", "PETG filament", transparent=True, food_safe=True, density=1.27),
        MaterialStock("plywood_3mm", "Plywood sheet", sheet_thicknesses=(3.0, 6.0), density=0.6),
        MaterialStock("acrylic_3mm", "Cast acrylic sheet", sheet_thicknesses=(3.0, 5.0),
                      transparent=True, density=1.18),
        MaterialStock("mdf_6mm", "MDF sheet", sheet_thicknesses=(3.0, 6.0, 9.0), density=0.75),
        MaterialStock("steel_wire", "Steel wire", wire_diameter=2.0, flexible=True, density=7.85),
        MaterialStock("aluminum_wire", "Aluminum wire", wire_diameter=3.0, flexible=True, density=2.70),
        MaterialStock("eps_foam", "EPS foam block", load_bearing=False, density=0.03),
        MaterialStock("xps_foam", "XPS foam block", load_bearing=False, density=0.035),
        MaterialStock("epoxy_resin", "Epoxy resin", transparent=True, heat_resistant=True, density=1.1),
        MaterialStock("silicone", "Casting silicone", flexible=True, food_safe=True, density=1.2),
        MaterialStock("clay", "Air-dry / kiln clay", density=1.6),
        MaterialStock("wool_felt", "Wool felt", load_bearing=False, density=0.3),
        MaterialStock("paper_pulp", "Papier-mache pulp", load_bearing=False, density=0.4),
        MaterialStock("cardstock", "Folded cardstock", load_bearing=False, density=0.7),
        MaterialStock("bamboo", "Bamboo strip", density=0.6),
    )
}


# --------------------------------------------------------------------------- #
# Workflow records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Workflow:
    """One fabrication workflow with its full standardized metadata bundle.

    Fields mirror the paper's per-workflow template (scene / CAD-logic / tutorial
    / export) reduced to declarative, comparable metadata:

      * ``keywords``     -- swatch keywords surfaced in the sidebar ("durable").
      * ``machines``     -- machine ids this workflow needs (any-of).
      * ``materials``    -- candidate material stock ids.
      * ``assembly_steps`` -- number of manual steps in the interactive guide.
      * ``checks``       -- feasibility-check ids (see fabworkflow_feasibility).
      * ``props``        -- capability tags for intent matching (bool/level).
      * ``cost``/``time``/``precision``/``skill`` -- ordinal levels 1..5
        (1 = low/fast/cheap, 5 = high/slow/expensive) for side-by-side compare.
    """

    id: str
    name: str
    category: str
    keywords: Tuple[str, ...]
    machines: Tuple[str, ...]
    materials: Tuple[str, ...]
    assembly_steps: int
    checks: Tuple[str, ...]
    props: Dict[str, object] = field(default_factory=dict)
    cost: int = 3
    time: int = 3
    precision: int = 3
    skill: int = 3
    notes: str = ""


def _wf(*args, **kwargs) -> Workflow:
    return Workflow(*args, **kwargs)


WORKFLOWS: Dict[str, Workflow] = {
    w.id: w
    for w in (
        _wf(
            "fdm_3d_printing", "FDM 3D Printing", "stacked_slice",
            ("familiar", "detailed", "slow"), ("prusa_mk3", "ender3", "prusa_xl"),
            ("pla", "abs", "petg"), 3,
            ("machine_fit", "print_time", "material_stock"),
            props={"durable": True, "detail": True, "transparent": False,
                   "lightweight": False, "food_safe": False, "heat_resistant": False,
                   "complex_geometry": True},
            cost=3, time=5, precision=4, skill=1,
            notes="Default novice choice; long print times, bed-size splits.",
        ),
        _wf(
            "laser_cut_interlocking", "Laser-Cut Interlocking", "interlocking",
            ("strong", "flat-pack", "no-glue"), ("laser_generic", "laser_large"),
            ("plywood_3mm", "acrylic_3mm", "mdf_6mm"), 5,
            ("machine_fit", "material_stock", "kerf"),
            props={"durable": True, "detail": True, "transparent": False,
                   "lightweight": True, "complex_geometry": False},
            cost=2, time=2, precision=4, skill=3,
            notes="Slot planar plates; requires kerf compensation and nesting.",
        ),
        _wf(
            "stacked_layers", "Laser-Cut Stacked Layers", "stacked_slice",
            ("layered", "sturdy"), ("laser_generic",),
            ("plywood_3mm", "mdf_6mm"), 4,
            ("machine_fit", "material_stock"),
            props={"durable": True, "food_safe": False, "complex_geometry": True},
            cost=2, time=3, precision=3, skill=2,
            notes="Stack sliced contours; adhesive between layers (not food safe).",
        ),
        _wf(
            "wire_forming", "Wire Forming", "wire_forming",
            ("aesthetic", "lightweight", "springy"), ("wire_bender",),
            ("steel_wire", "aluminum_wire"), 3,
            ("machine_fit", "wire_form"),
            props={"lightweight": True, "durable": False, "flexible": True,
                   "transparent": True, "complex_geometry": False},
            cost=2, time=2, precision=3, skill=3,
            notes="Flexible/springy frame; may not give a stable surface.",
        ),
        _wf(
            "hot_wire_foam_cutting", "Hot-Wire Foam Cutting", "stacked_slice",
            ("low-cost", "lightweight", "large"), ("foam_cutter",),
            ("eps_foam", "xps_foam"), 2,
            ("machine_fit", "foam_load"),
            props={"lightweight": True, "durable": False, "complex_geometry": True,
                   "large_scale": True},
            cost=1, time=1, precision=2, skill=2,
            notes="Cheap foam; limited load-bearing capacity.",
        ),
        _wf(
            "mold_making", "Mold Making (CNC)", "mold_casting",
            ("durable", "repeatable"), ("cnc_mill",),
            ("silicone", "epoxy_resin"), 6,
            ("machine_fit", "draft_angle"),
            props={"durable": True, "detail": True, "batch": True},
            cost=4, time=4, precision=4, skill=4,
            notes="Needs draft angles to demold; good for repeat production.",
        ),
        _wf(
            "silicone_molding", "Silicone Molding", "mold_casting",
            ("durable", "flexible", "food-safe"), ("manual",),
            ("silicone",), 5,
            ("draft_angle",),
            props={"durable": True, "flexible": True, "food_safe": True,
                   "heat_resistant": True, "batch": True},
            cost=3, time=4, precision=3, skill=3,
            notes="Flexible, heat-resistant, food-safe casts.",
        ),
        _wf(
            "stack_mold", "Stack Mold", "mold_casting",
            ("layered-mold", "repeatable"), ("laser_generic",),
            ("plywood_3mm", "silicone"), 6,
            ("machine_fit", "material_stock", "draft_angle"),
            props={"durable": True, "batch": True},
            cost=3, time=4, precision=3, skill=4,
            notes="Sliced stacked negative used as a casting mold (prior work).",
        ),
        _wf(
            "epoxy_laminating", "Epoxy Laminating", "other",
            ("strong", "glossy", "transparent"), ("laser_generic",),
            ("plywood_3mm", "epoxy_resin"), 5,
            ("machine_fit", "material_stock"),
            props={"durable": True, "transparent": True, "heat_resistant": True,
                   "glossy": True, "post_process": True},
            cost=4, time=4, precision=3, skill=4,
            notes="Laser-cut core, assemble, sand, seal with epoxy resin.",
        ),
        _wf(
            "laser_cut_clay_sculpture", "Laser-Cut Clay Sculpture", "stacked_slice",
            ("artistic", "sculptural"), ("laser_generic",),
            ("clay", "plywood_3mm"), 4,
            ("machine_fit", "material_stock"),
            props={"durable": True, "artistic": True},
            cost=2, time=3, precision=2, skill=3,
            notes="Arts-education workflow; laser-cut formers for clay.",
        ),
        _wf(
            "paper_mache", "Papier-Mache", "guide_structure",
            ("low-cost", "artistic", "lightweight"), ("manual",),
            ("paper_pulp",), 4,
            ("foam_load",),
            props={"lightweight": True, "durable": False, "artistic": True,
                   "low_cost": True},
            cost=1, time=3, precision=1, skill=1,
            notes="Build pulp over an armature; not load-bearing.",
        ),
        _wf(
            "paper_folding", "Paper Folding", "guide_structure",
            ("low-cost", "fast", "lightweight"), ("laser_generic", "manual"),
            ("cardstock",), 3,
            ("machine_fit", "material_stock"),
            props={"lightweight": True, "durable": False, "low_cost": True},
            cost=1, time=1, precision=2, skill=2,
            notes="Score-and-fold; resource-constrained environments.",
        ),
        _wf(
            "needle_felt_sculpture", "Needle-Felt Sculpture", "guide_structure",
            ("soft", "artistic"), ("manual",),
            ("wool_felt",), 4,
            ("foam_load",),
            props={"lightweight": True, "durable": False, "soft": True,
                   "artistic": True, "post_process": True},
            cost=2, time=4, precision=1, skill=3,
            notes="Soft sculpture; post-processing craft, not structural.",
        ),
        _wf(
            "wire_copper_electroplating", "Wire + Copper Electroplating", "wire_forming",
            ("decorative", "conductive", "durable"), ("wire_bender",),
            ("steel_wire", "aluminum_wire"), 6,
            ("machine_fit", "wire_form"),
            props={"durable": True, "lightweight": True, "conductive": True,
                   "post_process": True},
            cost=3, time=5, precision=3, skill=4,
            notes="Post-processing: electroplate a bent-wire frame with copper.",
        ),
        _wf(
            "bamboo_agents", "Bamboo Agents", "guide_structure",
            ("craft", "sustainable"), ("manual", "laser_generic"),
            ("bamboo",), 5,
            ("machine_fit", "material_stock"),
            props={"durable": True, "sustainable": True, "artistic": True},
            cost=2, time=4, precision=2, skill=4,
            notes="Digital-craft bamboo construction (prior research project).",
        ),
        _wf(
            "escape_loom", "Escape Loom (Hand Weaving)", "guide_structure",
            ("textile", "craft"), ("manual", "laser_generic"),
            ("cardstock",), 5,
            ("machine_fit",),
            props={"flexible": True, "textile": True, "artistic": True},
            cost=2, time=4, precision=2, skill=3,
            notes="Hand-weaving affordances (prior research project).",
        ),
    )
}


# --------------------------------------------------------------------------- #
# Query helpers
# --------------------------------------------------------------------------- #
def workflows_in_category(category: str) -> List[Workflow]:
    """All workflows belonging to a taxonomy category, ordered by id."""
    if category not in CATEGORIES:
        raise KeyError(f"unknown category {category!r}")
    return sorted(
        (w for w in WORKFLOWS.values() if w.category == category),
        key=lambda w: w.id,
    )


def workflows_for_machine(machine_id: str) -> List[Workflow]:
    """Workflows that can run on a given machine (Figure 15: constrain to the
    machines a learner actually has access to)."""
    if machine_id not in MACHINES:
        raise KeyError(f"unknown machine {machine_id!r}")
    return sorted(
        (w for w in WORKFLOWS.values() if machine_id in w.machines),
        key=lambda w: w.id,
    )


def available_workflows(machine_ids: List[str]) -> List[Workflow]:
    """Workflows runnable given a *set* of available machines (any-of match).

    This is the paper's core machine-constraint idea: "you give it the machines
    you have access to, then it generates workflows based on that."
    """
    avail = set(machine_ids)
    for mid in avail:
        if mid not in MACHINES:
            raise KeyError(f"unknown machine {mid!r}")
    return sorted(
        (w for w in WORKFLOWS.values() if avail.intersection(w.machines)),
        key=lambda w: w.id,
    )


def get_workflow(workflow_id: str) -> Workflow:
    if workflow_id not in WORKFLOWS:
        raise KeyError(f"unknown workflow {workflow_id!r}")
    return WORKFLOWS[workflow_id]
