"""Typed electronics IR plus rule-based circuit validation, mined from Forma-OSS.

This package gives HarnessCAD's device-level text-to-CAD briefs a typed
electrical layer parallel to the geometric op stream: a dataclass Hardware IR
(components, pins, nets, buses, power rails, assembly, mechanical placement
notes), deterministic derivations (power rails, buses, current draw, BOM
rollup), rule-based netlist validation, and heuristic enclosure-layout
seeding. HarnessCAD previously had no electronics/netlist IR at all.

Modules:

* ``hardware_ir`` -- the dataclass schema (Forma-OSS blueprint_core/models.py).
* ``circuit_validation`` -- the five electrical rules
  (Forma-OSS blueprint_core/validation.py).
* ``derive`` -- deterministic rail/bus/current/BOM derivations
  (Forma-OSS blueprint_core/agents/orchestrator.py).
* ``enclosure_layout`` -- heuristic mechanical placement seeding
  (Forma-OSS build_mechanical_render_data).

There is deliberately no dispatcher here. The catalogue, manufacturing and spec
surfaces each carry a ``registry`` because they arbitrate between *rival*
modules answering the same question; these four do not compete -- they compose
in one fixed order (parse an IR, derive from it, validate it, seed a layout).
A router over a straight line would be a surface with nothing to select, so the
package exports its schema and its entry points directly and nothing else.
"""

from __future__ import annotations

from harnesscad.domain.electronics.hardware_ir import (
    AssemblyStep,
    BusConnection,
    ComponentInstance,
    ComponentTemplate,
    ConnectionNet,
    FunctionalRequirements,
    HardwareIR,
    MechanicalNotes,
    MechanicalPlacement,
    MechanicalRotation3,
    MechanicalSource,
    MechanicalSpatialRelationship,
    MechanicalVector3,
    PinDefinition,
    PinMappingEntry,
    PinReference,
    PowerRail,
    ProjectOverview,
    ValidationIssue,
    ValidationSummary,
)
from harnesscad.domain.electronics.circuit_validation import (
    build_validation_summary,
    is_design_valid,
    validate_circuit,
)
from harnesscad.domain.electronics.derive import (
    bom_rollup,
    estimate_current_draw,
    extract_buses,
    extract_power_rails,
)
from harnesscad.domain.electronics.enclosure_layout import (
    derive_spatial_relationships,
    enrich_mechanical_layout,
    infer_render_dimensions,
    placement_layer,
    placement_position,
    placement_size,
)

__all__ = [
    # -- hardware_ir: the dataclass schema ---------------------------------
    "AssemblyStep",
    "BusConnection",
    "ComponentInstance",
    "ComponentTemplate",
    "ConnectionNet",
    "FunctionalRequirements",
    "HardwareIR",
    "MechanicalNotes",
    "MechanicalPlacement",
    "MechanicalRotation3",
    "MechanicalSource",
    "MechanicalSpatialRelationship",
    "MechanicalVector3",
    "PinDefinition",
    "PinMappingEntry",
    "PinReference",
    "PowerRail",
    "ProjectOverview",
    "ValidationIssue",
    "ValidationSummary",
    # -- circuit_validation: the five electrical rules ---------------------
    "build_validation_summary",
    "is_design_valid",
    "validate_circuit",
    # -- derive: rail / bus / current / BOM derivations --------------------
    "bom_rollup",
    "estimate_current_draw",
    "extract_buses",
    "extract_power_rails",
    # -- enclosure_layout: mechanical placement seeding --------------------
    "derive_spatial_relationships",
    "enrich_mechanical_layout",
    "infer_render_dimensions",
    "placement_layer",
    "placement_position",
    "placement_size",
]
