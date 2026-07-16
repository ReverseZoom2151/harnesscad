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
"""
