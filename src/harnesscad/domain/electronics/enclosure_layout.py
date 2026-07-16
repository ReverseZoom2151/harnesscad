"""Deterministic mechanical placement enrichment, mined from Forma-OSS.

Ported from Forma-OSS ``blueprint_core/agents/orchestrator.py``
(``build_mechanical_render_data`` and its helpers ``_infer_render_dimensions``,
``_placement_layer``, ``_placement_size``, ``_placement_position``,
``_row_position``, ``_dominant_axis``, ``_offset_for_axis``,
``_is_enclosure_component``), converted from pydantic to the stdlib dataclass
IR in ``harnesscad.domain.electronics.hardware_ir``.

Gap filled: HarnessCAD previously had no electronics/netlist IR at all, so its
device-level briefs had no bridge from a typed BOM into a spatial layout.
These heuristics seed that bridge: an envelope size inferred from project
keywords (or scaled from component count), per-component layer/size/position
presets (front-face displays, button rows, corner standoffs, battery/charger/
speaker/sensor/actuator/power positions), and controller-relative spatial
relationships.

This is *heuristic seeding*, distinct from
``harnesscad.agents.generation.layout_solver`` which is a constraint
optimizer: the presets here give a plausible deterministic starting layout
when the upstream agent output is sparse; the solver refines placements
against hard constraints. ``enrich_mechanical_layout`` only fills what is
missing -- existing placements and relationships are never touched.

Deterministic: same IR in -> same enriched IR out. No clock, no randomness.

Usage::

    from harnesscad.domain.electronics.enclosure_layout import enrich_mechanical_layout
    ir = enrich_mechanical_layout(ir)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Sequence

from harnesscad.domain.electronics.hardware_ir import (
    ComponentInstance,
    HardwareIR,
    MechanicalNotes,
    MechanicalPlacement,
    MechanicalRotation3,
    MechanicalSpatialRelationship,
    MechanicalVector3,
    ProjectOverview,
)

__all__ = [
    "infer_render_dimensions",
    "placement_layer",
    "placement_size",
    "placement_position",
    "derive_spatial_relationships",
    "enrich_mechanical_layout",
    "main",
]


def _mechanical_vector(x_mm: float, y_mm: float, z_mm: float) -> MechanicalVector3:
    return MechanicalVector3(
        x_mm=round(float(x_mm), 2),
        y_mm=round(float(y_mm), 2),
        z_mm=round(float(z_mm), 2),
    )


def _component_text(component: ComponentInstance) -> str:
    return (
        f"{component.ref_des} {component.name} "
        f"{component.part_number} {component.category}".lower()
    )


def _category_key(component: ComponentInstance) -> str:
    return component.category.strip().lower()


def _is_enclosure_component(component: ComponentInstance) -> bool:
    text = _component_text(component)
    if any(
        token in text
        for token in ["screw", "insert", "standoff", "button cap", "fastener"]
    ):
        return False
    return any(
        token in text
        for token in [
            "main enclosure",
            "enclosure shell",
            "project box",
            "shell",
            "housing",
            "case",
        ]
    )


def infer_render_dimensions(ir: HardwareIR) -> MechanicalVector3:
    """Infer the overall envelope: keyword-class presets, else a
    component-count-scaled fallback with the same clamps as the source."""
    if ir.mechanical and ir.mechanical.render_dimensions:
        return ir.mechanical.render_dimensions

    haystack = " ".join(
        [
            ir.overview.title if ir.overview else "",
            ir.overview.description if ir.overview else "",
            " ".join(ir.constraints or []),
            " ".join(ir.fabrication_notes or []),
        ]
    ).lower()

    if any(token in haystack for token in ["mp3", "audio", "pocket", "portable"]):
        return _mechanical_vector(100, 21, 54)
    if any(token in haystack for token in ["plant", "water", "soil", "garden"]):
        return _mechanical_vector(116, 82, 55)
    if any(token in haystack for token in ["thermostat", "nest", "hvac"]):
        return _mechanical_vector(86, 24, 86)
    if any(token in haystack for token in ["deadbolt", "lock", "servo"]):
        return _mechanical_vector(92, 64, 38)

    electrical_count = len(
        [
            component
            for component in ir.components
            if _category_key(component) not in {"mechanical", "3d print"}
        ]
    )
    width = max(92, min(150, 70 + electrical_count * 7))
    depth = max(48, min(92, 36 + electrical_count * 4))
    height = max(30, min(70, 24 + electrical_count * 3))
    return _mechanical_vector(width, depth, height)


def placement_layer(component: ComponentInstance) -> str:
    """Classify into enclosure / print / structural / mechanism / electrical."""
    key = _category_key(component)
    text = _component_text(component)

    if _is_enclosure_component(component):
        return "enclosure"
    if key == "3d print":
        return "print"
    if key == "mechanical":
        if any(token in text for token in ["screw", "insert", "standoff", "boss"]):
            return "structural"
        return "mechanism"
    return "electrical"


def placement_size(
    component: ComponentInstance, dimensions: MechanicalVector3
) -> MechanicalVector3:
    """Approximate the component envelope via token and category presets."""
    key = _category_key(component)
    text = _component_text(component)

    if _is_enclosure_component(component):
        return dimensions
    if any(
        token in text
        for token in ["front bezel", "faceplate", "acrylic", "window", "trim"]
    ):
        return _mechanical_vector(
            dimensions.x_mm * 0.82,
            max(2.0, dimensions.y_mm * 0.12),
            dimensions.z_mm * 0.72,
        )
    if any(token in text for token in ["back cover", "rear cover", "cover"]):
        return _mechanical_vector(
            dimensions.x_mm * 0.88,
            max(2.0, dimensions.y_mm * 0.12),
            dimensions.z_mm * 0.86,
        )
    if "battery" in text:
        return _mechanical_vector(
            min(48, dimensions.x_mm * 0.45),
            min(26, dimensions.y_mm * 0.65),
            min(9, dimensions.z_mm * 0.22),
        )
    if "speaker" in text:
        return _mechanical_vector(24, min(12, dimensions.y_mm * 0.45), 24)
    if "relay" in text:
        return _mechanical_vector(38, 26, 16)
    if "servo" in text:
        return _mechanical_vector(23, 12, 29)
    if any(token in text for token in ["oled", "display"]):
        return _mechanical_vector(34, 3, 18)
    if any(token in text for token in ["button", "switch", "cap"]):
        return _mechanical_vector(10, 7, 10)
    if any(token in text for token in ["usb-c", "usb"]):
        return _mechanical_vector(18, 8, 7)
    if any(token in text for token in ["screw", "insert", "standoff"]):
        return _mechanical_vector(5, 5, 8)
    if any(token in text for token in ["mount", "bracket", "plate"]):
        return _mechanical_vector(34, 4, 18)

    sizes = {
        "microcontroller": (38, 28, 5),
        "sensor": (20, 12, 14),
        "actuator": (30, 22, 14),
        "display": (34, 3, 18),
        "power": (42, 22, 8),
        "passives": (15, 12, 8),
        "communication": (28, 18, 5),
        "mechanical": (14, 10, 8),
        "3d print": (30, 5, 18),
    }
    x_mm, y_mm, z_mm = sizes.get(key, (22, 16, 6))
    return _mechanical_vector(x_mm, y_mm, z_mm)


def _row_position(index: int, count: int, span: float) -> float:
    if count <= 1:
        return 0.0
    return -span / 2 + span * (index / (count - 1))


def placement_position(
    component: ComponentInstance,
    components: List[ComponentInstance],
    dimensions: MechanicalVector3,
) -> MechanicalVector3:
    """Heuristic center position relative to the enclosure center: front-face
    displays, button rows, corner standoffs, and per-role interior spots."""
    key = _category_key(component)
    text = _component_text(component)
    width = dimensions.x_mm
    depth = dimensions.y_mm
    height = dimensions.z_mm

    if _is_enclosure_component(component):
        return _mechanical_vector(0, 0, 0)
    if any(
        token in text
        for token in ["front bezel", "faceplate", "trim plate", "acrylic cover", "window"]
    ):
        return _mechanical_vector(0, -depth * 0.46, height * 0.04)
    if any(token in text for token in ["back cover", "rear cover", "cover"]):
        return _mechanical_vector(0, depth * 0.46, 0)
    if any(token in text for token in ["oled mount", "display bezel"]):
        return _mechanical_vector(0, -depth * 0.36, height * 0.22)
    if any(token in text for token in ["controller mount", "esp32 mount", "board mount"]):
        return _mechanical_vector(0, -depth * 0.05, -height * 0.1)

    button_like = [
        item
        for item in components
        if any(token in _component_text(item) for token in ["button", "switch", "cap"])
    ]
    button_index = next(
        (index for index, item in enumerate(button_like) if item.ref_des == component.ref_des),
        -1,
    )
    if button_index >= 0:
        return _mechanical_vector(
            _row_position(button_index, len(button_like), width * 0.42),
            -depth * 0.43,
            -height * 0.12,
        )

    structural = [item for item in components if placement_layer(item) == "structural"]
    structural_index = next(
        (index for index, item in enumerate(structural) if item.ref_des == component.ref_des),
        -1,
    )
    if structural_index >= 0:
        corner_x = -width * 0.42 if structural_index % 2 == 0 else width * 0.42
        corner_z = -height * 0.36 if structural_index < 2 else height * 0.36
        return _mechanical_vector(corner_x, depth * 0.28, corner_z)

    if "display" in key or "oled" in text:
        return _mechanical_vector(0, -depth * 0.43, height * 0.24)
    if key == "microcontroller":
        return _mechanical_vector(0, 0, -height * 0.04)
    if "battery" in text:
        return _mechanical_vector(-width * 0.27, depth * 0.24, -height * 0.26)
    if any(token in text for token in ["charger", "usb-c", "usb"]):
        return _mechanical_vector(width * 0.28, -depth * 0.36, -height * 0.28)
    if "speaker" in text:
        return _mechanical_vector(width * 0.32, depth * 0.3, height * 0.2)
    if any(token in text for token in ["sd", "storage"]):
        return _mechanical_vector(-width * 0.3, -depth * 0.04, height * 0.04)
    if any(token in text for token in ["dac", "audio"]):
        return _mechanical_vector(width * 0.22, depth * 0.02, 0)
    if key == "sensor":
        sensors = [item for item in components if _category_key(item) == "sensor"]
        sensor_index = max(
            0,
            next(
                (index for index, item in enumerate(sensors) if item.ref_des == component.ref_des),
                0,
            ),
        )
        return _mechanical_vector(
            _row_position(sensor_index, len(sensors), width * 0.44),
            -depth * 0.42,
            height * 0.16,
        )
    if key == "actuator":
        actuators = [item for item in components if _category_key(item) == "actuator"]
        actuator_index = max(
            0,
            next(
                (index for index, item in enumerate(actuators) if item.ref_des == component.ref_des),
                0,
            ),
        )
        return _mechanical_vector(
            width * 0.3,
            depth * (0.12 - actuator_index * 0.18),
            -height * 0.04 + actuator_index * height * 0.18,
        )
    if key == "power":
        power_parts = [item for item in components if _category_key(item) == "power"]
        power_index = max(
            0,
            next(
                (index for index, item in enumerate(power_parts) if item.ref_des == component.ref_des),
                0,
            ),
        )
        return _mechanical_vector(
            -width * 0.28 + power_index * width * 0.22,
            depth * 0.22,
            -height * 0.25,
        )

    remaining = [
        item
        for item in components
        if _category_key(item) not in {"mechanical", "3d print"}
        and _category_key(item)
        not in {"microcontroller", "display", "sensor", "actuator", "power"}
    ]
    remaining_index = max(
        0,
        next(
            (index for index, item in enumerate(remaining) if item.ref_des == component.ref_des),
            0,
        ),
    )
    return _mechanical_vector(
        _row_position(remaining_index, len(remaining), width * 0.64),
        -depth * 0.16,
        -height * 0.03,
    )


def _dominant_axis(source: MechanicalPlacement, target: MechanicalPlacement) -> str:
    deltas = {
        "X": abs(target.position.x_mm - source.position.x_mm),
        "Y": abs(target.position.y_mm - source.position.y_mm),
        "Z": abs(target.position.z_mm - source.position.z_mm),
    }
    return max(deltas, key=deltas.get)


def _offset_for_axis(
    source: MechanicalPlacement, target: MechanicalPlacement, axis: str
) -> float:
    if axis == "X":
        return target.position.x_mm - source.position.x_mm
    if axis == "Y":
        return target.position.y_mm - source.position.y_mm
    return target.position.z_mm - source.position.z_mm


def derive_spatial_relationships(
    placements: List[MechanicalPlacement],
) -> List[MechanicalSpatialRelationship]:
    """Derive controller-relative offsets (capped at 9) plus a display-to-bezel
    alignment, exactly as the source seeds spatial_relationships."""
    placements_by_ref = {placement.ref_des: placement for placement in placements}
    controller = next(
        (
            placement
            for placement in placements
            if (placement.category or "").lower() == "microcontroller"
        ),
        None,
    )
    relationships: List[MechanicalSpatialRelationship] = []

    if controller:
        for placement in placements:
            if placement.ref_des == controller.ref_des or placement.layer == "enclosure":
                continue
            axis = _dominant_axis(controller, placement)
            relationships.append(
                MechanicalSpatialRelationship(
                    source_ref_des=controller.ref_des,
                    target_ref_des=placement.ref_des,
                    relation="spatial offset from controller",
                    axis=axis,
                    offset_mm=round(_offset_for_axis(controller, placement, axis), 2),
                    notes=(
                        f"{placement.ref_des} is placed along the {axis} axis "
                        f"relative to {controller.ref_des}."
                    ),
                )
            )
            if len(relationships) >= 9:
                break

    for placement in placements:
        text = f"{placement.ref_des} {placement.label or ''}".lower()
        if "display" in text or "oled" in text:
            bezel = next(
                (
                    candidate
                    for candidate in placements
                    if "bezel" in f"{candidate.label or ''}".lower()
                ),
                None,
            )
            if bezel and placement.ref_des != bezel.ref_des:
                relationships.append(
                    MechanicalSpatialRelationship(
                        source_ref_des=placement.ref_des,
                        target_ref_des=bezel.ref_des,
                        relation="aligned with display opening",
                        axis="Y",
                        offset_mm=round(bezel.position.y_mm - placement.position.y_mm, 2),
                        notes=(
                            "Display centerline is aligned to the front "
                            "bezel/window cutout."
                        ),
                    )
                )
                break

    return [
        relationship
        for relationship in relationships
        if relationship.source_ref_des in placements_by_ref
        and relationship.target_ref_des in placements_by_ref
    ]


def enrich_mechanical_layout(ir: HardwareIR) -> HardwareIR:
    """Populate render_dimensions, missing component_placements, and (when
    empty) spatial_relationships on a HardwareIR, never touching existing
    placements. Mirrors the source's build_mechanical_render_data."""
    if not ir.mechanical or not ir.components:
        return ir

    dimensions = infer_render_dimensions(ir)
    ir.mechanical.render_dimensions = dimensions

    existing_placement_refs = {
        placement.ref_des for placement in ir.mechanical.component_placements
    }
    generated_placements: List[MechanicalPlacement] = []
    for component in ir.components:
        if component.ref_des in existing_placement_refs:
            continue

        position = placement_position(component, ir.components, dimensions)
        generated_placements.append(
            MechanicalPlacement(
                ref_des=component.ref_des,
                label=component.name,
                category=component.category,
                layer=placement_layer(component),
                position=position,
                size=placement_size(component, dimensions),
                orientation_deg=MechanicalRotation3(),
                mounting_face=(
                    "front" if position.y_mm < -dimensions.y_mm * 0.32 else "internal"
                ),
                notes=component.rationale,
            )
        )

    if generated_placements:
        ir.mechanical.component_placements = [
            *ir.mechanical.component_placements,
            *generated_placements,
        ]

    if not ir.mechanical.spatial_relationships:
        ir.mechanical.spatial_relationships = derive_spatial_relationships(
            ir.mechanical.component_placements
        )

    metadata = ir.assembly_metadata or {}
    ir.assembly_metadata = {
        **metadata,
        "render_dimensions": dimensions.to_dict(),
        "component_placement_count": len(ir.mechanical.component_placements),
        "spatial_relationship_count": len(ir.mechanical.spatial_relationships),
    }
    return ir


# ==========================================
# Selfcheck fixtures
# ==========================================


def _synthetic_ir() -> HardwareIR:
    components = [
        ComponentInstance(
            ref_des="U1",
            part_number="ESP32-WROOM-32D",
            name="ESP32 Development Board",
            category="Microcontroller",
            rationale="Control loop.",
        ),
        ComponentInstance(
            ref_des="DS1",
            part_number="SSD1306-OLED",
            name="0.96in OLED Display",
            category="Display",
            rationale="Status readout.",
        ),
        ComponentInstance(
            ref_des="SW1",
            part_number="Tact-6mm",
            name="Push Button",
            category="Passives",
            rationale="User input.",
        ),
        ComponentInstance(
            ref_des="SW2",
            part_number="Tact-6mm",
            name="Push Button",
            category="Passives",
            rationale="User input.",
        ),
        ComponentInstance(
            ref_des="BAT1",
            part_number="LiPo-1000",
            name="3.7V LiPo Battery",
            category="Power",
            rationale="Portable power.",
        ),
        ComponentInstance(
            ref_des="MECH1",
            part_number="M3-Insert",
            name="M3 Heat-set Insert Standoff",
            category="Mechanical",
            rationale="Board mounting.",
        ),
        ComponentInstance(
            ref_des="ENC1",
            part_number="ENC-BOX",
            name="Main Enclosure Shell",
            category="3D Print",
            rationale="Housing.",
        ),
        ComponentInstance(
            ref_des="BEZ1",
            part_number="BEZ-1",
            name="Front Bezel",
            category="3D Print",
            rationale="Display window.",
        ),
    ]
    return HardwareIR(
        overview=ProjectOverview(
            title="Pocket Status Display",
            description="A portable pocket display gadget.",
            difficulty="Beginner",
            category="IoT",
        ),
        components=components,
        mechanical=MechanicalNotes(
            enclosure_type="3D Printed",
            mounting_guidance="Heat-set inserts.",
            manufacturability_rating="Easy",
            component_placements=[
                # Pre-existing placement must be preserved untouched.
                MechanicalPlacement(
                    ref_des="U1",
                    label="ESP32 (hand placed)",
                    category="Microcontroller",
                    position=MechanicalVector3(1.0, 2.0, 3.0),
                    size=MechanicalVector3(38, 28, 5),
                )
            ],
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` enriches a synthetic IR and asserts
    keyword envelope inference, layer classification, front-face/button-row/
    enclosure placements, preservation of the existing placement, and
    controller-relative relationships."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.electronics.enclosure_layout",
        description="Heuristic mechanical placement seeding for the Hardware "
        "IR (ported from Forma-OSS).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="enrich a synthetic IR and assert the heuristic placements.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the enriched mechanical data as JSON."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    ir = enrich_mechanical_layout(_synthetic_ir())
    mech = ir.mechanical
    assert mech is not None
    placements: Dict[str, MechanicalPlacement] = {
        placement.ref_des: placement for placement in mech.component_placements
    }

    dims = mech.render_dimensions
    # "pocket"/"portable" in the overview trips the audio/pocket preset.
    dims_ok = dims is not None and (dims.x_mm, dims.y_mm, dims.z_mm) == (100, 21, 54)

    # Existing hand placement preserved verbatim.
    preserved = placements["U1"]
    preserved_ok = (
        preserved.label == "ESP32 (hand placed)"
        and preserved.position.x_mm == 1.0
        and len(mech.component_placements) == 8
    )

    layers_ok = (
        placements["ENC1"].layer == "enclosure"
        and placements["BEZ1"].layer == "print"
        and placements["MECH1"].layer == "structural"
        and placements["DS1"].layer == "electrical"
    )

    # Display sits on the front face (negative Y) and is marked front-mounted.
    display = placements["DS1"]
    display_ok = (
        display.position.y_mm < 0
        and display.mounting_face == "front"
        and (display.size.x_mm, display.size.y_mm, display.size.z_mm) == (34, 3, 18)
    )

    # Two buttons form a symmetric row across X on the front face.
    row_ok = (
        placements["SW1"].position.x_mm == -placements["SW2"].position.x_mm
        and placements["SW1"].position.x_mm < 0
        and placements["SW1"].position.y_mm == placements["SW2"].position.y_mm
    )

    # Enclosure shell is centered and spans the full envelope.
    enclosure = placements["ENC1"]
    enclosure_ok = (
        (enclosure.position.x_mm, enclosure.position.y_mm, enclosure.position.z_mm)
        == (0, 0, 0)
        and enclosure.size.x_mm == dims.x_mm
    )

    relationships = mech.spatial_relationships
    controller_rel = [
        rel
        for rel in relationships
        if rel.source_ref_des == "U1" and rel.relation == "spatial offset from controller"
    ]
    bezel_rel = [
        rel for rel in relationships if rel.relation == "aligned with display opening"
    ]
    rel_ok = (
        len(controller_rel) >= 5
        and len(bezel_rel) == 1
        and bezel_rel[0].source_ref_des == "DS1"
        and bezel_rel[0].target_ref_des == "BEZ1"
        and all(rel.axis in ("X", "Y", "Z") for rel in relationships)
    )

    meta_ok = (
        ir.assembly_metadata.get("component_placement_count") == 8
        and ir.assembly_metadata.get("spatial_relationship_count") == len(relationships)
    )

    # Determinism: enriching an identical IR again yields the same document.
    ir2 = enrich_mechanical_layout(_synthetic_ir())
    deterministic_ok = ir2.to_dict() == ir.to_dict()

    if args.json:
        print(json.dumps(mech.to_dict(), indent=2, sort_keys=True))
    else:
        print("enclosure_layout selfcheck:")
        print("  envelope: %.0f x %.0f x %.0f mm" % (dims.x_mm, dims.y_mm, dims.z_mm))
        for placement in mech.component_placements:
            print(
                "  %-5s layer=%-10s pos=(%.1f, %.1f, %.1f) face=%s"
                % (
                    placement.ref_des,
                    placement.layer,
                    placement.position.x_mm,
                    placement.position.y_mm,
                    placement.position.z_mm,
                    placement.mounting_face,
                )
            )
        print("  relationships: %d" % len(relationships))

    ok = (
        dims_ok
        and preserved_ok
        and layers_ok
        and display_ok
        and row_ok
        and enclosure_ok
        and rel_ok
        and meta_ok
        and deterministic_ok
    )
    if not ok:
        print(
            "SELFCHECK FAILED: dims=%s preserved=%s layers=%s display=%s row=%s "
            "enclosure=%s rel=%s meta=%s deterministic=%s"
            % (
                dims_ok,
                preserved_ok,
                layers_ok,
                display_ok,
                row_ok,
                enclosure_ok,
                rel_ok,
                meta_ok,
                deterministic_ok,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        "enclosure_layout selfcheck OK: envelope, layers, placements, "
        "relationships, and determinism all verified"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
