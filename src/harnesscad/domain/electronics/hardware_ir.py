"""Typed Hardware IR for device-level electronics briefs.

Stdlib dataclasses provide tolerant ``to_dict``/``from_dict`` round-tripping:
missing keys fall back to
defaults, unknown keys are ignored, so partially-populated agent output still
loads.

Gap filled: HarnessCAD previously had no electronics/netlist IR at all. Its
briefs describe geometry (an op stream over solids); this schema gives the
same briefs a typed *electrical* layer -- components with pinouts, nets tying
pins together, buses, power rails, MCU pin mappings, assembly steps with
danger flags, and mechanical placement notes that bridge back into the
geometric world (positions/sizes in millimetres).

Design notes:

* ``FunctionalRequirements.missing_info`` is the anti-guess clarification
  hook -- unknowns are recorded as questions instead of being invented.
* ``ValidationSummary`` embeds categorized diagnostics *inside* the IR so the
  document is a record of what was checked, not just a snapshot.
* ``HardwareIR.is_valid`` mirrors "no CRITICAL issues" after validation.

Everything is deterministic and stdlib-only.

Usage::

    from harnesscad.domain.electronics.hardware_ir import HardwareIR
    ir = HardwareIR.from_dict(payload)
    payload_again = ir.to_dict()
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "PinDefinition",
    "ComponentTemplate",
    "ProjectOverview",
    "FunctionalRequirements",
    "ComponentInstance",
    "PinReference",
    "ConnectionNet",
    "BusConnection",
    "PowerRail",
    "PinMappingEntry",
    "AssemblyStep",
    "MechanicalVector3",
    "MechanicalRotation3",
    "MechanicalPlacement",
    "MechanicalSpatialRelationship",
    "MechanicalSource",
    "MechanicalNotes",
    "ValidationIssue",
    "ValidationSummary",
    "HardwareIR",
    "main",
]


def _opt_float(value: Any) -> Optional[float]:
    """Coerce a raw value to float, tolerating None and bad input."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any, default: float = 0.0) -> float:
    result = _opt_float(value)
    return default if result is None else result


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _str_list(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value]


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list_of(cls: Any, value: Any) -> List[Any]:
    """Build a list of dataclass instances from a raw list of dicts."""
    if not isinstance(value, (list, tuple)):
        return []
    out: List[Any] = []
    for item in value:
        if isinstance(item, dict):
            out.append(cls.from_dict(item))
        elif isinstance(item, cls):
            out.append(item)
    return out


# ==========================================
# 1. Component database schemas
# ==========================================


@dataclass
class PinDefinition:
    """A single physical pin: id, functional name, type, optional voltage."""

    pin_id: str = ""
    name: str = ""
    pin_type: str = ""  # Power, Ground, Digital, Analog, I2C, SPI, UART, PWM, Passive
    voltage: Optional[float] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pin_id": self.pin_id,
            "name": self.name,
            "pin_type": self.pin_type,
            "voltage": self.voltage,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PinDefinition":
        return cls(
            pin_id=_as_str(data.get("pin_id")),
            name=_as_str(data.get("name")),
            pin_type=_as_str(data.get("pin_type")),
            voltage=_opt_float(data.get("voltage")),
            description=_opt_str(data.get("description")),
        )


@dataclass
class ComponentTemplate:
    """A catalog part: part number, category, price, pinout, use cases."""

    part_number: str = ""
    name: str = ""
    category: str = ""
    description: str = ""
    price: float = 0.0
    sourcing_url: Optional[str] = None
    pins: List[PinDefinition] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "part_number": self.part_number,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "price": self.price,
            "sourcing_url": self.sourcing_url,
            "pins": [pin.to_dict() for pin in self.pins],
            "use_cases": list(self.use_cases),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentTemplate":
        return cls(
            part_number=_as_str(data.get("part_number")),
            name=_as_str(data.get("name")),
            category=_as_str(data.get("category")),
            description=_as_str(data.get("description")),
            price=_as_float(data.get("price")),
            sourcing_url=_opt_str(data.get("sourcing_url")),
            pins=_list_of(PinDefinition, data.get("pins")),
            use_cases=_str_list(data.get("use_cases")),
        )


# ==========================================
# 2. Project-level Hardware IR
# ==========================================


@dataclass
class ProjectOverview:
    """Project metadata: title, description, difficulty, cost, domain."""

    title: str = ""
    description: str = ""
    difficulty: str = ""  # Beginner, Intermediate, Advanced
    estimated_cost: float = 0.0
    category: str = ""  # IoT, Wearable, Automation, Robotics, Smart Home

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "difficulty": self.difficulty,
            "estimated_cost": self.estimated_cost,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectOverview":
        return cls(
            title=_as_str(data.get("title")),
            description=_as_str(data.get("description")),
            difficulty=_as_str(data.get("difficulty")),
            estimated_cost=_as_float(data.get("estimated_cost")),
            category=_as_str(data.get("category")),
        )


@dataclass
class FunctionalRequirements:
    """Extracted requirements. ``missing_info`` is the anti-guess hook:
    unknowns become clarifying questions instead of invented values."""

    requirements: List[str] = field(default_factory=list)
    power_needs: str = ""
    operating_voltage: float = 3.3
    physical_constraints: List[str] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirements": list(self.requirements),
            "power_needs": self.power_needs,
            "operating_voltage": self.operating_voltage,
            "physical_constraints": list(self.physical_constraints),
            "safety_notes": list(self.safety_notes),
            "missing_info": list(self.missing_info),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionalRequirements":
        return cls(
            requirements=_str_list(data.get("requirements")),
            power_needs=_as_str(data.get("power_needs")),
            operating_voltage=_as_float(data.get("operating_voltage"), 3.3),
            physical_constraints=_str_list(data.get("physical_constraints")),
            safety_notes=_str_list(data.get("safety_notes")),
            missing_info=_str_list(data.get("missing_info")),
        )


@dataclass
class ComponentInstance:
    """An instantiated BOM line: ref_des plus a full pinout."""

    ref_des: str = ""
    part_number: str = ""
    name: str = ""
    category: str = ""
    quantity: int = 1
    unit_price: float = 0.0
    sourcing_url: Optional[str] = None
    rationale: str = ""
    pins: List[PinDefinition] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref_des": self.ref_des,
            "part_number": self.part_number,
            "name": self.name,
            "category": self.category,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "sourcing_url": self.sourcing_url,
            "rationale": self.rationale,
            "pins": [pin.to_dict() for pin in self.pins],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentInstance":
        return cls(
            ref_des=_as_str(data.get("ref_des")),
            part_number=_as_str(data.get("part_number")),
            name=_as_str(data.get("name")),
            category=_as_str(data.get("category")),
            quantity=_as_int(data.get("quantity"), 1),
            unit_price=_as_float(data.get("unit_price")),
            sourcing_url=_opt_str(data.get("sourcing_url")),
            rationale=_as_str(data.get("rationale")),
            pins=_list_of(PinDefinition, data.get("pins")),
        )


@dataclass
class PinReference:
    """A (ref_des, pin_id) pair naming one component pin."""

    ref_des: str = ""
    pin_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"ref_des": self.ref_des, "pin_id": self.pin_id}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PinReference":
        return cls(
            ref_des=_as_str(data.get("ref_des")),
            pin_id=_as_str(data.get("pin_id")),
        )


@dataclass
class ConnectionNet:
    """An electrical net tying a set of component pins together."""

    net_id: str = ""
    name: str = ""
    net_type: str = ""  # Power, Ground, Analog, Digital, I2C, SPI, UART, PWM
    voltage: Optional[float] = None
    pins: List[PinReference] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "net_id": self.net_id,
            "name": self.name,
            "net_type": self.net_type,
            "voltage": self.voltage,
            "pins": [pin.to_dict() for pin in self.pins],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConnectionNet":
        return cls(
            net_id=_as_str(data.get("net_id")),
            name=_as_str(data.get("name")),
            net_type=_as_str(data.get("net_type")),
            voltage=_opt_float(data.get("voltage")),
            pins=_list_of(PinReference, data.get("pins")),
        )


@dataclass
class BusConnection:
    """A digital communication bus grouping related nets (I2C, SPI, ...)."""

    bus_id: str = ""
    bus_type: str = ""  # I2C, SPI, UART, CAN
    clock_frequency_hz: Optional[float] = None
    nets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bus_id": self.bus_id,
            "bus_type": self.bus_type,
            "clock_frequency_hz": self.clock_frequency_hz,
            "nets": list(self.nets),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BusConnection":
        return cls(
            bus_id=_as_str(data.get("bus_id")),
            bus_type=_as_str(data.get("bus_type")),
            clock_frequency_hz=_opt_float(data.get("clock_frequency_hz")),
            nets=_str_list(data.get("nets")),
        )


@dataclass
class PowerRail:
    """A summarized power delivery rail with a nominal capacity."""

    rail_id: str = ""
    voltage: float = 0.0
    max_current_capacity_ma: float = 0.0
    source_component: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rail_id": self.rail_id,
            "voltage": self.voltage,
            "max_current_capacity_ma": self.max_current_capacity_ma,
            "source_component": self.source_component,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PowerRail":
        return cls(
            rail_id=_as_str(data.get("rail_id")),
            voltage=_as_float(data.get("voltage")),
            max_current_capacity_ma=_as_float(data.get("max_current_capacity_ma")),
            source_component=_as_str(data.get("source_component")),
        )


@dataclass
class PinMappingEntry:
    """An MCU functional pin map row: GPIO -> peripheral signal -> net."""

    mcu_pin: str = ""
    connected_to: str = ""
    net_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mcu_pin": self.mcu_pin,
            "connected_to": self.connected_to,
            "net_name": self.net_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PinMappingEntry":
        return cls(
            mcu_pin=_as_str(data.get("mcu_pin")),
            connected_to=_as_str(data.get("connected_to")),
            net_name=_as_str(data.get("net_name")),
        )


@dataclass
class AssemblyStep:
    """A physical build instruction; ``danger_flag`` marks risky steps."""

    step_num: int = 0
    title: str = ""
    description: str = ""
    danger_flag: bool = False
    danger_message: Optional[str] = None
    affected_components: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_num": self.step_num,
            "title": self.title,
            "description": self.description,
            "danger_flag": self.danger_flag,
            "danger_message": self.danger_message,
            "affected_components": list(self.affected_components),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssemblyStep":
        return cls(
            step_num=_as_int(data.get("step_num")),
            title=_as_str(data.get("title")),
            description=_as_str(data.get("description")),
            danger_flag=_as_bool(data.get("danger_flag")),
            danger_message=_opt_str(data.get("danger_message")),
            affected_components=_str_list(data.get("affected_components")),
        )


# ==========================================
# 3. Mechanical bridge into the geometric world
# ==========================================


@dataclass
class MechanicalVector3:
    """A millimetre-unit vector: X width, Y depth, Z height."""

    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"x_mm": self.x_mm, "y_mm": self.y_mm, "z_mm": self.z_mm}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MechanicalVector3":
        return cls(
            x_mm=_as_float(data.get("x_mm")),
            y_mm=_as_float(data.get("y_mm")),
            z_mm=_as_float(data.get("z_mm")),
        )


@dataclass
class MechanicalRotation3:
    """Euler orientation in degrees around X, Y, and Z."""

    x_deg: float = 0.0
    y_deg: float = 0.0
    z_deg: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"x_deg": self.x_deg, "y_deg": self.y_deg, "z_deg": self.z_deg}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MechanicalRotation3":
        return cls(
            x_deg=_as_float(data.get("x_deg")),
            y_deg=_as_float(data.get("y_deg")),
            z_deg=_as_float(data.get("z_deg")),
        )


@dataclass
class MechanicalPlacement:
    """A per-component 3D placement relative to the enclosure center."""

    ref_des: str = ""
    label: Optional[str] = None
    category: Optional[str] = None
    layer: str = "electrical"  # electrical, mechanism, print, enclosure, structural, misc
    position: MechanicalVector3 = field(default_factory=MechanicalVector3)
    size: MechanicalVector3 = field(default_factory=MechanicalVector3)
    orientation_deg: MechanicalRotation3 = field(default_factory=MechanicalRotation3)
    mounting_face: Optional[str] = None  # front, back, floor, lid, left, right
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref_des": self.ref_des,
            "label": self.label,
            "category": self.category,
            "layer": self.layer,
            "position": self.position.to_dict(),
            "size": self.size.to_dict(),
            "orientation_deg": self.orientation_deg.to_dict(),
            "mounting_face": self.mounting_face,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MechanicalPlacement":
        return cls(
            ref_des=_as_str(data.get("ref_des")),
            label=_opt_str(data.get("label")),
            category=_opt_str(data.get("category")),
            layer=_as_str(data.get("layer"), "electrical"),
            position=MechanicalVector3.from_dict(_dict_or_empty(data.get("position"))),
            size=MechanicalVector3.from_dict(_dict_or_empty(data.get("size"))),
            orientation_deg=MechanicalRotation3.from_dict(
                _dict_or_empty(data.get("orientation_deg"))
            ),
            mounting_face=_opt_str(data.get("mounting_face")),
            notes=_opt_str(data.get("notes")),
        )


@dataclass
class MechanicalSpatialRelationship:
    """A physical offset/alignment relationship between two placements."""

    source_ref_des: str = ""
    target_ref_des: str = ""
    relation: str = ""  # centered-above, adjacent-to, mounted-on, aligned-with, ...
    axis: Optional[str] = None  # X, Y, or Z
    offset_mm: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_ref_des": self.source_ref_des,
            "target_ref_des": self.target_ref_des,
            "relation": self.relation,
            "axis": self.axis,
            "offset_mm": self.offset_mm,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MechanicalSpatialRelationship":
        return cls(
            source_ref_des=_as_str(data.get("source_ref_des")),
            target_ref_des=_as_str(data.get("target_ref_des")),
            relation=_as_str(data.get("relation")),
            axis=_opt_str(data.get("axis")),
            offset_mm=_opt_float(data.get("offset_mm")),
            notes=_opt_str(data.get("notes")),
        )


@dataclass
class MechanicalSource:
    """A CAD, enclosure, or fabrication source record."""

    name: str = ""
    source_type: str = ""  # Open STL, Paid STL, Vendor CAD, Reference CAD, Fabrication Estimate
    url: str = ""
    file_formats: List[str] = field(default_factory=list)
    license: Optional[str] = None
    estimated_unit_price_usd: float = 0.0
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_type": self.source_type,
            "url": self.url,
            "file_formats": list(self.file_formats),
            "license": self.license,
            "estimated_unit_price_usd": self.estimated_unit_price_usd,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MechanicalSource":
        return cls(
            name=_as_str(data.get("name")),
            source_type=_as_str(data.get("source_type")),
            url=_as_str(data.get("url")),
            file_formats=_str_list(data.get("file_formats")),
            license=_opt_str(data.get("license")),
            estimated_unit_price_usd=_as_float(data.get("estimated_unit_price_usd")),
            notes=_opt_str(data.get("notes")),
        )


@dataclass
class MechanicalNotes:
    """Enclosure and fabrication specifications plus the render contract."""

    enclosure_type: str = ""  # 3D Printed, Off-the-shelf, Custom Acrylic, ...
    mounting_guidance: str = ""
    fabrication_details: List[str] = field(default_factory=list)
    fabrication_cost_estimate_usd: float = 0.0
    cad_sources: List[MechanicalSource] = field(default_factory=list)
    manufacturability_rating: str = ""  # Easy, Moderate, Challenging
    render_dimensions: Optional[MechanicalVector3] = None
    component_placements: List[MechanicalPlacement] = field(default_factory=list)
    spatial_relationships: List[MechanicalSpatialRelationship] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enclosure_type": self.enclosure_type,
            "mounting_guidance": self.mounting_guidance,
            "fabrication_details": list(self.fabrication_details),
            "fabrication_cost_estimate_usd": self.fabrication_cost_estimate_usd,
            "cad_sources": [source.to_dict() for source in self.cad_sources],
            "manufacturability_rating": self.manufacturability_rating,
            "render_dimensions": (
                self.render_dimensions.to_dict() if self.render_dimensions else None
            ),
            "component_placements": [
                placement.to_dict() for placement in self.component_placements
            ],
            "spatial_relationships": [
                relationship.to_dict() for relationship in self.spatial_relationships
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MechanicalNotes":
        render_dimensions = data.get("render_dimensions")
        return cls(
            enclosure_type=_as_str(data.get("enclosure_type")),
            mounting_guidance=_as_str(data.get("mounting_guidance")),
            fabrication_details=_str_list(data.get("fabrication_details")),
            fabrication_cost_estimate_usd=_as_float(
                data.get("fabrication_cost_estimate_usd")
            ),
            cad_sources=_list_of(MechanicalSource, data.get("cad_sources")),
            manufacturability_rating=_as_str(data.get("manufacturability_rating")),
            render_dimensions=(
                MechanicalVector3.from_dict(render_dimensions)
                if isinstance(render_dimensions, dict)
                else None
            ),
            component_placements=_list_of(
                MechanicalPlacement, data.get("component_placements")
            ),
            spatial_relationships=_list_of(
                MechanicalSpatialRelationship, data.get("spatial_relationships")
            ),
        )


# ==========================================
# 4. Validation diagnostics
# ==========================================


@dataclass
class ValidationIssue:
    """One diagnostic: severity CRITICAL/WARNING/INFO plus a remedy."""

    severity: str = "INFO"
    category: str = ""  # Short Circuit, Voltage Mismatch, Unpowered IC, Pin Conflict, Overcurrent, Safety Block
    description: str = ""
    troubleshooting: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "troubleshooting": self.troubleshooting,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationIssue":
        return cls(
            severity=_as_str(data.get("severity"), "INFO"),
            category=_as_str(data.get("category")),
            description=_as_str(data.get("description")),
            troubleshooting=_as_str(data.get("troubleshooting")),
        )


@dataclass
class ValidationSummary:
    """Issues grouped into critical/warning/info lists."""

    critical: List[ValidationIssue] = field(default_factory=list)
    warning: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "critical": [issue.to_dict() for issue in self.critical],
            "warning": [issue.to_dict() for issue in self.warning],
            "info": [issue.to_dict() for issue in self.info],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationSummary":
        return cls(
            critical=_list_of(ValidationIssue, data.get("critical")),
            warning=_list_of(ValidationIssue, data.get("warning")),
            info=_list_of(ValidationIssue, data.get("info")),
        )


# ==========================================
# 5. The master document
# ==========================================


@dataclass
class HardwareIR:
    """The master typed document capturing an entire hardware design."""

    hardware_ir_version: str = "0.1"
    overview: Optional[ProjectOverview] = None
    requirements: Optional[FunctionalRequirements] = None
    components: List[ComponentInstance] = field(default_factory=list)
    nets: List[ConnectionNet] = field(default_factory=list)
    buses: List[BusConnection] = field(default_factory=list)
    pin_mappings: List[PinMappingEntry] = field(default_factory=list)
    assembly: List[AssemblyStep] = field(default_factory=list)
    mechanical: Optional[MechanicalNotes] = None
    constraints: List[str] = field(default_factory=list)
    power_rails: List[PowerRail] = field(default_factory=list)
    estimated_current_draw_ma: float = 0.0
    fabrication_notes: List[str] = field(default_factory=list)
    assembly_metadata: Dict[str, Any] = field(default_factory=dict)
    project_version_history: List[Dict[str, Any]] = field(default_factory=list)
    validation: ValidationSummary = field(default_factory=ValidationSummary)
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hardware_ir_version": self.hardware_ir_version,
            "overview": self.overview.to_dict() if self.overview else None,
            "requirements": self.requirements.to_dict() if self.requirements else None,
            "components": [component.to_dict() for component in self.components],
            "nets": [net.to_dict() for net in self.nets],
            "buses": [bus.to_dict() for bus in self.buses],
            "pin_mappings": [entry.to_dict() for entry in self.pin_mappings],
            "assembly": [step.to_dict() for step in self.assembly],
            "mechanical": self.mechanical.to_dict() if self.mechanical else None,
            "constraints": list(self.constraints),
            "power_rails": [rail.to_dict() for rail in self.power_rails],
            "estimated_current_draw_ma": self.estimated_current_draw_ma,
            "fabrication_notes": list(self.fabrication_notes),
            "assembly_metadata": dict(self.assembly_metadata),
            "project_version_history": [
                dict(entry) for entry in self.project_version_history
            ],
            "validation": self.validation.to_dict(),
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardwareIR":
        overview = data.get("overview")
        requirements = data.get("requirements")
        mechanical = data.get("mechanical")
        history = data.get("project_version_history")
        return cls(
            hardware_ir_version=_as_str(data.get("hardware_ir_version"), "0.1"),
            overview=(
                ProjectOverview.from_dict(overview)
                if isinstance(overview, dict)
                else None
            ),
            requirements=(
                FunctionalRequirements.from_dict(requirements)
                if isinstance(requirements, dict)
                else None
            ),
            components=_list_of(ComponentInstance, data.get("components")),
            nets=_list_of(ConnectionNet, data.get("nets")),
            buses=_list_of(BusConnection, data.get("buses")),
            pin_mappings=_list_of(PinMappingEntry, data.get("pin_mappings")),
            assembly=_list_of(AssemblyStep, data.get("assembly")),
            mechanical=(
                MechanicalNotes.from_dict(mechanical)
                if isinstance(mechanical, dict)
                else None
            ),
            constraints=_str_list(data.get("constraints")),
            power_rails=_list_of(PowerRail, data.get("power_rails")),
            estimated_current_draw_ma=_as_float(data.get("estimated_current_draw_ma")),
            fabrication_notes=_str_list(data.get("fabrication_notes")),
            assembly_metadata=_dict_or_empty(data.get("assembly_metadata")),
            project_version_history=[
                dict(entry)
                for entry in (history if isinstance(history, list) else [])
                if isinstance(entry, dict)
            ],
            validation=ValidationSummary.from_dict(
                _dict_or_empty(data.get("validation"))
            ),
            is_valid=_as_bool(data.get("is_valid"), True),
        )


# ==========================================
# 6. Selfcheck CLI
# ==========================================


def _synthetic_ir() -> HardwareIR:
    """Build a small but structurally complete synthetic HardwareIR."""
    mcu = ComponentInstance(
        ref_des="U1",
        part_number="ESP32-WROOM-32D",
        name="ESP32 Development Board",
        category="Microcontroller",
        unit_price=6.5,
        rationale="WiFi MCU for the control loop.",
        pins=[
            PinDefinition(pin_id="3V3", name="VCC", pin_type="Power", voltage=3.3),
            PinDefinition(pin_id="GND", name="GND", pin_type="Ground"),
            PinDefinition(pin_id="GPIO21", name="SDA", pin_type="I2C", voltage=3.3),
        ],
    )
    display = ComponentInstance(
        ref_des="DS1",
        part_number="SSD1306-OLED",
        name="0.96in OLED Display",
        category="Display",
        unit_price=3.0,
        rationale="Status readout.",
        pins=[
            PinDefinition(pin_id="VCC", name="VCC", pin_type="Power", voltage=3.3),
            PinDefinition(pin_id="GND", name="GND", pin_type="Ground"),
            PinDefinition(pin_id="SDA", name="SDA", pin_type="I2C", voltage=3.3),
        ],
    )
    ir = HardwareIR(
        overview=ProjectOverview(
            title="Desk Status Cube",
            description="A pocket status display cube.",
            difficulty="Beginner",
            estimated_cost=9.5,
            category="IoT",
        ),
        requirements=FunctionalRequirements(
            requirements=["Show status on OLED"],
            power_needs="5V USB",
            missing_info=["Which status feed should be shown?"],
        ),
        components=[mcu, display],
        nets=[
            ConnectionNet(
                net_id="NET_3V3",
                name="3.3V Power Rail",
                net_type="Power",
                voltage=3.3,
                pins=[
                    PinReference(ref_des="U1", pin_id="3V3"),
                    PinReference(ref_des="DS1", pin_id="VCC"),
                ],
            ),
            ConnectionNet(
                net_id="NET_I2C_SDA",
                name="I2C Data",
                net_type="I2C",
                pins=[
                    PinReference(ref_des="U1", pin_id="GPIO21"),
                    PinReference(ref_des="DS1", pin_id="SDA"),
                ],
            ),
        ],
        assembly=[
            AssemblyStep(
                step_num=1,
                title="Solder headers",
                description="Solder headers to the MCU board.",
                danger_flag=True,
                danger_message="Soldering iron is hot.",
                affected_components=["U1"],
            )
        ],
        mechanical=MechanicalNotes(
            enclosure_type="3D Printed",
            mounting_guidance="M3 standoffs at each corner.",
            manufacturability_rating="Easy",
            render_dimensions=MechanicalVector3(x_mm=92, y_mm=64, z_mm=38),
            component_placements=[
                MechanicalPlacement(
                    ref_des="U1",
                    label="ESP32",
                    category="Microcontroller",
                    position=MechanicalVector3(0, 0, -1.5),
                    size=MechanicalVector3(38, 28, 5),
                )
            ],
            spatial_relationships=[
                MechanicalSpatialRelationship(
                    source_ref_des="U1",
                    target_ref_des="DS1",
                    relation="spatial offset from controller",
                    axis="Y",
                    offset_mm=-27.5,
                )
            ],
        ),
        power_rails=[
            PowerRail(
                rail_id="RAIL_3V3",
                voltage=3.3,
                max_current_capacity_ma=500.0,
                source_component="U1",
            )
        ],
        estimated_current_draw_ma=105.0,
        assembly_metadata={"render_pipeline": "selfcheck"},
        project_version_history=[{"revision": 1, "note": "initial"}],
        validation=ValidationSummary(
            info=[
                ValidationIssue(
                    severity="INFO",
                    category="Note",
                    description="Synthetic design.",
                    troubleshooting="None required.",
                )
            ]
        ),
    )
    return ir


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` round-trips a synthetic HardwareIR
    through to_dict/from_dict (including a lossy payload with unknown and
    missing keys) and asserts the round trip is exact."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.electronics.hardware_ir",
        description="Typed Hardware IR dataclasses (ported from Forma-OSS).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="round-trip a synthetic HardwareIR through to_dict/from_dict.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the synthetic IR as JSON."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    ir = _synthetic_ir()
    payload = ir.to_dict()

    # Exact round trip.
    restored = HardwareIR.from_dict(payload)
    if restored.to_dict() != payload:
        print("SELFCHECK FAILED: round trip is not exact", file=sys.stderr)
        return 1

    # JSON round trip (proves everything is JSON-serializable).
    rehydrated = HardwareIR.from_dict(json.loads(json.dumps(payload)))
    if rehydrated.to_dict() != payload:
        print("SELFCHECK FAILED: JSON round trip is not exact", file=sys.stderr)
        return 1

    # Tolerant load: unknown keys ignored, missing keys defaulted.
    messy = dict(payload)
    messy["totally_unknown_key"] = {"ignored": True}
    del messy["buses"]
    del messy["is_valid"]
    tolerant = HardwareIR.from_dict(messy)
    if tolerant.buses != [] or tolerant.is_valid is not True:
        print("SELFCHECK FAILED: tolerant from_dict misbehaved", file=sys.stderr)
        return 1

    # Spot-check nested structure survived.
    ok = (
        restored.components[0].pins[0].voltage == 3.3
        and restored.mechanical is not None
        and restored.mechanical.render_dimensions is not None
        and restored.mechanical.render_dimensions.x_mm == 92
        and restored.requirements is not None
        and restored.requirements.missing_info
        and restored.assembly[0].danger_flag
        and restored.validation.info[0].severity == "INFO"
    )
    if not ok:
        print("SELFCHECK FAILED: nested fields did not survive", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("hardware_ir selfcheck OK:")
        print("  components: %d" % len(restored.components))
        print("  nets: %d" % len(restored.nets))
        print("  placements: %d" % len(restored.mechanical.component_placements))
        print("  missing_info: %s" % restored.requirements.missing_info[0])
        print("  round trips: exact + JSON + tolerant")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
