"""Rule-based netlist validation for the Hardware IR, mined from Forma-OSS.

Ported behavior-verbatim from Forma-OSS ``blueprint_core/validation.py``
(``validate_circuit`` + ``build_validation_summary``, documented in
``docs/validation.md``), converted from pydantic to the stdlib dataclass IR in
``harnesscad.domain.electronics.hardware_ir``.

Gap filled: HarnessCAD previously had no electronics/netlist IR at all, and
therefore no way to lint the electrical layer of a device-level text-to-CAD
brief. These five deterministic rules give it that verifier:

1. **Short circuit** -- a power pin and a ground pin on the same net
   (CRITICAL, listing the offending pins).
2. **Voltage mismatch** -- more than one distinct pin voltage on a net with a
   spread over 0.5V (WARNING, with level-shifter troubleshooting).
3. **Unpowered IC** -- active categories (microcontroller, sensor, display,
   actuator) whose power or ground pins appear in no net (CRITICAL each).
4. **Pin reuse conflict** -- a non-power/ground/passive pin appearing in more
   than one net (CRITICAL).
5. **Overcurrent risk** -- high-draw actuators (relay/servo/motor/pump by
   name or part number) sharing a 3.3V power net with the MCU's power pin
   (WARNING).

Severities follow the source taxonomy: CRITICAL must be fixed, WARNING is
risky but overridable, INFO is advisory. ``is_design_valid`` mirrors the
source's "valid means no CRITICAL issues".

Deterministic: same components + nets in -> same issues out, in a stable order.

Usage::

    from harnesscad.domain.electronics.circuit_validation import (
        validate_circuit, build_validation_summary, is_design_valid,
    )
    issues = validate_circuit(ir.components, ir.nets)
    ir.validation = build_validation_summary(issues)
    ir.is_valid = is_design_valid(issues)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

from harnesscad.domain.electronics.hardware_ir import (
    ComponentInstance,
    ConnectionNet,
    PinDefinition,
    PinReference,
    ValidationIssue,
    ValidationSummary,
)

__all__ = [
    "validate_circuit",
    "build_validation_summary",
    "is_design_valid",
    "main",
]

# Categories treated as active ICs that must be powered and grounded (Rule 3).
ACTIVE_CATEGORIES = ("microcontroller", "sensor", "display", "actuator")

# Exact part numbers always treated as high-draw actuators (Rule 5).
HIGH_DRAW_PART_NUMBERS = ("Relay-5V-1Ch", "SG90-Servo")

# Name/part keywords marking high-draw actuators (Rule 5).
HIGH_DRAW_KEYWORDS = ("relay", "servo", "motor", "pump")


def validate_circuit(
    components: List[ComponentInstance], nets: List[ConnectionNet]
) -> List[ValidationIssue]:
    """Run the five electrical/logical rules on the structured netlist.

    Returns a list of ValidationIssues (CRITICAL and WARNING) with
    troubleshooting advice, in a deterministic order.
    """
    issues: List[ValidationIssue] = []

    # Pre-index component pin attributes for fast lookup:
    # (ref_des, pin_id) -> PinDefinition
    pin_lookup: Dict[Tuple[str, str], PinDefinition] = {}
    component_lookup: Dict[str, ComponentInstance] = {}
    for comp in components:
        component_lookup[comp.ref_des] = comp
        for pin in comp.pins:
            pin_lookup[(comp.ref_des, pin.pin_id)] = pin

    # Pin -> nets reverse lookup for conflict detection:
    # (ref_des, pin_id) -> list of net_ids
    pin_to_nets: Dict[Tuple[str, str], List[str]] = {}
    for net in nets:
        for pin_ref in net.pins:
            key = (pin_ref.ref_des, pin_ref.pin_id)
            if key not in pin_to_nets:
                pin_to_nets[key] = []
            pin_to_nets[key].append(net.net_id)

    # ----------------------------------------------------
    # Rule 1: Short circuit (power pin directly to ground pin)
    # ----------------------------------------------------
    for net in nets:
        has_power = False
        has_ground = False
        power_pins: List[str] = []
        ground_pins: List[str] = []

        for pin_ref in net.pins:
            pin = pin_lookup.get((pin_ref.ref_des, pin_ref.pin_id))
            if pin:
                if pin.pin_type.lower() == "power":
                    has_power = True
                    power_pins.append(f"{pin_ref.ref_des}.{pin_ref.pin_id}")
                elif pin.pin_type.lower() == "ground":
                    has_ground = True
                    ground_pins.append(f"{pin_ref.ref_des}.{pin_ref.pin_id}")

        if has_power and has_ground:
            issues.append(
                ValidationIssue(
                    severity="CRITICAL",
                    category="Short Circuit",
                    description=(
                        f"Direct electrical short detected in net '{net.name}' "
                        f"({net.net_id}). Power pins [{', '.join(power_pins)}] are "
                        f"connected directly to Ground pins "
                        f"[{', '.join(ground_pins)}]."
                    ),
                    troubleshooting=(
                        "Separate the power rail connections from the ground "
                        "reference rail. Power pins must only connect to other "
                        "power nodes, never directly to GND."
                    ),
                )
            )

    # ----------------------------------------------------
    # Rule 2: Voltage mismatch on a shared net
    # ----------------------------------------------------
    for net in nets:
        voltages: Set[float] = set()
        connected_pins: List[str] = []

        for pin_ref in net.pins:
            pin = pin_lookup.get((pin_ref.ref_des, pin_ref.pin_id))
            if pin and pin.voltage is not None:
                voltages.add(pin.voltage)
                connected_pins.append(
                    f"{pin_ref.ref_des}.{pin_ref.pin_id} ({pin.voltage}V)"
                )

        # Multiple different voltages on the same signal rail.
        if len(voltages) > 1:
            max_v = max(voltages)
            min_v = min(voltages)
            # Only flag a significant spread (e.g. 5.0V and 3.3V together).
            if max_v - min_v > 0.5:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        category="Voltage Mismatch",
                        description=(
                            f"Potential voltage mismatch in net '{net.name}' "
                            f"({net.net_id}). Pins with different voltages are "
                            f"connected on the same net: "
                            f"{', '.join(connected_pins)}."
                        ),
                        troubleshooting=(
                            f"Use an active level-shifter (e.g., TXB0104) to "
                            f"bridge logic between {min_v}V and {max_v}V lines, "
                            f"or use a component operating at compatible voltages."
                        ),
                    )
                )

    # ----------------------------------------------------
    # Rule 3: Floating / unpowered active IC
    # ----------------------------------------------------
    for ref_des, comp in component_lookup.items():
        if comp.category.lower() in ACTIVE_CATEGORIES:
            has_power_pin = False
            has_ground_pin = False
            power_connected = False
            ground_connected = False
            p_pin_ids: List[str] = []
            g_pin_ids: List[str] = []

            for pin in comp.pins:
                if pin.pin_type.lower() == "power":
                    has_power_pin = True
                    p_pin_ids.append(pin.pin_id)
                    if (ref_des, pin.pin_id) in pin_to_nets:
                        power_connected = True
                elif pin.pin_type.lower() == "ground":
                    has_ground_pin = True
                    g_pin_ids.append(pin.pin_id)
                    if (ref_des, pin.pin_id) in pin_to_nets:
                        ground_connected = True

            if has_power_pin and not power_connected:
                issues.append(
                    ValidationIssue(
                        severity="CRITICAL",
                        category="Unpowered IC",
                        description=(
                            f"Active component '{comp.name}' ({ref_des}) is "
                            f"unpowered. None of its power pins "
                            f"[{', '.join(p_pin_ids)}] are connected to an "
                            f"active power net."
                        ),
                        troubleshooting=(
                            f"Connect one of the VCC/Power pins on {ref_des} to "
                            f"the main power rail (e.g., 3.3V or 5V net)."
                        ),
                    )
                )
            if has_ground_pin and not ground_connected:
                issues.append(
                    ValidationIssue(
                        severity="CRITICAL",
                        category="Unpowered IC",
                        description=(
                            f"Active component '{comp.name}' ({ref_des}) has no "
                            f"ground reference. None of its ground pins "
                            f"[{', '.join(g_pin_ids)}] are tied to the GND net."
                        ),
                        troubleshooting=(
                            f"Connect the GND/Ground pin on {ref_des} to the "
                            f"common system Ground net (GND)."
                        ),
                    )
                )

    # ----------------------------------------------------
    # Rule 4: Pin reuse conflict across independent nets
    # ----------------------------------------------------
    for (ref_des, pin_id), net_ids in pin_to_nets.items():
        # Exclude passive/power/ground buses which naturally share pins.
        pin = pin_lookup.get((ref_des, pin_id))
        if pin and pin.pin_type.lower() not in ("power", "ground", "passive"):
            if len(net_ids) > 1:
                comp = component_lookup.get(ref_des)
                comp_name = comp.name if comp else ref_des
                issues.append(
                    ValidationIssue(
                        severity="CRITICAL",
                        category="Pin Conflict",
                        description=(
                            f"Pin reuse conflict detected! Pin '{pin_id}' on "
                            f"'{comp_name}' ({ref_des}) is connected to multiple "
                            f"independent signal nets: {', '.join(net_ids)}."
                        ),
                        troubleshooting=(
                            f"Reassign pin '{pin_id}' to only belong to a single "
                            f"signal net. Signal pins cannot be shared directly "
                            f"across separate signal/communication lines."
                        ),
                    )
                )

    # ----------------------------------------------------
    # Rule 5: Overcurrent risk (power-hungry actuators on the MCU rail)
    # ----------------------------------------------------
    has_mcu = False
    mcu_ref: Optional[str] = None
    high_draw_actuator_refs: Dict[str, str] = {}

    for ref_des, comp in component_lookup.items():
        if comp.category.lower() == "microcontroller":
            has_mcu = True
            mcu_ref = ref_des
        else:
            component_text = f"{comp.name} {comp.part_number}".lower()
            is_high_draw_actuator = comp.part_number in HIGH_DRAW_PART_NUMBERS or any(
                keyword in component_text for keyword in HIGH_DRAW_KEYWORDS
            )
            if is_high_draw_actuator:
                high_draw_actuator_refs[ref_des] = f"{comp.name} ({ref_des})"

    if has_mcu and high_draw_actuator_refs:
        for net in nets:
            if net.net_type.lower() != "power" or net.voltage != 3.3:
                continue

            contains_mcu_power_pin = False
            powered_actuators: List[str] = []
            for pin_ref in net.pins:
                pin = pin_lookup.get((pin_ref.ref_des, pin_ref.pin_id))
                if not pin or pin.pin_type.lower() != "power":
                    continue
                if pin_ref.ref_des == mcu_ref:
                    contains_mcu_power_pin = True
                elif pin_ref.ref_des in high_draw_actuator_refs:
                    powered_actuators.append(
                        high_draw_actuator_refs[pin_ref.ref_des]
                    )

            if contains_mcu_power_pin and powered_actuators:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        category="Overcurrent Risk",
                        description=(
                            f"High-power actuator(s) "
                            f"[{', '.join(powered_actuators)}] are powered from "
                            f"the same 3.3V low-current output net '{net.name}' "
                            f"as the MCU ({mcu_ref}). Relays and servo motors "
                            f"draw peak currents that can crash the "
                            f"microcontroller or burn out its internal voltage "
                            f"regulator."
                        ),
                        troubleshooting=(
                            "Isolate the actuator power. Connect the servo/relay "
                            "power pin to a dedicated 5V input rail or external "
                            "power source, sharing only the ground reference "
                            "(GND) with the MCU."
                        ),
                    )
                )

    return issues


def build_validation_summary(issues: List[ValidationIssue]) -> ValidationSummary:
    """Group individual issues into critical, warning, and info lists."""
    critical = [issue for issue in issues if issue.severity.upper() == "CRITICAL"]
    warning = [issue for issue in issues if issue.severity.upper() == "WARNING"]
    info = [issue for issue in issues if issue.severity.upper() == "INFO"]
    return ValidationSummary(critical=critical, warning=warning, info=info)


def is_design_valid(issues: List[ValidationIssue]) -> bool:
    """A design is valid when no issue is CRITICAL (WARNING/INFO allowed)."""
    return not any(issue.severity.upper() == "CRITICAL" for issue in issues)


# ==========================================
# Selfcheck fixtures
# ==========================================


def _pin(pin_id: str, pin_type: str, voltage: Optional[float] = None) -> PinDefinition:
    return PinDefinition(pin_id=pin_id, name=pin_id, pin_type=pin_type, voltage=voltage)


def _bad_design() -> Tuple[List[ComponentInstance], List[ConnectionNet]]:
    """A tiny synthetic design that trips every one of the five rules."""
    mcu = ComponentInstance(
        ref_des="U1",
        part_number="ESP32-WROOM-32D",
        name="ESP32 Development Board",
        category="Microcontroller",
        pins=[
            _pin("3V3", "Power", 3.3),
            _pin("GND", "Ground"),
            _pin("GPIO4", "Digital", 3.3),
        ],
    )
    # Sensor with a 5V pin on the same net as a 3.3V pin (Rule 2) and its
    # power/ground pins deliberately left out of every net (Rule 3).
    sensor = ComponentInstance(
        ref_des="SEN1",
        part_number="DHT11",
        name="Temperature Sensor",
        category="Sensor",
        pins=[
            _pin("VCC", "Power", 5.0),
            _pin("GND", "Ground"),
            _pin("DATA", "Digital", 5.0),
        ],
    )
    servo = ComponentInstance(
        ref_des="M1",
        part_number="SG90-Servo",
        name="Micro Servo",
        category="Actuator",
        pins=[
            _pin("VCC", "Power", 3.3),
            _pin("GND", "Ground"),
            _pin("PWM", "PWM", 3.3),
        ],
    )
    nets = [
        # Rule 1: power and ground pins shorted on one net. Also Rule 5: the
        # MCU power pin shares the 3.3V power net with a servo power pin.
        ConnectionNet(
            net_id="NET_3V3",
            name="3.3V Power Rail",
            net_type="Power",
            voltage=3.3,
            pins=[
                PinReference(ref_des="U1", pin_id="3V3"),
                PinReference(ref_des="M1", pin_id="VCC"),
                PinReference(ref_des="U1", pin_id="GND"),
            ],
        ),
        # Rule 2: 3.3V GPIO and 5V sensor data pin on the same signal net.
        # Rule 4 (part 1): U1.GPIO4 is in this net...
        ConnectionNet(
            net_id="NET_SIG_A",
            name="Sensor Data",
            net_type="Digital",
            pins=[
                PinReference(ref_des="U1", pin_id="GPIO4"),
                PinReference(ref_des="SEN1", pin_id="DATA"),
            ],
        ),
        # Rule 4 (part 2): ...and also in this one.
        ConnectionNet(
            net_id="NET_SIG_B",
            name="Servo Signal",
            net_type="PWM",
            pins=[
                PinReference(ref_des="U1", pin_id="GPIO4"),
                PinReference(ref_des="M1", pin_id="PWM"),
            ],
        ),
        # Servo ground so M1 only trips Rule 3 for power via... actually M1's
        # power is connected (NET_3V3); tie its ground here too. SEN1 stays
        # fully unpowered so Rule 3 fires twice for it.
        ConnectionNet(
            net_id="NET_GND",
            name="Ground",
            net_type="Ground",
            pins=[
                PinReference(ref_des="M1", pin_id="GND"),
            ],
        ),
    ]
    return [mcu, sensor, servo], nets


def _good_design() -> Tuple[List[ComponentInstance], List[ConnectionNet]]:
    """A tiny synthetic design that passes every rule."""
    mcu = ComponentInstance(
        ref_des="U1",
        part_number="ESP32-WROOM-32D",
        name="ESP32 Development Board",
        category="Microcontroller",
        pins=[
            _pin("3V3", "Power", 3.3),
            _pin("GND", "Ground"),
            _pin("GPIO21", "I2C", 3.3),
        ],
    )
    display = ComponentInstance(
        ref_des="DS1",
        part_number="SSD1306-OLED",
        name="0.96in OLED Display",
        category="Display",
        pins=[
            _pin("VCC", "Power", 3.3),
            _pin("GND", "Ground"),
            _pin("SDA", "I2C", 3.3),
        ],
    )
    nets = [
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
            net_id="NET_GND",
            name="Ground",
            net_type="Ground",
            pins=[
                PinReference(ref_des="U1", pin_id="GND"),
                PinReference(ref_des="DS1", pin_id="GND"),
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
    ]
    return [mcu, display], nets


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` validates a synthetic design that
    trips all five rules and one that passes cleanly, and asserts the
    expected categories, severities, and validity verdicts."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.electronics.circuit_validation",
        description="Rule-based Hardware IR netlist validation "
        "(ported from Forma-OSS).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="run the five rules on synthetic failing and passing designs.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the issue lists as JSON."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    bad_components, bad_nets = _bad_design()
    bad_issues = validate_circuit(bad_components, bad_nets)
    bad_categories = {issue.category for issue in bad_issues}

    good_components, good_nets = _good_design()
    good_issues = validate_circuit(good_components, good_nets)

    summary = build_validation_summary(bad_issues)

    expected = {
        "Short Circuit",
        "Voltage Mismatch",
        "Unpowered IC",
        "Pin Conflict",
        "Overcurrent Risk",
    }
    ok = (
        bad_categories == expected
        and not is_design_valid(bad_issues)
        and good_issues == []
        and is_design_valid(good_issues)
        and len(summary.critical) >= 3  # short + 2x unpowered + pin conflict
        and len(summary.warning) == 2  # voltage mismatch + overcurrent
        and len(summary.info) == 0
    )

    if args.json:
        print(
            json.dumps(
                {
                    "bad_design_issues": [issue.to_dict() for issue in bad_issues],
                    "good_design_issues": [issue.to_dict() for issue in good_issues],
                    "summary": summary.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("circuit_validation selfcheck:")
        print("  bad design issues (%d):" % len(bad_issues))
        for issue in bad_issues:
            print("    [%s] %s" % (issue.severity, issue.category))
        print("  bad design valid: %s" % is_design_valid(bad_issues))
        print("  good design issues: %d" % len(good_issues))
        print("  good design valid: %s" % is_design_valid(good_issues))

    if not ok:
        print(
            "SELFCHECK FAILED: expected all 5 rule categories on the bad design "
            "and none on the good design; got %s" % sorted(bad_categories),
            file=sys.stderr,
        )
        return 1
    print("circuit_validation selfcheck OK: all 5 rules tripped, clean design passes")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
