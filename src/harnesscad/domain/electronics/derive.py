"""Deterministic Hardware IR derivations.

The helpers operate over the stdlib dataclass IR in
``harnesscad.domain.electronics.hardware_ir``.

Gap filled: HarnessCAD previously had no electronics/netlist IR at all, so it
had no way to enrich a device-level brief with derived electrical facts. These
functions compute the summary layers of the Hardware IR *deterministically*
from the components and nets (no model calls, no clock, no randomness):

* ``extract_power_rails`` -- every power net with a voltage becomes a
  PowerRail; the source component preference is category "power", then a
  BAT/USB/power-named ref_des, falling back to U1; rail ids are
  ``RAIL_{voltage with '.' -> 'V'}``; capacity is 500 mA at 3.3V, 1000 mA
  otherwise.
* ``extract_buses`` -- groups I2C nets into BUS_I2C_1 at 100 kHz and SPI nets
  into BUS_SPI_1 at 1 MHz.
* ``estimate_current_draw`` -- fixed per-category draws: MCU 80 mA, display
  25 mA, SG90 servo 250 mA, other actuator 70 mA, sensor 5 mA, generic red
  LED 15 mA.
* ``bom_rollup`` -- the deterministic BOM cost step: per-line quantities,
  unit and extended prices (rounded to 2 dp), component count, and total
  estimated electrical cost.

Usage::

    from harnesscad.domain.electronics.derive import (
        extract_power_rails, extract_buses, estimate_current_draw, bom_rollup,
    )
    ir.power_rails = extract_power_rails(ir.components, ir.nets)
    ir.buses = extract_buses(ir.nets)
    ir.estimated_current_draw_ma = estimate_current_draw(ir.components)
    bom = bom_rollup(ir.components)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from harnesscad.domain.electronics.hardware_ir import (
    BusConnection,
    ComponentInstance,
    ConnectionNet,
    PinDefinition,
    PinReference,
    PowerRail,
)

__all__ = [
    "extract_power_rails",
    "extract_buses",
    "estimate_current_draw",
    "bom_rollup",
    "main",
]


def extract_power_rails(
    components: List[ComponentInstance], nets: List[ConnectionNet]
) -> List[PowerRail]:
    """Derive PowerRail entries from the power nets that carry a voltage.

    Source component preference: a component with category "power" wins;
    otherwise a BAT1, "USB-Power", or power-named ref_des; fallback "U1".
    """
    rails: List[PowerRail] = []
    component_lookup = {component.ref_des: component for component in components}
    for net in nets:
        if net.net_type.lower() == "power" and net.voltage:
            source = None
            for pin_ref in net.pins:
                component = component_lookup.get(pin_ref.ref_des)
                if component and component.category.lower() == "power":
                    source = pin_ref.ref_des
                    break
                if pin_ref.ref_des == "BAT1":
                    source = "BAT1"
                elif pin_ref.ref_des == "USB-Power" or "power" in pin_ref.ref_des.lower():
                    source = pin_ref.ref_des
            if not source:
                source = "U1"

            rails.append(
                PowerRail(
                    rail_id=f"RAIL_{str(net.voltage).replace('.', 'V')}",
                    voltage=net.voltage,
                    max_current_capacity_ma=500.0 if net.voltage == 3.3 else 1000.0,
                    source_component=source,
                )
            )
    return rails


def extract_buses(nets: List[ConnectionNet]) -> List[BusConnection]:
    """Group I2C nets into BUS_I2C_1 (100 kHz) and SPI nets into BUS_SPI_1 (1 MHz)."""
    buses: List[BusConnection] = []
    i2c_nets = [net.net_id for net in nets if net.net_type.lower() == "i2c"]
    if i2c_nets:
        buses.append(
            BusConnection(
                bus_id="BUS_I2C_1",
                bus_type="I2C",
                clock_frequency_hz=100000.0,
                nets=i2c_nets,
            )
        )
    spi_nets = [net.net_id for net in nets if net.net_type.lower() == "spi"]
    if spi_nets:
        buses.append(
            BusConnection(
                bus_id="BUS_SPI_1",
                bus_type="SPI",
                clock_frequency_hz=1000000.0,
                nets=spi_nets,
            )
        )
    return buses


def estimate_current_draw(components: List[ComponentInstance]) -> float:
    """Sum fixed per-category peak current draws in milliamps."""
    draw = 0.0
    for comp in components:
        cat = comp.category.lower()
        if cat == "microcontroller":
            draw += 80.0
        elif cat == "display":
            draw += 25.0
        elif cat == "actuator":
            if comp.part_number == "SG90-Servo":
                draw += 250.0
            else:
                draw += 70.0  # relay coil
        elif cat == "sensor":
            draw += 5.0
        elif comp.part_number == "LED-Red-Generic":
            draw += 15.0
    return draw


def bom_rollup(components: List[ComponentInstance]) -> Dict[str, Any]:
    """The deterministic BOM cost step: line items plus totals.

    Returns a dict with ``line_items`` (ref_des, part_number, name, category,
    quantity, unit_price, extended_price rounded to 2 dp, sourcing_url),
    ``component_count`` (sum of quantities), and
    ``estimated_electrical_cost`` (sum of extended prices, 2 dp).
    """
    line_items: List[Dict[str, Any]] = []
    component_count = 0
    total_cost = 0.0
    for comp in components:
        quantity = comp.quantity if comp.quantity and comp.quantity > 0 else 1
        extended_price = round(comp.unit_price * quantity, 2)
        line_items.append(
            {
                "ref_des": comp.ref_des,
                "part_number": comp.part_number,
                "name": comp.name,
                "category": comp.category,
                "quantity": quantity,
                "unit_price": comp.unit_price,
                "extended_price": extended_price,
                "sourcing_url": comp.sourcing_url,
            }
        )
        component_count += quantity
        total_cost += extended_price
    return {
        "line_items": line_items,
        "component_count": component_count,
        "estimated_electrical_cost": round(total_cost, 2),
    }


# ==========================================
# Selfcheck fixtures
# ==========================================


def _pin(pin_id: str, pin_type: str, voltage: Optional[float] = None) -> PinDefinition:
    return PinDefinition(pin_id=pin_id, name=pin_id, pin_type=pin_type, voltage=voltage)


def _synthetic_design() -> tuple:
    components = [
        ComponentInstance(
            ref_des="U1",
            part_number="ESP32-WROOM-32D",
            name="ESP32 Development Board",
            category="Microcontroller",
            quantity=1,
            unit_price=6.5,
            pins=[_pin("3V3", "Power", 3.3), _pin("GND", "Ground")],
        ),
        ComponentInstance(
            ref_des="DS1",
            part_number="SSD1306-OLED",
            name="0.96in OLED Display",
            category="Display",
            quantity=1,
            unit_price=3.0,
        ),
        ComponentInstance(
            ref_des="M1",
            part_number="SG90-Servo",
            name="Micro Servo",
            category="Actuator",
            quantity=2,
            unit_price=2.25,
        ),
        ComponentInstance(
            ref_des="SEN1",
            part_number="DHT11",
            name="Temperature Sensor",
            category="Sensor",
            quantity=1,
            unit_price=1.8,
        ),
        ComponentInstance(
            ref_des="LED1",
            part_number="LED-Red-Generic",
            name="Red LED",
            category="Passives",
            quantity=1,
            unit_price=0.05,
        ),
        ComponentInstance(
            ref_des="BAT1",
            part_number="LiPo-1000",
            name="3.7V LiPo Battery",
            category="Power",
            quantity=1,
            unit_price=4.0,
        ),
    ]
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
            net_id="NET_5V",
            name="5V Power Rail",
            net_type="Power",
            voltage=5.0,
            pins=[
                PinReference(ref_des="BAT1", pin_id="VOUT"),
                PinReference(ref_des="M1", pin_id="VCC"),
            ],
        ),
        ConnectionNet(net_id="NET_I2C_SDA", name="I2C Data", net_type="I2C"),
        ConnectionNet(net_id="NET_I2C_SCL", name="I2C Clock", net_type="I2C"),
        ConnectionNet(net_id="NET_SPI_SCK", name="SPI Clock", net_type="SPI"),
    ]
    return components, nets


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` runs all four derivations on a
    synthetic design and asserts rail ids/capacities, bus grouping, the
    per-category current model, and the BOM totals."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.electronics.derive",
        description="Deterministic Hardware IR derivations "
        ".",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="derive rails/buses/current/BOM from a synthetic design.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the derived data as JSON."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    components, nets = _synthetic_design()

    rails = extract_power_rails(components, nets)
    buses = extract_buses(nets)
    draw = estimate_current_draw(components)
    bom = bom_rollup(components)

    # Rails: 3.3V net has no power-category component on it, no BAT/USB/power
    # ref, so it falls back to U1 at 500 mA; the 5V net finds BAT1 (category
    # Power) at 1000 mA.
    rails_ok = (
        len(rails) == 2
        and rails[0].rail_id == "RAIL_3V3"
        and rails[0].source_component == "U1"
        and rails[0].max_current_capacity_ma == 500.0
        and rails[1].rail_id == "RAIL_5V0"
        and rails[1].source_component == "BAT1"
        and rails[1].max_current_capacity_ma == 1000.0
    )

    buses_ok = (
        len(buses) == 2
        and buses[0].bus_id == "BUS_I2C_1"
        and buses[0].clock_frequency_hz == 100000.0
        and buses[0].nets == ["NET_I2C_SDA", "NET_I2C_SCL"]
        and buses[1].bus_id == "BUS_SPI_1"
        and buses[1].clock_frequency_hz == 1000000.0
        and buses[1].nets == ["NET_SPI_SCK"]
    )

    # 80 (mcu) + 25 (display) + 250 (servo, once per component line) + 5
    # (sensor) + 15 (LED part number).
    draw_ok = draw == 375.0

    # 6.5 + 3.0 + 2*2.25 + 1.8 + 0.05 + 4.0 = 19.85; count 1+1+2+1+1+1 = 7.
    bom_ok = (
        bom["component_count"] == 7
        and bom["estimated_electrical_cost"] == 19.85
        and bom["line_items"][2]["extended_price"] == 4.5
    )

    if args.json:
        print(
            json.dumps(
                {
                    "power_rails": [rail.to_dict() for rail in rails],
                    "buses": [bus.to_dict() for bus in buses],
                    "estimated_current_draw_ma": draw,
                    "bom": bom,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("derive selfcheck:")
        for rail in rails:
            print(
                "  rail %s: %.1fV, %.0f mA from %s"
                % (
                    rail.rail_id,
                    rail.voltage,
                    rail.max_current_capacity_ma,
                    rail.source_component,
                )
            )
        for bus in buses:
            print("  bus %s: %s nets=%s" % (bus.bus_id, bus.bus_type, bus.nets))
        print("  estimated draw: %.1f mA" % draw)
        print(
            "  BOM: %d parts, $%.2f"
            % (bom["component_count"], bom["estimated_electrical_cost"])
        )

    if not (rails_ok and buses_ok and draw_ok and bom_ok):
        print(
            "SELFCHECK FAILED: rails_ok=%s buses_ok=%s draw_ok=%s bom_ok=%s"
            % (rails_ok, buses_ok, draw_ok, bom_ok),
            file=sys.stderr,
        )
        return 1
    print("derive selfcheck OK: rails, buses, current draw, and BOM all match")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
