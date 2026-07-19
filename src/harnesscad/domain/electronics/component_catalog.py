"""Electronic component catalog: 14 stock parts with fully typed pinouts.

The rule engine in :mod:`harnesscad.domain.electronics.circuit_validation`
decides everything from pin metadata: rule 1 needs ``pin_type`` to tell a power
pin from a ground pin, rule 2 needs a numeric ``voltage`` on each pin, rule 3
needs to know which pins on a part are its supply pins, and rule 5 needs the
part's ``category`` and part number.  Until now nothing in HarnessCAD supplied
those facts, so the electrical rules could only lint a Hardware IR that a human
had hand-typed pin by pin -- and a hand-typed pinout is exactly the thing most
likely to contain the error the rules are meant to catch.  This module is the
missing ground truth: name a part, get its real pinout, and the validator's
verdict becomes a statement about the part rather than about the typist.

It is the electronics counterpart to
:mod:`harnesscad.domain.standards.part_catalog` (mechanical standard parts --
fasteners, bearings, extrusions), and follows the same shape: embedded tables,
a per-dataset :data:`PROVENANCE` block, bare-designation lookup, and a
``--selfcheck`` entry point.  The two catalogs do not overlap; ``part_catalog``
knows no pins and this one knows no threads.

Data provenance.  The part list, pin identifiers, pin types, nominal voltages,
categories, indicative prices and sourcing URLs come from the manufacturer
datasheets linked in :data:`_TEMPLATES` under ``sourcing_url``.  Pin numbers and
supply voltages are published measurements of physical parts, so they are cited
as data here; the tables below are independently structured around this
package's dataclasses.

Prices are indicative hobbyist single-unit USD as of the retrieval date and are
suitable for the BOM rollup in :mod:`harnesscad.domain.electronics.derive`
only as an order-of-magnitude estimate -- do not treat them as quotes.

Voltage convention.  ``voltage`` is the nominal *operating* level of the pin,
which for a supply pin that accepts a range is the low end of that range (the
BMP280 is listed at 3.3V though it accepts 1.8-3.6V).  This is deliberate:
rule 2 flags a spread greater than 0.5V between pins on one net, and using the
nominal level keeps a genuine 3.3V/5V logic collision visible while a part's
own tolerance band does not manufacture one.  A pin with no meaningful level --
a resistor lead -- carries ``None`` and rule 2 skips it.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.electronics.hardware_ir import (
    ComponentInstance,
    ComponentTemplate,
    PinDefinition,
)

__all__ = [
    "PROVENANCE",
    "PIN_TYPES",
    "component_template",
    "part_numbers",
    "categories",
    "parts_in_category",
    "parts_for_use_case",
    "use_cases",
    "pin",
    "power_pins",
    "ground_pins",
    "instantiate",
    "resolve",
    "main",
]

# ---------------------------------------------------------------------------
# Provenance.
# ---------------------------------------------------------------------------

PROVENANCE: Dict[str, Dict[str, str]] = {
    "component_templates": {
        "name": "component_templates",
        "version": "1",
        "source": (
            "Manufacturer datasheets (per-part sourcing_url), cross-checked "
            "against the Forma-OSS component seed "
            "supabase/migrations/20260618000200_seed_component_templates.sql"
        ),
        "license": (
            "Pin ids / types / voltages are datasheet facts cited as data; "
            "the Forma-OSS source file is MPL-2.0 and is not redistributed"
        ),
        "retrieved": "2026-07-19",
    },
}

# The closed pin-type vocabulary.  ``circuit_validation`` branches on the
# lower-cased form of these strings -- "power" and "ground" select the supply
# pins for rules 1, 3 and 5, and everything except power/ground/passive counts
# as a signal pin for the pin-reuse rule 4.  Adding a type here without
# teaching the rules about it would silently classify it as a signal pin, so
# the vocabulary is fixed and the selfcheck enforces it.
PIN_TYPES: Tuple[str, ...] = (
    "Power",
    "Ground",
    "Digital",
    "Analog",
    "I2C",
    "SPI",
    "UART",
    "PWM",
    "Passive",
)

# Categories.  ``circuit_validation.ACTIVE_CATEGORIES`` treats microcontroller,
# sensor, display and actuator as parts that must be powered and grounded;
# "Power" (sources) and "Passives" are exempt by design.
_CATEGORIES: Tuple[str, ...] = (
    "Microcontroller",
    "Sensor",
    "Actuator",
    "Display",
    "Power",
    "Passives",
)

# ---------------------------------------------------------------------------
# Part tables.  Pins are (pin_id, name, pin_type, voltage, description).
# ---------------------------------------------------------------------------

_PinRow = Tuple[str, str, str, Optional[float], str]

_TEMPLATES: Dict[str, Dict[str, object]] = {
    "ESP32-WROOM-32D": {
        "name": "ESP32 NodeMCU Development Board",
        "category": "Microcontroller",
        "description": (
            "WiFi + Bluetooth MCU module on a development carrier; dual-core "
            "Xtensa LX6, 3.3V logic with a 5V-tolerant VIN regulator input."
        ),
        "price": 4.50,
        "sourcing_url": "https://www.espressif.com/en/products/modules/esp32",
        "use_cases": [
            "automation", "bluetooth", "controller", "iot", "mcu",
            "robotics", "smart-home", "wifi",
        ],
        "pins": [
            ("3V3", "3.3V Power Out", "Power", 3.3, "3.3V regulated output"),
            ("GND", "Ground", "Ground", 0.0, "System ground reference"),
            ("EN", "Enable / Reset", "Passive", 3.3, "Reset pin, active low"),
            ("VP", "GPIO36 / ADC1_CH0", "Analog", 3.3, "ADC, input only"),
            ("VN", "GPIO39 / ADC1_CH3", "Analog", 3.3, "ADC, input only"),
            ("D34", "GPIO34 / ADC1_CH6", "Analog", 3.3, "Input only"),
            ("D35", "GPIO35 / ADC1_CH7", "Analog", 3.3, "Input only"),
            ("D32", "GPIO32 / ADC1_CH4", "Digital", 3.3, "General GPIO"),
            ("D33", "GPIO33 / ADC1_CH5", "Digital", 3.3, "General GPIO"),
            ("D25", "GPIO25 / DAC1", "Digital", 3.3, "DAC / general GPIO"),
            ("D26", "GPIO26 / DAC2", "Digital", 3.3, "DAC / general GPIO"),
            ("D27", "GPIO27", "Digital", 3.3, "General GPIO"),
            ("D14", "GPIO14 / HSPI CLK", "SPI", 3.3, "Secondary SPI clock"),
            ("D12", "GPIO12 / HSPI MISO", "SPI", 3.3, "Secondary SPI MISO"),
            ("D13", "GPIO13 / HSPI MOSI", "SPI", 3.3, "Secondary SPI MOSI"),
            ("D23", "GPIO23 / VSPI MOSI", "SPI", 3.3, "Primary SPI MOSI"),
            ("D22", "GPIO22 / I2C SCL", "I2C", 3.3, "Primary I2C clock"),
            ("D21", "GPIO21 / I2C SDA", "I2C", 3.3, "Primary I2C data"),
            ("TXD", "GPIO1 / UART0 TX", "UART", 3.3, "Serial transmit"),
            ("RXD", "GPIO3 / UART0 RX", "UART", 3.3, "Serial receive"),
            ("D19", "GPIO19 / VSPI MISO", "SPI", 3.3, "Primary SPI MISO"),
            ("D18", "GPIO18 / VSPI CLK", "SPI", 3.3, "Primary SPI clock"),
            ("D5", "GPIO5 / VSPI SS", "SPI", 3.3, "Primary SPI chip select"),
            ("VIN", "External Power In", "Power", 5.0, "5V unregulated input"),
        ],
    },
    "Arduino-Nano-V3": {
        "name": "Arduino Nano v3.0",
        "category": "Microcontroller",
        "description": (
            "Compact ATmega328P board with 5V logic; breadboard-friendly, no "
            "wireless."
        ),
        "price": 3.20,
        "sourcing_url": "https://store.arduino.cc/products/arduino-nano",
        "use_cases": [
            "basic-electronics", "learning", "mcu", "prototyping",
            "robotics", "wearable",
        ],
        "pins": [
            ("5V", "5V Power Out", "Power", 5.0, "5V regulated output"),
            ("3V3", "3.3V Power Out", "Power", 3.3, "3.3V regulated output"),
            ("GND", "Ground", "Ground", 0.0, "System ground"),
            ("VIN", "Voltage Input", "Power", 12.0, "7-12V input, regulated to 5V"),
            ("A0", "Analog 0", "Analog", 5.0, "Analog input 0"),
            ("A1", "Analog 1", "Analog", 5.0, "Analog input 1"),
            ("A2", "Analog 2", "Analog", 5.0, "Analog input 2"),
            ("A3", "Analog 3", "Analog", 5.0, "Analog input 3"),
            ("A4", "Analog 4 / I2C SDA", "I2C", 5.0, "I2C data / analog input 4"),
            ("A5", "Analog 5 / I2C SCL", "I2C", 5.0, "I2C clock / analog input 5"),
            ("D2", "Digital 2 / INT0", "Digital", 5.0, "GPIO / interrupt 0"),
            ("D3", "Digital 3 / PWM", "PWM", 5.0, "GPIO / PWM / interrupt 1"),
            ("D4", "Digital 4", "Digital", 5.0, "GPIO"),
            ("D5", "Digital 5 / PWM", "PWM", 5.0, "GPIO / PWM"),
            ("D6", "Digital 6 / PWM", "PWM", 5.0, "GPIO / PWM"),
            ("D7", "Digital 7", "Digital", 5.0, "GPIO"),
            ("D8", "Digital 8", "Digital", 5.0, "GPIO"),
            ("D9", "Digital 9 / PWM", "PWM", 5.0, "GPIO / PWM"),
            ("D10", "Digital 10 / SPI SS", "SPI", 5.0, "SPI slave select / PWM"),
            ("D11", "Digital 11 / SPI MOSI", "SPI", 5.0, "SPI MOSI / PWM"),
            ("D12", "Digital 12 / SPI MISO", "SPI", 5.0, "SPI MISO"),
            ("D13", "Digital 13 / SCK / LED", "Digital", 5.0, "SPI clock / onboard LED"),
        ],
    },
    "DHT22": {
        "name": "DHT22 Temperature and Humidity Sensor",
        "category": "Sensor",
        "description": (
            "Digital relative-humidity and temperature sensor on a single-wire "
            "bus; accepts 3.3-5.0V supply."
        ),
        "price": 2.80,
        "sourcing_url": (
            "https://www.sparkfun.com/datasheets/Sensors/Temperature/DHT22.pdf"
        ),
        "use_cases": [
            "environmental-monitor", "gardening", "humidity", "smart-home",
            "temperature", "weather-station",
        ],
        "pins": [
            ("VCC", "VCC Power", "Power", 3.3, "3.3V to 5.0V supply"),
            ("DATA", "Signal Out", "Digital", 3.3, "Single-wire data, needs pull-up"),
            ("NC", "No Connection", "Passive", None, "Do not connect"),
            ("GND", "Ground", "Ground", 0.0, "Power ground reference"),
        ],
    },
    "HC-SR04": {
        "name": "HC-SR04 Ultrasonic Distance Sensor",
        "category": "Sensor",
        "description": (
            "Ultrasonic rangefinder, 2cm to 400cm. 5V part with 5V logic on "
            "both TRIG and ECHO."
        ),
        "price": 1.50,
        "sourcing_url": (
            "https://cdn.sparkfun.com/datasheets/Sensors/Proximity/HCSR04.pdf"
        ),
        "use_cases": [
            "distance-sensing", "fluid-level", "obstacle-avoidance",
            "robotics", "security",
        ],
        "pins": [
            ("VCC", "5V Power Supply", "Power", 5.0, "Requires 5.0V nominal"),
            ("TRIG", "Trigger Input", "Digital", 5.0, "10us pulse starts a measurement"),
            ("ECHO", "Echo Output", "Digital", 5.0, "Pulse width equals round-trip time"),
            ("GND", "Ground", "Ground", 0.0, "Ground"),
        ],
    },
    "BMP280": {
        "name": "BMP280 Barometric Pressure and Temperature Sensor",
        "category": "Sensor",
        "description": (
            "Digital barometer / altimeter with selectable I2C or SPI "
            "interface; 1.8-3.6V supply."
        ),
        "price": 1.80,
        "sourcing_url": (
            "https://www.bosch-sensortec.com/products/environmental-sensors/"
            "pressure-sensors/bmp280/"
        ),
        "use_cases": [
            "altimeter", "barometer", "drones", "smart-watch", "weather-station",
        ],
        "pins": [
            ("VCC", "Power VCC", "Power", 3.3, "1.8V to 3.6V supply"),
            ("GND", "Ground", "Ground", 0.0, "Ground"),
            ("SCL", "I2C SCL / SPI SCK", "I2C", 3.3, "Clock"),
            ("SDA", "I2C SDA / SPI MOSI", "I2C", 3.3, "Data in/out"),
            ("CSB", "Chip Select", "SPI", 3.3, "SPI CSB active low; pull high for I2C"),
            ("SDO", "SPI MISO / I2C Address Select", "Digital", 3.3,
             "Address LSB in I2C mode, MISO in SPI mode"),
        ],
    },
    "MPU6050": {
        "name": "MPU-6050 6-Axis Accelerometer and Gyroscope",
        "category": "Sensor",
        "description": (
            "IMU with 3-axis accelerometer, 3-axis gyroscope and a digital "
            "motion processor; breakout carries a 3.3V regulator."
        ),
        "price": 2.20,
        "sourcing_url": (
            "https://invensense.tdk.com/products/motion-tracking/6-axis/mpu-6050/"
        ),
        "use_cases": [
            "balancing-robot", "drone-stability", "gesture-control",
            "motion-tracking", "robotics", "vr-headset",
        ],
        "pins": [
            ("VCC", "VCC Power", "Power", 3.3, "Onboard regulator accepts 3.3V or 5V"),
            ("GND", "Ground", "Ground", 0.0, "Ground"),
            ("SCL", "I2C Serial Clock", "I2C", 3.3, "I2C clock line"),
            ("SDA", "I2C Serial Data", "I2C", 3.3, "I2C data line"),
            ("XDA", "Auxiliary I2C Data", "I2C", 3.3, "Bus for an external magnetometer"),
            ("XCL", "Auxiliary I2C Clock", "I2C", 3.3, "Bus for an external magnetometer"),
            ("AD0", "I2C Address Select", "Digital", 3.3,
             "Address LSB: low = 0x68, high = 0x69"),
            ("INT", "Interrupt Out", "Digital", 3.3, "Motion interrupt output"),
        ],
    },
    "SG90-Servo": {
        "name": "SG90 Micro Servo Motor",
        "category": "Actuator",
        "description": (
            "180 degree hobby micro servo. Stall current is far above what an "
            "MCU 3.3V rail can source, so it wants its own 5V supply."
        ),
        "price": 2.00,
        "sourcing_url": "http://www.ee.ic.ac.uk/pjs99/ece3/parts/SG90Servo.pdf",
        "use_cases": [
            "hobbies", "rc-car", "robotic-arm", "robotics", "smart-door-lock",
        ],
        "pins": [
            ("5V", "Power VCC (red)", "Power", 5.0, "5.0V nominal power input"),
            ("GND", "Ground (brown)", "Ground", 0.0, "Power ground reference"),
            ("PWM", "Control Signal (orange)", "PWM", 5.0,
             "50Hz PWM, 1ms to 2ms pulse width"),
        ],
    },
    "Relay-5V-1Ch": {
        "name": "5V 1-Channel Optocoupled Relay Module",
        "category": "Actuator",
        "description": (
            "Switches mains or high-current DC loads from logic level. The "
            "COM/NO/NC terminals are mains-rated and must never be netted to "
            "logic."
        ),
        "price": 1.20,
        "sourcing_url": (
            "https://components101.com/switches/"
            "5v-single-channel-relay-module-pinout-features-datasheet"
        ),
        "use_cases": [
            "ac-switching", "home-automation", "motor-control", "smart-plug",
            "valve-control",
        ],
        "pins": [
            ("VCC", "Module Power", "Power", 5.0, "5V relay coil supply"),
            ("GND", "Module Ground", "Ground", 0.0, "System ground"),
            ("IN", "Signal Input", "Digital", 5.0, "Optocoupled coil trigger"),
            ("COM", "Switch Common", "Passive", 250.0, "High-power common pole"),
            ("NO", "Switch Normally Open", "Passive", 250.0,
             "Tied to COM only while energised"),
            ("NC", "Switch Normally Closed", "Passive", 250.0, "Tied to COM at rest"),
        ],
    },
    "SSD1306-I2C": {
        "name": "0.96 inch OLED Display (I2C)",
        "category": "Display",
        "description": "128x64 monochrome OLED on an SSD1306 controller, I2C interface.",
        "price": 2.50,
        "sourcing_url": (
            "https://components101.com/displays/"
            "096-inch-oled-display-module-pinout-datasheet"
        ),
        "use_cases": [
            "clock", "dashboard", "smart-home", "smart-thermostat",
            "user-interface",
        ],
        "pins": [
            ("VCC", "Power VCC", "Power", 3.3, "Module accepts 3.3V or 5V"),
            ("GND", "Ground", "Ground", 0.0, "Ground reference"),
            ("SCL", "I2C Serial Clock", "I2C", 3.3, "I2C clock"),
            ("SDA", "I2C Serial Data", "I2C", 3.3, "I2C data"),
        ],
    },
    "Battery-LiPo-3.7V": {
        "name": "3.7V Lithium Polymer Battery (1200mAh)",
        "category": "Power",
        "description": (
            "Single-cell rechargeable LiPo pack. Nominal 3.7V, so it is not a "
            "drop-in for a regulated 3.3V rail."
        ),
        "price": 5.50,
        "sourcing_url": (
            "https://components101.com/batteries/"
            "37v-lipo-battery-specification-datasheet"
        ),
        "use_cases": ["drones", "iot-nodes", "off-grid", "portable-power", "wearables"],
        "pins": [
            ("POS", "Positive Lead (red)", "Power", 3.7, "Positive terminal"),
            ("NEG", "Negative Lead (black)", "Ground", 0.0, "Negative terminal"),
        ],
    },
    "USB-5V-Plug": {
        "name": "5V USB Wall Power Supply",
        "category": "Power",
        "description": "Regulated 5V rail from a USB wall adapter, broken out to leads.",
        "price": 1.50,
        "sourcing_url": "https://en.wikipedia.org/wiki/USB",
        "use_cases": [
            "relay-controller", "sensors", "smart-home-hub", "stationary-power",
        ],
        "pins": [
            ("5V", "5V Power Line", "Power", 5.0, "5.0V regulated rail"),
            ("GND", "Ground", "Ground", 0.0, "Ground"),
        ],
    },
    "LED-Red-Generic": {
        "name": "Standard Red LED (5mm)",
        "category": "Passives",
        "description": (
            "5mm red indicator LED, roughly 2.0V forward drop. Needs a series "
            "current-limiting resistor."
        ),
        "price": 0.10,
        "sourcing_url": (
            "https://components101.com/diodes/5mm-red-led-pinout-specifications"
        ),
        "use_cases": ["blinky", "debugging", "diagnostics", "status-indicator"],
        "pins": [
            ("ANODE", "Anode (+), long lead", "Passive", 2.0,
             "Positive terminal, 1.8-2.2V forward drop"),
            ("CATHODE", "Cathode (-), flat lead", "Ground", 0.0, "Ground reference"),
        ],
    },
    "Resistor-220R": {
        "name": "220 Ohm Carbon Film Resistor (1/4W)",
        "category": "Passives",
        "description": (
            "Series current limit for a standard LED driven from a 3.3V or 5V "
            "logic pin."
        ),
        "price": 0.05,
        "sourcing_url": "https://components101.com/resistors/resistor-color-code",
        "use_cases": ["basic-circuit", "current-limiting", "led-protection"],
        "pins": [
            ("1", "Lead 1", "Passive", None, "Bidirectional passive lead"),
            ("2", "Lead 2", "Passive", None, "Bidirectional passive lead"),
        ],
    },
    "Resistor-10k": {
        "name": "10k Ohm Metal Film Resistor (1/4W)",
        "category": "Passives",
        "description": "Standard pull-up / pull-down value for logic and reset lines.",
        "price": 0.05,
        "sourcing_url": "https://components101.com/resistors/resistor-color-code",
        "use_cases": ["button-debouncing", "pull-down", "pull-up", "reset-line"],
        "pins": [
            ("1", "Lead 1", "Passive", None, "Bidirectional passive lead"),
            ("2", "Lead 2", "Passive", None, "Bidirectional passive lead"),
        ],
    },
}

# Aliases: the short names a brief is likely to use for a part whose catalog
# key carries a package or interface suffix.
_ALIASES: Dict[str, str] = {
    "ARDUINO-NANO": "Arduino-Nano-V3",
    "BATTERY-LIPO": "Battery-LiPo-3.7V",
    "ESP32": "ESP32-WROOM-32D",
    "LED": "LED-Red-Generic",
    "LIPO": "Battery-LiPo-3.7V",
    "NANO": "Arduino-Nano-V3",
    "RELAY": "Relay-5V-1Ch",
    "SG90": "SG90-Servo",
    "SSD1306": "SSD1306-I2C",
    "USB-5V": "USB-5V-Plug",
}


def _canonical(reference: str) -> str:
    """Map a user-facing reference onto a catalog key, or raise ``KeyError``."""
    text = reference.strip()
    if text in _TEMPLATES:
        return text
    upper = text.upper()
    for key in _TEMPLATES:
        if key.upper() == upper:
            return key
    if upper in _ALIASES:
        return _ALIASES[upper]
    raise KeyError('component "%s" not in catalog' % reference)


# ---------------------------------------------------------------------------
# Lookup.
# ---------------------------------------------------------------------------


def component_template(reference: str) -> ComponentTemplate:
    """Return the :class:`ComponentTemplate` for a part number or alias.

    A fresh object is built on every call so that callers can mutate the
    result (renaming pins for a variant, say) without corrupting the table.
    """
    key = _canonical(reference)
    row = _TEMPLATES[key]
    pins = [
        PinDefinition(
            pin_id=pin_id,
            name=name,
            pin_type=pin_type,
            voltage=voltage,
            description=description,
        )
        for (pin_id, name, pin_type, voltage, description) in row["pins"]  # type: ignore[union-attr]
    ]
    return ComponentTemplate(
        part_number=key,
        name=str(row["name"]),
        category=str(row["category"]),
        description=str(row["description"]),
        price=float(row["price"]),  # type: ignore[arg-type]
        sourcing_url=str(row["sourcing_url"]),
        pins=pins,
        use_cases=list(row["use_cases"]),  # type: ignore[arg-type]
    )


def part_numbers() -> List[str]:
    """Every catalog part number, sorted."""
    return sorted(_TEMPLATES)


def categories() -> List[str]:
    """Every category present in the catalog, sorted."""
    return sorted({str(row["category"]) for row in _TEMPLATES.values()})


def parts_in_category(category: str) -> List[str]:
    """Part numbers in one category, matched case-insensitively."""
    wanted = category.strip().lower()
    return sorted(
        key
        for key, row in _TEMPLATES.items()
        if str(row["category"]).lower() == wanted
    )


def use_cases() -> List[str]:
    """Every use-case tag across the catalog, sorted."""
    tags = set()
    for row in _TEMPLATES.values():
        tags.update(row["use_cases"])  # type: ignore[arg-type]
    return sorted(tags)


def parts_for_use_case(tag: str) -> List[str]:
    """Part numbers tagged with a use case, matched case-insensitively."""
    wanted = tag.strip().lower()
    return sorted(
        key
        for key, row in _TEMPLATES.items()
        if any(t.lower() == wanted for t in row["use_cases"])  # type: ignore[union-attr]
    )


def pin(reference: str, pin_id: str) -> PinDefinition:
    """One pin of one part, by id (case-insensitive), or raise ``KeyError``."""
    template = component_template(reference)
    wanted = pin_id.strip().lower()
    for candidate in template.pins:
        if candidate.pin_id.lower() == wanted:
            return candidate
    raise KeyError('pin "%s" not on component "%s"' % (pin_id, template.part_number))


def power_pins(reference: str) -> List[PinDefinition]:
    """The part's supply pins -- what rule 3 checks for a net connection."""
    return [p for p in component_template(reference).pins if p.pin_type == "Power"]


def ground_pins(reference: str) -> List[PinDefinition]:
    """The part's ground pins -- what rule 3 checks for a net connection."""
    return [p for p in component_template(reference).pins if p.pin_type == "Ground"]


def instantiate(
    reference: str,
    ref_des: str,
    quantity: int = 1,
    rationale: str = "",
) -> ComponentInstance:
    """Turn a catalog template into a placed BOM line with a reference designator.

    This is the bridge into the validator: ``validate_circuit`` takes
    :class:`ComponentInstance` objects, and an instance built here carries the
    datasheet pinout rather than whatever the caller remembered.
    """
    template = component_template(reference)
    return ComponentInstance(
        ref_des=ref_des,
        part_number=template.part_number,
        name=template.name,
        category=template.category,
        quantity=quantity,
        unit_price=template.price,
        sourcing_url=template.sourcing_url,
        rationale=rationale,
        pins=list(template.pins),
    )


def resolve(reference: str) -> Tuple[str, ComponentTemplate]:
    """Answer "which catalog part is this reference?" -- returns (category, template).

    Mirrors :func:`harnesscad.domain.standards.part_catalog.resolve` so the two
    catalogs can be probed the same way.
    """
    template = component_template(reference)
    return template.category, template


# ---------------------------------------------------------------------------
# Self-check.
# ---------------------------------------------------------------------------


def _check_tables() -> None:
    """The table is well-formed and speaks the vocabulary the rules understand."""
    assert len(_TEMPLATES) == 14, len(_TEMPLATES)
    assert part_numbers() == sorted(part_numbers())
    assert categories() == sorted(_CATEGORIES)

    total_pins = 0
    for key, row in _TEMPLATES.items():
        template = component_template(key)
        assert template.part_number == key
        assert template.name and template.description
        assert template.category in _CATEGORIES, (key, template.category)
        assert template.price > 0.0, key
        assert str(template.sourcing_url).startswith("http"), key
        assert template.use_cases == sorted(template.use_cases), key
        assert template.pins, key

        seen = set()
        for p in template.pins:
            # The closed vocabulary: an unknown pin_type would be silently
            # treated as a signal pin by rule 4 and ignored by rules 1/3/5.
            assert p.pin_type in PIN_TYPES, (key, p.pin_id, p.pin_type)
            assert p.pin_id and p.name and p.description, (key, p.pin_id)
            assert p.pin_id not in seen, (key, p.pin_id)
            seen.add(p.pin_id)
            # Rule 2 reads pin.voltage; a ground pin at anything but 0V would
            # make every ground net look like a mismatch.
            if p.pin_type == "Ground":
                assert p.voltage == 0.0, (key, p.pin_id)
            if p.voltage is not None:
                assert p.voltage >= 0.0, (key, p.pin_id)
            # Only leads with no defined level may omit a voltage.
            if p.voltage is None:
                assert p.pin_type == "Passive", (key, p.pin_id)
        total_pins += len(template.pins)

    assert total_pins == 91, total_pins
    assert len(component_template("ESP32-WROOM-32D").pins) == 24
    assert len(component_template("Arduino-Nano-V3").pins) == 22

    # Every active part the rules police has both a power and a ground pin,
    # otherwise rule 3 could never fire for it.
    for category in ("Microcontroller", "Sensor", "Actuator", "Display"):
        for key in parts_in_category(category):
            assert power_pins(key), key
            assert ground_pins(key), key

    # Lookup surface.
    assert _canonical("esp32") == "ESP32-WROOM-32D"
    assert _canonical("SG90") == "SG90-Servo"
    assert component_template("nano").part_number == "Arduino-Nano-V3"
    assert pin("ESP32", "d21").pin_type == "I2C"
    assert pin("HC-SR04", "ECHO").voltage == 5.0
    assert pin("Resistor-10k", "1").voltage is None
    assert resolve("SSD1306")[0] == "Display"
    assert parts_in_category("microcontroller") == [
        "Arduino-Nano-V3", "ESP32-WROOM-32D",
    ]
    assert "MPU6050" in parts_for_use_case("robotics")
    assert "robotics" in use_cases()
    try:
        component_template("STM32-Nucleo")
    except KeyError:
        pass
    else:
        raise AssertionError("an uncatalogued part must surface as a coverage gap")

    # Provenance, in the shape part_catalog uses.
    assert len(PROVENANCE) == 1
    for meta in PROVENANCE.values():
        for field_name in ("name", "version", "source", "license", "retrieved"):
            assert meta[field_name], field_name


def _check_drives_validation() -> None:
    """The catalog really is what makes ``circuit_validation`` executable.

    Three designs wired only from catalog parts: a 3.3V/5V collision that the
    voltage-mismatch rule must catch, the same design with a level-appropriate
    part substituted (must be silent), and one with the display's supply left
    off the rail (the unpowered-IC rule must catch it).  Nothing here types a
    pin by hand -- if the tables were wrong, these verdicts would flip.
    """
    from harnesscad.domain.electronics.circuit_validation import (
        build_validation_summary,
        is_design_valid,
        validate_circuit,
    )
    from harnesscad.domain.electronics.hardware_ir import (
        ConnectionNet,
        PinReference,
    )

    def _ref(ref_des: str, pin_id: str) -> PinReference:
        return PinReference(ref_des=ref_des, pin_id=pin_id)

    # -- Case A: a real-world beginner mistake ----------------------------
    # An HC-SR04 is a 5V part; hanging it off the ESP32's 3.3V rail and wiring
    # its 5V ECHO straight into a 3.3V GPIO is the classic way to cook an
    # ESP32 input. Both facts come from the catalog, not from this function.
    mcu = instantiate("ESP32-WROOM-32D", "U1")
    sonar = instantiate("HC-SR04", "S1")
    assert pin("ESP32-WROOM-32D", "3V3").voltage == 3.3
    assert pin("HC-SR04", "VCC").voltage == 5.0

    mismatched = [
        ConnectionNet(
            net_id="NET_PWR", name="MCU Rail", net_type="Power", voltage=3.3,
            pins=[_ref("U1", "3V3"), _ref("S1", "VCC")],
        ),
        ConnectionNet(
            net_id="NET_GND", name="Ground", net_type="Ground",
            pins=[_ref("U1", "GND"), _ref("S1", "GND")],
        ),
        ConnectionNet(
            net_id="NET_ECHO", name="Echo", net_type="Digital",
            pins=[_ref("U1", "D32"), _ref("S1", "ECHO")],
        ),
    ]
    issues = validate_circuit([mcu, sonar], mismatched)
    categories_hit = sorted({issue.category for issue in issues})
    assert categories_hit == ["Voltage Mismatch"], categories_hit
    assert len(issues) == 2, len(issues)  # the power rail and the echo line
    assert all(issue.severity == "WARNING" for issue in issues)
    assert "5.0V" in issues[0].description and "3.3V" in issues[0].description
    # A mismatch is a WARNING, so the design is still "valid" -- the rule
    # engine's own definition. Reporting it as a pass would be the lie.
    assert is_design_valid(issues)
    summary = build_validation_summary(issues)
    assert len(summary.warning) == 2 and not summary.critical

    # -- Case B: substitute a 3.3V-native sensor --------------------------
    # Same topology, BMP280 instead of HC-SR04. Nothing else changes, and the
    # warnings disappear -- so it was the catalog's voltages doing the work.
    baro = instantiate("BMP280", "S1")
    matched = [
        ConnectionNet(
            net_id="NET_PWR", name="MCU Rail", net_type="Power", voltage=3.3,
            pins=[_ref("U1", "3V3"), _ref("S1", "VCC")],
        ),
        ConnectionNet(
            net_id="NET_GND", name="Ground", net_type="Ground",
            pins=[_ref("U1", "GND"), _ref("S1", "GND")],
        ),
        ConnectionNet(
            net_id="NET_SDA", name="I2C Data", net_type="I2C",
            pins=[_ref("U1", "D21"), _ref("S1", "SDA")],
        ),
        ConnectionNet(
            net_id="NET_SCL", name="I2C Clock", net_type="I2C",
            pins=[_ref("U1", "D22"), _ref("S1", "SCL")],
        ),
    ]
    clean = validate_circuit([mcu, baro], matched)
    assert clean == [], [i.category for i in clean]
    assert is_design_valid(clean)

    # -- Case C: an active part left off the rail -------------------------
    display = instantiate("SSD1306-I2C", "DS1")
    starved = [
        ConnectionNet(
            net_id="NET_GND", name="Ground", net_type="Ground",
            pins=[_ref("U1", "GND"), _ref("DS1", "GND")],
        ),
        ConnectionNet(
            net_id="NET_SDA", name="I2C Data", net_type="I2C",
            pins=[_ref("U1", "D21"), _ref("DS1", "SDA")],
        ),
    ]
    starved_issues = validate_circuit([mcu, display], starved)
    unpowered = [i for i in starved_issues if i.category == "Unpowered IC"]
    assert len(unpowered) == 2, [i.description for i in starved_issues]
    assert {i.severity for i in unpowered} == {"CRITICAL"}
    assert not is_design_valid(starved_issues)

    # -- Instantiation carries the BOM facts too --------------------------
    assert mcu.unit_price == 4.50 and mcu.category == "Microcontroller"
    assert sonar.sourcing_url and sonar.sourcing_url.startswith("http")
    # Rule 5 keys off the part number the catalog supplies.
    servo = instantiate("SG90", "M1")
    assert servo.part_number == "SG90-Servo"


def _selfcheck() -> None:
    _check_tables()
    _check_drives_validation()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="component_catalog",
        description="Electronic component catalog: stock parts with typed "
                    "pinouts for Hardware IR construction and circuit "
                    "validation.",
    )
    parser.add_argument(
        "--selfcheck", action="store_true",
        help="assert the tables and prove they drive circuit_validation; exit 0",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="list every catalog part number with its category",
    )
    parser.add_argument(
        "reference", nargs="?",
        help="optional part number or alias to resolve (e.g. ESP32, SG90, BMP280)",
    )
    args = parser.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        pins = sum(len(row["pins"]) for row in _TEMPLATES.values())  # type: ignore[arg-type]
        print("component_catalog selfcheck: OK "
              "(%d parts, %d typed pins, %d categories; "
              "voltage-mismatch and unpowered-IC rules verified)"
              % (len(_TEMPLATES), pins, len(categories())))
        return 0

    if args.list:
        for key in part_numbers():
            template = component_template(key)
            print("%-20s %-16s %2d pins  %s"
                  % (key, template.category, len(template.pins), template.name))
        return 0

    if args.reference:
        try:
            category, template = resolve(args.reference)
        except KeyError as exc:
            print(str(exc))
            return 1
        print("%s: %s (%s)" % (category, template.part_number, template.name))
        for p in template.pins:
            volts = "-" if p.voltage is None else ("%.1fV" % p.voltage)
            print("  %-8s %-10s %-7s %s" % (p.pin_id, p.pin_type, volts, p.name))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
