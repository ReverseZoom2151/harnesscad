"""Tests for the deterministic Hardware IR derivations.

The module under test was ported from Forma-OSS
``blueprint_core/agents/orchestrator.py`` (MPL-2.0). The Forma-OSS test suite
contains NO coverage of extract_power_rails / extract_buses /
estimate_current_draw / the BOM cost step, so nothing was ported here; every
case below was derived by reading the harness implementation.

Property-based testing note: ``hypothesis`` is not installed in this repo and
must not be added, so the invariant tests enumerate small domains exhaustively
with ``itertools`` and use ``random.Random(20260719)`` for seeded shuffles.
FIXED SEED = 20260719.
"""

import itertools
import random
import unittest

from harnesscad.domain.electronics.derive import (
    bom_rollup,
    estimate_current_draw,
    extract_buses,
    extract_power_rails,
    main,
)
from harnesscad.domain.electronics.hardware_ir import (
    ComponentInstance,
    ConnectionNet,
    PinDefinition,
    PinReference,
)

FIXED_SEED = 20260719


def pin(pin_id, pin_type, voltage=None):
    return PinDefinition(pin_id=pin_id, name=pin_id, pin_type=pin_type, voltage=voltage)


def comp(ref_des, category="Passives", part_number="P", quantity=1, unit_price=0.0,
         name=None, sourcing_url=None):
    return ComponentInstance(
        ref_des=ref_des,
        part_number=part_number,
        name=name if name is not None else ref_des,
        category=category,
        quantity=quantity,
        unit_price=unit_price,
        sourcing_url=sourcing_url,
    )


def net(net_id, refs=(), net_type="Power", voltage=None):
    return ConnectionNet(
        net_id=net_id,
        name=net_id,
        net_type=net_type,
        voltage=voltage,
        pins=[PinReference(ref_des=r, pin_id="P") for r in refs],
    )


# ----------------------------------------------------------------------
# extract_power_rails
# ----------------------------------------------------------------------


class PowerRailTests(unittest.TestCase):
    def test_power_net_with_a_voltage_becomes_a_rail(self):
        rails = extract_power_rails([], [net("NET_3V3", ["U1"], voltage=3.3)])
        self.assertEqual(len(rails), 1)
        self.assertEqual(rails[0].rail_id, "RAIL_3V3")
        self.assertEqual(rails[0].voltage, 3.3)

    def test_non_power_net_type_produces_no_rail(self):
        # KNOWN-BAD: a 3.3V net that is not declared a power net is skipped.
        for net_type in ("Digital", "I2C", "Ground", "", "SPI"):
            with self.subTest(net_type=net_type):
                self.assertEqual(
                    extract_power_rails(
                        [], [net("N", ["U1"], net_type=net_type, voltage=3.3)]
                    ),
                    [],
                )

    def test_net_type_matching_is_case_insensitive(self):
        self.assertEqual(
            len(extract_power_rails([], [net("N", ["U1"], net_type="POWER", voltage=5.0)])),
            1,
        )

    def test_power_net_without_a_voltage_produces_no_rail(self):
        # KNOWN-BAD: voltage is None.
        self.assertEqual(extract_power_rails([], [net("N", ["U1"], voltage=None)]), [])

    def test_zero_volt_power_net_is_silently_skipped(self):
        # KNOWN-BAD-INPUT and a sharp edge: the guard is `net.voltage` (truthy),
        # not `is not None`, so a 0.0V power net yields no rail at all rather
        # than a 0V rail. Documented as observed behaviour.
        self.assertEqual(extract_power_rails([], [net("N", ["U1"], voltage=0.0)]), [])

    def test_negative_voltage_still_produces_a_rail(self):
        # There is no range check: a -5V rail is accepted, capacity 1000 mA.
        rails = extract_power_rails([], [net("N", ["U1"], voltage=-5.0)])
        self.assertEqual(len(rails), 1)
        self.assertEqual(rails[0].rail_id, "RAIL_-5V0")
        self.assertEqual(rails[0].max_current_capacity_ma, 1000.0)

    def test_rail_id_replaces_the_decimal_point_with_a_v(self):
        cases = {3.3: "RAIL_3V3", 5.0: "RAIL_5V0", 12.0: "RAIL_12V0", 1.8: "RAIL_1V8"}
        for voltage, expected in cases.items():
            with self.subTest(voltage=voltage):
                rails = extract_power_rails([], [net("N", ["U1"], voltage=voltage)])
                self.assertEqual(rails[0].rail_id, expected)

    def test_capacity_is_500ma_at_exactly_3v3_and_1000ma_otherwise(self):
        self.assertEqual(
            extract_power_rails([], [net("N", ["U1"], voltage=3.3)])[0]
            .max_current_capacity_ma,
            500.0,
        )
        for voltage in (3.2, 3.4, 5.0, 12.0):
            with self.subTest(voltage=voltage):
                self.assertEqual(
                    extract_power_rails([], [net("N", ["U1"], voltage=voltage)])[0]
                    .max_current_capacity_ma,
                    1000.0,
                )

    def test_source_defaults_to_u1_when_nothing_matches(self):
        components = [comp("A1", "Sensor"), comp("B2", "Display")]
        rails = extract_power_rails(components, [net("N", ["A1", "B2"], voltage=3.3)])
        self.assertEqual(rails[0].source_component, "U1")

    def test_source_defaults_to_u1_for_a_power_net_with_no_pins(self):
        self.assertEqual(
            extract_power_rails([], [net("N", [], voltage=5.0)])[0].source_component,
            "U1",
        )

    def test_power_category_component_wins_and_short_circuits_the_scan(self):
        components = [comp("PS1", "Power"), comp("BAT1", "Passives")]
        rails = extract_power_rails(components, [net("N", ["PS1", "BAT1"], voltage=5.0)])
        self.assertEqual(rails[0].source_component, "PS1")

    def test_power_category_matching_is_case_insensitive(self):
        components = [comp("PS1", "POWER")]
        rails = extract_power_rails(components, [net("N", ["PS1"], voltage=5.0)])
        self.assertEqual(rails[0].source_component, "PS1")

    def test_bat1_ref_des_is_recognised_without_a_component_record(self):
        rails = extract_power_rails([], [net("N", ["BAT1"], voltage=5.0)])
        self.assertEqual(rails[0].source_component, "BAT1")

    def test_ref_des_containing_power_is_recognised(self):
        for ref_des in ("USB-Power", "MAIN-POWER-IN", "power_brick"):
            with self.subTest(ref_des=ref_des):
                rails = extract_power_rails([], [net("N", [ref_des], voltage=5.0)])
                self.assertEqual(rails[0].source_component, ref_des)

    def test_a_later_matching_pin_overwrites_an_earlier_one(self):
        # Only the category-"power" branch breaks out of the loop; the BAT1 /
        # name-based branches keep scanning, so the LAST match wins.
        rails = extract_power_rails([], [net("N", ["BAT1", "USB-Power"], voltage=5.0)])
        self.assertEqual(rails[0].source_component, "USB-Power")

    def test_power_category_component_later_in_the_net_still_wins(self):
        components = [comp("PS1", "Power")]
        rails = extract_power_rails(components, [net("N", ["BAT1", "PS1"], voltage=5.0)])
        self.assertEqual(rails[0].source_component, "PS1")

    def test_multiple_power_nets_yield_rails_in_net_order(self):
        nets = [
            net("A", ["BAT1"], voltage=5.0),
            net("B", ["U1"], net_type="I2C", voltage=3.3),
            net("C", ["U1"], voltage=3.3),
        ]
        rails = extract_power_rails([], nets)
        self.assertEqual([r.rail_id for r in rails], ["RAIL_5V0", "RAIL_3V3"])
        self.assertEqual([r.source_component for r in rails], ["BAT1", "U1"])

    def test_two_nets_at_the_same_voltage_produce_two_rails_with_one_id(self):
        # Rail ids are not deduplicated; both nets emit RAIL_3V3.
        rails = extract_power_rails(
            [], [net("A", ["U1"], voltage=3.3), net("B", ["U2"], voltage=3.3)]
        )
        self.assertEqual([r.rail_id for r in rails], ["RAIL_3V3", "RAIL_3V3"])

    def test_empty_inputs_produce_no_rails(self):
        self.assertEqual(extract_power_rails([], []), [])


# ----------------------------------------------------------------------
# extract_buses
# ----------------------------------------------------------------------


class BusTests(unittest.TestCase):
    def test_i2c_nets_are_grouped_at_100khz(self):
        buses = extract_buses([net("SDA", net_type="I2C"), net("SCL", net_type="i2c")])
        self.assertEqual(len(buses), 1)
        self.assertEqual(buses[0].bus_id, "BUS_I2C_1")
        self.assertEqual(buses[0].bus_type, "I2C")
        self.assertEqual(buses[0].clock_frequency_hz, 100000.0)
        self.assertEqual(buses[0].nets, ["SDA", "SCL"])

    def test_spi_nets_are_grouped_at_1mhz(self):
        buses = extract_buses([net("SCK", net_type="SPI")])
        self.assertEqual(buses[0].bus_id, "BUS_SPI_1")
        self.assertEqual(buses[0].clock_frequency_hz, 1000000.0)

    def test_i2c_bus_always_precedes_spi_bus(self):
        buses = extract_buses([net("SCK", net_type="SPI"), net("SDA", net_type="I2C")])
        self.assertEqual([b.bus_id for b in buses], ["BUS_I2C_1", "BUS_SPI_1"])

    def test_net_ids_keep_their_input_order_inside_a_bus(self):
        nets = [net(n, net_type="I2C") for n in ("Z", "A", "M")]
        self.assertEqual(extract_buses(nets)[0].nets, ["Z", "A", "M"])

    def test_no_bus_nets_yields_an_empty_list(self):
        # KNOWN-BAD: unrelated net types (including UART and CAN, which the
        # function does NOT handle) produce no buses at all.
        self.assertEqual(
            extract_buses(
                [
                    net("P", net_type="Power", voltage=3.3),
                    net("U", net_type="UART"),
                    net("C", net_type="CAN"),
                    net("E", net_type=""),
                ]
            ),
            [],
        )

    def test_empty_net_list_yields_no_buses(self):
        self.assertEqual(extract_buses([]), [])

    def test_only_exact_type_names_match(self):
        # "I2C-SDA" is not the type "i2c", so it is not grouped.
        self.assertEqual(extract_buses([net("N", net_type="I2C-SDA")]), [])

    def test_bus_ids_are_never_incremented_for_a_second_group(self):
        # There is only ever one I2C bus id, no matter how many nets exist.
        nets = [net("N%d" % i, net_type="I2C") for i in range(5)]
        buses = extract_buses(nets)
        self.assertEqual(len(buses), 1)
        self.assertEqual(len(buses[0].nets), 5)


# ----------------------------------------------------------------------
# estimate_current_draw
# ----------------------------------------------------------------------


class CurrentDrawTests(unittest.TestCase):
    def test_per_category_draws(self):
        cases = [
            (comp("U1", "Microcontroller"), 80.0),
            (comp("DS1", "Display"), 25.0),
            (comp("M1", "Actuator", part_number="SG90-Servo"), 250.0),
            (comp("K1", "Actuator", part_number="Relay-5V-1Ch"), 70.0),
            (comp("SEN1", "Sensor"), 5.0),
            (comp("LED1", "Passives", part_number="LED-Red-Generic"), 15.0),
        ]
        for component, expected in cases:
            with self.subTest(ref_des=component.ref_des):
                self.assertEqual(estimate_current_draw([component]), expected)

    def test_unknown_category_and_part_number_draw_nothing(self):
        # KNOWN-BAD: an uncatalogued component silently contributes 0 mA, so
        # the estimate under-reports rather than refusing.
        self.assertEqual(
            estimate_current_draw(
                [comp("X1", "Mystery", part_number="NO-SUCH-PART"), comp("R1", "")]
            ),
            0.0,
        )

    def test_category_matching_is_case_insensitive(self):
        self.assertEqual(estimate_current_draw([comp("U1", "MICROCONTROLLER")]), 80.0)

    def test_led_part_number_is_ignored_when_the_category_already_matched(self):
        # The chain is elif-based: a Sensor named LED-Red-Generic draws 5 mA,
        # not 15 mA.
        self.assertEqual(
            estimate_current_draw([comp("X1", "Sensor", part_number="LED-Red-Generic")]),
            5.0,
        )

    def test_quantity_is_deliberately_ignored(self):
        # Divergence from the BOM roll-up: the current model is per component
        # LINE, not per unit, so two servos on one line still count 250 mA.
        self.assertEqual(
            estimate_current_draw(
                [comp("M1", "Actuator", part_number="SG90-Servo", quantity=2)]
            ),
            250.0,
        )

    def test_empty_component_list_draws_zero(self):
        self.assertEqual(estimate_current_draw([]), 0.0)

    def test_draws_sum_across_a_mixed_design(self):
        components = [
            comp("U1", "Microcontroller"),
            comp("DS1", "Display"),
            comp("M1", "Actuator", part_number="SG90-Servo"),
            comp("SEN1", "Sensor"),
            comp("LED1", "Passives", part_number="LED-Red-Generic"),
        ]
        self.assertEqual(estimate_current_draw(components), 375.0)


# ----------------------------------------------------------------------
# bom_rollup
# ----------------------------------------------------------------------


class BomRollupTests(unittest.TestCase):
    def test_single_line_item_fields(self):
        component = comp(
            "U1",
            "Microcontroller",
            part_number="ESP32",
            quantity=2,
            unit_price=6.5,
            name="ESP32 Board",
            sourcing_url="https://example.invalid/esp32",
        )
        bom = bom_rollup([component])
        self.assertEqual(
            bom["line_items"][0],
            {
                "ref_des": "U1",
                "part_number": "ESP32",
                "name": "ESP32 Board",
                "category": "Microcontroller",
                "quantity": 2,
                "unit_price": 6.5,
                "extended_price": 13.0,
                "sourcing_url": "https://example.invalid/esp32",
            },
        )
        self.assertEqual(bom["component_count"], 2)
        self.assertEqual(bom["estimated_electrical_cost"], 13.0)

    def test_zero_quantity_is_coerced_to_one(self):
        # KNOWN-BAD-INPUT: quantity 0 is treated as 1 rather than rejected.
        bom = bom_rollup([comp("R1", quantity=0, unit_price=2.0)])
        self.assertEqual(bom["line_items"][0]["quantity"], 1)
        self.assertEqual(bom["component_count"], 1)
        self.assertEqual(bom["estimated_electrical_cost"], 2.0)

    def test_negative_quantity_is_coerced_to_one(self):
        # KNOWN-BAD-INPUT: -3 units becomes 1 unit; no error is raised.
        bom = bom_rollup([comp("R1", quantity=-3, unit_price=2.0)])
        self.assertEqual(bom["line_items"][0]["quantity"], 1)
        self.assertEqual(bom["estimated_electrical_cost"], 2.0)

    def test_negative_unit_price_is_passed_through_unchecked(self):
        # KNOWN-BAD-INPUT: there is no price range check, so a negative price
        # produces a negative total. Recorded, not endorsed.
        bom = bom_rollup([comp("R1", quantity=2, unit_price=-1.5)])
        self.assertEqual(bom["estimated_electrical_cost"], -3.0)

    def test_extended_price_is_rounded_to_two_decimals(self):
        bom = bom_rollup([comp("R1", quantity=3, unit_price=0.333)])
        self.assertEqual(bom["line_items"][0]["extended_price"], 1.0)
        # The unit price itself is reported unrounded.
        self.assertEqual(bom["line_items"][0]["unit_price"], 0.333)

    def test_total_is_the_sum_of_rounded_extended_prices(self):
        bom = bom_rollup(
            [
                comp("A", quantity=1, unit_price=6.5),
                comp("B", quantity=1, unit_price=3.0),
                comp("C", quantity=2, unit_price=2.25),
                comp("D", quantity=1, unit_price=1.8),
                comp("E", quantity=1, unit_price=0.05),
                comp("F", quantity=1, unit_price=4.0),
            ]
        )
        self.assertEqual(bom["component_count"], 7)
        self.assertEqual(bom["estimated_electrical_cost"], 19.85)

    def test_empty_bom(self):
        self.assertEqual(
            bom_rollup([]),
            {"line_items": [], "component_count": 0, "estimated_electrical_cost": 0.0},
        )

    def test_missing_sourcing_url_stays_none(self):
        self.assertIsNone(bom_rollup([comp("R1")])["line_items"][0]["sourcing_url"])

    def test_duplicate_ref_des_produces_two_independent_line_items(self):
        # KNOWN-BAD-INPUT (duplicate refdes): bom_rollup has no dedup rule, so
        # the same designator is billed twice. Observed behaviour.
        bom = bom_rollup(
            [comp("R1", quantity=1, unit_price=1.0), comp("R1", quantity=1, unit_price=1.0)]
        )
        self.assertEqual(len(bom["line_items"]), 2)
        self.assertEqual(bom["component_count"], 2)


# ----------------------------------------------------------------------
# Invariants (stdlib substitute for property-based testing)
# ----------------------------------------------------------------------


class InvariantTests(unittest.TestCase):
    """hypothesis is unavailable in this repo, so invariants are enumerated
    exhaustively over small domains with itertools and shuffled with
    random.Random(20260719)."""

    POOL = [
        comp("U1", "Microcontroller", part_number="ESP32", quantity=1, unit_price=6.5),
        comp("DS1", "Display", part_number="SSD1306", quantity=1, unit_price=3.0),
        comp("M1", "Actuator", part_number="SG90-Servo", quantity=2, unit_price=2.25),
        comp("SEN1", "Sensor", part_number="DHT11", quantity=1, unit_price=1.8),
        comp("LED1", "Passives", part_number="LED-Red-Generic", quantity=4, unit_price=0.05),
    ]

    def test_current_draw_and_bom_totals_are_order_independent(self):
        # Exhaustive: all 120 permutations of the 5-component pool.
        base_draw = estimate_current_draw(self.POOL)
        base_bom = bom_rollup(self.POOL)
        for order in itertools.permutations(self.POOL):
            components = list(order)
            self.assertEqual(estimate_current_draw(components), base_draw)
            rolled = bom_rollup(components)
            self.assertEqual(rolled["component_count"], base_bom["component_count"])
            self.assertEqual(
                rolled["estimated_electrical_cost"],
                base_bom["estimated_electrical_cost"],
            )

    def test_current_draw_is_additive_over_every_subset(self):
        # Exhaustive over all 31 non-empty subsets of the pool.
        for size in range(1, len(self.POOL) + 1):
            for subset in itertools.combinations(self.POOL, size):
                expected = sum(estimate_current_draw([c]) for c in subset)
                self.assertEqual(estimate_current_draw(list(subset)), expected)

    def test_current_draw_is_monotonic_under_appending(self):
        running = []
        previous = 0.0
        for component in self.POOL:
            running.append(component)
            current = estimate_current_draw(running)
            self.assertGreaterEqual(current, previous)
            previous = current

    def test_component_count_equals_the_sum_of_line_quantities(self):
        for size in range(0, len(self.POOL) + 1):
            for subset in itertools.combinations(self.POOL, size):
                bom = bom_rollup(list(subset))
                self.assertEqual(
                    bom["component_count"],
                    sum(item["quantity"] for item in bom["line_items"]),
                )
                self.assertEqual(len(bom["line_items"]), len(subset))

    def test_bus_extraction_is_invariant_under_seeded_shuffles(self):
        # FIXED SEED 20260719.
        rng = random.Random(FIXED_SEED)
        nets = (
            [net("I%d" % i, net_type="I2C") for i in range(3)]
            + [net("S%d" % i, net_type="SPI") for i in range(2)]
            + [net("P", net_type="Power", voltage=3.3)]
        )
        for _ in range(25):
            shuffled = list(nets)
            rng.shuffle(shuffled)
            buses = extract_buses(shuffled)
            self.assertEqual([b.bus_id for b in buses], ["BUS_I2C_1", "BUS_SPI_1"])
            self.assertEqual(sorted(buses[0].nets), ["I0", "I1", "I2"])
            self.assertEqual(sorted(buses[1].nets), ["S0", "S1"])

    def test_derivations_do_not_mutate_their_inputs(self):
        components = list(self.POOL)
        nets = [
            net("NET_3V3", ["U1"], voltage=3.3),
            net("SDA", net_type="I2C"),
        ]
        before = ([c.to_dict() for c in components], [n.to_dict() for n in nets])
        extract_power_rails(components, nets)
        extract_buses(nets)
        estimate_current_draw(components)
        bom_rollup(components)
        after = ([c.to_dict() for c in components], [n.to_dict() for n in nets])
        self.assertEqual(after, before)

    def test_all_derivations_are_repeatable(self):
        nets = [net("NET_3V3", ["BAT1"], voltage=3.3), net("SDA", net_type="I2C")]
        first = (
            [r.to_dict() for r in extract_power_rails(self.POOL, nets)],
            [b.to_dict() for b in extract_buses(nets)],
            estimate_current_draw(self.POOL),
            bom_rollup(self.POOL),
        )
        for _ in range(5):
            again = (
                [r.to_dict() for r in extract_power_rails(self.POOL, nets)],
                [b.to_dict() for b in extract_buses(nets)],
                estimate_current_draw(self.POOL),
                bom_rollup(self.POOL),
            )
            self.assertEqual(again, first)

    def test_rail_count_equals_the_number_of_voltage_carrying_power_nets(self):
        # Exhaustive over every combination of net_type x voltage in a small
        # grid: 4 types x 4 voltages, taken two nets at a time.
        types = ("Power", "power", "Ground", "I2C")
        voltages = (None, 0.0, 3.3, 5.0)
        grid = [
            net("N", net_type=t, voltage=v)
            for t, v in itertools.product(types, voltages)
        ]
        for pair in itertools.combinations(grid, 2):
            expected = sum(
                1 for n in pair if n.net_type.lower() == "power" and n.voltage
            )
            self.assertEqual(len(extract_power_rails([], list(pair))), expected)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


class MainTests(unittest.TestCase):
    def test_selfcheck_passes(self):
        self.assertEqual(main(["--selfcheck"]), 0)

    def test_selfcheck_json_passes(self):
        self.assertEqual(main(["--selfcheck", "--json"]), 0)

    def test_no_arguments_prints_help_and_succeeds(self):
        self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()
