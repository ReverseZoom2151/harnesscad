"""Tests for the rule-based netlist validator.

The module under test was ported from Forma-OSS ``blueprint_core/validation.py``
(MPL-2.0). Its upstream test suite only exercised one case for
``validate_circuit`` (a power-to-ground short); that case is re-expressed here in
``ShortCircuitRuleTests.test_power_and_ground_pin_on_one_net_is_critical``. Every
other test below was derived by reading the harness implementation, and asserts
what the HARNESS does rather than what upstream did.

Property-based testing note: ``hypothesis`` is not installed in this repo and
must not be added, so the invariant tests use exhaustive ``itertools``
permutations over deliberately small designs plus a ``random.Random(20260719)``
shuffle over a fixed component pool. FIXED SEED = 20260719.
"""

import itertools
import random
import unittest

from harnesscad.domain.electronics.circuit_validation import (
    ACTIVE_CATEGORIES,
    HIGH_DRAW_KEYWORDS,
    HIGH_DRAW_PART_NUMBERS,
    build_validation_summary,
    is_design_valid,
    main,
    validate_circuit,
)
from harnesscad.domain.electronics.hardware_ir import (
    ComponentInstance,
    ConnectionNet,
    PinDefinition,
    PinReference,
    ValidationIssue,
)

FIXED_SEED = 20260719


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------


def pin(pin_id, pin_type, voltage=None):
    return PinDefinition(pin_id=pin_id, name=pin_id, pin_type=pin_type, voltage=voltage)


def comp(ref_des, category="Passives", pins=(), part_number="P", name=None):
    return ComponentInstance(
        ref_des=ref_des,
        part_number=part_number,
        name=name if name is not None else ref_des,
        category=category,
        pins=list(pins),
    )


def net(net_id, pin_pairs, net_type="Digital", voltage=None, name=None):
    return ConnectionNet(
        net_id=net_id,
        name=name if name is not None else net_id,
        net_type=net_type,
        voltage=voltage,
        pins=[PinReference(ref_des=r, pin_id=p) for r, p in pin_pairs],
    )


def categories(issues):
    return sorted({issue.category for issue in issues})


def _tokens(text):
    """Punctuation-insensitive bag of words, used to normalise descriptions."""
    return tuple(sorted(text.replace(",", " ").replace(".", " ").split()))


def fingerprint(issues):
    """Order-insensitive identity of an issue list.

    Descriptions are compared as sorted token bags rather than raw strings:
    the Pin Conflict message embeds ``', '.join(net_ids)`` in NET INPUT ORDER,
    so re-ordering the caller's net list legitimately re-orders that fragment.
    That is a formatting detail of the input ordering, not a difference in what
    was detected; see PinConflictRuleTests for the exact-string coverage and
    ``test_pin_conflict_description_follows_net_input_order`` below for the
    behaviour itself.
    """
    return sorted(
        (i.severity, i.category, _tokens(i.description), _tokens(i.troubleshooting))
        for i in issues
    )


def wired_mcu():
    """A microcontroller whose power and ground pins are both connected."""
    mcu = comp(
        "U1",
        "Microcontroller",
        [pin("3V3", "Power", 3.3), pin("GND", "Ground"), pin("GPIO4", "Digital", 3.3)],
        part_number="ESP32-WROOM-32D",
        name="ESP32",
    )
    nets = [
        net("NET_3V3", [("U1", "3V3")], net_type="Power", voltage=3.3),
        net("NET_GND", [("U1", "GND")], net_type="Ground"),
    ]
    return mcu, nets


# ----------------------------------------------------------------------
# Rule 1: short circuit
# ----------------------------------------------------------------------


class ShortCircuitRuleTests(unittest.TestCase):
    def test_power_and_ground_pin_on_one_net_is_critical(self):
        # KNOWN-BAD. Re-expression of the single validate_circuit case in the
        # Forma-OSS suite (tests/test_validation_models.py): an MCU with its
        # 3V3 and GND pins tied to the same net.
        components = [
            comp(
                "U1",
                "Microcontroller",
                [pin("3V3", "Power", 3.3), pin("GND", "Ground", 0.0)],
            )
        ]
        nets = [
            net(
                "NET_SHORT",
                [("U1", "3V3"), ("U1", "GND")],
                net_type="Power",
                voltage=3.3,
                name="Accidental short",
            )
        ]
        issues = validate_circuit(components, nets)
        shorts = [i for i in issues if i.category == "Short Circuit"]
        self.assertEqual(len(shorts), 1)
        self.assertEqual(shorts[0].severity, "CRITICAL")
        self.assertFalse(is_design_valid(issues))
        # Both offending pins are named in the description.
        self.assertIn("U1.3V3", shorts[0].description)
        self.assertIn("U1.GND", shorts[0].description)
        self.assertIn("NET_SHORT", shorts[0].description)

    def test_passing_counterpart_power_and_ground_on_separate_nets(self):
        mcu, nets = wired_mcu()
        issues = validate_circuit([mcu], nets)
        self.assertEqual([i for i in issues if i.category == "Short Circuit"], [])

    def test_pin_type_matching_is_case_insensitive(self):
        components = [
            comp("U1", "Passives", [pin("A", "POWER", 3.3), pin("B", "ground")])
        ]
        issues = validate_circuit(components, net_list_short())
        self.assertIn("Short Circuit", categories(issues))

    def test_multiple_power_and_ground_pins_yield_one_issue_per_net(self):
        components = [
            comp(
                "U1",
                "Passives",
                [
                    pin("A", "Power", 3.3),
                    pin("B", "Power", 3.3),
                    pin("C", "Ground"),
                ],
            )
        ]
        nets = [net("N1", [("U1", "A"), ("U1", "B"), ("U1", "C")])]
        shorts = [
            i for i in validate_circuit(components, nets) if i.category == "Short Circuit"
        ]
        self.assertEqual(len(shorts), 1)
        self.assertIn("U1.A, U1.B", shorts[0].description)

    def test_pin_reference_to_unknown_component_is_ignored(self):
        # KNOWN-BAD-INPUT (dangling reference): the ground pin belongs to a
        # component that is not in the component list, so pin_lookup misses and
        # no short is reported.
        components = [comp("U1", "Passives", [pin("A", "Power", 3.3)])]
        nets = [net("N1", [("U1", "A"), ("GHOST", "GND")])]
        self.assertEqual(validate_circuit(components, nets), [])

    def test_pin_reference_to_unknown_pin_id_is_ignored(self):
        components = [
            comp("U1", "Passives", [pin("A", "Power", 3.3), pin("B", "Ground")])
        ]
        nets = [net("N1", [("U1", "A"), ("U1", "NO_SUCH_PIN")])]
        self.assertEqual(validate_circuit(components, nets), [])


def net_list_short():
    return [net("N1", [("U1", "A"), ("U1", "B")])]


# ----------------------------------------------------------------------
# Rule 2: voltage mismatch (threshold: spread strictly greater than 0.5V)
# ----------------------------------------------------------------------


class VoltageMismatchRuleTests(unittest.TestCase):
    def _spread_issues(self, v_a, v_b):
        components = [
            comp("U1", "Passives", [pin("A", "Digital", v_a)]),
            comp("U2", "Passives", [pin("B", "Digital", v_b)]),
        ]
        nets = [net("N1", [("U1", "A"), ("U2", "B")])]
        return [
            i
            for i in validate_circuit(components, nets)
            if i.category == "Voltage Mismatch"
        ]

    def test_5v_and_3v3_on_one_net_warns(self):
        # KNOWN-BAD: classic 5V/3.3V logic collision.
        issues = self._spread_issues(5.0, 3.3)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "WARNING")
        self.assertIn("3.3V", issues[0].description)
        self.assertIn("5.0V", issues[0].description)
        # Design stays "valid" because this rule is only a WARNING.
        self.assertTrue(is_design_valid(issues))

    def test_spread_of_exactly_half_a_volt_is_not_flagged(self):
        # Boundary: the guard is `max - min > 0.5`, so exactly 0.5 passes.
        self.assertEqual(self._spread_issues(3.3, 3.8), [])

    def test_spread_just_over_half_a_volt_is_flagged(self):
        self.assertEqual(len(self._spread_issues(3.0, 3.6)), 1)

    def test_identical_voltages_are_not_flagged(self):
        self.assertEqual(self._spread_issues(3.3, 3.3), [])

    def test_pins_without_a_voltage_are_ignored(self):
        components = [
            comp("U1", "Passives", [pin("A", "Digital", 5.0)]),
            comp("U2", "Passives", [pin("B", "Digital", None)]),
        ]
        nets = [net("N1", [("U1", "A"), ("U2", "B")])]
        self.assertEqual(
            [
                i
                for i in validate_circuit(components, nets)
                if i.category == "Voltage Mismatch"
            ],
            [],
        )

    def test_troubleshooting_names_both_rail_voltages(self):
        issues = self._spread_issues(12.0, 3.3)
        self.assertIn("3.3V", issues[0].troubleshooting)
        self.assertIn("12.0V", issues[0].troubleshooting)


# ----------------------------------------------------------------------
# Rule 3: unpowered / ungrounded active IC
# ----------------------------------------------------------------------


class UnpoweredIcRuleTests(unittest.TestCase):
    def test_active_ic_with_no_nets_at_all_gets_two_critical_issues(self):
        # KNOWN-BAD: a completely floating sensor -> unpowered AND ungrounded.
        components = [
            comp("SEN1", "Sensor", [pin("VCC", "Power", 3.3), pin("GND", "Ground")])
        ]
        issues = validate_circuit(components, [])
        unpowered = [i for i in issues if i.category == "Unpowered IC"]
        self.assertEqual(len(unpowered), 2)
        self.assertTrue(all(i.severity == "CRITICAL" for i in unpowered))
        self.assertFalse(is_design_valid(issues))

    def test_power_connected_but_ground_floating_reports_only_the_ground(self):
        components = [
            comp("SEN1", "Sensor", [pin("VCC", "Power", 3.3), pin("GND", "Ground")])
        ]
        nets = [net("NET_3V3", [("SEN1", "VCC")], net_type="Power", voltage=3.3)]
        unpowered = [
            i
            for i in validate_circuit(components, nets)
            if i.category == "Unpowered IC"
        ]
        self.assertEqual(len(unpowered), 1)
        self.assertIn("no ground reference", unpowered[0].description)

    def test_fully_wired_active_ic_is_clean(self):
        mcu, nets = wired_mcu()
        self.assertEqual(
            [i for i in validate_circuit([mcu], nets) if i.category == "Unpowered IC"],
            [],
        )

    def test_all_four_active_categories_are_checked(self):
        # Every category in ACTIVE_CATEGORIES must trip the rule; the tuple is
        # read from the module so this test tracks the source list.
        for category in ACTIVE_CATEGORIES:
            with self.subTest(category=category):
                components = [comp("X1", category, [pin("VCC", "Power", 3.3)])]
                issues = validate_circuit(components, [])
                self.assertEqual(categories(issues), ["Unpowered IC"])

    def test_inactive_category_is_never_flagged(self):
        # Passing counterpart: a floating passive/connector is not an error.
        for category in ("Passives", "Connector", "Power", "", "MECHANICAL"):
            with self.subTest(category=category):
                components = [
                    comp("X1", category, [pin("VCC", "Power", 3.3), pin("G", "Ground")])
                ]
                self.assertEqual(validate_circuit(components, []), [])

    def test_active_ic_without_power_or_ground_pins_is_not_flagged(self):
        # The rule only fires when the component actually declares such a pin.
        components = [comp("SEN1", "Sensor", [pin("DATA", "Digital", 3.3)])]
        self.assertEqual(validate_circuit(components, []), [])

    def test_component_with_no_pins_at_all_is_not_flagged(self):
        self.assertEqual(validate_circuit([comp("SEN1", "Sensor", [])], []), [])

    def test_category_matching_is_case_insensitive(self):
        components = [comp("U1", "MICROCONTROLLER", [pin("VCC", "Power", 3.3)])]
        self.assertEqual(categories(validate_circuit(components, [])), ["Unpowered IC"])

    def test_connection_to_any_net_counts_as_powered(self):
        # Divergence worth noting: the rule only checks that the pin appears in
        # SOME net; it does not check that the net is actually a power net.
        components = [comp("SEN1", "Sensor", [pin("VCC", "Power", 3.3)])]
        nets = [net("NET_SIG", [("SEN1", "VCC")], net_type="Digital")]
        self.assertEqual(validate_circuit(components, nets), [])


# ----------------------------------------------------------------------
# Rule 4: pin reuse conflict
# ----------------------------------------------------------------------


class PinConflictRuleTests(unittest.TestCase):
    def test_signal_pin_in_two_nets_is_critical(self):
        # KNOWN-BAD: one GPIO driving two independent signal nets.
        components = [comp("U1", "Passives", [pin("GPIO4", "Digital", 3.3)])]
        nets = [net("NET_A", [("U1", "GPIO4")]), net("NET_B", [("U1", "GPIO4")])]
        issues = validate_circuit(components, nets)
        conflicts = [i for i in issues if i.category == "Pin Conflict"]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].severity, "CRITICAL")
        self.assertIn("NET_A, NET_B", conflicts[0].description)

    def test_signal_pin_in_one_net_is_clean(self):
        components = [comp("U1", "Passives", [pin("GPIO4", "Digital", 3.3)])]
        nets = [net("NET_A", [("U1", "GPIO4")])]
        self.assertEqual(validate_circuit(components, nets), [])

    def test_power_ground_and_passive_pins_may_be_shared(self):
        # Passing counterpart: bus-like pin types are exempted by design.
        for pin_type in ("Power", "Ground", "Passive", "PASSIVE"):
            with self.subTest(pin_type=pin_type):
                components = [comp("U1", "Passives", [pin("P", pin_type, 3.3)])]
                nets = [net("NET_A", [("U1", "P")]), net("NET_B", [("U1", "P")])]
                self.assertEqual(
                    [
                        i
                        for i in validate_circuit(components, nets)
                        if i.category == "Pin Conflict"
                    ],
                    [],
                )

    def test_pin_listed_twice_in_the_same_net_still_counts_twice(self):
        # KNOWN-BAD-INPUT (malformed net): a duplicated PinReference inside one
        # net appends the net_id twice, so the rule fires even though there is
        # only one net. Documented as current behaviour, not asserted as
        # correct design.
        components = [comp("U1", "Passives", [pin("GPIO4", "Digital", 3.3)])]
        nets = [net("NET_A", [("U1", "GPIO4"), ("U1", "GPIO4")])]
        conflicts = [
            i for i in validate_circuit(components, nets) if i.category == "Pin Conflict"
        ]
        self.assertEqual(len(conflicts), 1)
        self.assertIn("NET_A, NET_A", conflicts[0].description)

    def test_pin_conflict_description_follows_net_input_order(self):
        # Documented order-sensitivity: WHICH issues are raised does not depend
        # on net ordering, but the net_id list inside the Pin Conflict message
        # is joined in caller-supplied order.
        components = [comp("U1", "Passives", [pin("GPIO4", "Digital", 3.3)])]
        a = net("NET_A", [("U1", "GPIO4")])
        b = net("NET_B", [("U1", "GPIO4")])
        forward = validate_circuit(components, [a, b])[0].description
        reverse = validate_circuit(components, [b, a])[0].description
        self.assertIn("NET_A, NET_B", forward)
        self.assertIn("NET_B, NET_A", reverse)

    def test_conflict_on_pin_of_unknown_component_is_skipped(self):
        # pin_lookup miss -> `if pin and ...` is False -> no issue.
        self.assertEqual(
            validate_circuit(
                [], [net("NET_A", [("GHOST", "P")]), net("NET_B", [("GHOST", "P")])]
            ),
            [],
        )


# ----------------------------------------------------------------------
# Rule 5: overcurrent risk
# ----------------------------------------------------------------------


def overcurrent_design(
    actuator_part="SG90-Servo",
    actuator_name="Micro Servo",
    net_type="Power",
    net_voltage=3.3,
    include_mcu=True,
):
    components = []
    if include_mcu:
        components.append(
            comp(
                "U1",
                "Microcontroller",
                [pin("3V3", "Power", 3.3), pin("GND", "Ground")],
                part_number="ESP32-WROOM-32D",
                name="ESP32",
            )
        )
    components.append(
        comp(
            "M1",
            "Actuator",
            [pin("VCC", "Power", 3.3), pin("GND", "Ground")],
            part_number=actuator_part,
            name=actuator_name,
        )
    )
    pairs = [("M1", "VCC")]
    if include_mcu:
        pairs.insert(0, ("U1", "3V3"))
    nets = [
        net("NET_PWR", pairs, net_type=net_type, voltage=net_voltage),
        net(
            "NET_GND",
            ([("U1", "GND")] if include_mcu else []) + [("M1", "GND")],
            net_type="Ground",
        ),
    ]
    return components, nets


class OvercurrentRuleTests(unittest.TestCase):
    def _overcurrent(self, **kwargs):
        components, nets = overcurrent_design(**kwargs)
        return [
            i
            for i in validate_circuit(components, nets)
            if i.category == "Overcurrent Risk"
        ]

    def test_servo_sharing_the_mcu_3v3_rail_warns(self):
        # KNOWN-BAD: high-draw actuator on the MCU's 3.3V output.
        issues = self._overcurrent()
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "WARNING")
        self.assertIn("Micro Servo (M1)", issues[0].description)
        self.assertIn("U1", issues[0].description)

    def test_five_volt_rail_is_the_passing_counterpart(self):
        self.assertEqual(self._overcurrent(net_voltage=5.0), [])

    def test_net_with_no_voltage_is_skipped(self):
        self.assertEqual(self._overcurrent(net_voltage=None), [])

    def test_non_power_net_type_is_skipped(self):
        self.assertEqual(self._overcurrent(net_type="Digital"), [])

    def test_no_microcontroller_means_no_rule(self):
        self.assertEqual(self._overcurrent(include_mcu=False), [])

    def test_benign_actuator_on_the_same_rail_is_clean(self):
        # An actuator whose name/part number matches no high-draw keyword.
        self.assertEqual(
            self._overcurrent(actuator_part="PIEZO-BUZZ", actuator_name="Buzzer"), []
        )

    def test_every_high_draw_keyword_triggers_the_rule(self):
        for keyword in HIGH_DRAW_KEYWORDS:
            with self.subTest(keyword=keyword):
                issues = self._overcurrent(
                    actuator_part="GENERIC-1", actuator_name="Big %s unit" % keyword
                )
                self.assertEqual(len(issues), 1)

    def test_every_high_draw_part_number_triggers_the_rule(self):
        for part_number in HIGH_DRAW_PART_NUMBERS:
            with self.subTest(part_number=part_number):
                issues = self._overcurrent(
                    actuator_part=part_number, actuator_name="Anonymous Load"
                )
                self.assertEqual(len(issues), 1)

    def test_keyword_match_is_case_insensitive_and_matches_part_number_text(self):
        issues = self._overcurrent(actuator_part="RELAY-Board", actuator_name="Load")
        self.assertEqual(len(issues), 1)

    def test_a_microcontroller_is_never_classified_as_high_draw(self):
        # The classification loop skips microcontrollers entirely, so an MCU
        # literally named "Motor Controller" is not treated as the load.
        components = [
            comp(
                "U1",
                "Microcontroller",
                [pin("3V3", "Power", 3.3)],
                name="Motor Controller",
            )
        ]
        nets = [net("NET_PWR", [("U1", "3V3")], net_type="Power", voltage=3.3)]
        self.assertEqual(
            [
                i
                for i in validate_circuit(components, nets)
                if i.category == "Overcurrent Risk"
            ],
            [],
        )

    def test_actuator_alone_on_a_rail_without_the_mcu_pin_is_clean(self):
        components, nets = overcurrent_design()
        # Drop the MCU power pin from the shared rail.
        nets[0].pins = [PinReference(ref_des="M1", pin_id="VCC")]
        self.assertEqual(
            [
                i
                for i in validate_circuit(components, nets)
                if i.category == "Overcurrent Risk"
            ],
            [],
        )

    def test_only_power_type_pins_on_the_rail_are_considered(self):
        # The actuator's pin on the 3.3V net is a signal pin, not a power pin.
        components, nets = overcurrent_design()
        components[1].pins = [pin("VCC", "PWM", 3.3), pin("GND", "Ground")]
        self.assertEqual(
            [
                i
                for i in validate_circuit(components, nets)
                if i.category == "Overcurrent Risk"
            ],
            [],
        )


# ----------------------------------------------------------------------
# Degenerate and malformed inputs
# ----------------------------------------------------------------------


class DegenerateInputTests(unittest.TestCase):
    def test_empty_netlist_and_no_components(self):
        issues = validate_circuit([], [])
        self.assertEqual(issues, [])
        self.assertTrue(is_design_valid(issues))

    def test_components_but_no_nets_only_trips_the_active_ic_rule(self):
        components = [
            comp("U1", "Microcontroller", [pin("3V3", "Power", 3.3)]),
            comp("R1", "Passives", [pin("1", "Passive")]),
        ]
        self.assertEqual(categories(validate_circuit(components, [])), ["Unpowered IC"])

    def test_nets_but_no_components_produce_nothing(self):
        nets = [net("N1", [("U1", "A"), ("U1", "B")])]
        self.assertEqual(validate_circuit([], nets), [])

    def test_net_with_an_empty_pin_list(self):
        self.assertEqual(validate_circuit([], [net("N1", [])]), [])

    def test_duplicate_ref_des_last_component_wins_for_the_ic_rule(self):
        # KNOWN-BAD-INPUT (duplicate refdes). validate_circuit has no
        # duplicate-refdes rule; component_lookup is keyed by ref_des so the
        # SECOND U1 shadows the first for rules 3 and 5, while pin_lookup ends
        # up holding the union of both pinouts. Recorded as observed
        # behaviour, not endorsed: a real duplicate-refdes check would be a
        # separate rule.
        first = comp("U1", "Sensor", [pin("VCC", "Power", 3.3)])
        second = comp("U1", "Passives", [pin("OTHER", "Digital", 3.3)])
        issues = validate_circuit([first, second], [])
        # The shadowed Sensor is never examined, so no Unpowered IC fires.
        self.assertEqual(issues, [])
        # Reversing the order surfaces the sensor again.
        self.assertEqual(
            categories(validate_circuit([second, first], [])), ["Unpowered IC"]
        )

    def test_empty_string_pin_type_and_category_are_inert(self):
        components = [comp("U1", "", [pin("", "", None)])]
        self.assertEqual(validate_circuit(components, [net("N1", [("U1", "")])]), [])

    def test_the_five_rules_can_all_fire_on_one_design(self):
        # Uses the module's own bad-design shape: every category present.
        components, nets = _all_rules_design()
        self.assertEqual(
            categories(validate_circuit(components, nets)),
            [
                "Overcurrent Risk",
                "Pin Conflict",
                "Short Circuit",
                "Unpowered IC",
                "Voltage Mismatch",
            ],
        )


def _all_rules_design():
    mcu = comp(
        "U1",
        "Microcontroller",
        [pin("3V3", "Power", 3.3), pin("GND", "Ground"), pin("GPIO4", "Digital", 3.3)],
        part_number="ESP32-WROOM-32D",
        name="ESP32",
    )
    sensor = comp(
        "SEN1",
        "Sensor",
        [pin("VCC", "Power", 5.0), pin("GND", "Ground"), pin("DATA", "Digital", 5.0)],
        part_number="DHT11",
        name="Temp Sensor",
    )
    servo = comp(
        "M1",
        "Actuator",
        [pin("VCC", "Power", 3.3), pin("GND", "Ground"), pin("PWM", "PWM", 3.3)],
        part_number="SG90-Servo",
        name="Micro Servo",
    )
    nets = [
        net(
            "NET_3V3",
            [("U1", "3V3"), ("M1", "VCC"), ("U1", "GND")],
            net_type="Power",
            voltage=3.3,
        ),
        net("NET_SIG_A", [("U1", "GPIO4"), ("SEN1", "DATA")]),
        net("NET_SIG_B", [("U1", "GPIO4"), ("M1", "PWM")], net_type="PWM"),
        net("NET_GND", [("M1", "GND")], net_type="Ground"),
    ]
    return [mcu, sensor, servo], nets


# ----------------------------------------------------------------------
# Summary and validity helpers
# ----------------------------------------------------------------------


class SummaryTests(unittest.TestCase):
    def test_issues_are_bucketed_by_severity(self):
        issues = [
            ValidationIssue(severity="CRITICAL", category="Short Circuit"),
            ValidationIssue(severity="WARNING", category="Voltage Mismatch"),
            ValidationIssue(severity="INFO", category="Note"),
            ValidationIssue(severity="CRITICAL", category="Pin Conflict"),
        ]
        summary = build_validation_summary(issues)
        self.assertEqual(len(summary.critical), 2)
        self.assertEqual(len(summary.warning), 1)
        self.assertEqual(len(summary.info), 1)

    def test_severity_bucketing_is_case_insensitive(self):
        summary = build_validation_summary(
            [
                ValidationIssue(severity="critical"),
                ValidationIssue(severity="Warning"),
                ValidationIssue(severity="info"),
            ]
        )
        self.assertEqual(len(summary.critical), 1)
        self.assertEqual(len(summary.warning), 1)
        self.assertEqual(len(summary.info), 1)

    def test_unknown_severity_is_dropped_from_every_bucket(self):
        # KNOWN-BAD-INPUT: an out-of-taxonomy severity silently vanishes from
        # the summary (and is treated as non-critical by is_design_valid).
        summary = build_validation_summary([ValidationIssue(severity="FATAL")])
        self.assertEqual(summary.critical, [])
        self.assertEqual(summary.warning, [])
        self.assertEqual(summary.info, [])

    def test_empty_issue_list_gives_three_empty_buckets(self):
        summary = build_validation_summary([])
        self.assertEqual(summary.to_dict(), {"critical": [], "warning": [], "info": []})

    def test_summary_preserves_input_order_within_a_bucket(self):
        first = ValidationIssue(severity="CRITICAL", description="first")
        second = ValidationIssue(severity="CRITICAL", description="second")
        summary = build_validation_summary([first, second])
        self.assertEqual([i.description for i in summary.critical], ["first", "second"])


class IsDesignValidTests(unittest.TestCase):
    def test_no_issues_is_valid(self):
        self.assertTrue(is_design_valid([]))

    def test_warnings_and_info_alone_stay_valid(self):
        self.assertTrue(
            is_design_valid(
                [
                    ValidationIssue(severity="WARNING"),
                    ValidationIssue(severity="INFO"),
                ]
            )
        )

    def test_a_single_critical_invalidates(self):
        self.assertFalse(
            is_design_valid(
                [ValidationIssue(severity="WARNING"), ValidationIssue(severity="CRITICAL")]
            )
        )

    def test_lowercase_critical_still_invalidates(self):
        self.assertFalse(is_design_valid([ValidationIssue(severity="critical")]))


# ----------------------------------------------------------------------
# Invariants (stdlib substitute for property-based testing)
# ----------------------------------------------------------------------


class InvariantTests(unittest.TestCase):
    """hypothesis is unavailable in this repo, so these invariants are checked
    by exhaustive enumeration over small domains (itertools.permutations) and a
    seeded shuffle with random.Random(20260719) instead of generated data."""

    def test_result_set_is_independent_of_component_and_net_order(self):
        components, nets = _all_rules_design()
        baseline = fingerprint(validate_circuit(components, nets))
        # 3! component orders x 4! net orders = 144 exhaustive combinations.
        for comp_order in itertools.permutations(components):
            for net_order in itertools.permutations(nets):
                got = fingerprint(validate_circuit(list(comp_order), list(net_order)))
                self.assertEqual(got, baseline)

    def test_validate_circuit_is_deterministic_across_repeat_calls(self):
        components, nets = _all_rules_design()
        first = [i.to_dict() for i in validate_circuit(components, nets)]
        for _ in range(5):
            self.assertEqual(
                [i.to_dict() for i in validate_circuit(components, nets)], first
            )

    def test_validate_circuit_does_not_mutate_its_inputs(self):
        components, nets = _all_rules_design()
        before = (
            [c.to_dict() for c in components],
            [n.to_dict() for n in nets],
        )
        validate_circuit(components, nets)
        after = ([c.to_dict() for c in components], [n.to_dict() for n in nets])
        self.assertEqual(after, before)

    def test_adding_an_isolated_inert_component_never_adds_an_issue(self):
        # Exhaustive over a small pool of inert (non-active-category, fully
        # unconnected) components appended to a clean design.
        base_components, base_nets = wired_mcu()
        base = fingerprint(validate_circuit([base_components], base_nets))
        inert_pool = [
            comp("R%d" % n, "Passives", [pin("1", "Passive"), pin("2", "Passive")])
            for n in range(1, 4)
        ] + [
            comp("J1", "Connector", [pin("1", "Passive")]),
            comp("C1", "Passives", []),
        ]
        for size in range(1, 4):
            for extra in itertools.combinations(inert_pool, size):
                got = fingerprint(
                    validate_circuit([base_components] + list(extra), base_nets)
                )
                self.assertEqual(got, base)

    def test_seeded_shuffles_of_a_larger_design_agree(self):
        # FIXED SEED 20260719: 25 shuffles of a 3-component / 4-net design.
        rng = random.Random(FIXED_SEED)
        components, nets = _all_rules_design()
        baseline = fingerprint(validate_circuit(components, nets))
        for _ in range(25):
            shuffled_components = list(components)
            shuffled_nets = list(nets)
            rng.shuffle(shuffled_components)
            rng.shuffle(shuffled_nets)
            self.assertEqual(
                fingerprint(validate_circuit(shuffled_components, shuffled_nets)),
                baseline,
            )

    def test_summary_partitions_issues_without_loss_for_taxonomy_severities(self):
        # Exhaustive over every ordered triple of taxonomy severities.
        for combo in itertools.product(("CRITICAL", "WARNING", "INFO"), repeat=3):
            issues = [ValidationIssue(severity=s, description=str(n))
                      for n, s in enumerate(combo)]
            summary = build_validation_summary(issues)
            total = len(summary.critical) + len(summary.warning) + len(summary.info)
            self.assertEqual(total, len(issues))
            self.assertEqual(
                is_design_valid(issues), len(summary.critical) == 0
            )

    def test_every_issue_carries_a_severity_category_and_remedy(self):
        components, nets = _all_rules_design()
        for issue in validate_circuit(components, nets):
            self.assertIn(issue.severity, ("CRITICAL", "WARNING", "INFO"))
            self.assertTrue(issue.category)
            self.assertTrue(issue.description)
            self.assertTrue(issue.troubleshooting)


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
