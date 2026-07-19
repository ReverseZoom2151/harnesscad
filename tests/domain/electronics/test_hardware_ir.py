"""Tests for the tolerant Hardware IR dataclasses.

IMPORTANT CONTRACT NOTE (drives every "known-bad" case below): this module is a
*tolerant* loader. Unlike the pydantic original in Forma-OSS
(``blueprint_core/models.py``), the harness port has NO required fields, NO enum
validation, and NO referential-integrity checks. Its only refusal predicates are
the ones actually implemented by the coercion helpers:

  * ``_opt_float`` / ``_as_float``  -> unparseable numbers become None / default
  * ``_as_int``                     -> unparseable ints become the default
  * ``_str_list`` / ``_list_of``    -> non-sequences become [], non-dict entries
                                       inside a sequence are silently dropped
  * ``_dict_or_empty``              -> non-dicts become {}
  * nested optionals                -> a non-dict payload becomes None
  * ``from_dict`` on a non-mapping  -> AttributeError (the one hard failure)

So each malformed input below is paired with a well-formed counterpart, and the
assertion is on the *coerced* value, not on an exception. Deliberate divergence
from upstream is called out inline.
"""

import itertools
import json
import random
import unittest

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
    main,
)

# Every dataclass in the IR that exposes the to_dict/from_dict protocol.
ALL_NODES = [
    PinDefinition,
    ComponentTemplate,
    ProjectOverview,
    FunctionalRequirements,
    ComponentInstance,
    PinReference,
    ConnectionNet,
    BusConnection,
    PowerRail,
    PinMappingEntry,
    AssemblyStep,
    MechanicalVector3,
    MechanicalRotation3,
    MechanicalPlacement,
    MechanicalSpatialRelationship,
    MechanicalSource,
    MechanicalNotes,
    ValidationIssue,
    ValidationSummary,
    HardwareIR,
]

SEED = 20260719


def _populated_ir():
    """A structurally complete IR built independently of the module's own
    ``_synthetic_ir`` fixture, so the tests do not merely re-assert it."""
    return HardwareIR(
        hardware_ir_version="0.1",
        overview=ProjectOverview(
            title="Soil Probe",
            description="A battery powered soil moisture probe.",
            difficulty="Intermediate",
            estimated_cost=21.75,
            category="IoT",
        ),
        requirements=FunctionalRequirements(
            requirements=["Sample moisture every hour"],
            power_needs="LiPo",
            operating_voltage=3.3,
            physical_constraints=["Fits a 100mm tube"],
            safety_notes=["Do not submerge the MCU"],
            missing_info=["What soil type?"],
        ),
        components=[
            ComponentInstance(
                ref_des="U1",
                part_number="ESP32-C3",
                name="ESP32-C3 Board",
                category="Microcontroller",
                quantity=1,
                unit_price=4.25,
                sourcing_url="https://example.invalid/u1",
                rationale="Low power WiFi.",
                pins=[
                    PinDefinition("3V3", "VCC", "Power", 3.3, "rail"),
                    PinDefinition("GND", "GND", "Ground"),
                ],
            ),
            ComponentInstance(
                ref_des="S1",
                part_number="CAP-SOIL",
                name="Capacitive Soil Sensor",
                category="Sensor",
                quantity=2,
                unit_price=1.5,
            ),
        ],
        nets=[
            ConnectionNet(
                net_id="N1",
                name="3V3",
                net_type="Power",
                voltage=3.3,
                pins=[PinReference("U1", "3V3"), PinReference("S1", "VCC")],
            )
        ],
        buses=[BusConnection("B1", "I2C", 400000.0, ["N1"])],
        pin_mappings=[PinMappingEntry("GPIO4", "S1", "N1")],
        assembly=[
            AssemblyStep(
                step_num=1,
                title="Solder",
                description="Solder the header.",
                danger_flag=True,
                danger_message="Hot iron.",
                affected_components=["U1"],
            )
        ],
        mechanical=MechanicalNotes(
            enclosure_type="3D Printed",
            mounting_guidance="M3 inserts.",
            fabrication_details=["0.2mm layers"],
            fabrication_cost_estimate_usd=3.5,
            cad_sources=[
                MechanicalSource(
                    name="Tube",
                    source_type="Open STL",
                    url="https://example.invalid/stl",
                    file_formats=["stl", "step"],
                    license="CC-BY",
                    estimated_unit_price_usd=0.0,
                    notes="printable",
                )
            ],
            manufacturability_rating="Easy",
            render_dimensions=MechanicalVector3(116, 82, 55),
            component_placements=[
                MechanicalPlacement(
                    ref_des="U1",
                    label="MCU",
                    category="Microcontroller",
                    layer="electrical",
                    position=MechanicalVector3(0, 0, -1.5),
                    size=MechanicalVector3(38, 28, 5),
                    orientation_deg=MechanicalRotation3(0, 0, 90),
                    mounting_face="internal",
                    notes="centered",
                )
            ],
            spatial_relationships=[
                MechanicalSpatialRelationship("U1", "S1", "adjacent-to", "X", 12.5, "n")
            ],
        ),
        constraints=["battery only"],
        power_rails=[PowerRail("R1", 3.3, 500.0, "U1")],
        estimated_current_draw_ma=88.0,
        fabrication_notes=["print in PETG"],
        assembly_metadata={"pipeline": "test"},
        project_version_history=[{"revision": 1}],
        validation=ValidationSummary(
            critical=[ValidationIssue("CRITICAL", "Short Circuit", "d", "t")],
            warning=[ValidationIssue("WARNING", "Overcurrent", "d", "t")],
            info=[ValidationIssue("INFO", "Note", "d", "t")],
        ),
        is_valid=False,
    )


class RoundTripTests(unittest.TestCase):
    def test_default_instances_round_trip_for_every_node(self):
        # Table-driven over every dataclass in the IR: a defaulted instance must
        # survive to_dict -> from_dict unchanged.
        for cls in ALL_NODES:
            with self.subTest(cls=cls.__name__):
                node = cls()
                self.assertEqual(cls.from_dict(node.to_dict()), node)

    def test_default_payloads_round_trip_through_json(self):
        for cls in ALL_NODES:
            with self.subTest(cls=cls.__name__):
                payload = cls().to_dict()
                self.assertEqual(
                    cls.from_dict(json.loads(json.dumps(payload))), cls()
                )

    def test_populated_ir_round_trips_exactly(self):
        ir = _populated_ir()
        self.assertEqual(HardwareIR.from_dict(ir.to_dict()), ir)

    def test_populated_ir_round_trips_through_json(self):
        ir = _populated_ir()
        text = json.dumps(ir.to_dict(), sort_keys=True)
        self.assertEqual(HardwareIR.from_dict(json.loads(text)), ir)

    def test_round_trip_is_idempotent_on_the_payload(self):
        payload = _populated_ir().to_dict()
        once = HardwareIR.from_dict(payload).to_dict()
        twice = HardwareIR.from_dict(once).to_dict()
        self.assertEqual(payload, once)
        self.assertEqual(once, twice)

    def test_nested_leaves_survive_the_round_trip(self):
        restored = HardwareIR.from_dict(_populated_ir().to_dict())
        self.assertEqual(restored.components[0].pins[0].voltage, 3.3)
        self.assertEqual(restored.components[1].quantity, 2)
        self.assertEqual(restored.buses[0].clock_frequency_hz, 400000.0)
        self.assertEqual(restored.mechanical.cad_sources[0].file_formats,
                         ["stl", "step"])
        self.assertEqual(
            restored.mechanical.component_placements[0].orientation_deg.z_deg, 90
        )
        self.assertTrue(restored.assembly[0].danger_flag)
        self.assertEqual(restored.validation.critical[0].severity, "CRITICAL")
        self.assertFalse(restored.is_valid)

    def test_to_dict_copies_mutable_containers(self):
        ir = _populated_ir()
        payload = ir.to_dict()
        payload["constraints"].append("mutated")
        payload["assembly_metadata"]["pipeline"] = "mutated"
        self.assertEqual(ir.constraints, ["battery only"])
        self.assertEqual(ir.assembly_metadata, {"pipeline": "test"})

    def test_from_dict_accepts_already_built_children(self):
        # _list_of passes through instances of the target class untouched.
        pin = PinDefinition("A1", "SDA", "I2C", 3.3)
        net = ConnectionNet.from_dict({"net_id": "N", "pins": [PinReference("U1", "1")]})
        self.assertEqual(net.pins, [PinReference("U1", "1")])
        component = ComponentInstance.from_dict({"ref_des": "U1", "pins": [pin]})
        self.assertEqual(component.pins, [pin])


class MalformedPayloadTests(unittest.TestCase):
    """Known-bad vectors. Each is paired with a well-formed counterpart."""

    def test_from_dict_on_a_non_mapping_raises(self):
        # The single hard failure mode in the module: from_dict assumes .get.
        for bad in ["a string", 42, ["a", "list"], None]:
            with self.subTest(bad=bad):
                with self.assertRaises((AttributeError, TypeError)):
                    HardwareIR.from_dict(bad)
        # Counterpart: an empty mapping is accepted and fully defaulted.
        self.assertEqual(HardwareIR.from_dict({}), HardwareIR())

    def test_empty_payload_yields_all_defaults(self):
        for cls in ALL_NODES:
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.from_dict({}), cls())

    def test_unknown_keys_are_ignored(self):
        payload = _populated_ir().to_dict()
        payload["totally_unknown"] = {"nested": [1, 2, 3]}
        payload["components"][0]["not_a_field"] = "junk"
        restored = HardwareIR.from_dict(payload)
        self.assertEqual(restored, _populated_ir())

    def test_missing_keys_fall_back_to_defaults(self):
        payload = _populated_ir().to_dict()
        for key in ("buses", "is_valid", "hardware_ir_version", "nets",
                    "constraints", "validation", "estimated_current_draw_ma"):
            del payload[key]
        restored = HardwareIR.from_dict(payload)
        self.assertEqual(restored.buses, [])
        self.assertEqual(restored.nets, [])
        self.assertEqual(restored.constraints, [])
        self.assertIs(restored.is_valid, True)
        self.assertEqual(restored.hardware_ir_version, "0.1")
        self.assertEqual(restored.validation, ValidationSummary())
        self.assertEqual(restored.estimated_current_draw_ma, 0.0)

    def test_wrong_typed_numbers_are_coerced_or_defaulted(self):
        cases = [
            ("12.5", 12.5),          # numeric string parses
            (7, 7.0),                # int widens
            (True, 1.0),             # bool is an int in Python
            ("not a number", 0.0),   # bad -> _as_float default
            (None, 0.0),
            ([1, 2], 0.0),           # wrong container -> default
            ({"x": 1}, 0.0),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                rail = PowerRail.from_dict({"rail_id": "R", "voltage": raw})
                self.assertEqual(rail.voltage, expected)

    def test_optional_float_distinguishes_absent_from_bad(self):
        # _opt_float keeps None for both missing and unparseable, but a good
        # value must survive - the passing counterpart.
        self.assertIsNone(PinDefinition.from_dict({}).voltage)
        self.assertIsNone(PinDefinition.from_dict({"voltage": "high"}).voltage)
        self.assertIsNone(PinDefinition.from_dict({"voltage": [3.3]}).voltage)
        self.assertEqual(PinDefinition.from_dict({"voltage": "3.3"}).voltage, 3.3)
        self.assertEqual(PinDefinition.from_dict({"voltage": 0}).voltage, 0.0)

    def test_wrong_typed_ints_fall_back(self):
        self.assertEqual(ComponentInstance.from_dict({"quantity": "3"}).quantity, 3)
        self.assertEqual(ComponentInstance.from_dict({"quantity": 4.9}).quantity, 4)
        self.assertEqual(ComponentInstance.from_dict({"quantity": "many"}).quantity, 1)
        self.assertEqual(ComponentInstance.from_dict({"quantity": None}).quantity, 1)
        self.assertEqual(AssemblyStep.from_dict({"step_num": "bad"}).step_num, 0)

    def test_operating_voltage_keeps_its_non_zero_default(self):
        # FunctionalRequirements is the only node with a non-zero numeric
        # default; a bad value must land on 3.3, not 0.0.
        self.assertEqual(FunctionalRequirements.from_dict({}).operating_voltage, 3.3)
        self.assertEqual(
            FunctionalRequirements.from_dict(
                {"operating_voltage": "oops"}
            ).operating_voltage,
            3.3,
        )
        self.assertEqual(
            FunctionalRequirements.from_dict(
                {"operating_voltage": 5}
            ).operating_voltage,
            5.0,
        )

    def test_non_sequence_lists_become_empty(self):
        for bad in ["USB", 5, {"a": 1}, None, True]:
            with self.subTest(bad=bad):
                self.assertEqual(HardwareIR.from_dict({"components": bad}).components,
                                 [])
                self.assertEqual(HardwareIR.from_dict({"constraints": bad}).constraints,
                                 [])
        # Counterpart: a real list of dicts is materialised.
        ok = HardwareIR.from_dict({"components": [{"ref_des": "U1"}],
                                   "constraints": ["a", "b"]})
        self.assertEqual([c.ref_des for c in ok.components], ["U1"])
        self.assertEqual(ok.constraints, ["a", "b"])

    def test_non_dict_entries_inside_a_list_are_dropped_silently(self):
        # DIVERGENCE FROM UPSTREAM: pydantic would raise here; the port drops
        # the entry with no diagnostic at all.
        net = ConnectionNet.from_dict(
            {"net_id": "N1", "pins": [{"ref_des": "U1", "pin_id": "1"},
                                      "U2.3", None, 17, ["U3", "2"]]}
        )
        self.assertEqual(net.pins, [PinReference("U1", "1")])

    def test_str_list_stringifies_heterogeneous_entries(self):
        self.assertEqual(
            HardwareIR.from_dict({"constraints": [1, None, True, 2.5]}).constraints,
            ["1", "None", "True", "2.5"],
        )

    def test_non_dict_nested_optionals_become_none(self):
        for bad in ["overview", 3, ["a"], None]:
            with self.subTest(bad=bad):
                ir = HardwareIR.from_dict(
                    {"overview": bad, "requirements": bad, "mechanical": bad}
                )
                self.assertIsNone(ir.overview)
                self.assertIsNone(ir.requirements)
                self.assertIsNone(ir.mechanical)
        ok = HardwareIR.from_dict({"overview": {"title": "T"}})
        self.assertEqual(ok.overview.title, "T")

    def test_render_dimensions_only_accepts_a_mapping(self):
        self.assertIsNone(
            MechanicalNotes.from_dict({"render_dimensions": [1, 2, 3]}).render_dimensions
        )
        self.assertEqual(
            MechanicalNotes.from_dict(
                {"render_dimensions": {"x_mm": 1, "y_mm": 2, "z_mm": 3}}
            ).render_dimensions,
            MechanicalVector3(1.0, 2.0, 3.0),
        )

    def test_placement_vectors_survive_a_missing_or_bad_subdocument(self):
        placement = MechanicalPlacement.from_dict(
            {"ref_des": "U1", "position": "0,0,0", "size": None}
        )
        self.assertEqual(placement.position, MechanicalVector3())
        self.assertEqual(placement.size, MechanicalVector3())
        self.assertEqual(placement.orientation_deg, MechanicalRotation3())

    def test_layer_default_applies_only_to_a_missing_value(self):
        # _as_str substitutes the default for None only, so an explicit empty
        # string is preserved verbatim rather than becoming "electrical".
        self.assertEqual(MechanicalPlacement.from_dict({}).layer, "electrical")
        self.assertEqual(MechanicalPlacement.from_dict({"layer": None}).layer,
                         "electrical")
        self.assertEqual(MechanicalPlacement.from_dict({"layer": ""}).layer, "")
        self.assertEqual(MechanicalPlacement.from_dict({"layer": "print"}).layer,
                         "print")

    def test_unknown_enum_values_are_accepted_verbatim(self):
        # There is no enum validation anywhere in this module. Recording that
        # explicitly: "known-bad" enum strings pass straight through, which is a
        # deliberate divergence from the pydantic original.
        self.assertEqual(PinDefinition.from_dict({"pin_type": "Quantum"}).pin_type,
                         "Quantum")
        self.assertEqual(ConnectionNet.from_dict({"net_type": "TELEPATHY"}).net_type,
                         "TELEPATHY")
        self.assertEqual(BusConnection.from_dict({"bus_type": "SMOKE"}).bus_type,
                         "SMOKE")
        self.assertEqual(MechanicalPlacement.from_dict({"layer": "hyperspace"}).layer,
                         "hyperspace")
        self.assertEqual(
            MechanicalSpatialRelationship.from_dict({"axis": "W"}).axis, "W"
        )
        self.assertEqual(ValidationIssue.from_dict({"severity": "FATAL"}).severity,
                         "FATAL")
        # And an unknown enum still round-trips, so nothing is lost downstream.
        issue = ValidationIssue.from_dict({"severity": "FATAL"})
        self.assertEqual(ValidationIssue.from_dict(issue.to_dict()), issue)

    def test_severity_default_is_info_when_absent(self):
        self.assertEqual(ValidationIssue.from_dict({}).severity, "INFO")
        self.assertEqual(ValidationIssue.from_dict({"severity": None}).severity, "INFO")

    def test_booleans_coerce_truthily(self):
        self.assertIs(AssemblyStep.from_dict({"danger_flag": "no"}).danger_flag, True)
        self.assertIs(AssemblyStep.from_dict({"danger_flag": ""}).danger_flag, False)
        self.assertIs(AssemblyStep.from_dict({"danger_flag": 0}).danger_flag, False)
        self.assertIs(AssemblyStep.from_dict({}).danger_flag, False)
        self.assertIs(HardwareIR.from_dict({"is_valid": None}).is_valid, True)
        self.assertIs(HardwareIR.from_dict({"is_valid": 0}).is_valid, False)

    def test_assembly_metadata_rejects_non_mappings(self):
        self.assertEqual(
            HardwareIR.from_dict({"assembly_metadata": [("a", 1)]}).assembly_metadata,
            {},
        )
        self.assertEqual(
            HardwareIR.from_dict({"assembly_metadata": {"a": 1}}).assembly_metadata,
            {"a": 1},
        )

    def test_version_history_drops_non_mapping_entries(self):
        ir = HardwareIR.from_dict(
            {"project_version_history": [{"revision": 1}, "rev2", None, 3]}
        )
        self.assertEqual(ir.project_version_history, [{"revision": 1}])
        self.assertEqual(
            HardwareIR.from_dict({"project_version_history": "rev1"})
            .project_version_history,
            [],
        )

    def test_truncated_mechanical_payload_loads_instead_of_raising(self):
        # Ported in spirit from Forma-OSS tests/test_structured_repair.py, which
        # feeds a truncated MechanicalNotes JSON to a pydantic validator and
        # expects salvage/failure. DELIBERATE DIVERGENCE: the harness port has no
        # required fields, so a half-written record simply loads with defaults
        # and the caller gets no signal. Asserted here so the difference is
        # locked down rather than assumed.
        truncated = {
            "enclosure_type": "3D Printed",
            "component_placements": [
                {"ref_des": "U1", "position": {"x_mm": 1.0, "y_mm": 2.0, "z_mm": 3.0}},
                {"ref_des": "DS1", "position": {"x_mm": 4.0}},  # cut mid-record
            ],
            # manufacturability_rating never arrived
        }
        notes = MechanicalNotes.from_dict(truncated)
        self.assertEqual(notes.manufacturability_rating, "")
        self.assertEqual(len(notes.component_placements), 2)
        self.assertEqual(notes.component_placements[1].position,
                         MechanicalVector3(4.0, 0.0, 0.0))


class ReferentialIntegrityTests(unittest.TestCase):
    """Dangling cross-references. The IR stores refs as bare strings and
    performs no lookup, so these all load cleanly - documented, not asserted as
    desirable."""

    def test_net_pins_may_reference_absent_components(self):
        ir = HardwareIR.from_dict(
            {
                "components": [{"ref_des": "U1"}],
                "nets": [
                    {
                        "net_id": "N1",
                        "pins": [{"ref_des": "U1", "pin_id": "1"},
                                 {"ref_des": "GHOST", "pin_id": "9"}],
                    }
                ],
            }
        )
        refs = {c.ref_des for c in ir.components}
        dangling = [p.ref_des for p in ir.nets[0].pins if p.ref_des not in refs]
        self.assertEqual(dangling, ["GHOST"])
        self.assertIs(ir.is_valid, True)  # nothing flags it

    def test_bus_may_reference_absent_nets(self):
        ir = HardwareIR.from_dict(
            {"nets": [{"net_id": "N1"}], "buses": [{"bus_id": "B1",
                                                    "nets": ["N1", "N_MISSING"]}]}
        )
        self.assertEqual(ir.buses[0].nets, ["N1", "N_MISSING"])

    def test_placement_and_relationship_may_reference_absent_components(self):
        ir = HardwareIR.from_dict(
            {
                "components": [{"ref_des": "U1"}],
                "mechanical": {
                    "component_placements": [{"ref_des": "NOPE"}],
                    "spatial_relationships": [
                        {"source_ref_des": "U1", "target_ref_des": "ALSO_NOPE"}
                    ],
                },
            }
        )
        self.assertEqual(ir.mechanical.component_placements[0].ref_des, "NOPE")
        self.assertEqual(
            ir.mechanical.spatial_relationships[0].target_ref_des, "ALSO_NOPE"
        )

    def test_duplicate_ref_des_is_not_rejected(self):
        ir = HardwareIR.from_dict(
            {"components": [{"ref_des": "U1"}, {"ref_des": "U1"}]}
        )
        self.assertEqual([c.ref_des for c in ir.components], ["U1", "U1"])

    def test_pin_mapping_may_name_an_unknown_net(self):
        ir = HardwareIR.from_dict({"pin_mappings": [{"mcu_pin": "GPIO4",
                                                     "net_name": "NOT_A_NET"}]})
        self.assertEqual(ir.pin_mappings[0].net_name, "NOT_A_NET")


class GeneratedRoundTripTests(unittest.TestCase):
    """Substitutes for property-based testing: `hypothesis` is not available in
    this environment, so invariants are checked exhaustively over small,
    hand-enumerated domains (itertools.product) and over pseudo-random payloads
    from random.Random(SEED) with SEED = 20260719, making every run identical."""

    def test_exhaustive_scalar_coercion_grid(self):
        raws = [None, "", "0", "3.3", "junk", 0, 1, 2.5, True, False, [], {}]
        for raw in raws:
            with self.subTest(raw=repr(raw)):
                payload = {"voltage": raw, "max_current_capacity_ma": raw,
                           "rail_id": raw, "source_component": raw}
                rail = PowerRail.from_dict(payload)
                # Whatever the coercion produced, it must be exactly typed and
                # must round-trip.
                self.assertIsInstance(rail.voltage, float)
                self.assertIsInstance(rail.rail_id, str)
                self.assertEqual(PowerRail.from_dict(rail.to_dict()), rail)

    def test_exhaustive_optional_presence_grid_on_placement(self):
        # 2^4 = 16 combinations of present/absent optional sub-documents.
        keys = ["label", "category", "mounting_face", "notes"]
        for flags in itertools.product([False, True], repeat=len(keys)):
            with self.subTest(flags=flags):
                payload = {"ref_des": "U1"}
                for key, present in zip(keys, flags):
                    if present:
                        payload[key] = "v-" + key
                placement = MechanicalPlacement.from_dict(payload)
                self.assertEqual(
                    MechanicalPlacement.from_dict(placement.to_dict()), placement
                )
                for key, present in zip(keys, flags):
                    value = getattr(placement, key)
                    if present:
                        self.assertEqual(value, "v-" + key)
                    else:
                        self.assertIsNone(value)

    def _random_ir_payload(self, rng):
        n = rng.randint(0, 4)
        categories = ["Microcontroller", "Sensor", "Power", "Display", "Passives"]
        components = [
            {
                "ref_des": "C%d" % index,
                "part_number": "PN%d" % rng.randint(0, 99),
                "name": rng.choice(["Board", "Probe", "Cell", "Screen"]),
                "category": rng.choice(categories),
                "quantity": rng.choice([1, 2, "3", "bad", None]),
                "unit_price": rng.choice([0, 1.25, "2.5", "oops", None]),
                "pins": [
                    {"pin_id": "P%d" % pin, "pin_type": rng.choice(["Power", "I2C"]),
                     "voltage": rng.choice([3.3, 5, None, "bad"])}
                    for pin in range(rng.randint(0, 3))
                ],
            }
            for index in range(n)
        ]
        return {
            "overview": {"title": "T%d" % rng.randint(0, 9),
                         "estimated_cost": rng.choice([0, 1.5, "x"])},
            "components": components,
            "constraints": [str(rng.randint(0, 9)) for _ in range(rng.randint(0, 3))],
            "estimated_current_draw_ma": rng.choice([0, 12.5, "bad"]),
            "is_valid": rng.choice([True, False, None, 0]),
        }

    def test_random_payloads_reach_a_fixed_point_after_one_load(self):
        rng = random.Random(SEED)
        for index in range(200):
            payload = self._random_ir_payload(rng)
            with self.subTest(index=index):
                once = HardwareIR.from_dict(payload)
                twice = HardwareIR.from_dict(once.to_dict())
                # parse(serialise(parse(x))) == parse(x): the loader is
                # idempotent even when the raw input is messy.
                self.assertEqual(twice, once)
                self.assertEqual(twice.to_dict(), once.to_dict())

    def test_random_payloads_survive_json(self):
        rng = random.Random(SEED)
        for index in range(100):
            payload = self._random_ir_payload(rng)
            with self.subTest(index=index):
                ir = HardwareIR.from_dict(payload)
                text = json.dumps(ir.to_dict(), sort_keys=True)
                self.assertEqual(HardwareIR.from_dict(json.loads(text)), ir)

    def test_random_payloads_never_leak_raw_types(self):
        rng = random.Random(SEED)
        for index in range(100):
            ir = HardwareIR.from_dict(self._random_ir_payload(rng))
            with self.subTest(index=index):
                self.assertIsInstance(ir.estimated_current_draw_ma, float)
                self.assertIsInstance(ir.is_valid, bool)
                for component in ir.components:
                    self.assertIsInstance(component.quantity, int)
                    self.assertIsInstance(component.unit_price, float)
                    for pin in component.pins:
                        self.assertTrue(pin.voltage is None
                                        or isinstance(pin.voltage, float))


class CliTests(unittest.TestCase):
    def test_selfcheck_passes(self):
        self.assertEqual(main(["--selfcheck"]), 0)

    def test_selfcheck_json_passes(self):
        self.assertEqual(main(["--selfcheck", "--json"]), 0)

    def test_no_args_prints_help_and_succeeds(self):
        self.assertEqual(main([]), 0)

    def test_unknown_flag_exits(self):
        with self.assertRaises(SystemExit):
            main(["--not-a-flag"])


if __name__ == "__main__":
    unittest.main()
