"""Tests for the heuristic mechanical placement seeding.

CONTRACT NOTE (drives the "known-bad" cases below): this module has no
validator and raises nothing. Read end to end, its only refusal-shaped
behaviours are:

  * ``enrich_mechanical_layout`` bails out (returns the IR untouched) when
    ``ir.mechanical`` is None or ``ir.components`` is empty;
  * ``infer_render_dimensions`` refuses to overwrite an envelope that is
    already set, and clamps the count-scaled fallback to
    92..150 x 48..92 x 30..70;
  * existing ``component_placements`` are never regenerated;
  * ``derive_spatial_relationships`` drops any relationship whose endpoints are
    not both present, skips enclosure-layer placements, and caps
    controller-relative rows at 9.

There is *no* overlap rejection, *no* bounds enforcement and *no* clearance
rule anywhere in the file, so the classic layout known-bad vectors cannot be
expressed as "the module refuses X". They are expressed instead as assertions
on what it actually produces, and two of them are genuine defects marked
``expectedFailure`` below.
"""

import itertools
import random
import unittest

from harnesscad.domain.electronics.enclosure_layout import (
    derive_spatial_relationships,
    enrich_mechanical_layout,
    infer_render_dimensions,
    main,
    placement_layer,
    placement_position,
    placement_size,
)
from harnesscad.domain.electronics.hardware_ir import (
    ComponentInstance,
    HardwareIR,
    MechanicalNotes,
    MechanicalPlacement,
    MechanicalVector3,
    ProjectOverview,
)

SEED = 20260719
DIMS = MechanicalVector3(100.0, 50.0, 40.0)


def component(ref_des, name="", category="", part_number="", rationale=""):
    return ComponentInstance(
        ref_des=ref_des,
        part_number=part_number,
        name=name,
        category=category,
        rationale=rationale,
    )


def build_ir(components, mechanical=None, title="", description="",
             constraints=None, fabrication_notes=None):
    return HardwareIR(
        overview=ProjectOverview(title=title, description=description),
        components=list(components),
        mechanical=MechanicalNotes() if mechanical is None else mechanical,
        constraints=list(constraints or []),
        fabrication_notes=list(fabrication_notes or []),
    )


def placements_by_ref(ir):
    return {p.ref_des: p for p in ir.mechanical.component_placements}


def extents(placement, axis):
    """(low, high) of a placement along 'x'/'y'/'z' in millimetres."""
    center = getattr(placement.position, axis + "_mm")
    half = getattr(placement.size, axis + "_mm") / 2.0
    return center - half, center + half


def boxes_overlap(a, b):
    for axis in ("x", "y", "z"):
        a_low, a_high = extents(a, axis)
        b_low, b_high = extents(b, axis)
        if a_high <= b_low or b_high <= a_low:
            return False
    return True


class InferDimensionsTests(unittest.TestCase):
    def test_existing_envelope_is_never_overwritten(self):
        preset = MechanicalVector3(7.0, 8.0, 9.0)
        ir = build_ir([component("U1")],
                      mechanical=MechanicalNotes(render_dimensions=preset),
                      title="portable pocket audio")
        self.assertIs(infer_render_dimensions(ir), preset)

    def test_keyword_presets(self):
        cases = [
            ("A pocket mp3 player", (100.0, 21.0, 54.0)),
            ("Portable audio thing", (100.0, 21.0, 54.0)),
            ("Soil moisture plant watering rig", (116.0, 82.0, 55.0)),
            ("Smart thermostat for hvac", (86.0, 24.0, 86.0)),
            ("Servo deadbolt lock", (92.0, 64.0, 38.0)),
        ]
        for description, expected in cases:
            with self.subTest(description=description):
                ir = build_ir([], description=description)
                dims = infer_render_dimensions(ir)
                self.assertEqual((dims.x_mm, dims.y_mm, dims.z_mm), expected)

    def test_preset_precedence_is_first_match_wins(self):
        # A brief hitting several keyword classes resolves to the audio preset,
        # which is tested first in the source.
        ir = build_ir([], description="a portable plant thermostat lock")
        dims = infer_render_dimensions(ir)
        self.assertEqual((dims.x_mm, dims.y_mm, dims.z_mm), (100.0, 21.0, 54.0))

    def test_keywords_are_read_from_constraints_and_fabrication_notes(self):
        self.assertEqual(
            infer_render_dimensions(build_ir([], constraints=["must fit a POCKET"])).x_mm,
            100.0,
        )
        self.assertEqual(
            infer_render_dimensions(
                build_ir([], fabrication_notes=["Soil probe housing"])
            ).x_mm,
            116.0,
        )

    def test_missing_overview_does_not_explode(self):
        # KNOWN-BAD: overview is Optional and the haystack builder must tolerate
        # it being absent. Falls through to the count-scaled branch.
        ir = HardwareIR(components=[component("U1")], mechanical=MechanicalNotes())
        self.assertIsNone(ir.overview)
        dims = infer_render_dimensions(ir)
        self.assertEqual((dims.x_mm, dims.y_mm, dims.z_mm), (92.0, 48.0, 30.0))

    def test_fallback_is_clamped_at_both_ends(self):
        empty = infer_render_dimensions(build_ir([]))
        self.assertEqual((empty.x_mm, empty.y_mm, empty.z_mm), (92.0, 48.0, 30.0))
        huge = build_ir([component("C%d" % i, category="Sensor") for i in range(40)])
        big = infer_render_dimensions(huge)
        self.assertEqual((big.x_mm, big.y_mm, big.z_mm), (150.0, 92.0, 70.0))

    def test_fallback_counts_only_electrical_components(self):
        # Mechanical and 3D-print parts are excluded from the scaling count, so
        # a BOM made entirely of them lands on the minimum envelope.
        parts = [component("M%d" % i, category="Mechanical") for i in range(10)]
        parts += [component("P%d" % i, category="3D Print") for i in range(10)]
        dims = infer_render_dimensions(build_ir(parts))
        self.assertEqual((dims.x_mm, dims.y_mm, dims.z_mm), (92.0, 48.0, 30.0))

    def test_fallback_grows_monotonically_with_component_count(self):
        previous = None
        for count in range(0, 16):
            parts = [component("C%d" % i, category="Sensor") for i in range(count)]
            dims = infer_render_dimensions(build_ir(parts))
            if previous is not None:
                self.assertGreaterEqual(dims.x_mm, previous.x_mm)
                self.assertGreaterEqual(dims.y_mm, previous.y_mm)
                self.assertGreaterEqual(dims.z_mm, previous.z_mm)
            previous = dims

    def test_category_matching_is_case_and_space_insensitive(self):
        parts = [component("C%d" % i, category="  3D PRINT  ") for i in range(10)]
        dims = infer_render_dimensions(build_ir(parts))
        self.assertEqual(dims.x_mm, 92.0)


class PlacementLayerTests(unittest.TestCase):
    def test_layer_table(self):
        cases = [
            (component("E1", name="Main Enclosure Shell", category="3D Print"),
             "enclosure"),
            (component("E2", name="Project Box", category="Mechanical"), "enclosure"),
            (component("E3", name="ABS Housing", category="Mechanical"), "enclosure"),
            (component("B1", name="Front Bezel", category="3D Print"), "print"),
            (component("M1", name="M3 Heat-set Insert", category="Mechanical"),
             "structural"),
            (component("M2", name="Standoff", category="Mechanical"), "structural"),
            (component("M3", name="Hinge", category="Mechanical"), "mechanism"),
            (component("U1", name="ESP32", category="Microcontroller"), "electrical"),
            (component("X1"), "electrical"),  # wholly empty component
        ]
        for part, expected in cases:
            with self.subTest(ref_des=part.ref_des):
                self.assertEqual(placement_layer(part), expected)

    def test_fastener_tokens_veto_the_enclosure_classification(self):
        # KNOWN-BAD-ish: "Housing Screw" contains an enclosure token, but the
        # fastener veto list wins, so it is structural rather than enclosure.
        screw = component("M9", name="Housing Screw", category="Mechanical")
        self.assertEqual(placement_layer(screw), "structural")
        # Passing counterpart without the veto token.
        self.assertEqual(
            placement_layer(component("M8", name="Housing", category="Mechanical")),
            "enclosure",
        )
        # The multi-word "button cap" veto matters on its own: this part names
        # the "case" it belongs to but is a button cap, not the shell.
        cap = component("K1", name="Button Cap for Case", category="Mechanical")
        self.assertEqual(placement_layer(cap), "mechanism")

    def test_classification_reads_every_text_field(self):
        for kwargs in [{"name": "shell"}, {"part_number": "shell"},
                       {"category": "shell"}]:
            with self.subTest(**kwargs):
                self.assertEqual(placement_layer(component("Z1", **kwargs)),
                                 "enclosure")
        self.assertEqual(placement_layer(component("shell")), "enclosure")


class PlacementSizeTests(unittest.TestCase):
    def test_enclosure_takes_the_whole_envelope(self):
        shell = component("E1", name="Enclosure Shell", category="3D Print")
        self.assertEqual(placement_size(shell, DIMS), DIMS)

    def test_enclosure_size_aliases_the_envelope_object(self):
        # Documented hazard, not a contract: the enclosure branch returns the
        # caller's vector itself rather than a copy, so the placement size and
        # render_dimensions are the same object.
        shell = component("E1", name="Enclosure Shell", category="3D Print")
        self.assertIs(placement_size(shell, DIMS), DIMS)

    def test_token_presets(self):
        cases = [
            ("Front Bezel", "3D Print", (82.0, 6.0, 28.8)),
            ("Back Cover", "3D Print", (88.0, 6.0, 34.4)),
            ("LiPo Battery", "Power", (45.0, 26.0, 8.8)),
            ("Speaker", "Audio", (24.0, 12.0, 24.0)),
            ("Relay Module", "Actuator", (38.0, 26.0, 16.0)),
            ("SG90 Servo", "Actuator", (23.0, 12.0, 29.0)),
            ("OLED Screen", "Display", (34.0, 3.0, 18.0)),
            ("Tactile Switch", "Passives", (10.0, 7.0, 10.0)),
            ("USB-C Port", "Power", (18.0, 8.0, 7.0)),
            ("M3 Standoff", "Mechanical", (5.0, 5.0, 8.0)),
            ("Mount Bracket", "3D Print", (34.0, 4.0, 18.0)),
        ]
        for name, category, expected in cases:
            with self.subTest(name=name):
                size = placement_size(component("Z1", name=name, category=category),
                                      DIMS)
                self.assertEqual((size.x_mm, size.y_mm, size.z_mm), expected)

    def test_category_presets_and_unknown_category_fallback(self):
        cases = [
            ("Microcontroller", (38.0, 28.0, 5.0)),
            ("Sensor", (20.0, 12.0, 14.0)),
            ("Actuator", (30.0, 22.0, 14.0)),
            ("Power", (42.0, 22.0, 8.0)),
            ("Passives", (15.0, 12.0, 8.0)),
            ("Communication", (28.0, 18.0, 5.0)),
            ("Mechanical", (14.0, 10.0, 8.0)),
            ("3D Print", (30.0, 5.0, 18.0)),
            ("Antimatter Trap", (22.0, 16.0, 6.0)),  # unknown enum -> fallback
            ("", (22.0, 16.0, 6.0)),
        ]
        for category, expected in cases:
            with self.subTest(category=category):
                size = placement_size(component("Z1", category=category), DIMS)
                self.assertEqual((size.x_mm, size.y_mm, size.z_mm), expected)

    def test_bezel_and_battery_thickness_floors_hold_on_a_flat_envelope(self):
        # KNOWN-BAD: a degenerate envelope with zero depth. The bezel branch has
        # a max(2.0, ...) floor so the panel keeps a printable thickness.
        flat = MechanicalVector3(100.0, 0.0, 40.0)
        bezel = placement_size(component("B1", name="Front Bezel"), flat)
        self.assertEqual(bezel.y_mm, 2.0)
        cover = placement_size(component("B2", name="Back Cover"), flat)
        self.assertEqual(cover.y_mm, 2.0)

    def test_zero_envelope_yields_zero_sized_scaled_parts(self):
        # KNOWN-BAD: zero-size enclosure. Nothing refuses it; parts sized by a
        # fraction of the envelope collapse to zero on the scaled axes, while
        # absolute presets are unaffected.
        zero = MechanicalVector3(0.0, 0.0, 0.0)
        bezel = placement_size(component("B1", name="Front Bezel"), zero)
        self.assertEqual((bezel.x_mm, bezel.z_mm), (0.0, 0.0))
        self.assertEqual(bezel.y_mm, 2.0)
        button = placement_size(component("S1", name="Push Button"), zero)
        self.assertEqual((button.x_mm, button.y_mm, button.z_mm), (10.0, 7.0, 10.0))

    def test_negative_envelope_propagates_without_complaint(self):
        # KNOWN-BAD: negative dimensions. There is no positivity check, so the
        # negative simply flows through; recorded so a future guard is a visible
        # behaviour change.
        negative = MechanicalVector3(-100.0, -50.0, -40.0)
        cover = placement_size(component("B2", name="Back Cover"), negative)
        self.assertEqual(cover.x_mm, -88.0)
        self.assertEqual(cover.y_mm, 2.0)  # the max() floor still applies

    def test_sizes_are_rounded_to_two_decimals(self):
        odd = MechanicalVector3(101.111, 50.0, 40.0)
        bezel = placement_size(component("B1", name="Front Bezel"), odd)
        self.assertEqual(bezel.x_mm, round(101.111 * 0.82, 2))
        self.assertEqual(bezel.x_mm, 82.91)


class PlacementPositionTests(unittest.TestCase):
    def test_enclosure_sits_at_the_origin(self):
        shell = component("E1", name="Enclosure Shell", category="3D Print")
        self.assertEqual(placement_position(shell, [shell], DIMS),
                         MechanicalVector3(0.0, 0.0, 0.0))

    def test_front_panels_go_to_negative_y_and_covers_to_positive_y(self):
        bezel = component("B1", name="Front Bezel", category="3D Print")
        cover = component("B2", name="Back Cover", category="3D Print")
        self.assertLess(placement_position(bezel, [bezel], DIMS).y_mm, 0)
        self.assertGreater(placement_position(cover, [cover], DIMS).y_mm, 0)

    def test_single_button_is_centered_and_a_row_is_symmetric(self):
        one = component("SW1", name="Push Button", category="Passives")
        self.assertEqual(placement_position(one, [one], DIMS).x_mm, 0.0)

        buttons = [component("SW%d" % i, name="Push Button", category="Passives")
                   for i in range(1, 4)]
        xs = [placement_position(b, buttons, DIMS).x_mm for b in buttons]
        self.assertEqual(xs[0], -xs[2])
        self.assertEqual(xs[1], 0.0)
        self.assertEqual(sorted(xs), xs)
        ys = {placement_position(b, buttons, DIMS).y_mm for b in buttons}
        self.assertEqual(len(ys), 1)

    def test_structural_parts_take_four_distinct_corners(self):
        studs = [component("M%d" % i, name="M3 Standoff", category="Mechanical")
                 for i in range(1, 5)]
        corners = {
            (placement_position(s, studs, DIMS).x_mm,
             placement_position(s, studs, DIMS).z_mm)
            for s in studs
        }
        self.assertEqual(len(corners), 4)

    def test_component_absent_from_the_peer_list_falls_back_to_index_zero(self):
        # KNOWN-BAD: a placement query for a component that is not in the peer
        # list at all (a dangling reference). The `next(..., 0)` default means
        # it silently takes slot 0 rather than failing.
        peers = [component("S%d" % i, category="Sensor") for i in range(1, 4)]
        orphan = component("S99", category="Sensor")
        self.assertEqual(
            placement_position(orphan, peers, DIMS),
            placement_position(peers[0], peers, DIMS),
        )

    def test_empty_peer_list_does_not_divide_by_zero(self):
        lonely = component("R1", category="Passives")
        position = placement_position(lonely, [], DIMS)
        self.assertEqual(position.x_mm, 0.0)

    def test_role_positions_land_on_the_documented_sides(self):
        battery = component("BAT1", name="LiPo Battery", category="Power")
        charger = component("J1", name="USB-C Charger", category="Power")
        speaker = component("SPK1", name="Speaker", category="Audio")
        self.assertLess(placement_position(battery, [battery], DIMS).x_mm, 0)
        self.assertGreater(placement_position(charger, [charger], DIMS).x_mm, 0)
        self.assertGreater(placement_position(speaker, [speaker], DIMS).z_mm, 0)

    def test_zero_envelope_collapses_every_position_to_the_origin(self):
        zero = MechanicalVector3(0.0, 0.0, 0.0)
        parts = [
            component("U1", category="Microcontroller"),
            component("S1", category="Sensor"),
            component("BAT1", name="Battery", category="Power"),
        ]
        for part in parts:
            with self.subTest(ref_des=part.ref_des):
                self.assertEqual(placement_position(part, parts, zero),
                                 MechanicalVector3(0.0, 0.0, 0.0))


class DeriveSpatialRelationshipTests(unittest.TestCase):
    @staticmethod
    def placement(ref_des, category=None, layer="electrical", label=None,
                  x=0.0, y=0.0, z=0.0):
        return MechanicalPlacement(
            ref_des=ref_des,
            label=label,
            category=category,
            layer=layer,
            position=MechanicalVector3(x, y, z),
            size=MechanicalVector3(1.0, 1.0, 1.0),
        )

    def test_no_controller_yields_no_offset_relationships(self):
        rows = derive_spatial_relationships(
            [self.placement("S1", category="Sensor", x=10.0),
             self.placement("S2", category="Sensor", x=20.0)]
        )
        self.assertEqual(rows, [])

    def test_empty_input(self):
        self.assertEqual(derive_spatial_relationships([]), [])

    def test_dominant_axis_and_signed_offset(self):
        controller = self.placement("U1", category="Microcontroller")
        cases = [
            (self.placement("A1", x=9.0, y=1.0, z=2.0), "X", 9.0),
            (self.placement("A2", x=1.0, y=-9.0, z=2.0), "Y", -9.0),
            (self.placement("A3", x=1.0, y=2.0, z=-9.0), "Z", -9.0),
        ]
        for target, axis, offset in cases:
            with self.subTest(ref_des=target.ref_des):
                rows = derive_spatial_relationships([controller, target])
                self.assertEqual(rows[0].axis, axis)
                self.assertEqual(rows[0].offset_mm, offset)
                self.assertEqual(rows[0].source_ref_des, "U1")

    def test_axis_tie_resolves_to_x(self):
        controller = self.placement("U1", category="Microcontroller")
        target = self.placement("A1", x=5.0, y=5.0, z=5.0)
        rows = derive_spatial_relationships([controller, target])
        self.assertEqual(rows[0].axis, "X")

    def test_enclosure_layer_and_self_are_skipped(self):
        rows = derive_spatial_relationships(
            [self.placement("U1", category="Microcontroller"),
             self.placement("E1", layer="enclosure", x=30.0),
             self.placement("S1", x=10.0)]
        )
        self.assertEqual([r.target_ref_des for r in rows], ["S1"])

    def test_controller_rows_are_capped_at_nine(self):
        rows = derive_spatial_relationships(
            [self.placement("U1", category="Microcontroller")]
            + [self.placement("A%d" % i, x=float(i)) for i in range(30)]
        )
        self.assertEqual(len(rows), 9)

    def test_display_to_bezel_alignment_is_emitted_once(self):
        rows = derive_spatial_relationships(
            [self.placement("DS1", label="OLED Display", y=-20.0),
             self.placement("DS2", label="Second Display", y=-19.0),
             self.placement("BEZ1", label="Front Bezel", y=-23.0)]
        )
        aligned = [r for r in rows if r.relation == "aligned with display opening"]
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0].source_ref_des, "DS1")
        self.assertEqual(aligned[0].target_ref_des, "BEZ1")
        self.assertEqual(aligned[0].offset_mm, -3.0)

    def test_no_bezel_means_no_alignment_row(self):
        rows = derive_spatial_relationships(
            [self.placement("DS1", label="OLED Display", y=-20.0)]
        )
        self.assertEqual(rows, [])

    def test_case_insensitive_matching_of_controller_and_display(self):
        rows = derive_spatial_relationships(
            [self.placement("U1", category="MICROCONTROLLER"),
             self.placement("DS1", label="oled panel", y=-20.0),
             self.placement("BEZ1", label="FRONT BEZEL", y=-23.0)]
        )
        self.assertTrue(any(r.source_ref_des == "U1" for r in rows))
        self.assertTrue(any(r.relation == "aligned with display opening"
                            for r in rows))

    def test_none_category_placements_are_not_mistaken_for_a_controller(self):
        rows = derive_spatial_relationships(
            [self.placement("A1", category=None, x=5.0),
             self.placement("A2", category=None, x=9.0)]
        )
        self.assertEqual(rows, [])


class EnrichLayoutTests(unittest.TestCase):
    def test_missing_mechanical_block_returns_the_ir_untouched(self):
        ir = HardwareIR(components=[component("U1", category="Microcontroller")])
        self.assertIs(enrich_mechanical_layout(ir), ir)
        self.assertIsNone(ir.mechanical)
        self.assertEqual(ir.assembly_metadata, {})

    def test_empty_component_list_returns_the_ir_untouched(self):
        ir = build_ir([])
        self.assertIs(enrich_mechanical_layout(ir), ir)
        self.assertIsNone(ir.mechanical.render_dimensions)
        self.assertEqual(ir.mechanical.component_placements, [])
        self.assertEqual(ir.assembly_metadata, {})

    def test_every_component_gets_exactly_one_placement(self):
        parts = [
            component("U1", name="ESP32", category="Microcontroller"),
            component("DS1", name="OLED Display", category="Display"),
            component("S1", name="Soil Probe", category="Sensor"),
            component("E1", name="Enclosure Shell", category="3D Print"),
        ]
        ir = enrich_mechanical_layout(build_ir(parts))
        self.assertEqual(len(ir.mechanical.component_placements), len(parts))
        self.assertEqual({p.ref_des for p in ir.mechanical.component_placements},
                         {c.ref_des for c in parts})

    def test_existing_placements_are_preserved_verbatim(self):
        hand = MechanicalPlacement(
            ref_des="U1",
            label="hand placed",
            category="Microcontroller",
            position=MechanicalVector3(1.0, 2.0, 3.0),
            size=MechanicalVector3(4.0, 5.0, 6.0),
        )
        parts = [component("U1", category="Microcontroller"),
                 component("S1", category="Sensor")]
        ir = enrich_mechanical_layout(
            build_ir(parts, mechanical=MechanicalNotes(component_placements=[hand]))
        )
        kept = placements_by_ref(ir)["U1"]
        self.assertIs(kept, hand)
        self.assertEqual(kept.label, "hand placed")
        self.assertEqual(kept.position, MechanicalVector3(1.0, 2.0, 3.0))
        self.assertEqual(len(ir.mechanical.component_placements), 2)

    def test_a_placement_for_an_unknown_ref_des_is_left_in_place(self):
        # KNOWN-BAD: the mechanical block carries a placement for a component
        # that is not in the BOM. Nothing prunes it, so the count exceeds the
        # component count.
        ghost = MechanicalPlacement(ref_des="GHOST")
        ir = enrich_mechanical_layout(
            build_ir([component("U1", category="Microcontroller")],
                     mechanical=MechanicalNotes(component_placements=[ghost]))
        )
        self.assertIn("GHOST", placements_by_ref(ir))
        self.assertEqual(len(ir.mechanical.component_placements), 2)
        self.assertEqual(ir.assembly_metadata["component_placement_count"], 2)

    def test_existing_spatial_relationships_are_not_regenerated(self):
        existing = derive_spatial_relationships(
            [MechanicalPlacement(ref_des="U1", category="Microcontroller"),
             MechanicalPlacement(ref_des="S1",
                                 position=MechanicalVector3(9.0, 0.0, 0.0))]
        )
        self.assertEqual(len(existing), 1)
        ir = enrich_mechanical_layout(
            build_ir([component("U1", category="Microcontroller"),
                      component("S1", category="Sensor")],
                     mechanical=MechanicalNotes(spatial_relationships=existing))
        )
        self.assertIs(ir.mechanical.spatial_relationships, existing)

    def test_relationship_endpoints_all_resolve_to_real_placements(self):
        parts = [component("U1", category="Microcontroller"),
                 component("DS1", name="OLED Display", category="Display"),
                 component("BEZ1", name="Front Bezel", category="3D Print"),
                 component("E1", name="Enclosure Shell", category="3D Print")]
        ir = enrich_mechanical_layout(build_ir(parts))
        refs = set(placements_by_ref(ir))
        for row in ir.mechanical.spatial_relationships:
            self.assertIn(row.source_ref_des, refs)
            self.assertIn(row.target_ref_des, refs)
            self.assertIn(row.axis, ("X", "Y", "Z"))

    def test_mounting_face_tracks_the_front_threshold(self):
        parts = [component("DS1", name="OLED Display", category="Display"),
                 component("U1", name="ESP32", category="Microcontroller")]
        ir = enrich_mechanical_layout(build_ir(parts))
        by_ref = placements_by_ref(ir)
        depth = ir.mechanical.render_dimensions.y_mm
        for placement in by_ref.values():
            expected = ("front" if placement.position.y_mm < -depth * 0.32
                        else "internal")
            self.assertEqual(placement.mounting_face, expected)
        self.assertEqual(by_ref["DS1"].mounting_face, "front")
        self.assertEqual(by_ref["U1"].mounting_face, "internal")

    def test_metadata_counts_match_the_document(self):
        parts = [component("U1", category="Microcontroller"),
                 component("S1", category="Sensor")]
        ir = enrich_mechanical_layout(build_ir(parts))
        self.assertEqual(ir.assembly_metadata["component_placement_count"],
                         len(ir.mechanical.component_placements))
        self.assertEqual(ir.assembly_metadata["spatial_relationship_count"],
                         len(ir.mechanical.spatial_relationships))
        self.assertEqual(ir.assembly_metadata["render_dimensions"],
                         ir.mechanical.render_dimensions.to_dict())

    def test_existing_metadata_is_merged_not_replaced(self):
        ir = build_ir([component("U1", category="Microcontroller")])
        ir.assembly_metadata = {"pipeline": "upstream"}
        enrich_mechanical_layout(ir)
        self.assertEqual(ir.assembly_metadata["pipeline"], "upstream")
        self.assertIn("render_dimensions", ir.assembly_metadata)

    def test_notes_and_labels_come_from_the_component(self):
        part = component("U1", name="ESP32 Board", category="Microcontroller",
                         rationale="runs the loop")
        ir = enrich_mechanical_layout(build_ir([part]))
        placement = placements_by_ref(ir)["U1"]
        self.assertEqual(placement.label, "ESP32 Board")
        self.assertEqual(placement.category, "Microcontroller")
        self.assertEqual(placement.notes, "runs the loop")
        self.assertEqual(placement.orientation_deg.x_deg, 0.0)

    def test_enrichment_is_deterministic_and_idempotent(self):
        parts = [component("U1", name="ESP32", category="Microcontroller"),
                 component("DS1", name="OLED Display", category="Display"),
                 component("SW1", name="Push Button", category="Passives"),
                 component("E1", name="Enclosure Shell", category="3D Print")]
        first = enrich_mechanical_layout(build_ir(parts)).to_dict()
        second = enrich_mechanical_layout(build_ir(parts)).to_dict()
        self.assertEqual(first, second)
        # Re-enriching an already enriched IR must not change it.
        ir = enrich_mechanical_layout(build_ir(parts))
        again = enrich_mechanical_layout(ir).to_dict()
        self.assertEqual(again, first)

    def test_duplicate_ref_des_collapses_to_one_placement(self):
        # KNOWN-BAD: two BOM lines share a ref_des. The first generated
        # placement claims it, and because the dedup set is snapshotted before
        # the loop the second is emitted too - both at identical coordinates.
        parts = [component("S1", name="Probe", category="Sensor"),
                 component("S1", name="Probe", category="Sensor")]
        ir = enrich_mechanical_layout(build_ir(parts))
        rows = [p for p in ir.mechanical.component_placements if p.ref_des == "S1"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].position, rows[1].position)  # perfectly co-located

    def test_selfcheck_fixture_agrees_with_the_public_helpers(self):
        # Cross-check: the module's own CLI selfcheck must be consistent with
        # what the individual helpers produce for the same inputs.
        self.assertEqual(main(["--selfcheck"]), 0)


class GeneratedLayoutInvariantTests(unittest.TestCase):
    """Substitutes for property-based testing: `hypothesis` is unavailable and
    must not be added, so invariants are checked over small enumerated domains
    (itertools) and pseudo-random component sets from random.Random(SEED) with
    SEED = 20260719 - identical on every run.

    Scope note: the checked invariant is containment on X (plus Y for parts the
    module itself marks 'internal'). Front-mounted parts are placed deliberately
    proud of the front face, so Y containment does not apply to them. Z
    containment is NOT checked here because it is genuinely violated - see
    DisplayHeightDefectTests. Global non-overlap is likewise not an invariant of
    this module - see PowerRowDefectTests.
    """

    CATEGORIES = ["Microcontroller", "Sensor", "Display", "Communication",
                  "Passives"]

    def _random_bom(self, rng):
        # At most three parts per category: beyond that the row-spreading
        # heuristics are known to break (see PowerRowDefectTests).
        parts = []
        index = 0
        for category in self.CATEGORIES:
            for _ in range(rng.randint(0, 3)):
                index += 1
                parts.append(component("C%d" % index, name="Part %d" % index,
                                       category=category))
        rng.shuffle(parts)
        return parts

    def _assert_contained(self, ir, axes=("x",)):
        dims = ir.mechanical.render_dimensions
        for placement in ir.mechanical.component_placements:
            for axis in axes:
                low, high = extents(placement, axis)
                half = getattr(dims, axis + "_mm") / 2.0
                self.assertGreaterEqual(low, -half - 1e-6,
                                        "%s escapes -%s" % (placement.ref_des, axis))
                self.assertLessEqual(high, half + 1e-6,
                                     "%s escapes +%s" % (placement.ref_des, axis))
            if placement.mounting_face == "internal":
                low, high = extents(placement, "y")
                half = dims.y_mm / 2.0
                self.assertGreaterEqual(low, -half - 1e-6, placement.ref_des)
                self.assertLessEqual(high, half + 1e-6, placement.ref_des)

    def test_random_boms_stay_inside_the_envelope(self):
        rng = random.Random(SEED)
        for trial in range(120):
            parts = self._random_bom(rng)
            if not parts:
                continue
            with self.subTest(trial=trial, refs=[p.ref_des for p in parts]):
                self._assert_contained(enrich_mechanical_layout(build_ir(parts)))

    def test_exhaustive_small_category_pairs_stay_inside_the_envelope(self):
        for first, second in itertools.product(self.CATEGORIES, repeat=2):
            with self.subTest(first=first, second=second):
                parts = [component("A1", name="A", category=first),
                         component("B1", name="B", category=second)]
                self._assert_contained(enrich_mechanical_layout(build_ir(parts)))

    def test_exhaustive_counts_of_one_category_stay_inside_the_envelope(self):
        for category in self.CATEGORIES:
            for count in range(1, 4):
                with self.subTest(category=category, count=count):
                    parts = [component("C%d" % i, name="P%d" % i, category=category)
                             for i in range(count)]
                    self._assert_contained(enrich_mechanical_layout(build_ir(parts)))

    def test_random_boms_always_produce_complete_well_formed_placements(self):
        rng = random.Random(SEED)
        valid_layers = {"electrical", "mechanism", "print", "enclosure",
                        "structural"}
        for trial in range(120):
            parts = self._random_bom(rng)
            if not parts:
                continue
            ir = enrich_mechanical_layout(build_ir(parts))
            with self.subTest(trial=trial):
                self.assertEqual(len(ir.mechanical.component_placements), len(parts))
                for placement in ir.mechanical.component_placements:
                    self.assertIn(placement.layer, valid_layers)
                    self.assertIn(placement.mounting_face, ("front", "internal"))
                    for axis in ("x_mm", "y_mm", "z_mm"):
                        self.assertGreater(getattr(placement.size, axis), 0.0)
                        self.assertEqual(round(getattr(placement.position, axis), 2),
                                         getattr(placement.position, axis))

    def test_module_makes_no_non_overlap_guarantee(self):
        # Recorded rather than asserted as desirable: a display and a sensor on
        # a default 92mm envelope share volume. This documents that callers must
        # run the constraint solver (harnesscad.agents.generation.layout_solver)
        # if they need collision-free placements.
        parts = [component("DS1", name="OLED Display", category="Display"),
                 component("S1", name="Probe", category="Sensor")]
        ir = enrich_mechanical_layout(build_ir(parts))
        by_ref = placements_by_ref(ir)
        self.assertTrue(boxes_overlap(by_ref["DS1"], by_ref["S1"]))


class DisplayHeightDefectTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_bug_display_pokes_through_the_top_of_a_minimum_envelope(self):
        # REAL BUG. Reproducing input: a single
        #   ComponentInstance(ref_des="DS1", name="OLED Display",
        #                     category="Display")
        # with no render_dimensions, so infer_render_dimensions returns the
        # clamped minimum envelope 92 x 48 x 30. placement_position puts the
        # display at z = +height*0.24 = +7.2 and placement_size gives it a fixed
        # 18mm height, so its top edge is at 16.2mm against a half-height of
        # 15.0mm: the screen protrudes 1.2mm through the lid. The z offset is a
        # fraction of the envelope but the size is absolute, so the two do not
        # scale together; the defect appears for every envelope with
        # height < 34.6mm, which includes the module's own minimum.
        ir = enrich_mechanical_layout(
            build_ir([component("DS1", name="OLED Display", category="Display")])
        )
        dims = ir.mechanical.render_dimensions
        self.assertEqual((dims.x_mm, dims.y_mm, dims.z_mm), (92.0, 48.0, 30.0))
        low, high = extents(placements_by_ref(ir)["DS1"], "z")
        self.assertLessEqual(high, dims.z_mm / 2.0 + 1e-6)
        self.assertGreaterEqual(low, -dims.z_mm / 2.0 - 1e-6)

    def test_passing_counterpart_a_tall_envelope_contains_the_display(self):
        ir = enrich_mechanical_layout(
            build_ir([component("DS1", name="OLED Display", category="Display")],
                     mechanical=MechanicalNotes(
                         render_dimensions=MechanicalVector3(100.0, 50.0, 60.0)))
        )
        _, high = extents(placements_by_ref(ir)["DS1"], "z")
        self.assertLessEqual(high, 30.0 + 1e-6)


class PowerRowDefectTests(unittest.TestCase):
    """Two real defects in the ``key == "power"`` row heuristic, kept as
    expectedFailure so the code is not changed to make them pass."""

    @staticmethod
    def _four_power_parts_ir():
        parts = [component("P%d" % i, name="Regulator", category="Power")
                 for i in range(4)]
        return enrich_mechanical_layout(build_ir(parts))

    @unittest.expectedFailure
    def test_bug_fourth_power_part_is_pushed_outside_the_enclosure(self):
        # REAL BUG. Reproducing input: four ComponentInstances with
        # category="Power", name="Regulator", ref_des P0..P3 and no
        # render_dimensions, i.e. the count-scaled envelope x_mm = 98.
        # placement_position's power branch is -width*0.28 + index*width*0.22,
        # which is unbounded in `index`; at index 3 the centre is x = +37.24
        # with a 42mm-wide preset, so the right edge lands at 58.24mm against a
        # half-width of 49.0mm - the part protrudes 9.24mm through the wall.
        # Every other row heuristic (buttons, sensors, structural, remaining)
        # uses _row_position, which spreads over a bounded span; only the power
        # branch multiplies the index without a clamp.
        ir = self._four_power_parts_ir()
        half_width = ir.mechanical.render_dimensions.x_mm / 2.0
        for placement in ir.mechanical.component_placements:
            low, high = extents(placement, "x")
            self.assertGreaterEqual(low, -half_width - 1e-6, placement.ref_des)
            self.assertLessEqual(high, half_width + 1e-6, placement.ref_des)

    @unittest.expectedFailure
    def test_bug_power_parts_in_a_row_interpenetrate(self):
        # REAL BUG (same reproducing input as above). The power row pitch is
        # width*0.22 = 21.56mm while the power size preset is 42mm wide, so
        # consecutive regulators overlap by ~20mm in X while sharing an
        # identical Y and Z. Any two adjacent power parts occupy the same
        # volume, which is not a plausible seed layout even for a heuristic.
        ir = self._four_power_parts_ir()
        rows = ir.mechanical.component_placements
        for left, right in itertools.combinations(rows, 2):
            self.assertFalse(boxes_overlap(left, right),
                             "%s overlaps %s" % (left.ref_des, right.ref_des))


class TokenMatchingDefectTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_bug_capacitor_is_classified_as_a_button(self):
        # REAL BUG. Reproducing input:
        #   ComponentInstance(ref_des="C1", part_number="CAP-100",
        #                     name="100uF Capacitor", category="Passives")
        # placement_size and placement_position both test for the bare substring
        # "cap" (intended for "button cap") against the concatenated
        # ref_des/name/part_number/category text. "Capacitor" and "CAP-100"
        # contain it, so an electrolytic capacitor is given the 10x7x10 button
        # envelope and placed in the front-face button row at y = -depth*0.43,
        # where it is then marked mounting_face="front" - a bulk capacitor
        # sticking out of the user-facing panel. Same trap fires for any part
        # whose text contains "cap" (e.g. "Capacitive Soil Sensor",
        # "capacity"). The fix would be to match "button cap" as a phrase.
        cap = component("C1", name="100uF Capacitor", category="Passives",
                        part_number="CAP-100")
        size = placement_size(cap, DIMS)
        self.assertNotEqual((size.x_mm, size.y_mm, size.z_mm), (10.0, 7.0, 10.0))

    def test_passing_counterpart_a_real_button_cap_is_button_sized(self):
        cap = component("SW1", name="Button Cap", category="Mechanical")
        size = placement_size(cap, DIMS)
        self.assertEqual((size.x_mm, size.y_mm, size.z_mm), (10.0, 7.0, 10.0))


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
