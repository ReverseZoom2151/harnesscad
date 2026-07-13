"""The catalogue surface: parts + the standards knowledge base."""

import unittest

from harnesscad.core.loop import HarnessSession
from harnesscad.domain.library import registry as C
from harnesscad.io.backends.stub import StubBackend


class TestDiscovery(unittest.TestCase):
    def test_discovers_more_than_five_real_modules(self):
        routed = C.routed_modules()
        self.assertGreater(len(routed), 5, routed)

    def test_every_library_and_standards_module_has_a_route(self):
        self.assertEqual(C.unadapted(), [])

    def test_discovery_is_deterministic(self):
        self.assertEqual(C.discover(), C.discover())


class TestPartsCatalogue(unittest.TestCase):
    def test_the_catalogue_ships_execution_verified_parts(self):
        names = C.part_names()
        self.assertGreater(len(names), 3)
        for card in C.catalogue().cards():
            self.assertTrue(card.verified, card.name)

    def test_retrieval_is_by_function(self):
        hits = C.find_part("mounting")
        self.assertTrue(hits)
        self.assertIn("mounting", hits[0]["tags"])

    def test_a_part_instantiates_to_ops_that_apply(self):
        ops = C.instantiate("flange")
        session = HarnessSession(StubBackend())
        result = session.apply_ops(ops)
        self.assertTrue(result.ok, [d.message for d in result.diagnostics])

    def test_an_out_of_range_parameter_is_refused(self):
        with self.assertRaises(ValueError):
            C.instantiate("flange", thickness=-5.0)

    def test_an_unknown_part_raises(self):
        with self.assertRaises(KeyError):
            C.instantiate("teleporter")

    def test_a_family_sweep_verifies_every_variant(self):
        manifest = C.family("flange", {"thickness": [6.0, 8.0, 10.0]})
        self.assertEqual(len(manifest.entries), 3)
        self.assertEqual(len(manifest.accepted), 3)
        # provenance: each accepted variant carries a digest
        for entry in manifest.accepted:
            self.assertTrue(entry.digest)

    def test_a_family_of_an_unknown_part_raises(self):
        with self.assertRaises(C.CatalogueError):
            C.family("teleporter", {"x": [1.0]})


class TestStandards(unittest.TestCase):
    def test_thread_lookup_is_the_table_not_a_guess(self):
        t = C.thread("M6")
        self.assertEqual(t["radius"], 3.0)
        self.assertEqual(t["units"], "mm")

    def test_an_unknown_thread_raises(self):
        with self.assertRaises(KeyError):
            C.thread("M999")

    def test_heatsert_bore_schedule_and_wall_fit(self):
        loose = C.heatsert("M4", wall_thickness=30.0)
        tight = C.heatsert("M4", wall_thickness=1.0)
        self.assertTrue(loose["fits_in_wall"])
        self.assertFalse(tight["fits_in_wall"])
        self.assertGreater(loose["bore_depth"], 0.0)

    def test_aci_colour_round_trips(self):
        by_name = C.aci("red")
        by_index = C.aci(by_name["index"])
        by_rgb = C.aci(list(by_name["rgb"]))
        self.assertEqual(by_name, by_index)
        self.assertEqual(by_name, by_rgb)

    def test_rules_are_ingested_and_conflicts_detected(self):
        rules = C.ingest_rules(
            "Wall thickness shall be at least 1.2 mm.\n"
            "Wall thickness shall be at most 0.8 mm.\n",
            "DFM", "1.0")
        self.assertTrue(rules)
        self.assertTrue(C.rule_conflicts(rules))

    def test_a_fresh_standards_registry_is_empty(self):
        self.assertEqual(C.standards().standards(), [])


class TestNamesAndConcepts(unittest.TestCase):
    def test_default_cad_names_are_dropped(self):
        out = C.normalize_names(["Part1", "bracket_left", "Boss-Extrude1"])
        self.assertEqual(out["user_names"], ["bracket_left"])
        self.assertIn("Part1", out["default_names"])

    def test_ppmi_scores_names_that_co_occur(self):
        model = C.name_semantics([
            ["hex_bolt", "hex_nut", "flat_washer"],
            ["hex_bolt", "cap_screw"],
            ["cap_screw", "hex_nut"],
        ])
        self.assertGreater(model.pair_score("hex_bolt", "cap_screw"), 0.0)

    def test_relative_edits_resolve_to_numbers(self):
        self.assertEqual(C.resolve_relative(10.0, "+10%").resolved, 11.0)
        self.assertEqual(C.resolve_relative(10.0, "*1.5").resolved, 15.0)

    def test_a_vague_relative_token_is_not_guessed(self):
        self.assertIsNone(C.resolve_relative(10.0, "a bit bigger"))

    def test_dual_channel_retrieval_fuses_by_rank(self):
        hits = C.retrieve({"a": 0.1, "b": 0.5}, {"a": 0.4, "b": 0.2})
        self.assertEqual([h.item_id for h in hits], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
