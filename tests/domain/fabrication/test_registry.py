"""The manufacturing surface: can this be made, and how?"""

import unittest

from harnesscad.domain.fabrication import registry as F


class TestDiscovery(unittest.TestCase):
    def test_discovers_more_than_five_real_modules(self):
        self.assertGreater(len(F.routed_modules()), 5, F.routed_modules())

    def test_every_fabrication_module_has_a_route(self):
        self.assertEqual(F.unadapted(), [])

    def test_discovery_is_deterministic(self):
        self.assertEqual(F.discover(), F.discover())


class TestWorkflows(unittest.TestCase):
    def test_the_taxonomy_is_real(self):
        ids = {w["id"] for w in F.workflows()}
        self.assertIn("fdm_3d_printing", ids)
        self.assertIn("ender3", F.machines())

    def test_feasibility_findings_are_per_process(self):
        part = F.part_spec((100.0, 60.0, 8.0), volume_mm3=48000.0)
        findings = F.analyze("fdm_3d_printing", part, machine_id="ender3",
                             material="pla")
        self.assertTrue(findings)
        checks = {f["check"] for f in findings}
        self.assertIn("machine_fit", checks)
        self.assertIn("print_time", checks)

    def test_a_part_too_big_for_the_bed_is_an_honest_finding(self):
        huge = F.part_spec((5000.0, 5000.0, 5000.0), volume_mm3=1e11)
        findings = F.analyze("fdm_3d_printing", huge, machine_id="ender3")
        self.assertFalse(all(f["ok"] for f in findings))

    def test_comparison_is_a_table_not_a_score(self):
        table = F.compare(["fdm_3d_printing", "laser_cut_interlocking"])
        self.assertEqual(set(table),
                         {"fdm_3d_printing", "laser_cut_interlocking"})

    def test_analysis_is_deterministic(self):
        part = F.part_spec((100.0, 60.0, 8.0), volume_mm3=48000.0)
        self.assertEqual(F.analyze("fdm_3d_printing", part, machine_id="ender3"),
                         F.analyze("fdm_3d_printing", part, machine_id="ender3"))


class TestReadinessGate(unittest.TestCase):
    def test_a_watertight_single_body_is_ready(self):
        report = F.readiness({"hole_count": 0, "component_count": 1,
                              "volume_mm3": 48000.0})
        self.assertTrue(report["ready"])

    def test_a_leaking_mesh_is_not_ready(self):
        report = F.readiness({"hole_count": 12, "component_count": 1,
                              "volume_mm3": 48000.0}, allow_warnings=False)
        self.assertFalse(report["ready"])
        self.assertTrue(report["findings"])


class TestFlatpackAndBricks(unittest.TestCase):
    def test_a_cabinet_decomposes_into_panels(self):
        ps = F.panels(600.0, 400.0, 300.0, 18.0)
        self.assertGreater(len(ps), 3)
        self.assertGreater(ps[0]["total_material_area"], 0.0)

    def test_a_panel_bigger_than_the_bed_must_be_split(self):
        tight = F.nest(600.0, 400.0, 300.0, 18.0, bed_w=400.0, bed_h=350.0)
        roomy = F.nest(600.0, 400.0, 300.0, 18.0, bed_w=2000.0, bed_h=2000.0)
        self.assertTrue(tight["needs_split"])
        self.assertFalse(roomy["needs_split"])

    def test_a_bed_that_cannot_fit_the_short_side_FAILS_rather_than_pretends(self):
        # Splitting the long side cannot rescue a panel whose SHORT side already
        # exceeds the bed. The module says so instead of emitting a bad layout.
        with self.assertRaises(ValueError):
            F.nest(600.0, 400.0, 300.0, 18.0, bed_w=100.0, bed_h=100.0)

    def test_bricks_cover_the_voxels_exactly(self):
        voxels = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
        laid = F.bricks(voxels)
        self.assertTrue(laid)
        for brick in laid:
            self.assertTrue(brick["valid_part"])
            self.assertTrue(brick["covers_exactly"])

    def test_brick_colours_come_from_the_lego_palette(self):
        voxels = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
        colors = F.brick_colors(voxels, {c: [(196, 40, 27)] for c in voxels})
        self.assertEqual(colors, ["bright_red"])

    def test_legolization_is_deterministic_in_its_seed(self):
        voxels = [(0, 0, 0), (1, 0, 0)]
        self.assertEqual(F.bricks(voxels, seed=1), F.bricks(voxels, seed=1))


class TestFeaturesAndExport(unittest.TestCase):
    def test_difficulty_from_a_feature_histogram(self):
        easy = F.difficulty({"through_hole": 2})
        hard = F.difficulty({"through_hole": 20, "pocket": 8, "slot": 6})
        self.assertEqual(easy["level"], "easy")
        self.assertNotEqual(hard["level"], "easy")

    def test_a_feature_only_declares_the_attributes_it_has(self):
        out = F.feature_attributes("through_hole", {"diameter": 6.0,
                                                    "depth": 8.0})
        self.assertIn("diameter", out["declares"])
        self.assertEqual(out["attributes"]["diameter"], 6.0)

    def test_export_is_PLANNED_and_never_executed(self):
        plan = F.export_plan("cube(10);", "stl")
        self.assertEqual(plan["argv"][0], "openscad")
        self.assertIn("stl", plan["format"])
        self.assertTrue(plan["cache_key"])
        # same source + format -> same key (a plan, not a side effect)
        self.assertEqual(plan["cache_key"],
                         F.export_plan("cube(10);", "stl")["cache_key"])

    def test_an_export_result_is_classified_not_rerun(self):
        self.assertTrue(F.classify_export(0, "")["success"])
        self.assertFalse(F.classify_export(1, "ERROR: bad")["success"])


if __name__ == "__main__":
    unittest.main()
