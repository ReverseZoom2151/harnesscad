import unittest

from harnesscad.eval.quality.geometry.mesh_stability import mesh_stability_metrics
from harnesscad.eval.quality.physics.schedule import GenerationPhase, PhysicsSchedule
from harnesscad.eval.verifiers.standability import evaluate_standability


class StandabilityTests(unittest.TestCase):
    contacts = ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))

    def test_centered_com_has_positive_margin_and_deterministic_samples(self):
        a = evaluate_standability((0, 0, 1), self.contacts)
        b = evaluate_standability((0, 0, 1), reversed(self.contacts))
        self.assertAlmostEqual(a.margin, 1.0)
        self.assertTrue(a.supported)
        self.assertTrue(a.robust)
        self.assertEqual(a, b)
        self.assertEqual(len(a.tilt_samples), 20)

    def test_outside_and_degenerate_support(self):
        outside = evaluate_standability((2, 0, 1), self.contacts)
        self.assertLess(outside.margin, 0)
        self.assertIn("com_outside_support", outside.diagnostics)
        line = evaluate_standability((0, 0, 1), ((-1, 0, 0), (1, 0, 0)))
        self.assertIsNone(line.margin)
        self.assertIn("degenerate_support", line.diagnostics)

    def test_near_edge_com_can_be_supported_but_tilt_unstable(self):
        report = evaluate_standability((0.9999, 0, 10), self.contacts)
        self.assertTrue(report.supported)
        self.assertFalse(report.robust)
        self.assertLess(report.minimum_potential_delta, 0)


class MeshTests(unittest.TestCase):
    def test_planar_mesh_normals_and_bottom_roughness(self):
        vertices = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
        result = mesh_stability_metrics(vertices, ((0, 1, 2), (0, 2, 3)))
        self.assertAlmostEqual(result.normal_inconsistency, 0)
        self.assertAlmostEqual(result.bottom_roughness, 0)
        self.assertEqual(result.adjacent_pairs, 1)

    def test_crease_and_degenerate_input(self):
        vertices = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        result = mesh_stability_metrics(vertices, ((0, 1, 2), (0, 3, 1)))
        self.assertGreater(result.normal_inconsistency, 0)
        empty = mesh_stability_metrics((), ())
        self.assertIsNone(empty.normal_inconsistency)
        self.assertIn("no_bottom_vertices", empty.diagnostics)


class ScheduleTests(unittest.TestCase):
    def test_only_runs_on_refine_cadence(self):
        schedule = PhysicsSchedule(every_n=10, first_refine_iteration=3)
        self.assertFalse(schedule.should_run(GenerationPhase.COARSE, 3))
        self.assertTrue(schedule.should_run(GenerationPhase.REFINE, 3))
        self.assertFalse(schedule.should_run(GenerationPhase.REFINE, 12))
        self.assertTrue(schedule.should_run(GenerationPhase.REFINE, 13))

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            PhysicsSchedule(every_n=0)
        with self.assertRaises(ValueError):
            PhysicsSchedule().should_run(GenerationPhase.REFINE, -1)


if __name__ == "__main__":
    unittest.main()
