"""Tests for PHT-CAD Progressive Hierarchical Tuning curriculum."""

import unittest

from harnesscad.domain.reconstruction import pht_progressive_tuning as pht


class ScheduleTest(unittest.TestCase):
    def test_three_stages_ordered(self):
        sched = pht.schedule()
        self.assertEqual(len(sched), 3)
        self.assertEqual([s.name for s in sched], list(pht.STAGE_NAMES))
        self.assertEqual([s.index for s in sched], [1, 2, 3])

    def test_stage_hyperparams(self):
        self.assertAlmostEqual(pht.stage(1).learning_rate, 1e-4)
        self.assertEqual(pht.stage(1).max_tokens, 4096)
        self.assertEqual(pht.stage(2).max_tokens, 8192)
        self.assertAlmostEqual(pht.stage(3).learning_rate, 2e-5)
        self.assertAlmostEqual(pht.stage(3).replay_fraction, 0.5)

    def test_stage_levels_coarse_to_fine(self):
        self.assertEqual(pht.stage(1).refines, "type")
        self.assertEqual(pht.stage(2).refines, "coarse_params")
        self.assertEqual(pht.stage(3).refines, "fine_params")

    def test_bad_index(self):
        with self.assertRaises(ValueError):
            pht.stage(0)
        with self.assertRaises(ValueError):
            pht.stage(4)


class MixtureTest(unittest.TestCase):
    def test_stage1_no_replay(self):
        sizes = {"single_primitive": 100, "sketch_structural": 200}
        mix = pht.build_mixture(1, sizes)
        self.assertEqual(mix, {"single_primitive": 100})

    def test_stage3_replays_half_of_stage2(self):
        sizes = {"dimensional_annotated": 40, "sketch_structural": 200}
        mix = pht.build_mixture(3, sizes)
        self.assertEqual(mix["dimensional_annotated"], 40)
        self.assertEqual(mix["sketch_structural"], 100)

    def test_stage2_no_replay(self):
        sizes = {"sketch_structural": 200, "single_primitive": 100}
        mix = pht.build_mixture(2, sizes)
        self.assertEqual(mix, {"sketch_structural": 200})


class RefineTest(unittest.TestCase):
    def test_coarse_snaps_to_wide_grid(self):
        p = pht.Prediction("line", level="type")
        p = pht.refine(p, "coarse_params", [123.0, 77.0])
        self.assertEqual(p.level, "coarse_params")
        # snapped to nearest multiple of 50
        self.assertEqual(p.params, [100.0, 100.0])

    def test_fine_keeps_precision(self):
        p = pht.Prediction("line", level="coarse_params", params=[100.0, 100.0])
        p = pht.refine(p, "fine_params", [123.4, 77.6])
        self.assertEqual(p.params, [123.0, 78.0])  # grid 1.0

    def test_no_backward_refine(self):
        p = pht.Prediction("circle", level="fine_params")
        with self.assertRaises(ValueError):
            pht.refine(p, "coarse_params", [1.0])

    def test_unknown_level(self):
        p = pht.Prediction("circle")
        with self.assertRaises(ValueError):
            pht.refine(p, "bogus")

    def test_full_pipeline(self):
        p = pht.run_pipeline("circle", [500.4, 250.9, 33.2])
        self.assertEqual(p.level, "fine_params")
        self.assertEqual(p.kind, "circle")
        self.assertEqual(p.params, [500.0, 251.0, 33.0])


class AblationTest(unittest.TestCase):
    def test_full_pipeline_best(self):
        self.assertAlmostEqual(pht.ablation_accuracy(0.87, None), 0.87)

    def test_stage_drops_monotone(self):
        base = 0.87
        no1 = pht.ablation_accuracy(base, "primitive_perception")
        no2 = pht.ablation_accuracy(base, "structural_perception")
        self.assertAlmostEqual(no1, 0.75)
        self.assertAlmostEqual(no2, 0.72)
        # removing stage 2 hurts more than removing stage 1
        self.assertLess(no2, no1)

    def test_bad_stage(self):
        with self.assertRaises(ValueError):
            pht.ablation_accuracy(0.8, "annotation_geometry_alignment")


if __name__ == "__main__":
    unittest.main()
