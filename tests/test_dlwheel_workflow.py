"""Tests for quality.dlwheel_workflow (paper 112 seven-stage schema)."""

import unittest

from quality import dlwheel_workflow as wf


class SchemaTests(unittest.TestCase):
    def test_seven_stages(self):
        self.assertEqual(len(wf.STAGES), 7)
        self.assertEqual([s.number for s in wf.STAGES], list(range(1, 8)))

    def test_get_stage(self):
        self.assertEqual(wf.get_stage(1).name, "2D generative design")
        self.assertEqual(wf.get_stage(5).name, "CAE automation")
        with self.assertRaises(ValueError):
            wf.get_stage(99)

    def test_groups(self):
        gen = wf.stages_in_group("design generation")
        ev = wf.stages_in_group("design evaluation")
        self.assertEqual([s.number for s in gen], [1, 2, 3, 4])
        self.assertEqual([s.number for s in ev], [5, 6, 7])
        with self.assertRaises(ValueError):
            wf.stages_in_group("nope")


class OrderValidationTests(unittest.TestCase):
    def test_canonical_valid(self):
        self.assertTrue(wf.validate_order(wf.canonical_order()))

    def test_topological_equals_canonical(self):
        self.assertEqual(wf.topological_order(), list(range(1, 8)))

    def test_out_of_order_fails(self):
        # Running stage 5 before stage 4 (needs cad_3d_models) is invalid.
        with self.assertRaises(ValueError):
            wf.validate_order([1, 2, 3, 5])

    def test_duplicate_fails(self):
        with self.assertRaises(ValueError):
            wf.validate_order([1, 1])

    def test_unknown_stage_fails(self):
        with self.assertRaises(ValueError):
            wf.validate_order([1, 42])

    def test_partial_prefix_valid(self):
        self.assertTrue(wf.validate_order([1, 2, 3, 4]))


class UpstreamTests(unittest.TestCase):
    def test_upstream_of_last(self):
        # Stage 7 transitively depends on everything before it.
        self.assertEqual(wf.required_upstream(7), [1, 2, 3, 4, 5, 6])

    def test_upstream_of_first(self):
        self.assertEqual(wf.required_upstream(1), [])

    def test_upstream_of_six(self):
        # Stage 6 needs modal_labels (5) and latent_encoder (2); tracing back
        # pulls in 1..5.
        self.assertEqual(wf.required_upstream(6), [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
