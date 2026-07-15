"""Tests for design-state extraction and editability scoring (Arko-T)."""

import unittest

from harnesscad.eval.quality.sequence import design_state as ds


RICH_PROGRAM = [
    {"OP": "sketch", "params": {"plane": "XY"}},
    {"OP": "circle", "params": {"radius": 5.0}},
    {"OP": "extrude", "params": {"height": 10.0}},
    {"OP": "hole", "params": {"diameter": 3.0}, "refs": ["face_top"]},
    {"OP": "symmetric"},
    {"OP": "fillet", "params": {"radius": 1.0}, "refs": ["edge_3"]},
]

BARE_PROGRAM = [
    {"OP": "sketch"},
    {"OP": "extrude"},
]


class ExtractTest(unittest.TestCase):
    def test_features_captured(self):
        state = ds.extract_design_state(RICH_PROGRAM)
        self.assertIn("hole", state.features)
        self.assertIn("fillet", state.features)

    def test_parameters_captured(self):
        state = ds.extract_design_state(RICH_PROGRAM)
        self.assertIn("radius", state.parameters)
        self.assertIn("height", state.parameters)

    def test_constraints_captured(self):
        state = ds.extract_design_state(RICH_PROGRAM)
        self.assertIn("symmetric", state.constraints)

    def test_attachments_captured(self):
        state = ds.extract_design_state(RICH_PROGRAM)
        self.assertIn("face_top", state.attachments)
        self.assertIn("edge_3", state.attachments)

    def test_history_order(self):
        state = ds.extract_design_state(RICH_PROGRAM)
        self.assertEqual(state.history[0], "sketch")
        self.assertEqual(state.history[-1], "fillet")


class EditabilityTest(unittest.TestCase):
    def test_rich_program_full_score(self):
        state = ds.extract_design_state(RICH_PROGRAM)
        self.assertEqual(ds.editability_score(state), 1.0)

    def test_bare_program_low_score(self):
        state = ds.extract_design_state(BARE_PROGRAM)
        # no named params, no features, no constraints, no attachments
        self.assertEqual(ds.editability_score(state), 0.0)


class ConstructionOrderTest(unittest.TestCase):
    def test_canonical_order_scores_one(self):
        state = ds.extract_design_state(RICH_PROGRAM)
        self.assertEqual(ds.construction_order_score(state), 1.0)

    def test_out_of_order_penalized(self):
        prog = [
            {"OP": "fillet"},   # finishing (3)
            {"OP": "sketch"},   # sketch (0)  -> decreasing
            {"OP": "extrude"},  # (1)
        ]
        state = ds.extract_design_state(prog)
        self.assertLess(ds.construction_order_score(state), 1.0)

    def test_short_history_defaults_one(self):
        state = ds.extract_design_state([{"OP": "sketch"}])
        self.assertEqual(ds.construction_order_score(state), 1.0)


if __name__ == "__main__":
    unittest.main()
