import unittest

from harnesscad.data.dataengine.annotation_reconcile import ModalityDescription, reconcile_descriptions
from harnesscad.data.dataengine.command_balance import command_balance
from harnesscad.data.dataengine.reverse_description import ordered_recovery_ratio, verify_with_reflection
from harnesscad.eval.quality.sequence_confidence import CommandConfidence, assess_sequence_confidence


class ReverseDescriptionTests(unittest.TestCase):
    def test_lcs_ratio_handles_reorder_and_empty(self):
        self.assertEqual(ordered_recovery_ratio(["a", "b"], ["b", "a"]), 0.5)
        self.assertEqual(ordered_recovery_ratio([], []), 1.0)

    def test_bounded_reflection_stops_on_pass(self):
        answers = {"v1": ["a"], "v2": ["a", "b"]}
        result = verify_with_reflection(
            ["a", "b"], "v1", lambda text: answers[text],
            lambda text, feedback, ratio: "v2",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(len(result.attempts), 2)

    def test_reflection_exhaustion(self):
        result = verify_with_reflection(
            ["a", "b"], "bad", lambda text: ["x"],
            lambda text, feedback, ratio: text, maximum_reflections=2,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(len(result.attempts), 3)


class ReconcileTests(unittest.TestCase):
    def test_agreement_routes_automatically(self):
        result = reconcile_descriptions([
            ModalityDescription("image", "round flange", 0.9),
            ModalityDescription("point_cloud", "round metal flange", 0.8),
        ])
        self.assertEqual(result.route, "auto_pass")

    def test_conflict_or_missing_modality_routes_to_review(self):
        conflict = reconcile_descriptions([
            ModalityDescription("image", "cube", 0.9),
            ModalityDescription("point_cloud", "cylinder", 0.9),
        ])
        self.assertIn("contradiction:cube/cylinder", conflict.conflicts)
        missing = reconcile_descriptions([ModalityDescription("image", "part", 0.9)])
        self.assertIn("insufficient_modalities", missing.conflicts)


class ConfidenceTests(unittest.TestCase):
    def test_selective_context_names_only_low_fields(self):
        result = assess_sequence_confidence([
            CommandConfidence(0, "extrude", 0.95, {"depth": 0.4, "draft": 0.9})
        ])
        self.assertEqual(result.low_confidence, ("0:extrude:arg:depth",))
        self.assertIn("depth", result.correction_context)

    def test_invalid_confidence_rejected(self):
        with self.assertRaises(ValueError):
            CommandConfidence(0, "x", 2.0, {})


class BalanceTests(unittest.TestCase):
    def test_skew_weights_and_zero_safe_vocabulary(self):
        report = command_balance(
            [["line", "line", "line"], ["circle"]],
            vocabulary=("line", "circle", "spline"),
            rare_frequency=0.3,
        )
        self.assertGreater(report.inverse_weights["circle"], report.inverse_weights["line"])
        self.assertEqual(report.inverse_weights["spline"], 0.0)
        self.assertIn("spline", report.rare_commands)

    def test_empty_is_defined(self):
        report = command_balance([], vocabulary=("line",))
        self.assertEqual(report.frequencies["line"], 0.0)


if __name__ == "__main__":
    unittest.main()
