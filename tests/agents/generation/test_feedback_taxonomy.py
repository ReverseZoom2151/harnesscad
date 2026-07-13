import unittest

from harnesscad.agents.generation.feedback_taxonomy import (
    FEEDBACK_TYPES, ERROR_TYPES, classify_feedback, feedback_distribution,
    normalize_error_type, error_distribution, majority_vote,
)


class TestFeedbackClassify(unittest.TestCase):
    def test_dimensional(self):
        self.assertEqual(classify_feedback("Increase the height and reduce width"),
                         "dimensional")

    def test_positional(self):
        self.assertEqual(classify_feedback("center the object and align with base"),
                         "positional")

    def test_structural(self):
        self.assertEqual(classify_feedback("make it cylindrical and adjust the corner shape"),
                         "structural")

    def test_default_structural(self):
        self.assertEqual(classify_feedback("qwerty zxcv"), "structural")

    def test_three_types(self):
        self.assertEqual(set(FEEDBACK_TYPES), {"structural", "dimensional", "positional"})


class TestFeedbackDistribution(unittest.TestCase):
    def test_distribution(self):
        d = feedback_distribution([
            "make it cylindrical",
            "increase height",
            "increase width",
        ])
        self.assertEqual(d["total"], 3)
        self.assertEqual(d["counts"]["dimensional"], 2)
        self.assertAlmostEqual(d["fractions"]["structural"], 1 / 3)


class TestErrorTypes(unittest.TestCase):
    def test_five_types(self):
        self.assertEqual(len(ERROR_TYPES), 5)

    def test_normalize_aliases(self):
        self.assertEqual(normalize_error_type("structural"), "structural_configuration")
        self.assertEqual(normalize_error_type("Failure Rate"), "failure")
        self.assertEqual(normalize_error_type("spatial"), "spatial_precision")

    def test_unknown(self):
        with self.assertRaises(ValueError):
            normalize_error_type("gibberish")

    def test_distribution(self):
        d = error_distribution(["structural", "structural", "logical", "correct"])
        self.assertEqual(d["counts"]["structural_configuration"], 2)
        self.assertAlmostEqual(d["fractions"]["logical"], 0.25)


class TestMajorityVote(unittest.TestCase):
    def test_majority(self):
        self.assertEqual(majority_vote(["structural", "structural", "logical"]),
                         "structural_configuration")

    def test_tie_breaks_by_order(self):
        # tie between structural_configuration and spatial_precision -> structural first
        self.assertEqual(majority_vote(["structural", "spatial"]),
                         "structural_configuration")

    def test_empty(self):
        with self.assertRaises(ValueError):
            majority_vote([])


if __name__ == "__main__":
    unittest.main()
