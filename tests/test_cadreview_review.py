import unittest

from harnesscad.domain.programs.review import cadreview_errorgen as eg
from harnesscad.domain.programs.review import cadreview_review as rv
from harnesscad.domain.programs.review import cadreview_taxonomy as tax
from harnesscad.domain.programs.review.cadreview_detect import detect

PROGRAM = """
$fn = 32;
size = 10;
module leg(h) { cube([2, 2, h]); }
translate([0, 0, 5]) cube([size, size, 2]);
rotate([0, 0, 45]) cylinder(r=3, h=8);
translate([10, 0, 0]) sphere(r=4);
for (i = [0 : 3]) translate([i * 5, 0, 0]) cube([1, 1, 1]);
"""


class TestReviewReport(unittest.TestCase):
    def test_report_schema_keys(self):
        inj = eg.inject(PROGRAM, tax.ROTATION_ERROR, seed=1)
        rep = rv.build_report(inj.source, PROGRAM).to_dict()
        for key in ("error type", "erroneous code block ID", "feedback",
                    "correct code"):
            self.assertIn(key, rep)
        self.assertEqual(rep["error type"], "Rotation error")

    def test_feedback_word_cap(self):
        inj = eg.inject(PROGRAM, tax.POSITION_ERROR, seed=1)
        rep = rv.build_report(inj.source, PROGRAM)
        self.assertLessEqual(len(rep.feedback.split()), 75)

    def test_no_error_uses_predefined_feedback(self):
        rep = rv.build_report(PROGRAM, PROGRAM, seed=3)
        self.assertEqual(rep.error_type, "No error")
        self.assertIn(rep.feedback, rv.PREDEFINED_FEEDBACK)

    def test_predefined_feedback_count(self):
        self.assertEqual(len(rv.PREDEFINED_FEEDBACK), 10)

    def test_correct_code_is_clean(self):
        inj = eg.inject(PROGRAM, tax.SIZE_ERROR, seed=4)
        rep = rv.build_report(inj.source, PROGRAM)
        self.assertTrue(detect(rep.correct_code, PROGRAM).ok)


class TestScorer(unittest.TestCase):
    def test_diagnostic_reward_both_required(self):
        self.assertEqual(rv.diagnostic_reward("rotation_error", 2,
                                              "rotation_error", 2), 1)
        # right type, wrong block -> 0
        self.assertEqual(rv.diagnostic_reward("rotation_error", 3,
                                              "rotation_error", 2), 0)
        # wrong type, right block -> 0
        self.assertEqual(rv.diagnostic_reward("size_error", 2,
                                              "rotation_error", 2), 0)

    def test_reward_accepts_labels_and_ids(self):
        self.assertEqual(rv.diagnostic_reward("Rotation error", 2,
                                              "rotation_error", 2), 1)

    def test_score_dataset_perfect(self):
        preds, golds = [], []
        for i, et in enumerate(tax.ERROR_TYPES):
            inj = eg.inject(PROGRAM, et, seed=i)
            r = detect(inj.source, PROGRAM)
            preds.append((r.primary.error_type.id, r.primary.block_id))
            golds.append((inj.error_type.id, inj.block_id))
        stats = rv.score_dataset(preds, golds)
        self.assertEqual(stats["n"], 8)
        self.assertEqual(stats["acc"], 1.0)
        self.assertEqual(stats["type_acc"], 1.0)
        self.assertEqual(stats["block_acc"], 1.0)

    def test_score_dataset_partial(self):
        preds = [("size_error", 1), ("rotation_error", 9)]
        golds = [("size_error", 1), ("rotation_error", 3)]
        stats = rv.score_dataset(preds, golds)
        self.assertEqual(stats["acc"], 0.5)
        self.assertEqual(stats["type_acc"], 1.0)
        self.assertEqual(stats["block_acc"], 0.5)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            rv.score_dataset([("a", 1)], [])


if __name__ == "__main__":
    unittest.main()
