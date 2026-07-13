import unittest

from harnesscad.domain.programs.review import errorgen as eg
from harnesscad.domain.programs.review import taxonomy as tax
from harnesscad.domain.programs.review.correct import correct
from harnesscad.domain.programs.review.detect import detect

PROGRAM = """
$fn = 32;
size = 10;
module leg(h) { cube([2, 2, h]); }
translate([0, 0, 5]) cube([size, size, 2]);
rotate([0, 0, 45]) cylinder(r=3, h=8);
translate([10, 0, 0]) sphere(r=4);
for (i = [0 : 3]) translate([i * 5, 0, 0]) cube([1, 1, 1]);
"""


class TestCorrect(unittest.TestCase):
    def test_roundtrip_all_error_types(self):
        # inject -> detect -> correct -> the corrected program must be clean.
        for i, et in enumerate(tax.ERROR_TYPES):
            inj = eg.inject(PROGRAM, et, seed=i + 3)
            review = detect(inj.source, PROGRAM)
            fix = correct(inj.source, PROGRAM, review)
            recheck = detect(fix.source, PROGRAM)
            self.assertTrue(recheck.ok,
                            f"{et.label}: still {recheck.primary.error_type.label}")

    def test_suggestion_per_detection(self):
        inj = eg.inject(PROGRAM, tax.ROTATION_ERROR, seed=1)
        review = detect(inj.source, PROGRAM)
        fix = correct(inj.source, PROGRAM, review)
        self.assertEqual(len(fix.suggestions), len(review.detections))
        self.assertEqual(fix.suggestions[0].fix_action, "fix_rotation")
        self.assertIn("Block", fix.suggestions[0].instruction)

    def test_no_error_leaves_no_suggestions(self):
        review = detect(PROGRAM, PROGRAM)
        fix = correct(PROGRAM, PROGRAM, review)
        self.assertEqual(fix.suggestions, [])

    def test_to_dict(self):
        inj = eg.inject(PROGRAM, tax.SIZE_ERROR, seed=2)
        review = detect(inj.source, PROGRAM)
        d = correct(inj.source, PROGRAM, review).to_dict()
        self.assertIn("source", d)
        self.assertIn("suggestions", d)


if __name__ == "__main__":
    unittest.main()
