import unittest

from harnesscad.domain.programs.review import errorgen as eg
from harnesscad.domain.programs.review import taxonomy as tax
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


class TestErrorGen(unittest.TestCase):
    def test_all_eight_injectable(self):
        types = eg.injectable_types(PROGRAM)
        self.assertEqual({t.id for t in types}, set(tax.ids()))

    def test_injection_changes_source(self):
        for et in tax.ERROR_TYPES:
            inj = eg.inject(PROGRAM, et, seed=1)
            self.assertIsNotNone(inj, et.label)
            self.assertNotEqual(inj.source.strip(), PROGRAM.strip(), et.label)

    def test_deterministic(self):
        a = eg.inject(PROGRAM, tax.SIZE_ERROR, seed=7)
        b = eg.inject(PROGRAM, tax.SIZE_ERROR, seed=7)
        self.assertEqual(a.source, b.source)

    def test_detector_recovers_ground_truth(self):
        # Inject each error type, then the deterministic detector must recover
        # BOTH the error type and the block id (the paper's Acc criterion).
        for i, et in enumerate(tax.ERROR_TYPES):
            inj = eg.inject(PROGRAM, et, seed=i)
            review = detect(inj.source, PROGRAM)
            self.assertEqual(review.primary.error_type.id, inj.error_type.id,
                             f"type mismatch for {et.label}")
            self.assertEqual(review.primary.block_id, inj.block_id,
                             f"block mismatch for {et.label}")

    def test_inject_all(self):
        samples = eg.inject_all(PROGRAM, seed=0)
        self.assertEqual(len(samples), 8)

    def test_not_injectable_returns_none(self):
        # A program with no control flow cannot host a logic error.
        simple = "cube([1, 1, 1]);"
        self.assertNotIn(tax.LOGIC_ERROR, eg.injectable_types(simple))
        self.assertIsNone(eg.inject(simple, tax.LOGIC_ERROR))


if __name__ == "__main__":
    unittest.main()
