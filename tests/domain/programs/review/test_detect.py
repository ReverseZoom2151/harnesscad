import unittest

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


class TestDetect(unittest.TestCase):
    def test_identical_is_no_error(self):
        review = detect(PROGRAM, PROGRAM)
        self.assertTrue(review.ok)
        self.assertEqual(review.primary.error_type.id, tax.NO_ERROR.id)
        self.assertEqual(review.detections, [])

    def test_primitive_error(self):
        bad = PROGRAM.replace("cube([size, size, 2])", "cylinder([size, size, 2])")
        r = detect(bad, PROGRAM)
        self.assertEqual(r.primary.error_type.id, tax.PRIMITIVE_ERROR.id)
        self.assertEqual(r.primary.block_id, 2)

    def test_rotation_error(self):
        bad = PROGRAM.replace("rotate([0, 0, 45])", "rotate([0, 0, 330])")
        r = detect(bad, PROGRAM)
        self.assertEqual(r.primary.error_type.id, tax.ROTATION_ERROR.id)
        self.assertEqual(r.primary.block_id, 3)

    def test_position_error(self):
        bad = PROGRAM.replace("translate([10, 0, 0]) sphere",
                              "translate([40, 0, 0]) sphere")
        r = detect(bad, PROGRAM)
        self.assertEqual(r.primary.error_type.id, tax.POSITION_ERROR.id)
        self.assertEqual(r.primary.block_id, 4)

    def test_size_error(self):
        bad = PROGRAM.replace("sphere(r=4)", "sphere(r=9)")
        r = detect(bad, PROGRAM)
        self.assertEqual(r.primary.error_type.id, tax.SIZE_ERROR.id)
        self.assertEqual(r.primary.block_id, 4)

    def test_constant_error(self):
        bad = PROGRAM.replace("size = 10;", "size = 25;")
        r = detect(bad, PROGRAM)
        self.assertEqual(r.primary.error_type.id, tax.CONSTANT_ERROR.id)
        self.assertEqual(r.primary.block_id, 0)

    def test_logic_error(self):
        bad = PROGRAM.replace("[0 : 3]", "[0 : 9]")
        r = detect(bad, PROGRAM)
        self.assertEqual(r.primary.error_type.id, tax.LOGIC_ERROR.id)
        self.assertEqual(r.primary.block_id, 5)

    def test_missing_block(self):
        bad = PROGRAM.replace("translate([10, 0, 0]) sphere(r=4);", "")
        r = detect(bad, PROGRAM)
        self.assertEqual(r.primary.error_type.id, tax.MISSING_BLOCK.id)
        self.assertEqual(r.primary.block_id, 4)

    def test_redundant_block(self):
        bad = PROGRAM + "\ntranslate([137, 149, 151]) cube([1, 1, 1]);\n"
        r = detect(bad, PROGRAM)
        self.assertEqual(r.primary.error_type.id, tax.REDUNDANT_BLOCK.id)

    def test_to_dict(self):
        r = detect(PROGRAM, PROGRAM)
        d = r.to_dict()
        self.assertIn("primary", d)
        self.assertTrue(d["ok"])


if __name__ == "__main__":
    unittest.main()
