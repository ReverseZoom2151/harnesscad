import unittest

from harnesscad.domain.programs import cadreview_blocks as cb

PROGRAM = """
$fn = 32;
size = 10;
module leg(h) { cube([2, 2, h]); }
translate([0, 0, 5]) cube([size, size, 2]);
rotate([0, 0, 45]) cylinder(r=3, h=8);
translate([10, 0, 0]) sphere(r=4);
for (i = [0 : 3]) translate([i * 5, 0, 0]) cube([1, 1, 1]);
"""


class TestBlocks(unittest.TestCase):
    def test_strip_comments(self):
        src = 'a = 1; // trailing\n/* block */ b = 2;'
        out = cb._strip_comments(src)
        self.assertNotIn("trailing", out)
        self.assertNotIn("block", out)
        self.assertIn("a = 1;", out)

    def test_comments_not_stripped_inside_string(self):
        src = 'text("a // b");'
        self.assertIn("a // b", cb._strip_comments(src))

    def test_statement_split(self):
        stmts = cb.split_statements(PROGRAM)
        # 2 assignments + module + 3 shape statements + for = 7
        self.assertEqual(len(stmts), 7)

    def test_segment_block_zero_is_macros(self):
        blocks = cb.segment(PROGRAM)
        self.assertEqual(blocks[0].id, 0)
        self.assertEqual(blocks[0].kind, "assignment")
        self.assertIn("$fn", blocks[0].text)
        self.assertIn("size = 10", blocks[0].text)

    def test_block_kinds(self):
        blocks = cb.segment(PROGRAM)
        kinds = {b.id: b.kind for b in blocks}
        self.assertEqual(kinds[1], "module")
        self.assertEqual(kinds[2], "transform")   # translate ... cube
        self.assertEqual(kinds[3], "transform")   # rotate ... cylinder
        self.assertEqual(kinds[5], "control_flow")  # for ...

    def test_contiguous_ids(self):
        blocks = cb.segment(PROGRAM)
        self.assertEqual([b.id for b in blocks], list(range(len(blocks))))

    def test_annotate_has_block_comments(self):
        text = cb.annotate(PROGRAM)
        self.assertIn("// Block 0", text)
        self.assertIn("// Block 5", text)

    def test_empty_program(self):
        self.assertEqual(cb.segment(""), [])
        self.assertEqual(cb.segment("   \n  "), [])


if __name__ == "__main__":
    unittest.main()
