import unittest

from fabrication.lego_brick_library import (
    LIBRARY,
    ALLOWED_ORIENTED,
    Brick,
    BrickFormatError,
    brick_to_line,
    is_library_part,
    is_raster_ordered,
    is_wellformed_line,
    line_to_brick,
    parse,
    raster_scan_sorted,
    serialize,
    valid_next_chars,
)


class TestLibrary(unittest.TestCase):
    def test_eight_standard_parts(self):
        self.assertEqual(len(LIBRARY), 8)
        for hw in [(1, 1), (1, 2), (1, 4), (1, 6), (1, 8), (2, 2), (2, 4), (2, 6)]:
            self.assertIn(hw, LIBRARY)

    def test_orientation_independent(self):
        self.assertTrue(is_library_part(2, 4))
        self.assertTrue(is_library_part(4, 2))
        self.assertFalse(is_library_part(3, 3))
        self.assertFalse(is_library_part(2, 8))

    def test_allowed_oriented_has_both_rotations(self):
        self.assertIn((2, 4), ALLOWED_ORIENTED)
        self.assertIn((4, 2), ALLOWED_ORIENTED)
        # 1x1 square: only one entry needed but set collapses duplicates
        self.assertIn((1, 1), ALLOWED_ORIENTED)


class TestBrick(unittest.TestCase):
    def test_cells_and_footprint(self):
        b = Brick(h=2, w=4, x=1, y=1, z=0)
        cells = b.cells()
        self.assertEqual(len(cells), 8)
        self.assertIn((1, 1, 0), cells)
        self.assertIn((2, 4, 0), cells)
        self.assertEqual(b.footprint(), (2, 4))

    def test_in_bounds(self):
        b = Brick(h=2, w=4, x=18, y=16, z=0)
        self.assertTrue(b.in_bounds(20, 20, 20))
        self.assertFalse(Brick(h=2, w=4, x=19, y=16, z=0).in_bounds(20, 20, 20))
        self.assertFalse(Brick(h=1, w=1, x=0, y=0, z=20).in_bounds(20, 20, 20))


class TestCodec(unittest.TestCase):
    def test_roundtrip(self):
        b = Brick(h=2, w=6, x=3, y=4, z=5)
        self.assertEqual(brick_to_line(b), "2x6 (3,4,5)")
        self.assertEqual(line_to_brick("2x6 (3,4,5)"), b)

    def test_orientation_preserved(self):
        self.assertEqual(line_to_brick("4x2 (0,0,0)"), Brick(4, 2, 0, 0, 0))
        self.assertNotEqual(line_to_brick("4x2 (0,0,0)"), line_to_brick("2x4 (0,0,0)"))

    def test_serialize_parse_design(self):
        bricks = [Brick(1, 2, 0, 0, 0), Brick(2, 4, 0, 0, 1)]
        text = serialize(bricks)
        self.assertEqual(parse(text), bricks)

    def test_parse_ignores_blank_lines(self):
        self.assertEqual(len(parse("1x1 (0,0,0)\n\n1x2 (0,0,1)\n")), 2)

    def test_bad_format_raises(self):
        for bad in ["2-4 (0,0,0)", "2x4 0,0,0", "2x4 (0,0)", "2x4 (a,0,0)", "x4 (0,0,0)"]:
            with self.assertRaises(BrickFormatError):
                line_to_brick(bad)


class TestRasterScan(unittest.TestCase):
    def test_sort_bottom_to_top(self):
        bricks = [Brick(1, 1, 5, 5, 2), Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 1, 0)]
        s = raster_scan_sorted(bricks)
        self.assertEqual([b.z for b in s], [0, 0, 2])
        self.assertTrue(is_raster_ordered(s))
        self.assertFalse(is_raster_ordered(bricks))


class TestFormatFSA(unittest.TestCase):
    def test_wellformed(self):
        self.assertTrue(is_wellformed_line("2x4 (0,0,0)"))
        self.assertTrue(is_wellformed_line("  2x4 (0,0,0)  "))
        self.assertFalse(is_wellformed_line("2x4  (0,0,0)"))
        self.assertFalse(is_wellformed_line("2x4 (0,0,0"))

    def test_valid_next_chars_start(self):
        self.assertEqual(valid_next_chars(""), frozenset("0123456789"))

    def test_valid_next_chars_after_first_digit(self):
        nxt = valid_next_chars("2")
        self.assertIn("x", nxt)
        self.assertIn("4", nxt)

    def test_valid_next_chars_expects_x(self):
        self.assertEqual(valid_next_chars("2x"), frozenset("0123456789"))

    def test_valid_next_chars_expects_open_paren(self):
        self.assertEqual(valid_next_chars("2x4 "), frozenset("("))

    def test_valid_next_chars_after_complete(self):
        self.assertEqual(valid_next_chars("2x4 (0,0,0)"), frozenset())

    def test_valid_next_chars_illegal_prefix(self):
        self.assertEqual(valid_next_chars("2y"), frozenset())

    def test_valid_next_chars_comma_then_close(self):
        # after two coords and their commas, third coord then ')'
        nxt = valid_next_chars("2x4 (0,0,0")
        self.assertIn(")", nxt)
        self.assertIn("0", nxt)


if __name__ == "__main__":
    unittest.main()
