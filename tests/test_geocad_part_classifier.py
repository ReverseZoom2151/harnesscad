"""Tests for GeoCAD simple/complex local-part routing."""

import unittest

from harnesscad.domain.reconstruction.translate.flexcad_text import curve, loop
from harnesscad.domain.reconstruction.recognize import geocad_part_classifier as pc


def _lines(n):
    return loop(*[curve("line", 0, 0) for _ in range(n)])


class ClassifyTest(unittest.TestCase):
    def test_triangle_simple(self):
        r = pc.classify_loop(_lines(3))
        self.assertTrue(r.is_simple)
        self.assertEqual(r.family, pc.FAMILY_TRIANGLE)
        self.assertEqual(r.branch, pc.BRANCH_VERTEX)

    def test_quadrilateral_simple(self):
        r = pc.classify_loop(_lines(4))
        self.assertEqual(r.family, pc.FAMILY_QUADRILATERAL)
        self.assertTrue(r.is_simple)

    def test_circle_simple(self):
        r = pc.classify_loop(loop(curve("circle", 0, 0, 1, 0, 0, 1, 2, 2)))
        self.assertEqual(r.family, pc.FAMILY_CIRCLE)
        self.assertTrue(r.is_simple)

    def test_sector_simple(self):
        lp = loop(curve("line", 0, 0), curve("arc", 0, 0, 1, 1), curve("line", 0, 0))
        r = pc.classify_loop(lp)
        self.assertEqual(r.family, pc.FAMILY_SECTOR)
        self.assertTrue(r.is_simple)

    def test_semicircle_simple(self):
        lp = loop(curve("line", 0, 0), curve("arc", 0, 0, 1, 1))
        r = pc.classify_loop(lp)
        self.assertEqual(r.family, pc.FAMILY_SEMICIRCLE)

    def test_complex(self):
        # 5 lines -> not a simple template.
        r = pc.classify_loop(_lines(5))
        self.assertFalse(r.is_simple)
        self.assertEqual(r.family, pc.FAMILY_COMPLEX)
        self.assertEqual(r.branch, pc.BRANCH_VLLM)

    def test_complex_mixed(self):
        lp = loop(curve("line", 0, 0), curve("arc", 0, 0, 1, 1),
                  curve("line", 0, 0), curve("arc", 0, 0, 1, 1))
        self.assertFalse(pc.classify_loop(lp).is_simple)


class PartitionTest(unittest.TestCase):
    def test_partition(self):
        loops = [_lines(3), _lines(5), _lines(4), _lines(7)]
        simple, complex_ = pc.partition(loops)
        self.assertEqual(simple, [0, 2])
        self.assertEqual(complex_, [1, 3])


if __name__ == "__main__":
    unittest.main()
