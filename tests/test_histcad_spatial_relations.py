import unittest
import math

from harnesscad.data.dataengine.histcad_spatial_relations import (
    OBB, sat_overlap, classify_contact, relative_position_labels,
    analyze_parts,
)


def _box(cx, cy, cz, hx=1.0, hy=1.0, hz=1.0):
    return OBB((cx, cy, cz), (hx, hy, hz))


class TestSAT(unittest.TestCase):
    def test_separate(self):
        a, b = _box(0, 0, 0), _box(10, 0, 0)
        collides, gap, _ = sat_overlap(a, b)
        self.assertFalse(collides)
        self.assertGreater(gap, 0)

    def test_overlap(self):
        a, b = _box(0, 0, 0), _box(0.5, 0, 0)
        collides, gap, _ = sat_overlap(a, b)
        self.assertTrue(collides)
        self.assertLess(gap, 0)

    def test_touch(self):
        a, b = _box(0, 0, 0), _box(2, 0, 0)  # half=1 each, centers 2 apart
        collides, gap, touching = sat_overlap(a, b)
        self.assertTrue(collides)
        self.assertAlmostEqual(gap, 0.0, places=6)
        self.assertTrue(touching)


class TestClassify(unittest.TestCase):
    def test_separate(self):
        self.assertEqual(classify_contact(_box(0, 0, 0), _box(10, 0, 0)), "separate")

    def test_touch(self):
        self.assertEqual(classify_contact(_box(0, 0, 0), _box(2, 0, 0)), "touch")

    def test_intersect(self):
        self.assertEqual(classify_contact(_box(0, 0, 0), _box(0.5, 0, 0)), "intersect")

    def test_contain(self):
        big = _box(0, 0, 0, 5, 5, 5)
        small = _box(0, 0, 0, 1, 1, 1)
        self.assertEqual(classify_contact(big, small), "contain")

    def test_contained(self):
        big = _box(0, 0, 0, 5, 5, 5)
        small = _box(0, 0, 0, 1, 1, 1)
        self.assertEqual(classify_contact(small, big), "contained")


class TestDirection(unittest.TestCase):
    def test_right_above_front(self):
        labels = relative_position_labels(_box(0, 0, 0), _box(5, 5, 5))
        self.assertEqual(set(labels), {"right", "above", "front"})

    def test_left_below_back(self):
        labels = relative_position_labels(_box(0, 0, 0), _box(-5, -5, -5))
        self.assertEqual(set(labels), {"left", "below", "back"})

    def test_aligned(self):
        self.assertEqual(relative_position_labels(_box(0, 0, 0), _box(0, 0, 0)), ())


class TestOrientedBox(unittest.TestCase):
    def test_rotated_still_separates(self):
        # rotate b 45 deg about z; still far away
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
        b = OBB((10, 0, 0), (1, 1, 1),
                ((c, s, 0), (-s, c, 0), (0, 0, 1)))
        self.assertEqual(classify_contact(_box(0, 0, 0), b), "separate")

    def test_from_aabb(self):
        o = OBB.from_aabb((0, 0, 0), (2, 4, 6))
        self.assertEqual(o.center, (1.0, 2.0, 3.0))
        self.assertEqual(o.half, (1.0, 2.0, 3.0))


class TestAnalyzeParts(unittest.TestCase):
    def test_pairwise(self):
        rels = analyze_parts([_box(0, 0, 0), _box(2, 0, 0), _box(10, 0, 0)])
        self.assertEqual(len(rels), 3)
        self.assertEqual(rels[0].rel_type, "touch")   # 0-1
        self.assertEqual(rels[2].rel_type, "separate")  # 1-2


if __name__ == "__main__":
    unittest.main()
