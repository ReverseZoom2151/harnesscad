"""Tests for HistCAD 19-type sketch constraint-satisfaction checker."""

import unittest

from harnesscad.domain.geometry.sketch import constraint_satisfaction as cs
from harnesscad.domain.geometry.sketch.constraint_satisfaction import Constraint as C


class TestCoverage(unittest.TestCase):
    def test_all_19_types(self):
        self.assertEqual(len(cs.CONSTRAINT_TYPES), 19)

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            cs.satisfies(C("wat", ()))


class TestGeometric(unittest.TestCase):
    def test_coincident(self):
        self.assertTrue(cs.satisfies(C("coincident", ((0, 0), (0, 0)))))
        self.assertFalse(cs.satisfies(C("coincident", ((0, 0), (1, 0)))))

    def test_horizontal_vertical(self):
        self.assertTrue(cs.satisfies(C("horizontal", [((0, 5), (3, 5))])))
        self.assertFalse(cs.satisfies(C("horizontal", [((0, 5), (3, 6))])))
        self.assertTrue(cs.satisfies(C("vertical", [((2, 0), (2, 9))])))

    def test_parallel_perpendicular(self):
        l1 = ((0, 0), (1, 0))
        l2 = ((0, 1), (2, 1))
        l3 = ((0, 0), (0, 1))
        self.assertTrue(cs.satisfies(C("parallel", [l1, l2])))
        self.assertTrue(cs.satisfies(C("perpendicular", [l1, l3])))
        self.assertTrue(cs.satisfies(C("normal", [l1, l3])))
        self.assertFalse(cs.satisfies(C("perpendicular", [l1, l2])))

    def test_concentric_tangent(self):
        c1 = {"center": (0, 0), "radius": 1.0}
        c2 = {"center": (0, 0), "radius": 2.0}
        self.assertTrue(cs.satisfies(C("concentric", [c1, c2])))
        ext = {"center": (3, 0), "radius": 1.0}
        # d=3 is neither r1+r2 (=2, external) nor |r1-r2| (=0, internal): NOT tangent.
        self.assertFalse(cs.satisfies(C("tangent", [c1, ext])))
        # external tangent: centers 2 apart, radii 1 and 1
        a = {"center": (0, 0), "radius": 1.0}
        b = {"center": (2, 0), "radius": 1.0}
        self.assertTrue(cs.satisfies(C("tangent", [a, b])))


class TestDimensional(unittest.TestCase):
    def test_length_distance(self):
        self.assertTrue(cs.satisfies(C("length", [((0, 0), (3, 4))], value=5.0)))
        self.assertTrue(cs.satisfies(C("distance", ((0, 0), (0, 7)), value=7.0)))
        self.assertFalse(cs.satisfies(C("length", [((0, 0), (3, 4))], value=4.0)))

    def test_diameter_radius(self):
        circ = {"center": (0, 0), "radius": 2.5}
        self.assertTrue(cs.satisfies(C("radius", [circ], value=2.5)))
        self.assertTrue(cs.satisfies(C("diameter", [circ], value=5.0)))

    def test_angle(self):
        l1 = ((0, 0), (1, 0))
        l2 = ((0, 0), (1, 1))
        self.assertTrue(cs.satisfies(C("angle", [l1, l2], value=45.0)))

    def test_minor_major_radius(self):
        ell = {"minor_radius": 1.0, "major_radius": 3.0}
        self.assertTrue(cs.satisfies(C("minor_radius", [ell], value=1.0)))
        self.assertTrue(cs.satisfies(C("major_radius", [ell], value=3.0)))


class TestAnchoringRelational(unittest.TestCase):
    def test_fix(self):
        self.assertTrue(cs.satisfies(C("fix", ((1, 2), (1, 2)))))
        self.assertFalse(cs.satisfies(C("fix", ((1, 2), (1, 3)))))

    def test_midpoint(self):
        self.assertTrue(cs.satisfies(C("midpoint", ((1, 1), (0, 0), (2, 2)))))
        self.assertFalse(cs.satisfies(C("midpoint", ((1, 0), (0, 0), (2, 2)))))

    def test_equal_lines_and_circles(self):
        self.assertTrue(cs.satisfies(C("equal", (((0, 0), (2, 0)), ((5, 5), (5, 7)))) ))  # len 2 vs 2
        c1 = {"center": (0, 0), "radius": 1.0}
        c2 = {"center": (9, 9), "radius": 1.0}
        self.assertTrue(cs.satisfies(C("equal", (c1, c2))))

    def test_mirror(self):
        axis = ((0, 0), (0, 1))  # y-axis
        self.assertTrue(cs.satisfies(C("mirror", ((1, 3), (-1, 3), axis))))
        self.assertFalse(cs.satisfies(C("mirror", ((1, 3), (1, 3), axis))))


class TestRate(unittest.TestCase):
    def test_rate(self):
        good = C("horizontal", [((0, 0), (1, 0))])
        bad = C("vertical", [((0, 0), (1, 0))])
        self.assertEqual(cs.satisfaction_rate([good, bad]), 0.5)
        self.assertEqual(cs.satisfaction_rate([]), 0.0)


if __name__ == "__main__":
    unittest.main()
