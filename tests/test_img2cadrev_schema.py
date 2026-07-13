import math
import unittest

from harnesscad.domain.reconstruction.sequences.img2cadrev_schema import (
    LINE, ARC, CIRCLE, EXTRUDE_JOIN, EXTRUDE_CUT,
    COMMAND_TYPES, ATTRIBUTE_DIM,
    is_sketch_command, is_extrude_command, is_command_type,
    attribute_dim, validate_attribute_vector, extrusion_is_cut,
    sample_arc, profile_polygon, signed_area, is_counter_clockwise,
    profile_is_closed, is_simple_polygon, profile_is_valid,
)


class SchemaBasicsTest(unittest.TestCase):
    def test_command_vocabulary(self):
        self.assertEqual(len(COMMAND_TYPES), 5)
        self.assertIn(EXTRUDE_JOIN, COMMAND_TYPES)
        self.assertIn(EXTRUDE_CUT, COMMAND_TYPES)

    def test_join_and_cut_distinct(self):
        self.assertNotEqual(EXTRUDE_JOIN, EXTRUDE_CUT)
        self.assertTrue(extrusion_is_cut(EXTRUDE_CUT))
        self.assertFalse(extrusion_is_cut(EXTRUDE_JOIN))
        with self.assertRaises(ValueError):
            extrusion_is_cut(LINE)

    def test_attribute_dims(self):
        self.assertEqual(attribute_dim(LINE), 2)
        self.assertEqual(attribute_dim(ARC), 3)
        self.assertEqual(attribute_dim(CIRCLE), 3)
        self.assertEqual(attribute_dim(EXTRUDE_JOIN), 7)
        self.assertEqual(attribute_dim(EXTRUDE_CUT), 7)

    def test_classifiers(self):
        self.assertTrue(is_sketch_command(LINE))
        self.assertFalse(is_sketch_command(EXTRUDE_JOIN))
        self.assertTrue(is_extrude_command(EXTRUDE_CUT))
        self.assertTrue(is_command_type(ARC))
        self.assertFalse(is_command_type("Z"))

    def test_validate_attribute_vector(self):
        validate_attribute_vector(LINE, [1.0, 2.0])
        with self.assertRaises(ValueError):
            validate_attribute_vector(LINE, [1.0])
        with self.assertRaises(ValueError):
            validate_attribute_vector(ARC, [1.0, 2.0, "x"])

    def test_attribute_dim_unknown(self):
        with self.assertRaises(KeyError):
            attribute_dim("nope")


class GeometryTest(unittest.TestCase):
    def _square(self):
        # CCW unit square via lines from origin.
        return [
            (LINE, [1.0, 0.0]),
            (LINE, [1.0, 1.0]),
            (LINE, [0.0, 1.0]),
            (LINE, [0.0, 0.0]),
        ]

    def test_signed_area_ccw(self):
        poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.assertAlmostEqual(signed_area(poly), 1.0)
        self.assertTrue(is_counter_clockwise(poly))

    def test_signed_area_cw(self):
        poly = [(0, 0), (0, 1), (1, 1), (1, 0)]
        self.assertAlmostEqual(signed_area(poly), -1.0)
        self.assertFalse(is_counter_clockwise(poly))

    def test_profile_polygon_square(self):
        poly = profile_polygon(self._square())
        self.assertEqual(len(poly), 4)
        self.assertAlmostEqual(signed_area(poly), 1.0)

    def test_profile_is_closed(self):
        self.assertTrue(profile_is_closed(self._square()))
        open_chain = [(LINE, [1.0, 0.0]), (LINE, [1.0, 1.0])]
        self.assertFalse(profile_is_closed(open_chain))

    def test_circle_profile(self):
        cmds = [(CIRCLE, [0.0, 0.0, 1.0])]
        self.assertTrue(profile_is_closed(cmds))
        poly = profile_polygon(cmds)
        # Area of sampled unit circle is close to pi.
        self.assertGreater(signed_area(poly), 2.5)
        self.assertTrue(profile_is_valid(cmds))

    def test_valid_square(self):
        self.assertTrue(profile_is_valid(self._square()))

    def test_self_intersecting_not_valid(self):
        # Bowtie: crosses itself -> not simple.
        bowtie = [
            (LINE, [2.0, 2.0]),
            (LINE, [2.0, 0.0]),
            (LINE, [0.0, 2.0]),
            (LINE, [0.0, 0.0]),
        ]
        poly = profile_polygon(bowtie)
        self.assertFalse(is_simple_polygon(poly))
        self.assertFalse(profile_is_valid(bowtie))

    def test_cw_square_not_valid(self):
        cw = [
            (LINE, [0.0, 1.0]),
            (LINE, [1.0, 1.0]),
            (LINE, [1.0, 0.0]),
            (LINE, [0.0, 0.0]),
        ]
        self.assertTrue(profile_is_closed(cw))
        self.assertFalse(profile_is_valid(cw))  # clockwise

    def test_sample_arc_semicircle(self):
        # Semicircle from (1,0) to (-1,0), sweep pi -> passes near (0,1).
        pts = sample_arc((1.0, 0.0), (-1.0, 0.0), math.pi, n=8)
        self.assertEqual(len(pts), 9)
        # Some sampled point should be near the top of the unit circle.
        top = max(pts, key=lambda p: p[1])
        self.assertAlmostEqual(top[1], 1.0, places=5)

    def test_sample_arc_degenerate(self):
        pts = sample_arc((0.0, 0.0), (1.0, 0.0), 0.0)
        self.assertEqual(pts, [(0.0, 0.0), (1.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
