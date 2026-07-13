"""Tests for geometry.joinable_joint_axis."""

import math
import unittest

from harnesscad.domain.geometry.kinematics.joint_axis import (
    AxisLineError,
    angle_between,
    as_vec3,
    axis_lines_colinear,
    closest_point_on_line,
    direction_between,
    dist_point_to_line,
    find_axis_line,
    find_axis_line_from_edge,
    find_axis_line_from_face,
    joint_axis_error,
    normalize,
)


class VectorHelperTests(unittest.TestCase):
    def test_as_vec3_from_dict(self):
        self.assertEqual(as_vec3({"x": 1, "y": 2, "z": 3}), (1.0, 2.0, 3.0))

    def test_as_vec3_prefixed_keys(self):
        entity = {"origin_x": 4, "origin_y": 5, "origin_z": 6}
        self.assertEqual(as_vec3(entity, "origin"), (4.0, 5.0, 6.0))

    def test_as_vec3_sequence_and_bad_length(self):
        self.assertEqual(as_vec3([0, 0, 1]), (0.0, 0.0, 1.0))
        with self.assertRaises(AxisLineError):
            as_vec3([1, 2])

    def test_normalize_zero_vector(self):
        self.assertEqual(normalize((0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))

    def test_normalize_unit(self):
        v = normalize((0.0, 3.0, 4.0))
        self.assertAlmostEqual(v[1], 0.6)
        self.assertAlmostEqual(v[2], 0.8)

    def test_angle_between(self):
        a = angle_between((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        self.assertAlmostEqual(a, math.pi / 2)
        self.assertAlmostEqual(
            angle_between((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), 0.0)

    def test_direction_between_coincident(self):
        self.assertEqual(
            direction_between((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
            (0.0, 0.0, 0.0))

    def test_point_to_line(self):
        d = dist_point_to_line((0.0, 2.0, 0.0), (0.0, 0.0, 0.0),
                               (0.0, 0.0, 1.0))
        self.assertAlmostEqual(d, 2.0)
        foot = closest_point_on_line((3.0, 2.0, 5.0), (0.0, 0.0, 0.0),
                                     (0.0, 0.0, 2.0))
        self.assertAlmostEqual(foot[2], 5.0)
        self.assertAlmostEqual(foot[0], 0.0)

    def test_point_on_line_has_zero_distance(self):
        d = dist_point_to_line((0.0, 0.0, 7.0), (0.0, 0.0, 0.0),
                               (0.0, 0.0, 1.0))
        self.assertAlmostEqual(d, 0.0)


class FaceAxisTests(unittest.TestCase):
    def test_planar_face_uses_centroid_and_normal(self):
        face = {
            "surface_type": "PlaneSurfaceType",
            "centroid": {"x": 1.0, "y": 2.0, "z": 0.0},
            "normal": {"x": 0.0, "y": 0.0, "z": 5.0},
        }
        origin, direction = find_axis_line_from_face(face)
        self.assertEqual(origin, (1.0, 2.0, 0.0))
        self.assertEqual(direction, (0.0, 0.0, 1.0))

    def test_cylindrical_face_uses_origin_and_axis(self):
        face = {
            "surface_type": "CylinderSurfaceType",
            "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
            "axis": {"x": 0.0, "y": 2.0, "z": 0.0},
        }
        origin, direction = find_axis_line(face)
        self.assertEqual(origin, (0.0, 0.0, 0.0))
        self.assertEqual(direction, (0.0, 1.0, 0.0))

    def test_sphere_uses_default_z_axis(self):
        face = {"surface_type": "SphereSurfaceType",
                "origin": {"x": 1.0, "y": 1.0, "z": 1.0}}
        origin, direction = find_axis_line_from_face(face)
        self.assertEqual(direction, (0.0, 0.0, 1.0))
        self.assertEqual(origin, (1.0, 1.0, 1.0))

    def test_unsupported_surface(self):
        with self.assertRaises(AxisLineError):
            find_axis_line_from_face({"surface_type": "NurbsSurfaceType"})


class EdgeAxisTests(unittest.TestCase):
    def test_linear_edge(self):
        edge = {
            "curve_type": "Line3DCurveType",
            "start_point": {"x": 0.0, "y": 0.0, "z": 0.0},
            "end_point": {"x": 0.0, "y": 0.0, "z": 4.0},
        }
        origin, direction = find_axis_line(edge)
        self.assertEqual(origin, (0.0, 0.0, 0.0))
        self.assertEqual(direction, (0.0, 0.0, 1.0))

    def test_circular_edge(self):
        edge = {
            "curve_type": "Circle3DCurveType",
            "center": {"x": 1.0, "y": 0.0, "z": 0.0},
            "normal": {"x": 1.0, "y": 0.0, "z": 0.0},
        }
        origin, direction = find_axis_line_from_edge(edge)
        self.assertEqual(origin, (1.0, 0.0, 0.0))
        self.assertEqual(direction, (1.0, 0.0, 0.0))

    def test_degenerate_edge_rejected(self):
        with self.assertRaises(AxisLineError):
            find_axis_line({"curve_type": "Line3DCurveType",
                            "is_degenerate": True})

    def test_entity_without_type(self):
        with self.assertRaises(AxisLineError):
            find_axis_line({"area": 1.0})


class AxisComparisonTests(unittest.TestCase):
    def test_identical_axes_are_colinear(self):
        line = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        angle, dist = joint_axis_error(line, line)
        self.assertAlmostEqual(angle, 0.0)
        self.assertAlmostEqual(dist, 0.0)
        self.assertTrue(axis_lines_colinear(line, line))

    def test_reversed_direction_still_colinear(self):
        a = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        b = ((0.0, 0.0, 5.0), (0.0, 0.0, -1.0))
        angle, dist = joint_axis_error(a, b)
        self.assertAlmostEqual(angle, 0.0)
        self.assertAlmostEqual(dist, 0.0)
        self.assertTrue(axis_lines_colinear(a, b))

    def test_offset_parallel_axis_not_colinear(self):
        a = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        b = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        angle, dist = joint_axis_error(a, b)
        self.assertAlmostEqual(angle, 0.0)
        self.assertAlmostEqual(dist, 1.0)
        self.assertFalse(axis_lines_colinear(a, b))
        self.assertTrue(axis_lines_colinear(a, b, distance_tol=2.0))

    def test_perpendicular_axis_angle(self):
        a = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        b = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        angle, dist = joint_axis_error(a, b)
        self.assertAlmostEqual(angle, 90.0)
        self.assertAlmostEqual(dist, 0.0)
        self.assertFalse(axis_lines_colinear(a, b))

    def test_zero_direction_falls_back_to_point_distance(self):
        a = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        b = ((3.0, 4.0, 0.0), (0.0, 0.0, 0.0))
        _, dist = joint_axis_error(a, b)
        self.assertAlmostEqual(dist, 5.0)


if __name__ == "__main__":
    unittest.main()
