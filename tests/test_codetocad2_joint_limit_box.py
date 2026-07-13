"""Tests for the per-axis 6-DOF joint limit box."""

import math
import unittest

from geometry.codetocad2_joint_limit_box import (
    AxisLimit,
    JointLimitBox,
    LimitError,
    ball,
    cylindrical,
    free_joint,
    planar,
    prismatic,
    revolute,
    rigid,
    wrap_angle,
)


class TestAxisLimit(unittest.TestCase):
    def test_kinds(self):
        self.assertTrue(AxisLimit.free().is_free)
        self.assertTrue(AxisLimit.locked(0.0).is_locked)
        self.assertTrue(AxisLimit(-1.0, 2.0).is_ranged)

    def test_clamp_linear(self):
        limit = AxisLimit(-1.0, 2.0)
        self.assertEqual(limit.clamp(0.5), 0.5)
        self.assertEqual(limit.clamp(-9.0), -1.0)
        self.assertEqual(limit.clamp(9.0), 2.0)

    def test_clamp_free_is_identity(self):
        self.assertEqual(AxisLimit.free().clamp(1234.5), 1234.5)

    def test_half_open_limit(self):
        limit = AxisLimit(0.0, None)
        self.assertEqual(limit.clamp(-3.0), 0.0)
        self.assertEqual(limit.clamp(3.0), 3.0)

    def test_locked_clamps_to_value(self):
        limit = AxisLimit.locked(4.0)
        self.assertEqual(limit.clamp(-100.0), 4.0)
        self.assertEqual(limit.dof, 0)

    def test_angular_clamp_takes_nearest_branch(self):
        # hinge from -30 to +90 degrees; a proposed 350 degrees is really -10.
        limit = AxisLimit(math.radians(-30.0), math.radians(90.0), angular=True)
        clamped = limit.clamp(math.radians(350.0))
        self.assertAlmostEqual(math.degrees(clamped), -10.0, places=9)

    def test_angular_clamp_out_of_range(self):
        limit = AxisLimit(math.radians(-30.0), math.radians(90.0), angular=True)
        clamped = limit.clamp(math.radians(180.0))
        self.assertAlmostEqual(math.degrees(clamped), 90.0, places=9)

    def test_bad_order_rejected(self):
        with self.assertRaises(LimitError):
            AxisLimit(3.0, 1.0)

    def test_intersect(self):
        merged = AxisLimit(-1.0, 4.0).intersect(AxisLimit(0.0, 2.0))
        self.assertEqual(merged.as_pair(), (0.0, 2.0))
        with self.assertRaises(LimitError):
            AxisLimit(-1.0, 0.0).intersect(AxisLimit(1.0, 2.0))

    def test_span_and_contains(self):
        limit = AxisLimit(0.0, 2.0)
        self.assertEqual(limit.span, 2.0)
        self.assertTrue(limit.contains(1.0))
        self.assertFalse(limit.contains(3.0))
        self.assertIsNone(AxisLimit.free().span)

    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(3.0 * math.pi), -math.pi, places=9)
        self.assertAlmostEqual(wrap_angle(0.25), 0.25, places=12)


class TestJointLimitBox(unittest.TestCase):
    def test_from_xyz_matches_adapter_signature(self):
        box = JointLimitBox.from_xyz(
            limit_location_xyz=[(0.0, 0.0), (0.0, 0.0), (0.0, 10.0)],
            limit_rotation_xyz=[(0.0, 0.0), (0.0, 0.0), None],
        )
        self.assertEqual(box.limit("z").as_pair(), (0.0, 10.0))
        self.assertTrue(box.limit("rz").is_free)
        self.assertEqual(box.dof, 2)
        self.assertEqual(box.classify(), "cylindrical")

    def test_clamp_pose(self):
        box = JointLimitBox.from_xyz(
            limit_location_xyz=[(0.0, 0.0), (0.0, 0.0), (0.0, 10.0)],
            limit_rotation_xyz=[(0.0, 0.0), (0.0, 0.0), (0.0, math.pi)],
        )
        clamped = box.clamp((5.0, -2.0, 12.0, 1.0, 1.0, -0.5))
        self.assertEqual(clamped[:3], (0.0, 0.0, 10.0))
        self.assertEqual(clamped[3], 0.0)
        self.assertEqual(clamped[4], 0.0)
        self.assertAlmostEqual(clamped[5], 0.0, places=9)

    def test_clamp_is_idempotent(self):
        box = revolute("y", -1.0, 1.0)
        once = box.clamp((0.0, 0.0, 0.0, 0.0, 5.0, 0.0))
        twice = box.clamp(once)
        self.assertEqual(once, twice)
        self.assertTrue(box.contains(once))

    def test_bad_pose_length(self):
        with self.assertRaises(LimitError):
            rigid().clamp((0.0, 0.0, 0.0))

    def test_classification(self):
        self.assertEqual(rigid().classify(), "rigid")
        self.assertEqual(free_joint().classify(), "free")
        self.assertEqual(revolute("z").classify(), "revolute")
        self.assertEqual(prismatic("x", 0.0, 3.0).classify(), "prismatic")
        self.assertEqual(cylindrical("y").classify(), "cylindrical")
        self.assertEqual(planar("z").classify(), "planar")
        self.assertEqual(ball().classify(), "ball")

    def test_generic_pattern(self):
        box = JointLimitBox.locked_box().with_free("x").with_free("ry")
        self.assertEqual(box.classify(), "generic")
        self.assertIsNone(box.axis_of())

    def test_dof_counts(self):
        self.assertEqual(rigid().dof, 0)
        self.assertEqual(revolute("x").dof, 1)
        self.assertEqual(planar("z").translational_dof, 2)
        self.assertEqual(planar("z").rotational_dof, 1)
        self.assertEqual(free_joint().dof, 6)

    def test_axis_of(self):
        self.assertEqual(revolute("y").axis_of(), "y")
        self.assertEqual(prismatic("x").axis_of(), "x")
        self.assertEqual(cylindrical("z").axis_of(), "z")

    def test_movable_names(self):
        self.assertEqual(cylindrical("z").movable(), ("z", "rz"))
        self.assertEqual(rigid().movable(), ())

    def test_bounded(self):
        self.assertTrue(rigid().is_bounded())
        self.assertFalse(revolute("z").is_bounded())
        self.assertTrue(revolute("z", -1.0, 1.0).is_bounded())

    def test_intersect_boxes(self):
        hinge = revolute("z", -2.0, 2.0)
        tighter = revolute("z", 0.0, 1.0)
        merged = hinge.intersect(tighter)
        self.assertEqual(merged.limit("rz").as_pair(), (0.0, 1.0))
        self.assertEqual(merged.classify(), "revolute")

    def test_ball_with_ranges(self):
        box = ball(angular_range_x=(-0.5, 0.5), angular_range_z=(-1.0, 1.0))
        clamped = box.clamp((1.0, 1.0, 1.0, 2.0, 0.3, -3.0))
        self.assertEqual(clamped[:3], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(clamped[3], 0.5, places=9)
        self.assertAlmostEqual(clamped[4], 0.3, places=9)
        self.assertAlmostEqual(clamped[5], -1.0, places=9)

    def test_as_dict_and_equality(self):
        box = revolute("z", -1.0, 1.0)
        self.assertEqual(box.as_dict()["rz"], (-1.0, 1.0))
        self.assertEqual(box, revolute("z", -1.0, 1.0))
        self.assertEqual(hash(box), hash(revolute("z", -1.0, 1.0)))

    def test_bad_triple(self):
        with self.assertRaises(LimitError):
            JointLimitBox.from_xyz(limit_location_xyz=[(0.0, 1.0)])
        with self.assertRaises(LimitError):
            JointLimitBox.locked_box().index_of("q")


if __name__ == "__main__":
    unittest.main()
