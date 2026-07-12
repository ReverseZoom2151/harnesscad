"""Tests for geometry.joinable_joint_motion."""

import math
import unittest

from geometry.joinable_joint_axis import axis_lines_colinear, dot
from geometry.joinable_joint_motion import (
    JOINT_TYPES,
    JointPose,
    UnknownJointType,
    axis_plane_basis,
    is_pose_parameter_free,
    joint_constrained_dof,
    joint_free_dof,
    joint_free_parameters,
    joint_pose_transform,
    normalize_joint_type,
    pose_matrix,
    project_pose,
    sample_joint_motion,
)
from geometry.joinable_joint_transform import (
    apply_joint_transform_to_axis,
    transform_point,
)


class VocabularyTests(unittest.TestCase):
    def test_all_types_have_dof_and_parameters(self):
        for name in JOINT_TYPES:
            self.assertIsInstance(joint_free_dof(name), int)
            self.assertIsInstance(joint_free_parameters(name), tuple)

    def test_dof_counts(self):
        self.assertEqual(joint_free_dof("rigid"), 0)
        self.assertEqual(joint_free_dof("revolute"), 1)
        self.assertEqual(joint_free_dof("slider"), 1)
        self.assertEqual(joint_free_dof("cylindrical"), 2)
        self.assertEqual(joint_free_dof("pin_slot"), 2)
        self.assertEqual(joint_free_dof("planar"), 3)

    def test_constrained_dof_complements_free_dof(self):
        for name in JOINT_TYPES:
            self.assertEqual(joint_free_dof(name) + joint_constrained_dof(name),
                             6)

    def test_aliases(self):
        self.assertEqual(normalize_joint_type("Prismatic"), "slider")
        self.assertEqual(normalize_joint_type("Pin-Slot"), "pin_slot")
        self.assertEqual(normalize_joint_type("RevoluteJointType"), "revolute")
        self.assertEqual(normalize_joint_type("fixed"), "rigid")

    def test_unknown_type(self):
        with self.assertRaises(UnknownJointType):
            normalize_joint_type("wobbly")

    def test_free_parameter_predicate(self):
        self.assertTrue(is_pose_parameter_free("revolute", "rotation"))
        self.assertFalse(is_pose_parameter_free("revolute", "offset"))
        self.assertTrue(is_pose_parameter_free("cylindrical", "offset"))
        self.assertFalse(is_pose_parameter_free("rigid", "rotation"))
        with self.assertRaises(ValueError):
            is_pose_parameter_free("rigid", "wiggle")

    def test_free_parameter_count_matches_dof_for_parameterised_types(self):
        for name in ("rigid", "revolute", "slider", "cylindrical", "pin_slot",
                     "planar"):
            self.assertEqual(len(joint_free_parameters(name)),
                             joint_free_dof(name))


class ProjectPoseTests(unittest.TestCase):
    def test_rigid_zeroes_everything_but_flip(self):
        pose = JointPose(rotation=1.0, offset=2.0, slide_u=3.0, slide_v=4.0,
                         flip=True)
        projected = project_pose("rigid", pose)
        self.assertEqual(projected, JointPose(flip=True))

    def test_revolute_keeps_rotation_only(self):
        pose = JointPose(rotation=1.0, offset=2.0)
        projected = project_pose("revolute", pose)
        self.assertEqual(projected.rotation, 1.0)
        self.assertEqual(projected.offset, 0.0)

    def test_slider_keeps_offset_only(self):
        projected = project_pose("slider", JointPose(rotation=1.0, offset=2.0))
        self.assertEqual(projected.rotation, 0.0)
        self.assertEqual(projected.offset, 2.0)

    def test_planar_drops_offset(self):
        pose = JointPose(rotation=0.5, offset=9.0, slide_u=1.0, slide_v=2.0)
        projected = project_pose("planar", pose)
        self.assertEqual(projected.offset, 0.0)
        self.assertEqual(projected.slide_u, 1.0)
        self.assertEqual(projected.slide_v, 2.0)

    def test_pose_dict_and_repr(self):
        pose = JointPose(rotation=1.0)
        self.assertEqual(pose.as_dict()["rotation"], 1.0)
        self.assertIn("JointPose", repr(pose))


class SampleMotionTests(unittest.TestCase):
    def test_rigid_has_single_pose(self):
        poses = sample_joint_motion("rigid", steps=5)
        self.assertEqual(poses, [JointPose()])

    def test_revolute_sample_count_and_neutral_first(self):
        poses = sample_joint_motion("revolute", steps=4)
        self.assertEqual(len(poses), 4)
        self.assertEqual(poses[0], JointPose())
        self.assertTrue(all(p.offset == 0.0 for p in poses))

    def test_cylindrical_is_product_of_two_parameters(self):
        poses = sample_joint_motion("cylindrical", steps=3)
        self.assertEqual(len(poses), 9)

    def test_planar_is_product_of_three(self):
        poses = sample_joint_motion("planar", steps=2)
        self.assertEqual(len(poses), 8)

    def test_flip_doubles_the_sample(self):
        poses = sample_joint_motion("slider", steps=3, include_flip=True)
        self.assertEqual(len(poses), 6)
        self.assertTrue(any(p.flip for p in poses))

    def test_offset_range_respected(self):
        poses = sample_joint_motion("slider", steps=3, offset_range=2.0)
        offsets = sorted(p.offset for p in poses)
        self.assertAlmostEqual(offsets[0], -2.0)
        self.assertAlmostEqual(offsets[-1], 2.0)

    def test_deterministic(self):
        a = sample_joint_motion("cylindrical", steps=3)
        b = sample_joint_motion("cylindrical", steps=3)
        self.assertEqual(a, b)

    def test_bad_steps(self):
        with self.assertRaises(ValueError):
            sample_joint_motion("revolute", steps=0)


class BasisAndMatrixTests(unittest.TestCase):
    def test_basis_is_orthonormal_and_normal_to_axis(self):
        axis = (0.0, 0.0, 1.0)
        u, v = axis_plane_basis(axis)
        self.assertAlmostEqual(dot(u, v), 0.0)
        self.assertAlmostEqual(dot(u, axis), 0.0)
        self.assertAlmostEqual(dot(v, axis), 0.0)
        self.assertAlmostEqual(dot(u, u), 1.0)

    def test_basis_zero_axis(self):
        with self.assertRaises(ValueError):
            axis_plane_basis((0.0, 0.0, 0.0))

    def test_pose_matrix_slide_translates_in_plane(self):
        pose = JointPose(slide_u=2.0)
        mat = pose_matrix(pose, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        moved = transform_point(mat, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(moved[2], 0.0)
        self.assertAlmostEqual(math.sqrt(moved[0] ** 2 + moved[1] ** 2), 2.0)


class PoseTransformTests(unittest.TestCase):
    def test_rigid_pose_ignores_illegal_parameters(self):
        axis1 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        axis2 = ((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))
        pose = JointPose(rotation=1.0, offset=5.0)
        mat = joint_pose_transform("rigid", pose, axis1, axis2)
        moved = apply_joint_transform_to_axis(mat, axis1)
        for got, want in zip(moved[0], axis2[0]):
            self.assertAlmostEqual(got, want)

    def test_slider_offset_moves_along_axis(self):
        axis1 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        axis2 = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        mat = joint_pose_transform("slider", JointPose(offset=3.0), axis1, axis2)
        moved = apply_joint_transform_to_axis(mat, axis1)
        self.assertAlmostEqual(moved[0][2], 3.0)
        self.assertTrue(axis_lines_colinear(moved, axis2))

    def test_revolute_pose_keeps_axis_colinear(self):
        axis1 = ((1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        axis2 = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        mat = joint_pose_transform("revolute", JointPose(rotation=math.pi / 3),
                                   axis1, axis2)
        moved = apply_joint_transform_to_axis(mat, axis1)
        self.assertTrue(axis_lines_colinear(moved, axis2))

    def test_pin_slot_slides_off_the_axis(self):
        axis1 = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        axis2 = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        mat = joint_pose_transform("pin_slot", JointPose(slide_u=2.0),
                                   axis1, axis2)
        moved = apply_joint_transform_to_axis(mat, axis1)
        self.assertFalse(axis_lines_colinear(moved, axis2))
        self.assertAlmostEqual(moved[0][2], 0.0)


if __name__ == "__main__":
    unittest.main()
