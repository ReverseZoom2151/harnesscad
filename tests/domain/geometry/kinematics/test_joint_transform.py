"""Tests for geometry.joinable_joint_transform."""

import math
import unittest

from harnesscad.domain.geometry.kinematics.joint_axis import axis_lines_colinear, joint_axis_error
from harnesscad.domain.geometry.kinematics.joint_transform import (
    align_vectors,
    apply_joint_transform_to_axis,
    identity_matrix,
    joint_alignment_matrix,
    joint_transform_from_parameters,
    matmul,
    offset_parameter_matrix,
    rotation_matrix_about_axis,
    rotation_parameter_matrix,
    transform_point,
    transform_vector,
)


def _apply3(rot3, v):
    return tuple(sum(rot3[i][j] * v[j] for j in range(3)) for i in range(3))


class MatrixTests(unittest.TestCase):
    def test_identity_matmul(self):
        i = identity_matrix()
        self.assertEqual(matmul(i, i), i)

    def test_transform_point_translation(self):
        m = offset_parameter_matrix(2.0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertEqual(transform_point(m, (1.0, 1.0, 1.0)), (1.0, 1.0, 3.0))

    def test_transform_vector_ignores_translation(self):
        m = offset_parameter_matrix(9.0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertEqual(transform_vector(m, (1.0, 0.0, 0.0)), (1.0, 0.0, 0.0))

    def test_rotation_about_z(self):
        rot = rotation_matrix_about_axis(math.pi / 2, (0.0, 0.0, 1.0))
        v = _apply3(rot, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(v[0], 0.0)
        self.assertAlmostEqual(v[1], 1.0)


class AlignVectorsTests(unittest.TestCase):
    def test_align_orthogonal(self):
        rot = align_vectors((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        v = _apply3(rot, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(v[0], 0.0)
        self.assertAlmostEqual(v[1], 1.0)
        self.assertAlmostEqual(v[2], 0.0)

    def test_align_parallel_is_identity(self):
        rot = align_vectors((0.0, 0.0, 2.0), (0.0, 0.0, 5.0))
        self.assertEqual(rot[0][0], 1.0)
        self.assertEqual(rot[1][1], 1.0)
        self.assertEqual(rot[2][2], 1.0)

    def test_align_antiparallel(self):
        rot = align_vectors((0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
        v = _apply3(rot, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(v[2], -1.0)

    def test_align_is_a_rotation(self):
        rot = align_vectors((1.0, 2.0, 3.0), (-3.0, 1.0, 0.5))
        # Orthonormal rows.
        for i in range(3):
            self.assertAlmostEqual(sum(c * c for c in rot[i]), 1.0)

    def test_zero_vector_rejected(self):
        with self.assertRaises(ValueError):
            align_vectors((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))


class JointAlignmentTests(unittest.TestCase):
    def test_alignment_makes_axes_colinear(self):
        axis1 = ((1.0, 2.0, 3.0), (1.0, 0.0, 0.0))
        axis2 = ((-4.0, 0.0, 1.0), (0.0, 0.0, 1.0))
        mat = joint_alignment_matrix(axis1[0], axis1[1], axis2[0], axis2[1])
        moved = apply_joint_transform_to_axis(mat, axis1)
        angle, dist = joint_axis_error(moved, axis2)
        self.assertAlmostEqual(angle, 0.0, places=9)
        self.assertAlmostEqual(dist, 0.0, places=9)
        self.assertTrue(axis_lines_colinear(moved, axis2))

    def test_alignment_maps_origin_onto_origin(self):
        axis1 = ((1.0, 2.0, 3.0), (0.0, 1.0, 0.0))
        axis2 = ((5.0, 5.0, 5.0), (0.0, 0.0, 1.0))
        mat = joint_alignment_matrix(axis1[0], axis1[1], axis2[0], axis2[1])
        moved_origin = transform_point(mat, axis1[0])
        for got, want in zip(moved_origin, axis2[0]):
            self.assertAlmostEqual(got, want)


class ParameterMatrixTests(unittest.TestCase):
    def test_rotation_parameter_keeps_axis_fixed(self):
        origin = (1.0, 1.0, 0.0)
        direction = (0.0, 0.0, 1.0)
        mat = rotation_parameter_matrix(math.pi / 3, origin, direction)
        moved = transform_point(mat, origin)
        for got, want in zip(moved, origin):
            self.assertAlmostEqual(got, want)

    def test_rotation_parameter_rotates_offaxis_point(self):
        mat = rotation_parameter_matrix(math.pi, (0.0, 0.0, 0.0),
                                        (0.0, 0.0, 1.0))
        moved = transform_point(mat, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(moved[0], -1.0)
        self.assertAlmostEqual(moved[1], 0.0)

    def test_offset_no_flip_is_pure_translation(self):
        mat = offset_parameter_matrix(3.0, (0.0, 0.0, 0.0), (0.0, 0.0, 2.0))
        moved = transform_point(mat, (1.0, 2.0, 0.0))
        self.assertAlmostEqual(moved[2], 3.0)
        self.assertAlmostEqual(moved[0], 1.0)

    def test_flip_reflects_through_origin_plane(self):
        mat = offset_parameter_matrix(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                                      flip=True)
        moved = transform_point(mat, (1.0, 2.0, 4.0))
        self.assertAlmostEqual(moved[0], 1.0)
        self.assertAlmostEqual(moved[1], 2.0)
        self.assertAlmostEqual(moved[2], -4.0)

    def test_flip_about_shifted_plane(self):
        mat = offset_parameter_matrix(0.0, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0),
                                      flip=True)
        moved = transform_point(mat, (0.0, 0.0, 3.0))
        self.assertAlmostEqual(moved[2], -1.0)

    def test_flip_is_involutive(self):
        mat = offset_parameter_matrix(0.0, (0.0, 0.0, 2.0), (0.0, 1.0, 0.0),
                                      flip=True)
        p = (1.0, 5.0, 2.0)
        back = transform_point(mat, transform_point(mat, p))
        for got, want in zip(back, p):
            self.assertAlmostEqual(got, want)


class FullTransformTests(unittest.TestCase):
    def test_default_parameters_equal_alignment(self):
        axis1 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        axis2 = ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        full = joint_transform_from_parameters(axis1[0], axis1[1],
                                               axis2[0], axis2[1])
        align = joint_alignment_matrix(axis1[0], axis1[1], axis2[0], axis2[1])
        for row_a, row_b in zip(full, align):
            for a, b in zip(row_a, row_b):
                self.assertAlmostEqual(a, b)

    def test_offset_moves_along_axis_and_keeps_colinearity(self):
        axis1 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        axis2 = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        mat = joint_transform_from_parameters(axis1[0], axis1[1], axis2[0],
                                              axis2[1], offset=4.0)
        moved = apply_joint_transform_to_axis(mat, axis1)
        self.assertAlmostEqual(moved[0][2], 4.0)
        self.assertTrue(axis_lines_colinear(moved, axis2))

    def test_rotation_about_axis_preserves_colinearity(self):
        axis1 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        axis2 = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        mat = joint_transform_from_parameters(axis1[0], axis1[1], axis2[0],
                                              axis2[1], rotation=math.pi / 4)
        moved = apply_joint_transform_to_axis(mat, axis1)
        self.assertTrue(axis_lines_colinear(moved, axis2))

    def test_flip_preserves_axis_line(self):
        axis1 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        axis2 = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        mat = joint_transform_from_parameters(axis1[0], axis1[1], axis2[0],
                                              axis2[1], flip=True)
        moved = apply_joint_transform_to_axis(mat, axis1)
        self.assertTrue(axis_lines_colinear(moved, axis2))

    def test_deterministic(self):
        args = ((0.0, 1.0, 2.0), (0.0, 1.0, 0.0), (3.0, 0.0, 0.0),
                (1.0, 1.0, 0.0))
        a = joint_transform_from_parameters(*args, offset=1.5,
                                            rotation=0.7, flip=True)
        b = joint_transform_from_parameters(*args, offset=1.5,
                                            rotation=0.7, flip=True)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
