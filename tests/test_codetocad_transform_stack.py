import math
import unittest

from harnesscad.domain.geometry.topology.landmarks import BoundaryBox
from harnesscad.domain.geometry.transforms.affine import (
    IDENTITY,
    TransformError,
    apply_direction,
    apply_point,
    apply_points,
    compose,
    invert_rigid,
    is_close,
    matmul,
    mirror,
    rotation_around_axis,
    rotation_euler,
    rotation_x,
    rotation_y,
    rotation_z,
    scaling,
    transform_box,
    translation,
)


def assert_point(test, actual, expected):
    for a, e in zip(actual, expected):
        test.assertAlmostEqual(a, e, places=9)


class TestPrimitives(unittest.TestCase):
    def test_translation_units(self):
        matrix = translation("10mm", 0, "1cm")
        assert_point(self, apply_point(matrix, (0, 0, 0)), (0.01, 0.0, 0.01))

    def test_translation_ignored_for_directions(self):
        matrix = translation(1, 2, 3)
        assert_point(self, apply_direction(matrix, (1, 0, 0)), (1.0, 0.0, 0.0))

    def test_scaling(self):
        matrix = scaling(2.0, 1.0, 0.5)
        assert_point(self, apply_point(matrix, (1, 1, 1)), (2.0, 1.0, 0.5))

    def test_rotation_z_90(self):
        matrix = rotation_z("90deg")
        assert_point(self, apply_point(matrix, (1, 0, 0)), (0.0, 1.0, 0.0))

    def test_rotation_x_90(self):
        matrix = rotation_x("90deg")
        assert_point(self, apply_point(matrix, (0, 1, 0)), (0.0, 0.0, 1.0))

    def test_rotation_y_90(self):
        matrix = rotation_y(math.pi / 2)
        assert_point(self, apply_point(matrix, (0, 0, 1)), (1.0, 0.0, 0.0))

    def test_identity(self):
        assert_point(self, apply_point(IDENTITY, (1, 2, 3)), (1.0, 2.0, 3.0))


class TestEuler(unittest.TestCase):
    def test_order_is_x_then_y_then_z(self):
        matrix = rotation_euler(x="90deg", y=0, z="90deg")
        expected = matmul(rotation_z("90deg"), rotation_x("90deg"))
        self.assertTrue(is_close(matrix, expected))
        assert_point(self, apply_point(matrix, (0, 1, 0)), (0.0, 0.0, 1.0))

    def test_zero(self):
        self.assertTrue(is_close(rotation_euler(), IDENTITY))

    def test_full_turn(self):
        self.assertTrue(is_close(rotation_euler(z="1turn"), IDENTITY, tolerance=1e-12))


class TestRotationAroundAxis(unittest.TestCase):
    def test_offset_axis(self):
        # Rotate 180deg about the vertical line x=1, y=0.
        matrix = rotation_around_axis((1, 0, 0), (0, 0, 1), "180deg")
        assert_point(self, apply_point(matrix, (2, 0, 0)), (0.0, 0.0, 0.0))
        assert_point(self, apply_point(matrix, (1, 0, 5)), (1.0, 0.0, 5.0))

    def test_matches_rotation_z_through_origin(self):
        matrix = rotation_around_axis((0, 0, 0), (0, 0, 1), "30deg")
        self.assertTrue(is_close(matrix, rotation_z("30deg")))

    def test_unit_expression_point(self):
        matrix = rotation_around_axis(("10mm", 0, 0), (0, 0, 1), "180deg")
        assert_point(self, apply_point(matrix, (0.02, 0, 0)), (0.0, 0.0, 0.0))

    def test_degenerate_axis(self):
        with self.assertRaises(TransformError):
            rotation_around_axis((0, 0, 0), (0, 0, 0), "90deg")

    def test_normalisation(self):
        unnormalised = rotation_around_axis((0, 0, 0), (0, 0, 7), "45deg")
        self.assertTrue(is_close(unnormalised, rotation_z("45deg")))


class TestMirror(unittest.TestCase):
    def test_planes(self):
        assert_point(self, apply_point(mirror("xy"), (1, 2, 3)), (1.0, 2.0, -3.0))
        assert_point(self, apply_point(mirror("xz"), (1, 2, 3)), (1.0, -2.0, 3.0))
        assert_point(self, apply_point(mirror("yz"), (1, 2, 3)), (-1.0, 2.0, 3.0))

    def test_through_point(self):
        matrix = mirror("yz", through=(1, 0, 0))
        assert_point(self, apply_point(matrix, (3, 0, 0)), (-1.0, 0.0, 0.0))
        assert_point(self, apply_point(matrix, (1, 5, 5)), (1.0, 5.0, 5.0))

    def test_involution(self):
        matrix = mirror("xy")
        self.assertTrue(is_close(matmul(matrix, matrix), IDENTITY))

    def test_unknown_plane(self):
        with self.assertRaises(TransformError):
            mirror("zz")


class TestCompose(unittest.TestCase):
    def test_application_order(self):
        # Rotate first, then translate.
        matrix = compose(rotation_z("90deg"), translation(1, 0, 0))
        assert_point(self, apply_point(matrix, (1, 0, 0)), (1.0, 1.0, 0.0))

    def test_reverse_order_differs(self):
        first = compose(rotation_z("90deg"), translation(1, 0, 0))
        second = compose(translation(1, 0, 0), rotation_z("90deg"))
        self.assertFalse(is_close(first, second))
        assert_point(self, apply_point(second, (0, 0, 0)), (0.0, 1.0, 0.0))

    def test_empty_compose_is_identity(self):
        self.assertTrue(is_close(compose(), IDENTITY))

    def test_apply_points(self):
        matrix = translation(1, 1, 1)
        result = apply_points(matrix, [(0, 0, 0), (1, 1, 1)])
        self.assertEqual(len(result), 2)
        assert_point(self, result[1], (2.0, 2.0, 2.0))


class TestBoxTransform(unittest.TestCase):
    def test_translate_box(self):
        box = BoundaryBox.from_extents((0, 0, 0), (2, 2, 2))
        moved = transform_box(translation(1, 0, 0), box)
        self.assertAlmostEqual(moved.x.min, 0.0)
        self.assertAlmostEqual(moved.x.max, 2.0)

    def test_rotate_box_refits_aabb(self):
        box = BoundaryBox.from_extents((0, 0, 0), (2, 4, 1))
        rotated = transform_box(rotation_z("90deg"), box)
        self.assertAlmostEqual(rotated.size[0], 4.0, places=9)
        self.assertAlmostEqual(rotated.size[1], 2.0, places=9)
        self.assertAlmostEqual(rotated.size[2], 1.0, places=9)

    def test_scale_box(self):
        box = BoundaryBox.from_extents((0, 0, 0), (2, 2, 2))
        scaled = transform_box(scaling(2.0, 2.0, 2.0), box)
        self.assertAlmostEqual(scaled.size[0], 4.0)


class TestInvert(unittest.TestCase):
    def test_round_trip(self):
        matrix = compose(rotation_euler(x="30deg", z="45deg"), translation(1, 2, 3))
        inverse = invert_rigid(matrix)
        self.assertTrue(is_close(matmul(inverse, matrix), IDENTITY, tolerance=1e-9))
        point = apply_point(inverse, apply_point(matrix, (0.5, -1.0, 2.0)))
        assert_point(self, point, (0.5, -1.0, 2.0))

    def test_rejects_scaled_matrix(self):
        with self.assertRaises(TransformError):
            invert_rigid(scaling(2.0, 2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
