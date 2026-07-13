import math
import unittest

from harnesscad.domain.geometry.graphcad_orientation import (
    IDENTITY,
    align_z_to,
    apply,
    axis_direction,
    compose,
    is_rotation,
    node_rotation,
    normalize,
    parse_orientation,
    parse_rotation,
    radial_from,
    resolve_orientation,
    resolve_rotation,
    rotation_about,
    tilt_then_spin,
)

Z = (0.0, 0.0, 1.0)


def assert_close(case, actual, expected, places=9):
    for got, want in zip(actual, expected):
        case.assertAlmostEqual(got, want, places=places)


class VectorTests(unittest.TestCase):
    def test_normalize(self):
        assert_close(self, normalize((0.0, 3.0, 4.0)), (0.0, 0.6, 0.8))

    def test_normalize_zero(self):
        with self.assertRaises(ValueError):
            normalize((0.0, 0.0, 0.0))

    def test_axis_tokens(self):
        self.assertEqual(axis_direction("+X"), (1.0, 0.0, 0.0))
        self.assertEqual(axis_direction("-Z"), (0.0, 0.0, -1.0))
        self.assertEqual(axis_direction("y"), (0.0, 1.0, 0.0))
        self.assertEqual(axis_direction("–Y"), (0.0, -1.0, 0.0))  # en dash from the spec

    def test_unknown_axis(self):
        with self.assertRaises(ValueError):
            axis_direction("+W")

    def test_radial_from(self):
        assert_close(self, radial_from((2.0, 0.0, 0.0), (0.0, 0.0, 0.0)), (1.0, 0.0, 0.0))


class RotationTests(unittest.TestCase):
    def test_rotation_about_z(self):
        matrix = rotation_about((0.0, 0.0, 1.0), 90.0)
        assert_close(self, apply(matrix, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))
        self.assertTrue(is_rotation(matrix))

    def test_align_z_to_each_axis(self):
        for token in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
            direction = axis_direction(token)
            matrix = align_z_to(direction)
            assert_close(self, apply(matrix, Z), direction)
            self.assertTrue(is_rotation(matrix))

    def test_align_z_to_plus_z_is_identity(self):
        self.assertEqual(align_z_to((0.0, 0.0, 5.0)), IDENTITY)

    def test_align_z_to_minus_z_is_a_half_turn(self):
        matrix = align_z_to((0.0, 0.0, -1.0))
        assert_close(self, apply(matrix, Z), (0.0, 0.0, -1.0))
        self.assertTrue(is_rotation(matrix))

    def test_align_z_to_oblique(self):
        direction = normalize((1.0, 1.0, 1.0))
        assert_close(self, apply(align_z_to(direction), Z), direction)

    def test_compose_order(self):
        spin = rotation_about((0.0, 0.0, 1.0), 90.0)
        tilt = rotation_about((1.0, 0.0, 0.0), 90.0)
        # compose(spin, tilt) applies tilt first: +Z -> -Y -> ... -> (0,0,1)x
        assert_close(self, apply(compose(spin, tilt), Z), (1.0, 0.0, 0.0))

    def test_is_rotation_rejects_scaling(self):
        self.assertFalse(is_rotation(((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))))


class ParseTests(unittest.TestCase):
    def test_parse_axis_orientation(self):
        kind, payload = parse_orientation("orientation=axis:+Y")
        self.assertEqual(kind, "axis")
        self.assertEqual(payload, (0.0, 1.0, 0.0))

    def test_parse_relation_orientation(self):
        self.assertEqual(
            parse_orientation("orientation = axis:radial_from body"),
            ("radial_from", "body"),
        )
        self.assertEqual(
            parse_orientation("axis:tangent_to shell"), ("tangent_to", "shell")
        )
        self.assertEqual(parse_orientation("orientation=normal:plate"), ("normal", "plate"))

    def test_parse_orientation_rejects_junk(self):
        with self.assertRaises(ValueError):
            parse_orientation("orientation=sideways")

    def test_parse_rotation_axis_angle(self):
        self.assertEqual(parse_rotation("rotation = axis:X,30°"), ("axis_angle", ("X", 30.0)))

    def test_parse_tilt_then_spin(self):
        kind, payload = parse_rotation(
            "rotation = tilt_then_spin(tilt=X,15°, spin=Z,240°)"
        )
        self.assertEqual(kind, "tilt_then_spin")
        self.assertEqual(payload, ("X", 15.0, "Z", 240.0))

    def test_parse_rotation_rejects_junk(self):
        with self.assertRaises(ValueError):
            parse_rotation("rotation=a bit")


class ResolveTests(unittest.TestCase):
    def test_axis_orientation(self):
        matrix = resolve_orientation("orientation=axis:+X")
        assert_close(self, apply(matrix, Z), (1.0, 0.0, 0.0))

    def test_radial_from_points_away(self):
        matrix = resolve_orientation(
            "orientation=axis:radial_from body", center=(1.0, 0.0, 0.0),
            reference_center=(0.0, 0.0, 0.0)
        )
        assert_close(self, apply(matrix, Z), (1.0, 0.0, 0.0))

    def test_normal_points_at_the_target(self):
        matrix = resolve_orientation(
            "orientation=normal:body", center=(1.0, 0.0, 0.0),
            reference_center=(0.0, 0.0, 0.0)
        )
        assert_close(self, apply(matrix, Z), (-1.0, 0.0, 0.0))

    def test_tangent_is_perpendicular_to_radial(self):
        center = (2.0, 0.0, 0.0)
        reference = (0.0, 0.0, 0.0)
        matrix = resolve_orientation(
            "orientation=axis:tangent_to body", center=center, reference_center=reference
        )
        local_z = apply(matrix, Z)
        radial = radial_from(center, reference)
        self.assertAlmostEqual(sum(a * b for a, b in zip(local_z, radial)), 0.0)

    def test_relation_needs_centers(self):
        with self.assertRaises(ValueError):
            resolve_orientation("orientation=axis:radial_from body")

    def test_resolve_rotation(self):
        matrix = resolve_rotation("rotation=axis:Z,90")
        assert_close(self, apply(matrix, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))

    def test_tilt_then_spin_distributes_legs(self):
        directions = []
        for index in range(3):
            matrix = tilt_then_spin("X", 15.0, "Z", index * 120.0)
            self.assertTrue(is_rotation(matrix))
            directions.append(apply(matrix, Z))
        # Every leg keeps the same tilt from vertical.
        for direction in directions:
            self.assertAlmostEqual(direction[2], math.cos(math.radians(15.0)))
        # And they are evenly spread in azimuth.
        first = math.atan2(directions[0][1], directions[0][0])
        second = math.atan2(directions[1][1], directions[1][0])
        spread = math.degrees((second - first) % (2 * math.pi))
        self.assertAlmostEqual(spread, 120.0, places=6)


class NodeRotationTests(unittest.TestCase):
    def test_native_pose_by_default(self):
        self.assertEqual(node_rotation(), IDENTITY)

    def test_orientation_only(self):
        assert_close(self, apply(node_rotation("orientation=axis:-Y"), Z), (0.0, -1.0, 0.0))

    def test_orientation_then_rotation(self):
        matrix = node_rotation("orientation=axis:+X", "rotation=axis:Z,90")
        # rotation is about the *local* Z, which orientation already sent to +X,
        # so local +Z still lands on world +X.
        assert_close(self, apply(matrix, Z), (1.0, 0.0, 0.0))
        self.assertTrue(is_rotation(matrix))

    def test_rotation_only_tilts_from_native_pose(self):
        matrix = node_rotation(rotation="rotation=axis:X,90")
        assert_close(self, apply(matrix, Z), (0.0, -1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
