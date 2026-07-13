"""Tests for the SSRL geometric self-supervision pretext-target constructor."""

from __future__ import annotations

import math
import unittest

from harnesscad.data.dataengine.augment.ssrl_rasterization_pretext import (
    UVBox,
    RasterSample,
    bounding_box,
    reparameterize,
    reparameterizer,
    point_in_polygon,
    distance_to_boundary,
    signed_distance,
    eval_plane,
    eval_cylinder,
    eval_sphere,
    eval_cone,
    eval_surface,
    boundary_biased_points,
    build_targets,
    nearest_opposite_sdf,
    l2_reconstruction_loss,
)

# A unit square clip centred away from the origin, at arbitrary scale.
SQUARE = [(2.0, 3.0), (6.0, 3.0), (6.0, 7.0), (2.0, 7.0)]
UNIT_SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]


class BoundingBoxTests(unittest.TestCase):
    def test_bbox(self):
        box = bounding_box(SQUARE)
        self.assertEqual(box, UVBox(2.0, 3.0, 6.0, 7.0))
        self.assertEqual(box.u_span, 4.0)
        self.assertEqual(box.v_span, 4.0)

    def test_bbox_rejects_degenerate(self):
        with self.assertRaises(ValueError):
            bounding_box([(0.0, 0.0), (1.0, 1.0)])


class ReparameterizeTests(unittest.TestCase):
    def test_maps_bbox_to_unit_square(self):
        out = reparameterize(SQUARE)
        self.assertEqual(out, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    def test_already_unit_square_is_fixed_point(self):
        self.assertEqual(reparameterize(UNIT_SQUARE), UNIT_SQUARE)

    def test_degenerate_axis_maps_to_half(self):
        # Zero v-span polygon (colinear in v) -> v maps to 0.5.
        poly = [(0.0, 5.0), (2.0, 5.0), (4.0, 5.0)]
        out = reparameterize(poly)
        for _, v in out:
            self.assertEqual(v, 0.5)

    def test_reparameterizer_matches_reparameterize(self):
        fn = reparameterizer(SQUARE)
        self.assertEqual([fn(p) for p in SQUARE], reparameterize(SQUARE))
        # An interior point maps proportionally.
        self.assertEqual(fn((4.0, 5.0)), (0.5, 0.5))


class PointInPolygonTests(unittest.TestCase):
    def test_interior_and_exterior(self):
        self.assertTrue(point_in_polygon((0.5, 0.5), UNIT_SQUARE))
        self.assertFalse(point_in_polygon((1.5, 0.5), UNIT_SQUARE))
        self.assertFalse(point_in_polygon((-0.2, 0.5), UNIT_SQUARE))

    def test_boundary_counts_inside(self):
        self.assertTrue(point_in_polygon((0.5, 0.0), UNIT_SQUARE))
        self.assertTrue(point_in_polygon((0.0, 0.0), UNIT_SQUARE))

    def test_triangle(self):
        self.assertTrue(point_in_polygon((0.2, 0.2), TRIANGLE))
        self.assertFalse(point_in_polygon((0.8, 0.8), TRIANGLE))


class SignedDistanceTests(unittest.TestCase):
    def test_center_negative_and_magnitude(self):
        # Centre of the unit square: distance 0.5 to the nearest edge, inside.
        self.assertAlmostEqual(signed_distance((0.5, 0.5), UNIT_SQUARE), -0.5)

    def test_outside_positive(self):
        self.assertAlmostEqual(signed_distance((2.0, 0.5), UNIT_SQUARE), 1.0)

    def test_on_boundary_zero(self):
        self.assertAlmostEqual(signed_distance((0.5, 0.0), UNIT_SQUARE), 0.0)

    def test_distance_to_boundary_unsigned(self):
        self.assertAlmostEqual(distance_to_boundary((0.5, 0.5), UNIT_SQUARE), 0.5)
        self.assertAlmostEqual(distance_to_boundary((2.0, 0.5), UNIT_SQUARE), 1.0)


class SurfaceTests(unittest.TestCase):
    def test_plane(self):
        self.assertEqual(eval_plane(2.0, 3.0), (2.0, 3.0, 0.0))
        self.assertEqual(
            eval_plane(1.0, 1.0, origin=(1.0, 0.0, 0.0),
                       x_axis=(0.0, 1.0, 0.0), y_axis=(0.0, 0.0, 1.0)),
            (1.0, 1.0, 1.0),
        )

    def test_cylinder(self):
        x, y, z = eval_cylinder(0.0, 5.0, radius=2.0)
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, 5.0)

    def test_sphere_radius_preserved(self):
        x, y, z = eval_sphere(0.7, 1.2, radius=3.0)
        self.assertAlmostEqual(math.sqrt(x * x + y * y + z * z), 3.0)

    def test_cone_apex(self):
        x, y, z = eval_cone(1.0, 0.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, 0.0)

    def test_dispatch_and_unknown(self):
        self.assertEqual(eval_surface("plane", 1.0, 2.0), (1.0, 2.0, 0.0))
        with self.assertRaises(ValueError):
            eval_surface("torus", 0.0, 0.0)


class BoundaryBiasedSamplingTests(unittest.TestCase):
    def test_count_and_determinism(self):
        a = boundary_biased_points(UNIT_SQUARE, 50, seed=7)
        b = boundary_biased_points(UNIT_SQUARE, 50, seed=7)
        self.assertEqual(len(a), 50)
        self.assertEqual(a, b)

    def test_different_seed_differs(self):
        a = boundary_biased_points(UNIT_SQUARE, 50, seed=1)
        b = boundary_biased_points(UNIT_SQUARE, 50, seed=2)
        self.assertNotEqual(a, b)

    def test_boundary_points_are_near_zero_level(self):
        pts = boundary_biased_points(UNIT_SQUARE, 40, seed=3,
                                     boundary_fraction=0.5, pool_multiplier=20)
        # The nearest-boundary half should have small |sdf|.
        near = pts[:20]
        far_like = pts[20:]
        near_mean = sum(abs(signed_distance(p, UNIT_SQUARE)) for p in near) / 20
        rest_mean = sum(abs(signed_distance(p, UNIT_SQUARE))
                        for p in far_like) / len(far_like)
        self.assertLess(near_mean, rest_mean)

    def test_rejects_bad_args(self):
        with self.assertRaises(ValueError):
            boundary_biased_points(UNIT_SQUARE, 0, seed=1)
        with self.assertRaises(ValueError):
            boundary_biased_points(UNIT_SQUARE, 10, seed=1, boundary_fraction=2.0)


class BuildTargetsTests(unittest.TestCase):
    def test_targets_shape_and_determinism(self):
        t1 = build_targets(SQUARE, "plane", 30, seed=11)
        t2 = build_targets(SQUARE, "plane", 30, seed=11)
        self.assertEqual(len(t1), 30)
        self.assertTrue(all(isinstance(s, RasterSample) for s in t1))
        self.assertEqual([s.sdf for s in t1], [s.sdf for s in t2])

    def test_reparam_normalises_uv_into_square(self):
        # With reparam, sampled uv should lie in the padded unit square.
        t = build_targets(SQUARE, "plane", 40, seed=5)
        for s in t:
            self.assertGreaterEqual(s.u, -0.1 - 1e-9)
            self.assertLessEqual(s.u, 1.1 + 1e-9)

    def test_plane_xyz_matches_uv(self):
        t = build_targets(UNIT_SQUARE, "plane", 10, seed=2, reparam=False)
        for s in t:
            self.assertAlmostEqual(s.xyz[0], s.u)
            self.assertAlmostEqual(s.xyz[1], s.v)
            self.assertAlmostEqual(s.xyz[2], 0.0)

    def test_surface_params_passed(self):
        t = build_targets(UNIT_SQUARE, "cylinder", 5, seed=1, reparam=False,
                          surface_params={"radius": 4.0})
        for s in t:
            planar_r = math.hypot(s.xyz[0], s.xyz[1])
            self.assertAlmostEqual(planar_r, 4.0)


class NearestOppositeSDFTests(unittest.TestCase):
    def test_sign_matches_membership(self):
        pts = [(0.5, 0.5), (2.0, 0.5), (0.1, 0.1), (5.0, 5.0)]
        sd = nearest_opposite_sdf(pts, UNIT_SQUARE)
        self.assertLess(sd[0], 0.0)   # inside
        self.assertGreater(sd[1], 0.0)  # outside
        self.assertLess(sd[2], 0.0)   # inside
        self.assertGreater(sd[3], 0.0)  # outside

    def test_all_inside_falls_back_to_boundary(self):
        pts = [(0.4, 0.4), (0.6, 0.6)]
        sd = nearest_opposite_sdf(pts, UNIT_SQUARE)
        # No outside point available -> exact boundary distance, negative sign.
        self.assertAlmostEqual(sd[0], -distance_to_boundary(pts[0], UNIT_SQUARE))


class L2LossTests(unittest.TestCase):
    def test_zero_loss_on_perfect_prediction(self):
        t = build_targets(UNIT_SQUARE, "plane", 8, seed=4, reparam=False)
        preds = [(s.xyz[0], s.xyz[1], s.xyz[2], s.sdf) for s in t]
        self.assertAlmostEqual(l2_reconstruction_loss(preds, t), 0.0)

    def test_positive_loss_and_validation(self):
        t = build_targets(UNIT_SQUARE, "plane", 4, seed=4, reparam=False)
        preds = [(0.0, 0.0, 0.0, 0.0) for _ in t]
        self.assertGreater(l2_reconstruction_loss(preds, t), 0.0)
        with self.assertRaises(ValueError):
            l2_reconstruction_loss([(0.0, 0.0, 0.0)], t[:1])
        with self.assertRaises(ValueError):
            l2_reconstruction_loss(preds[:1], t)


if __name__ == "__main__":
    unittest.main()
