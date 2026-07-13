import math
import unittest
from dataclasses import replace

from harnesscad.domain.geometry.cqcontrib_enclosure import (
    EnclosureError,
    EnclosureSpec,
    fillet_order,
    plan_enclosure,
    rounded_rect_area,
    validate_spec,
)

DEFAULT = EnclosureSpec()  # the contrib Parametric_Enclosure defaults


class TestRoundedRectArea(unittest.TestCase):
    def test_zero_radius_is_rectangle(self):
        self.assertAlmostEqual(rounded_rect_area(10, 20, 0), 200.0)

    def test_full_radius_is_circle_capped_rect(self):
        # radius = half the smaller side
        a = rounded_rect_area(10, 10, 5)
        self.assertAlmostEqual(a, 100.0 - (4 - math.pi) * 25.0)
        self.assertAlmostEqual(a, math.pi * 25.0)

    def test_invalid(self):
        with self.assertRaises(EnclosureError):
            rounded_rect_area(10, 20, 6)
        with self.assertRaises(EnclosureError):
            rounded_rect_area(0, 20, 1)


class TestFilletOrder(unittest.TestCase):
    def test_larger_first(self):
        self.assertEqual(fillet_order(10.0, 2.0), (10.0, 2.0))
        self.assertEqual(fillet_order(2.0, 10.0), (10.0, 2.0))

    def test_equal_keeps_side_first(self):
        self.assertEqual(fillet_order(3.0, 3.0), (3.0, 3.0))

    def test_negative(self):
        with self.assertRaises(EnclosureError):
            fillet_order(-1.0, 2.0)


class TestValidation(unittest.TestCase):
    def test_default_spec_is_valid(self):
        self.assertEqual(validate_spec(DEFAULT), [])

    def test_side_radius_must_exceed_thickness(self):
        errs = validate_spec(replace(DEFAULT, side_radius=2.0))
        self.assertTrue(any("side radius must exceed" in e for e in errs))

    def test_posts_outside_footprint(self):
        errs = validate_spec(replace(DEFAULT, screwpost_inset=4.0))
        self.assertTrue(any("stick out" in e for e in errs))

    def test_post_od_le_id(self):
        errs = validate_spec(replace(DEFAULT, screwpost_od=4.0))
        self.assertTrue(any("OD must exceed" in e for e in errs))

    def test_thick_walls(self):
        errs = validate_spec(replace(DEFAULT, thickness=60.0))
        self.assertTrue(errs)

    def test_plan_raises_on_invalid(self):
        with self.assertRaises(EnclosureError):
            plan_enclosure(replace(DEFAULT, side_radius=1.0))

    def test_errors_sorted(self):
        errs = validate_spec(replace(DEFAULT, thickness=60.0, screwpost_od=1.0))
        self.assertEqual(errs, sorted(errs))


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.plan = plan_enclosure(DEFAULT)

    def test_inner_dims(self):
        self.assertAlmostEqual(self.plan.inner_width, 94.0)
        self.assertAlmostEqual(self.plan.inner_length, 144.0)
        self.assertAlmostEqual(self.plan.inner_height, 44.0)
        self.assertAlmostEqual(self.plan.inner_side_radius, 7.0)

    def test_post_centers(self):
        c = self.plan.post_centers
        self.assertEqual(len(c), 4)
        self.assertIn((-38.0, -63.0), c)
        self.assertIn((38.0, 63.0), c)
        self.assertEqual(c, tuple(sorted(c)))

    def test_post_height_and_split(self):
        self.assertAlmostEqual(self.plan.post_height, 100.0 - 100.0 + 48.0)
        self.assertAlmostEqual(self.plan.post_height, 48.0)
        self.assertAlmostEqual(self.plan.lid_split_z, 47.0)

    def test_lip_footprint_matches_cavity(self):
        self.assertAlmostEqual(self.plan.lip_width, self.plan.inner_width)
        self.assertAlmostEqual(self.plan.lip_length, self.plan.inner_length)

    def test_volumes_positive_and_ordered(self):
        p = self.plan
        self.assertGreater(p.outer_volume, p.cavity_volume)
        self.assertGreater(p.cavity_volume, 0.0)
        self.assertGreater(p.post_volume, 0.0)
        self.assertGreater(p.lid_volume, 0.0)
        self.assertGreater(p.body_volume, 0.0)

    def test_post_volume_formula(self):
        ring = math.pi * (25.0 - 4.0)
        self.assertAlmostEqual(self.plan.post_volume, 4.0 * ring * 48.0)

    def test_fillet_order_from_spec(self):
        self.assertEqual(self.plan.fillet_order, (10.0, 2.0))

    def test_determinism(self):
        self.assertEqual(plan_enclosure(DEFAULT), plan_enclosure(DEFAULT))

    def test_scaling_thickness_shrinks_cavity(self):
        thick = plan_enclosure(replace(DEFAULT, thickness=5.0))
        self.assertLess(thick.cavity_volume, self.plan.cavity_volume)


if __name__ == "__main__":
    unittest.main()
