"""Tests for verifiers.spatialhero_plausibility."""

import unittest

from verifiers.spatialhero_plausibility import (
    AABB,
    check_constraints,
    check_physical_plausibility,
    fill_ratio,
    all_constraints_pass,
)


def _cube(side: float) -> AABB:
    return AABB(0.0, side, 0.0, side, 0.0, side)


class TestAABB(unittest.TestCase):
    def test_extents_and_volume(self):
        b = AABB(0.0, 2.0, 0.0, 3.0, 0.0, 4.0)
        self.assertEqual(b.width, 2.0)
        self.assertEqual(b.depth, 3.0)
        self.assertEqual(b.height, 4.0)
        self.assertEqual(b.volume, 24.0)
        self.assertEqual(b.extents(), (2.0, 3.0, 4.0))

    def test_center(self):
        b = AABB(0.0, 2.0, -1.0, 1.0, 0.0, 10.0)
        self.assertEqual(b.center, (1.0, 0.0, 5.0))


class TestFillRatio(unittest.TestCase):
    def test_solid_cube(self):
        b = _cube(10.0)
        self.assertAlmostEqual(fill_ratio(1000.0, b), 1.0)

    def test_half_filled(self):
        b = _cube(10.0)
        self.assertAlmostEqual(fill_ratio(500.0, b), 0.5)

    def test_nonpositive_volume(self):
        self.assertIsNone(fill_ratio(0.0, _cube(10.0)))
        self.assertIsNone(fill_ratio(-5.0, _cube(10.0)))


class TestPlausibility(unittest.TestCase):
    def test_normal_cube_is_plausible(self):
        b = _cube(10.0)
        # solid volume half the bbox -> no issues, no warnings
        r = check_physical_plausibility(500.0, 600.0, b)
        self.assertTrue(r["plausible"])
        self.assertEqual(r["issues"], [])
        self.assertEqual(r["warnings"], [])

    def test_tiny_dimension_warns(self):
        b = AABB(0.0, 0.05, 0.0, 10.0, 0.0, 10.0)
        r = check_physical_plausibility(2.5, 40.0, b)
        self.assertTrue(any("small" in w for w in r["warnings"]))
        self.assertTrue(r["plausible"])  # warning is non-fatal

    def test_huge_dimension_warns(self):
        b = AABB(0.0, 20000.0, 0.0, 10.0, 0.0, 10.0)
        r = check_physical_plausibility(1000.0, 500.0, b)
        self.assertTrue(any("large" in w for w in r["warnings"]))

    def test_extreme_aspect_ratio_warns(self):
        b = AABB(0.0, 1000.0, 0.0, 1.0, 0.0, 1.0)
        r = check_physical_plausibility(500.0, 4000.0, b)
        self.assertTrue(any("aspect" in w.lower() for w in r["warnings"]))

    def test_low_fill_ratio_warns(self):
        b = _cube(100.0)  # bbox volume 1e6
        r = check_physical_plausibility(100.0, 500.0, b)  # fill ratio 1e-4
        self.assertTrue(any("fill ratio" in w.lower() for w in r["warnings"]))

    def test_high_fill_ratio_is_issue(self):
        b = _cube(10.0)  # bbox volume 1000
        r = check_physical_plausibility(990.0, 600.0, b)  # fill 0.99 > 0.95
        self.assertFalse(r["plausible"])
        self.assertTrue(len(r["issues"]) >= 1)

    def test_high_sa_to_vol_warns(self):
        b = _cube(10.0)
        r = check_physical_plausibility(1.0, 5000.0, b)  # sa/vol = 5000
        self.assertTrue(any("surface area" in w.lower() for w in r["warnings"]))

    def test_metrics_reported(self):
        b = _cube(10.0)
        r = check_physical_plausibility(500.0, 600.0, b)
        self.assertIn("fill_ratio", r["metrics"])
        self.assertAlmostEqual(r["metrics"]["fill_ratio"], 0.5)
        self.assertEqual(len(r["metrics"]["aspect_ratios"]), 3)


class TestConstraints(unittest.TestCase):
    def test_dimension_caps(self):
        b = AABB(0.0, 10.0, 0.0, 5.0, 0.0, 20.0)
        res = check_constraints(b, 1000.0, {"max_width": 15.0, "max_height": 15.0})
        self.assertTrue(res["max_width"])
        self.assertFalse(res["max_height"])  # 20 > 15

    def test_volume_bounds(self):
        b = _cube(10.0)
        res = check_constraints(b, 800.0, {"min_volume": 500.0, "max_volume": 1000.0})
        self.assertTrue(res["min_volume"])
        self.assertTrue(res["max_volume"])

    def test_topology_constraints(self):
        b = _cube(10.0)
        res = check_constraints(
            b, 800.0,
            {"must_be_closed": True, "min_faces": 6},
            num_faces=6, is_closed=True,
        )
        self.assertTrue(res["must_be_closed"])
        self.assertTrue(res["min_faces"])

    def test_topology_omitted_without_measurement(self):
        b = _cube(10.0)
        res = check_constraints(b, 800.0, {"must_be_closed": True, "min_faces": 6})
        self.assertNotIn("must_be_closed", res)
        self.assertNotIn("min_faces", res)

    def test_all_pass_helper(self):
        self.assertTrue(all_constraints_pass({}))
        self.assertTrue(all_constraints_pass({"a": True, "b": True}))
        self.assertFalse(all_constraints_pass({"a": True, "b": False}))


if __name__ == "__main__":
    unittest.main()
