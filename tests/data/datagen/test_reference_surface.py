"""Tests for datagen.designproc_reference_surface."""

import math
import unittest

from harnesscad.data.datagen import reference_surface as rs


class TestHeightFields(unittest.TestCase):
    def test_gaussian_peak_at_origin(self):
        # At origin the Gaussian equals its full height; decays outward.
        h0 = rs.gaussian_height(0.0, 0.0, 100.0, 7.0)
        self.assertAlmostEqual(h0, 7.0)
        h1 = rs.gaussian_height(40.0, 0.0, 100.0, 7.0)
        self.assertLess(h1, h0)

    def test_saddle_sign_flip(self):
        # Hyperbolic paraboloid: opposite signs along x vs y axes.
        self.assertGreater(rs.saddle_height(10.0, 0.0, 0.004), 0.0)
        self.assertLess(rs.saddle_height(0.0, 10.0, 0.004), 0.0)
        self.assertAlmostEqual(rs.saddle_height(0.0, 0.0, 0.004), 0.0)

    def test_wave_and_ripple(self):
        self.assertAlmostEqual(rs.wave_height(0.0, 0.0, 10.0, 0.05), 0.0)
        self.assertAlmostEqual(
            rs.ripple_height(0.0, 0.0, 8.0, 0.06), 8.0)  # cos(0) = 1


class TestNet(unittest.TestCase):
    def test_net_shape_and_span(self):
        net = rs.make_net("saddle", {"curv": 0.004}, resolution=8, span=100.0)
        self.assertEqual(len(net), 8)
        self.assertTrue(all(len(row) == 8 for row in net))
        # Corner x/y should reach +-span/2.
        xs = [pt[0] for row in net for pt in row]
        self.assertAlmostEqual(min(xs), -50.0)
        self.assertAlmostEqual(max(xs), 50.0)

    def test_resolution_guard(self):
        with self.assertRaises(ValueError):
            rs.make_net("saddle", resolution=1)

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            rs.make_net("torus")

    def test_curved_surfaces_are_non_planar(self):
        for kind in rs.SURFACE_TYPES:
            net = rs.make_net(kind, resolution=8)
            self.assertTrue(rs.is_curved(net), kind)

    def test_flat_wave_is_not_curved(self):
        # Zero amplitude wave collapses to a plane.
        net = rs.make_net("wave", {"amp": 0.0, "freq": 0.05}, resolution=6)
        self.assertFalse(rs.is_curved(net))


class TestScriptEmission(unittest.TestCase):
    def test_script_contains_cadquery_calls(self):
        script = rs.emit_script("gaussian", resolution=100)
        self.assertIn("import cadquery as cq", script)
        self.assertIn("makeSplineApprox", script)
        self.assertIn("cq.Vector(x, y, z)", script)
        self.assertIn("gaussian.step", script)

    def test_each_kind_emits_its_height_expr(self):
        self.assertIn("math.exp", rs.emit_script("gaussian"))
        self.assertIn("x**2 - y**2", rs.emit_script("saddle"))
        self.assertIn("math.sin", rs.emit_script("wave"))
        self.assertIn("math.hypot", rs.emit_script("ripple"))

    def test_line_count_positive(self):
        self.assertGreater(rs.script_line_count(rs.emit_script("saddle")), 5)

    def test_script_deterministic(self):
        a = rs.emit_script("ripple", {"amp": 5.0, "freq": 0.07})
        b = rs.emit_script("ripple", {"amp": 5.0, "freq": 0.07})
        self.assertEqual(a, b)


class TestParameterVariation(unittest.TestCase):
    def test_sweep_monotonic_first_key(self):
        sweep = rs.sweep_parameters("saddle", 5)
        curvs = [p["curv"] for p in sweep]
        self.assertEqual(curvs, sorted(curvs))
        self.assertLess(curvs[0], curvs[-1])  # shallow -> deep

    def test_sweep_single(self):
        sweep = rs.sweep_parameters("saddle", 1)
        self.assertEqual(len(sweep), 1)

    def test_vary_deterministic(self):
        a = rs.vary_parameters("wave", 123, 4)
        b = rs.vary_parameters("wave", 123, 4)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 4)

    def test_vary_within_ranges(self):
        for p in rs.vary_parameters("gaussian", 7, 20):
            self.assertGreaterEqual(p["height"], 3.0)
            self.assertLessEqual(p["height"], 15.0)

    def test_vary_seed_changes_output(self):
        self.assertNotEqual(
            rs.vary_parameters("ripple", 1, 5),
            rs.vary_parameters("ripple", 2, 5))


class TestSurfaceFamily(unittest.TestCase):
    def test_family_size_and_determinism(self):
        fam1 = rs.surface_family(42, per_kind=3)
        fam2 = rs.surface_family(42, per_kind=3)
        self.assertEqual(len(fam1), 3 * len(rs.SURFACE_TYPES))
        self.assertEqual([f["params"] for f in fam1],
                         [f["params"] for f in fam2])

    def test_family_all_curved(self):
        for f in rs.surface_family(9, per_kind=2):
            self.assertTrue(f["curved"], f["kind"])
            self.assertIn("makeSplineApprox", f["script"])

    def test_per_kind_guard(self):
        with self.assertRaises(ValueError):
            rs.surface_family(1, per_kind=0)


if __name__ == "__main__":
    unittest.main()
