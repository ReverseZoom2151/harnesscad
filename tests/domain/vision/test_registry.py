"""The vision surface: pixels -> CISP ops, and back."""

import unittest

from harnesscad.core.cisp.ops import AddCircle, AddLine, NewSketch
from harnesscad.core.loop import HarnessSession
from harnesscad.domain.vision import registry as V
from harnesscad.io.backends.stub import StubBackend


def square_image(n=32, lo=8, hi=24):
    """A greyscale image with a bright square outline -- deterministic input."""
    img = [[0.0] * n for _ in range(n)]
    for i in range(lo, hi):
        img[lo][i] = 1.0
        img[hi - 1][i] = 1.0
        img[i][lo] = 1.0
        img[i][hi - 1] = 1.0
    return img


class TestDiscovery(unittest.TestCase):
    def test_discovers_more_than_five_real_modules(self):
        self.assertGreater(len(V.routed_modules()), 5, V.routed_modules())

    def test_every_vision_module_has_a_route(self):
        self.assertEqual(V.unadapted(), [])

    def test_discovery_is_deterministic(self):
        self.assertEqual(V.discover(), V.discover())


class TestInverseLeg(unittest.TestCase):
    def test_an_image_traces_to_ops_that_apply(self):
        result = V.trace(square_image())
        self.assertGreater(result["edge_pixels"], 0)
        self.assertTrue(result["primitives"])

        session = HarnessSession(StubBackend())
        applied = session.apply_ops(result["ops"])
        self.assertTrue(applied.ok,
                        [d.message for d in applied.diagnostics])
        self.assertEqual(applied.applied, len(result["ops"]))

    def test_the_traced_ops_are_real_cisp(self):
        ops = V.trace(square_image())["ops"]
        self.assertIsInstance(ops[0], NewSketch)
        self.assertTrue(all(isinstance(o, (NewSketch, AddLine, AddCircle))
                            for o in ops))

    def test_tracing_is_deterministic(self):
        a = [o.to_dict() for o in V.trace(square_image())["ops"]]
        b = [o.to_dict() for o in V.trace(square_image())["ops"]]
        self.assertEqual(a, b)

    def test_a_blank_image_traces_to_nothing_rather_than_hallucinating(self):
        blank = [[0.0] * 16 for _ in range(16)]
        result = V.trace(blank)
        self.assertEqual(result["primitives"], [])


class TestCalibration(unittest.TestCase):
    def test_without_a_calibration_the_ops_are_in_PIXELS(self):
        px = V.trace(square_image())["ops"]
        mm = V.trace(square_image(), calibration=V.calibrate(16.0, 40.0))["ops"]
        self.assertNotEqual([o.to_dict() for o in px],
                            [o.to_dict() for o in mm])

    def test_a_reference_of_known_width_gives_mm_per_pixel(self):
        cal = V.calibrate(16.0, 40.0)
        self.assertAlmostEqual(cal.mm_per_pixel, 2.5)

    def test_measurement_uses_the_calibration(self):
        cal = V.calibrate(16.0, 40.0)
        w, h = V.measure(cal, [(0.0, 0.0), (16.0, 8.0)])
        self.assertAlmostEqual(w, 40.0)
        self.assertAlmostEqual(h, 20.0)


class TestForwardLeg(unittest.TestCase):
    def test_a_sketch_rasterises(self):
        image = V.rasterize(
            [{"type": "line", "start": (1.0, 1.0), "end": (60.0, 60.0)}],
            resolution=32)
        self.assertGreater(image.occupancy, 0)

    def test_the_patch_tokeniser_round_trips_exactly(self):
        grid = V.rasterize(
            [{"type": "line", "start": (1.0, 1.0), "end": (60.0, 60.0)}],
            resolution=32).to_grid()
        tokens = V.tokens(grid, 8)
        self.assertEqual(tokens["num_patches"], 16)
        self.assertEqual(
            V.detokenize(tokens["tokens"], 8, tokens["per_side"]), grid)


class TestModelBoundary(unittest.TestCase):
    def test_masking_is_deterministic_in_its_seed(self):
        grid = tuple(tuple(1 for _ in range(16)) for _ in range(16))
        self.assertEqual(V.mask(grid, 8, 0.5, seed=1).visible,
                         V.mask(grid, 8, 0.5, seed=1).visible)

    def test_point_sampling_is_deterministic_in_its_seed(self):
        grid = tuple(tuple(1 for _ in range(8)) for _ in range(8))
        self.assertEqual(V.sample_points(grid, 4, seed=2),
                         V.sample_points(grid, 4, seed=2))

    def test_the_pose_grid_is_a_stable_id_space(self):
        poses = V.poses()
        self.assertEqual(len(poses), 60)
        self.assertEqual([p["id"] for p in poses], list(range(60)))

    def test_the_residual_guard_bounds_a_correction(self):
        corrected, issues = V.guard([1.0, 2.0], [0.1, 0.1])
        self.assertEqual(issues, ())
        # The residual is MAGNITUDE-BOUNDED: a model does not get to move the
        # geometry as far as it likes.
        for base, out in zip((1.0, 2.0), corrected):
            self.assertGreater(out, base)
            self.assertLess(out - base, 0.1)

    def test_the_residual_guard_rejects_a_non_finite_correction(self):
        _out, issues = V.guard([1.0, 2.0], [float("inf"), 0.1])
        self.assertTrue(issues)

    def test_the_residual_guard_rejects_a_shape_mismatch(self):
        _out, issues = V.guard([1.0, 2.0], [0.1])
        self.assertTrue(issues)


if __name__ == "__main__":
    unittest.main()
