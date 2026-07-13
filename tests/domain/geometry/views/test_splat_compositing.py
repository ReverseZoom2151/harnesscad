"""Tests for geometry.turbo3d_splat_compositing."""

import unittest

from harnesscad.domain.geometry.views.splat_compositing import (
    alpha_from_kernel,
    composite_front_to_back,
    sort_front_to_back,
    tile_bins,
    tile_grid_shape,
)


class TileBinningTest(unittest.TestCase):
    def test_grid_shape_rounds_up(self):
        self.assertEqual(tile_grid_shape(256, 256, 16), (16, 16))
        self.assertEqual(tile_grid_shape(255, 100, 16), (16, 7))

    def test_single_tile(self):
        bins = tile_bins((2.0, 2.0, 10.0, 10.0), 64, 64, 16)
        self.assertEqual(bins, [(0, 0)])

    def test_box_spans_multiple_tiles(self):
        # spans columns 0..1 and rows 0..1 -> 4 tiles, row-major
        bins = tile_bins((10.0, 10.0, 20.0, 20.0), 64, 64, 16)
        self.assertEqual(bins, [(0, 0), (1, 0), (0, 1), (1, 1)])

    def test_fully_outside_is_culled(self):
        self.assertEqual(tile_bins((100.0, 100.0, 120.0, 120.0), 64, 64, 16), [])
        self.assertEqual(tile_bins((-50.0, -50.0, -1.0, -1.0), 64, 64, 16), [])

    def test_clipped_to_image(self):
        # box extends past the right/bottom edge; only valid tiles returned
        bins = tile_bins((-10.0, -10.0, 300.0, 300.0), 32, 32, 16)
        self.assertEqual(bins, [(0, 0), (1, 0), (0, 1), (1, 1)])

    def test_bad_bbox_raises(self):
        with self.assertRaises(ValueError):
            tile_bins((10.0, 0.0, 5.0, 5.0), 64, 64, 16)
        with self.assertRaises(ValueError):
            tile_bins((0.0, 0.0, 5.0, 5.0), 64, 64, 0)


class AlphaFromKernelTest(unittest.TestCase):
    def test_product(self):
        self.assertAlmostEqual(alpha_from_kernel(0.5, 0.4), 0.2)

    def test_clamp_high(self):
        self.assertEqual(alpha_from_kernel(1.0, 1.0), 0.999)

    def test_zero(self):
        self.assertEqual(alpha_from_kernel(0.0, 1.0), 0.0)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            alpha_from_kernel(1.5, 0.5)
        with self.assertRaises(ValueError):
            alpha_from_kernel(0.5, -0.1)


class SortFrontToBackTest(unittest.TestCase):
    def test_ascending_depth(self):
        self.assertEqual(sort_front_to_back([3.0, 1.0, 2.0]), [1, 2, 0])

    def test_stable_on_ties(self):
        self.assertEqual(sort_front_to_back([1.0, 1.0, 0.5]), [2, 0, 1])


class CompositeFrontToBackTest(unittest.TestCase):
    def test_single_opaque_splat(self):
        color, alpha = composite_front_to_back([[1.0, 0.0, 0.0]], [1.0])
        self.assertAlmostEqual(color[0], 1.0)
        self.assertAlmostEqual(alpha, 1.0)

    def test_two_layer_over(self):
        # front alpha 0.5 red, back alpha 1.0 blue
        color, alpha = composite_front_to_back(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0.5, 1.0]
        )
        # red contrib 0.5, blue contrib 0.5*1.0 = 0.5
        self.assertAlmostEqual(color[0], 0.5)
        self.assertAlmostEqual(color[2], 0.5)
        self.assertAlmostEqual(alpha, 1.0)

    def test_transmittance_accumulation(self):
        # three splats alpha 0.5 each -> alpha = 1 - 0.5^3 = 0.875
        _, alpha = composite_front_to_back(
            [[1.0], [1.0], [1.0]], [0.5, 0.5, 0.5]
        )
        self.assertAlmostEqual(alpha, 0.875)

    def test_order_matters(self):
        near_first, _ = composite_front_to_back(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [0.5, 0.5]
        )
        far_first, _ = composite_front_to_back(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], [0.5, 0.5]
        )
        self.assertNotAlmostEqual(near_first[0], far_first[0])

    def test_background_blend(self):
        # transparent scene -> pure background
        color, alpha = composite_front_to_back(
            [], [], background=[0.2, 0.3, 0.4]
        )
        self.assertAlmostEqual(color[0], 0.2)
        self.assertAlmostEqual(color[2], 0.4)
        self.assertAlmostEqual(alpha, 0.0)

    def test_partial_background(self):
        color, _ = composite_front_to_back(
            [[1.0]], [0.25], background=[0.0]
        )
        # front 0.25*1 + background 0.75*0 = 0.25
        self.assertAlmostEqual(color[0], 0.25)

    def test_early_termination(self):
        # after enough opaque layers, later ones do not change result
        many = [[float(i)] for i in range(50)]
        alphas = [0.9] * 50
        color, alpha = composite_front_to_back(many, alphas, min_transmittance=1e-3)
        # near-fully opaque; alpha close to 1
        self.assertGreaterEqual(alpha, 0.999)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            composite_front_to_back([[1.0]], [0.5, 0.5])

    def test_bad_alpha_raises(self):
        with self.assertRaises(ValueError):
            composite_front_to_back([[1.0]], [1.5])


if __name__ == "__main__":
    unittest.main()
