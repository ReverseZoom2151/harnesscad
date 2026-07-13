import unittest

from harnesscad.domain.numeric.mamtiff_pyramid import (
    scale_lengths, downsample, upsample, build_pyramid,
    laplacian_pyramid, reconstruct_laplacian, pyramid_pool_flops,
)


class TestScaleLengths(unittest.TestCase):
    def test_256_schedule(self):
        self.assertEqual(scale_lengths(256, 3, 2), (256, 128, 64))

    def test_floor_at_one(self):
        self.assertEqual(scale_lengths(3, 4, 2), (3, 1, 1, 1))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            scale_lengths(0, 3)
        with self.assertRaises(ValueError):
            scale_lengths(256, 0)
        with self.assertRaises(ValueError):
            scale_lengths(256, 3, 1)


class TestDownsample(unittest.TestCase):
    def test_avg_pool_half(self):
        seq = ((0.0, 0.0), (2.0, 4.0), (1.0, 1.0), (3.0, 3.0))
        out = downsample(seq, 2, "avg")
        self.assertEqual(out, ((1.0, 2.0), (2.0, 2.0)))

    def test_max_pool(self):
        seq = ((0.0, 5.0), (2.0, 4.0), (1.0, 1.0), (3.0, 9.0))
        out = downsample(seq, 2, "max")
        self.assertEqual(out, ((2.0, 5.0), (3.0, 9.0)))

    def test_partial_trailing_window(self):
        seq = ((0.0,), (2.0,), (10.0,))
        out = downsample(seq, 2, "avg")
        # ceil(3/2) = 2 outputs; last window has one token
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], (1.0,))
        self.assertEqual(out[1], (10.0,))

    def test_bad_factor(self):
        with self.assertRaises(ValueError):
            downsample(((1.0,),), 1)

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            downsample(((1.0,), (2.0,)), 2, "nope")


class TestUpsample(unittest.TestCase):
    def test_nearest_double(self):
        seq = ((0.0,), (10.0,))
        out = upsample(seq, 4, "nearest")
        self.assertEqual(len(out), 4)
        self.assertEqual(out[0], (0.0,))
        self.assertEqual(out[-1], (10.0,))

    def test_linear_endpoints_preserved(self):
        seq = ((0.0,), (10.0,))
        out = upsample(seq, 5, "linear")
        self.assertEqual(out[0], (0.0,))
        self.assertEqual(out[-1], (10.0,))
        # midpoint of a 5-sample linear ramp from 0..10 is 5
        self.assertAlmostEqual(out[2][0], 5.0)

    def test_identity_when_equal(self):
        seq = ((1.0,), (2.0,))
        self.assertEqual(upsample(seq, 2), seq)

    def test_single_token_repeat(self):
        self.assertEqual(upsample(((7.0,),), 3), ((7.0,), (7.0,), (7.0,)))

    def test_shrink_rejected(self):
        with self.assertRaises(ValueError):
            upsample(((1.0,), (2.0,), (3.0,)), 2)


class TestBuildPyramid(unittest.TestCase):
    def test_level0_is_input(self):
        seq = tuple((float(i),) for i in range(8))
        pyr = build_pyramid(seq, 4, 2, "avg")
        self.assertEqual(pyr[0], seq)
        self.assertEqual(len(pyr), 4)
        self.assertEqual([len(p) for p in pyr], [8, 4, 2, 1])

    def test_stops_at_one(self):
        seq = ((1.0,), (2.0,))
        pyr = build_pyramid(seq, 5, 2)
        # once length 1 is reached it repeats
        self.assertTrue(all(len(p) == 1 for p in pyr[2:]))


class TestLaplacian(unittest.TestCase):
    def test_lossless_reconstruction(self):
        seq = tuple((float(i), float(2 * i)) for i in range(8))
        details, coarsest = laplacian_pyramid(seq, 4, 2, "linear")
        rec = reconstruct_laplacian(details, coarsest, "linear")
        self.assertEqual(len(rec), len(seq))
        for a, b in zip(rec, seq):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=9)

    def test_detail_bands_count(self):
        seq = tuple((float(i),) for i in range(8))
        details, coarsest = laplacian_pyramid(seq, 3, 2)
        self.assertEqual(len(details), 2)
        self.assertEqual(len(coarsest), 2)


class TestFlops(unittest.TestCase):
    def test_bounded_by_geometric_series(self):
        n, d, lv = 256, 64, 4
        flops = pyramid_pool_flops(n, d, lv, 2)
        # strictly less than 2 * n * d (geometric bound factor/(factor-1)=2)
        self.assertLess(flops, 2 * n * d)
        self.assertGreater(flops, n * d)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            pyramid_pool_flops(0, 1, 1)


if __name__ == "__main__":
    unittest.main()
