"""Tests for numeric.makeashape_wavelet_transform (3D DWT + round-trip)."""
import math
import random
import unittest

from harnesscad.domain.numeric.wavelet_transform import (
    Grid3D, WAVELETS, SUBBAND_NAMES, DETAIL_NAMES,
    wavelet_forward_1d, wavelet_inverse_1d,
    dwt3_level, idwt3_level, dwt3, idwt3, WaveletDecomposition,
)


def _rng_grid(dims, seed=0):
    rng = random.Random(seed)
    n = dims[0] * dims[1] * dims[2]
    return Grid3D(dims, [rng.uniform(-3.0, 3.0) for _ in range(n)])


class OneDFilterBankTests(unittest.TestCase):
    def test_haar_lengths_and_roundtrip(self):
        x = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
        low, high = wavelet_forward_1d(x, "haar")
        self.assertEqual(len(low), 3)
        self.assertEqual(len(high), 3)
        back = wavelet_inverse_1d(low, high, "haar")
        for a, b in zip(x, back):
            self.assertAlmostEqual(a, b, places=10)

    def test_haar_is_orthonormal_energy_preserving(self):
        x = [4.0, -1.0, 2.0, 7.0]
        low, high = wavelet_forward_1d(x, "haar")
        e_in = sum(v * v for v in x)
        e_out = sum(v * v for v in low) + sum(v * v for v in high)
        self.assertAlmostEqual(e_in, e_out, places=10)

    def test_bior53_roundtrip(self):
        rng = random.Random(1)
        x = [rng.uniform(-5, 5) for _ in range(8)]
        low, high = wavelet_forward_1d(x, "bior53")
        back = wavelet_inverse_1d(low, high, "bior53")
        for a, b in zip(x, back):
            self.assertAlmostEqual(a, b, places=10)

    def test_odd_length_rejected(self):
        with self.assertRaises(ValueError):
            wavelet_forward_1d([1.0, 2.0, 3.0], "haar")

    def test_unknown_wavelet_rejected(self):
        with self.assertRaises(ValueError):
            wavelet_forward_1d([1.0, 2.0], "db4")


class SingleLevel3DTests(unittest.TestCase):
    def test_subband_dims_and_names(self):
        grid = _rng_grid((4, 4, 4), seed=2)
        subbands, sub = dwt3_level(grid, "haar")
        self.assertEqual(set(subbands), set(SUBBAND_NAMES))
        self.assertEqual(sub, (2, 2, 2))
        for name in SUBBAND_NAMES:
            self.assertEqual(subbands[name].dims, (2, 2, 2))

    def test_single_level_roundtrip_haar(self):
        grid = _rng_grid((4, 6, 8), seed=3)
        subbands, _ = dwt3_level(grid, "haar")
        recon = idwt3_level(subbands, "haar")
        self.assertEqual(recon.dims, grid.dims)
        for a, b in zip(grid.data, recon.data):
            self.assertAlmostEqual(a, b, places=9)

    def test_single_level_roundtrip_bior53(self):
        grid = _rng_grid((4, 4, 4), seed=4)
        subbands, _ = dwt3_level(grid, "bior53")
        recon = idwt3_level(subbands, "bior53")
        for a, b in zip(grid.data, recon.data):
            self.assertAlmostEqual(a, b, places=9)

    def test_constant_grid_has_zero_detail(self):
        grid = Grid3D((4, 4, 4), [2.5] * 64)
        subbands, _ = dwt3_level(grid, "haar")
        for name in DETAIL_NAMES:
            self.assertAlmostEqual(subbands[name].max_abs(), 0.0, places=10)
        # All energy is in the coarse band.
        self.assertGreater(subbands["LLL"].max_abs(), 0.0)

    def test_odd_dim_rejected(self):
        with self.assertRaises(ValueError):
            dwt3_level(_rng_grid((3, 4, 4)), "haar")


class MultiLevelTests(unittest.TestCase):
    def test_multilevel_shapes(self):
        grid = _rng_grid((8, 8, 8), seed=5)
        decomp = dwt3(grid, levels=3, wavelet="haar")
        self.assertEqual(len(decomp.details), 3)
        self.assertEqual(decomp.coarse.dims, (1, 1, 1))
        # finest first
        self.assertEqual(decomp.details[0]["HHH"].dims, (4, 4, 4))
        self.assertEqual(decomp.details[1]["HHH"].dims, (2, 2, 2))
        self.assertEqual(decomp.details[2]["HHH"].dims, (1, 1, 1))

    def test_multilevel_roundtrip(self):
        grid = _rng_grid((8, 8, 8), seed=6)
        for wavelet in WAVELETS:
            decomp = dwt3(grid, levels=3, wavelet=wavelet)
            recon = idwt3(decomp)
            self.assertEqual(recon.dims, grid.dims)
            for a, b in zip(grid.data, recon.data):
                self.assertAlmostEqual(a, b, places=8)

    def test_levels_must_divide(self):
        with self.assertRaises(ValueError):
            dwt3(_rng_grid((8, 8, 8)), levels=4)  # 16 does not divide 8

    def test_from_function_and_get(self):
        grid = Grid3D.from_function((2, 3, 4), lambda x, y, z: x * 100 + y * 10 + z)
        self.assertEqual(grid.get(1, 2, 3), 123.0)
        self.assertEqual(grid.dims, (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
