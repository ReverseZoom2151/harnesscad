"""Tests for numeric.makeashape_wavelet_tree (filtering, packing, adaptive)."""
import random
import unittest

from harnesscad.domain.numeric.wavelet_transform import Grid3D, dwt3, idwt3, DETAIL_NAMES
from harnesscad.domain.numeric.wavelet_tree import (
    sibling_information, top_k_locations, truncate_top_k,
    compress_decomposition_top_k, detail_coefficient_count, nonzero_detail_count,
    importance_mask, adaptive_coordinate_set, complement_coordinate_set,
    coordinate_set_as_binary_mask, sample_complement,
    pack_diffusible, unpack_diffusible, PACKED_CHANNELS,
)


def _rng_grid(dims, seed=0):
    rng = random.Random(seed)
    n = dims[0] * dims[1] * dims[2]
    return Grid3D(dims, [rng.uniform(-3.0, 3.0) for _ in range(n)])


def _detail_level(dims, seed):
    rng = random.Random(seed)
    n = dims[0] * dims[1] * dims[2]
    return {name: Grid3D(dims, [rng.uniform(-1, 1) for _ in range(n)]) for name in DETAIL_NAMES}


class SiblingFilteringTests(unittest.TestCase):
    def test_sibling_information_takes_max(self):
        dims = (1, 1, 1)
        detail = {n: Grid3D(dims, [0.0]) for n in DETAIL_NAMES}
        detail["HHH"] = Grid3D(dims, [-5.0])
        detail["LLH"] = Grid3D(dims, [2.0])
        info = sibling_information(detail)
        self.assertAlmostEqual(info[(0, 0, 0)], 5.0)

    def test_top_k_selects_largest(self):
        detail = _detail_level((2, 2, 2), seed=1)
        # Force a known maximum location.
        for n in DETAIL_NAMES:
            data = list(detail[n].data)
            data[0] = 0.0
            detail[n] = Grid3D((2, 2, 2), data)
        detail["HHH"] = Grid3D((2, 2, 2), [9.0] + list(detail["HHH"].data[1:]))
        locs = top_k_locations(detail, 1)
        self.assertEqual(locs, [(0, 0, 0)])

    def test_top_k_is_deterministic(self):
        detail = _detail_level((3, 3, 3), seed=2)
        self.assertEqual(top_k_locations(detail, 5), top_k_locations(detail, 5))

    def test_truncate_keeps_only_top_k(self):
        detail = _detail_level((2, 2, 2), seed=3)
        trunc = truncate_top_k(detail, 2)
        kept = set(top_k_locations(detail, 2))
        nx, ny, nz = (2, 2, 2)
        for name in DETAIL_NAMES:
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        if (ix, iy, iz) in kept:
                            self.assertEqual(trunc[name].get(ix, iy, iz),
                                             detail[name].get(ix, iy, iz))
                        else:
                            self.assertEqual(trunc[name].get(ix, iy, iz), 0.0)

    def test_truncate_all_locations_is_lossless(self):
        grid = _rng_grid((8, 8, 8), seed=4)
        decomp = dwt3(grid, levels=2, wavelet="haar")
        # keep every location (k = number of D0 cells)
        big_k = 4 * 4 * 4
        comp = compress_decomposition_top_k(decomp, big_k)
        recon = idwt3(comp)
        for a, b in zip(grid.data, recon.data):
            self.assertAlmostEqual(a, b, places=8)

    def test_compression_reduces_nonzero_count(self):
        grid = _rng_grid((8, 8, 8), seed=5)
        decomp = dwt3(grid, levels=2, wavelet="haar")
        full = nonzero_detail_count(decomp)
        comp = compress_decomposition_top_k(decomp, 2)
        self.assertLess(nonzero_detail_count(comp), full)
        self.assertGreater(detail_coefficient_count(decomp), 0)


class AdaptiveCoordinateTests(unittest.TestCase):
    def test_importance_mask_threshold(self):
        band = Grid3D((2, 1, 1), [32.0, 0.5])  # v=32, thresh=1.0
        mask = importance_mask(band, ratio=32.0)
        self.assertIn((0, 0, 0), mask)
        self.assertNotIn((1, 0, 0), mask)

    def test_importance_mask_zero_band_empty(self):
        band = Grid3D((2, 2, 2), [0.0] * 8)
        self.assertEqual(importance_mask(band), set())

    def test_adaptive_set_and_complement_partition(self):
        detail = _detail_level((3, 3, 3), seed=6)
        p0 = adaptive_coordinate_set(detail)
        comp = complement_coordinate_set(detail, p0)
        self.assertEqual(len(p0) + len(comp), 27)
        self.assertEqual(p0 & comp, set())

    def test_binary_mask_marks_coords(self):
        coords = {(0, 0, 0), (1, 1, 1)}
        mask = coordinate_set_as_binary_mask(coords, (2, 2, 2))
        self.assertEqual(mask.get(0, 0, 0), 1.0)
        self.assertEqual(mask.get(1, 1, 1), 1.0)
        self.assertEqual(mask.get(0, 1, 0), 0.0)
        self.assertAlmostEqual(sum(mask.data), 2.0)

    def test_sample_complement_deterministic_and_bounded(self):
        comp = {(i, 0, 0) for i in range(10)}
        a = sample_complement(comp, 3, seed=7)
        b = sample_complement(comp, 3, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)
        # requesting more than available returns all
        self.assertEqual(len(sample_complement(comp, 100, seed=7)), 10)


class PackingTests(unittest.TestCase):
    def test_packed_channel_count(self):
        self.assertEqual(PACKED_CHANNELS, 64)

    def test_pack_unpack_roundtrip(self):
        grid = _rng_grid((8, 8, 8), seed=8)
        decomp = dwt3(grid, levels=3, wavelet="haar")
        packed = pack_diffusible(decomp)
        # coarse dims for 3 levels of an 8^3 grid is 1^3
        self.assertEqual(packed.dims, (1, 1, 1))
        self.assertEqual(packed.channels, 64)
        c0, d0, d1 = unpack_diffusible(packed)
        # C0 matches
        for a, b in zip(c0.data, decomp.coarse.data):
            self.assertAlmostEqual(a, b, places=10)
        # D0 == coarsest detail, D1 == next finer
        for name in DETAIL_NAMES:
            for a, b in zip(d0[name].data, decomp.details[-1][name].data):
                self.assertAlmostEqual(a, b, places=10)
            for a, b in zip(d1[name].data, decomp.details[-2][name].data):
                self.assertAlmostEqual(a, b, places=10)

    def test_pack_reduces_spatial_resolution(self):
        grid = _rng_grid((8, 8, 8), seed=9)
        decomp = dwt3(grid, levels=2, wavelet="haar")
        packed = pack_diffusible(decomp)
        self.assertEqual(packed.dims, (2, 2, 2))
        # spatial cells shrink from 8^3 while channels grow to 64
        self.assertEqual(len(packed.data), 2 * 2 * 2 * 64)

    def test_pack_requires_two_levels(self):
        grid = _rng_grid((4, 4, 4), seed=10)
        decomp = dwt3(grid, levels=1, wavelet="haar")
        with self.assertRaises(ValueError):
            pack_diffusible(decomp)


if __name__ == "__main__":
    unittest.main()
