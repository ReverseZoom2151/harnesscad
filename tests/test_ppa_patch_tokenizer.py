"""Tests for deterministic PPA image-patch tokenisation."""

import unittest

from harnesscad.domain.vision import ppa_patch_tokenizer as pt


class TestPatchGrid(unittest.TestCase):
    def test_128_into_32(self):
        g = pt.patch_grid(128, 32)
        self.assertEqual(g.per_side, 4)
        self.assertEqual(g.num_patches, 16)
        self.assertEqual(g.token_dim, 32 * 32)

    def test_non_divisible_raises(self):
        with self.assertRaises(ValueError):
            pt.patch_grid(128, 30)

    def test_nonpositive_raises(self):
        with self.assertRaises(ValueError):
            pt.patch_grid(0, 4)


class TestTokenize(unittest.TestCase):
    def setUp(self):
        # 4x4 grid, values = row*10 + col so patches are easy to verify
        self.grid = tuple(tuple(r * 10 + c for c in range(4)) for r in range(4))

    def test_token_count_and_dim(self):
        tokens = pt.tokenize(self.grid, 2)
        self.assertEqual(len(tokens), 4)       # 2x2 patches
        self.assertEqual(len(tokens[0]), 4)    # 2x2 pixels each

    def test_top_left_patch_contents(self):
        tokens = pt.tokenize(self.grid, 2)
        # top-left patch rows 0-1 cols 0-1: (0,1,10,11)
        self.assertEqual(tokens[0], (0, 1, 10, 11))
        # top-right patch cols 2-3: (2,3,12,13)
        self.assertEqual(tokens[1], (2, 3, 12, 13))
        # bottom-left patch rows 2-3 cols 0-1: (20,21,30,31)
        self.assertEqual(tokens[2], (20, 21, 30, 31))

    def test_deterministic(self):
        self.assertEqual(pt.tokenize(self.grid, 2), pt.tokenize(self.grid, 2))

    def test_full_image_single_patch(self):
        tokens = pt.tokenize(self.grid, 4)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(len(tokens[0]), 16)

    def test_non_square_grid_raises(self):
        with self.assertRaises(ValueError):
            pt.tokenize([[1, 2, 3], [4, 5, 6]], 1)


class TestRoundTrip(unittest.TestCase):
    def test_detokenize_inverts(self):
        grid = tuple(tuple(r * 8 + c for c in range(8)) for r in range(8))
        tokens = pt.tokenize(grid, 4)
        g = pt.patch_grid(8, 4)
        rebuilt = pt.detokenize(tokens, 4, g.per_side)
        self.assertEqual(rebuilt, grid)

    def test_detokenize_bad_count(self):
        with self.assertRaises(ValueError):
            pt.detokenize([(1,)], 1, 3)


class TestOccupancyAndPixels(unittest.TestCase):
    def test_occupancy(self):
        grid = ((1, 1, 0, 0),
                (1, 1, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0))
        occ = pt.patch_occupancy(grid, 2)
        self.assertEqual(occ[0], 1.0)   # fully lit top-left patch
        self.assertEqual(occ[1], 0.0)
        self.assertEqual(occ[3], 0.0)

    def test_grid_from_pixels(self):
        grid = pt.grid_from_pixels(3, {(0, 0), (2, 1)})
        self.assertEqual(grid[0][0], 1)   # y=0,x=0
        self.assertEqual(grid[1][2], 1)   # y=1,x=2
        self.assertEqual(grid[2][2], 0)
        # out-of-range pixels ignored
        grid2 = pt.grid_from_pixels(2, {(5, 5)})
        self.assertEqual(grid2, ((0, 0), (0, 0)))

    def test_pixels_then_tokenize(self):
        grid = pt.grid_from_pixels(4, {(0, 0), (3, 3)})
        tokens = pt.tokenize(grid, 2)
        self.assertEqual(tokens[0], (1, 0, 0, 0))   # (0,0) lit in top-left patch
        self.assertEqual(tokens[3], (0, 0, 0, 1))   # (3,3) lit in bottom-right patch


if __name__ == "__main__":
    unittest.main()
