"""Tests for drawings.rastercad_codec."""

from __future__ import annotations

import unittest

from harnesscad.domain.drawings.rastercad_codec import (
    TokenStream,
    decode_blocks,
    decode_tokens,
    encode_blocks,
    encode_tokens,
    latent_shape,
    roundtrip_tokens,
)


class TestLatentShape(unittest.TestCase):
    def test_recad_vae_shape(self) -> None:
        # RECAD: 32x32 sketch, downsample factor 8 -> 4x4 latent grid.
        self.assertEqual(latent_shape(32, 8), 4)

    def test_ceil_behaviour(self) -> None:
        self.assertEqual(latent_shape(33, 8), 5)
        self.assertEqual(latent_shape(8, 8), 1)
        self.assertEqual(latent_shape(1, 8), 1)

    def test_invalid(self) -> None:
        with self.assertRaises(ValueError):
            latent_shape(0, 8)
        with self.assertRaises(ValueError):
            latent_shape(8, 0)


class TestBlockCodec(unittest.TestCase):
    def test_encode_shape(self) -> None:
        grid = [[0] * 32 for _ in range(32)]
        coarse = encode_blocks(grid, factor=8, levels=5)
        self.assertEqual(len(coarse), 4)
        self.assertEqual(len(coarse[0]), 4)

    def test_empty_canvas_all_zero(self) -> None:
        grid = [[0] * 16 for _ in range(16)]
        coarse = encode_blocks(grid, factor=8, levels=5)
        self.assertTrue(all(v == 0 for row in coarse for v in row))

    def test_full_canvas_max_level(self) -> None:
        grid = [[1] * 16 for _ in range(16)]
        coarse = encode_blocks(grid, factor=8, levels=5)
        self.assertTrue(all(v == 4 for row in coarse for v in row))

    def test_block_constant_roundtrip_exact(self) -> None:
        # A canvas whose blocks are each fully 0 or fully 1 must reconstruct
        # exactly through the lossy block codec.
        grid = [[0] * 16 for _ in range(16)]
        for r in range(8):
            for c in range(8):
                grid[r][c] = 1  # top-left block fully filled
        coarse = encode_blocks(grid, factor=8, levels=5)
        recon = decode_blocks(coarse, factor=8, out_height=16, out_width=16, levels=5)
        self.assertEqual(recon, grid)

    def test_half_full_block_quantised(self) -> None:
        # A block half full -> mid level; decode at threshold 0.5 fills it.
        grid = [[0] * 8 for _ in range(8)]
        for r in range(4):
            for c in range(8):
                grid[r][c] = 1  # exactly half the 8x8 block
        coarse = encode_blocks(grid, factor=8, levels=5)
        self.assertEqual(coarse[0][0], 2)  # 0.5 * 4 = 2
        recon = decode_blocks(coarse, factor=8, out_height=8, out_width=8, levels=5)
        self.assertEqual(len(recon), 8)
        self.assertTrue(all(v == 1 for row in recon for v in row))

    def test_non_divisible_size(self) -> None:
        grid = [[1] * 10 for _ in range(10)]
        coarse = encode_blocks(grid, factor=8, levels=5)
        self.assertEqual(len(coarse), 2)
        self.assertEqual(len(coarse[0]), 2)
        recon = decode_blocks(coarse, factor=8, out_height=10, out_width=10, levels=5)
        self.assertEqual(len(recon), 10)
        self.assertEqual(len(recon[0]), 10)

    def test_decode_level_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            decode_blocks([[9]], factor=2, levels=5)

    def test_encode_rejects_non_binary(self) -> None:
        with self.assertRaises(ValueError):
            encode_blocks([[0, 2]], factor=1)

    def test_levels_must_be_at_least_two(self) -> None:
        with self.assertRaises(ValueError):
            encode_blocks([[1]], factor=1, levels=1)


class TestTokenCodec(unittest.TestCase):
    def test_roundtrip_identity_various(self) -> None:
        grids = [
            [[0]],
            [[1]],
            [[0, 1, 0, 1]],
            [[1, 1, 1], [0, 0, 0], [1, 0, 1]],
            [[0] * 5 for _ in range(7)],
            [[1] * 5 for _ in range(7)],
        ]
        for g in grids:
            self.assertEqual(roundtrip_tokens(g), g)

    def test_starts_with_zero_run_convention(self) -> None:
        stream = encode_tokens([[0, 0, 1, 1]])
        self.assertEqual(stream.runs, [2, 2])

    def test_starts_with_one_leading_zero_run(self) -> None:
        stream = encode_tokens([[1, 1, 0]])
        # Leading zero-length run encodes a canvas beginning with 1.
        self.assertEqual(stream.runs[0], 0)
        self.assertEqual(decode_tokens(stream), [[1, 1, 0]])

    def test_runs_sum_to_pixel_count(self) -> None:
        grid = [[0, 1, 1], [1, 0, 0]]
        stream = encode_tokens(grid)
        self.assertEqual(sum(stream.runs), 6)

    def test_decode_bad_sum(self) -> None:
        with self.assertRaises(ValueError):
            decode_tokens(TokenStream(height=2, width=2, runs=[1]))

    def test_decode_negative_run(self) -> None:
        with self.assertRaises(ValueError):
            decode_tokens(TokenStream(height=1, width=2, runs=[-1, 3]))

    def test_encode_rejects_non_binary(self) -> None:
        with self.assertRaises(ValueError):
            encode_tokens([[0, 5]])

    def test_ragged_grid_rejected(self) -> None:
        with self.assertRaises(ValueError):
            encode_tokens([[0, 1], [0]])


if __name__ == "__main__":
    unittest.main()
