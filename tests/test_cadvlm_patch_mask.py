import unittest

from harnesscad.domain.vision.cadvlm_patch_mask import (
    MaskedImage, apply_mask, mask_count, masked_indices, masked_mse, mse,
    patch_count, patchify, unpatchify,
)


def _grid(n):
    return tuple(tuple(r * n + c for c in range(n)) for r in range(n))


class CadVLMPatchMaskTests(unittest.TestCase):
    def test_patchify_unpatchify_roundtrip(self):
        grid = _grid(6)
        tiles = patchify(grid, patch=3)
        self.assertEqual(len(tiles), 4)
        self.assertEqual(unpatchify(tiles, patch=3), grid)

    def test_patch_count_matches_paper_geometry(self):
        # 224 / 32 = 7 -> 49 patches (Sec 4.1 / 5.1).
        self.assertEqual(patch_count(224, 32), 49)

    def test_patchify_rejects_non_divisible(self):
        with self.assertRaises(ValueError):
            patchify(_grid(5), patch=2)
        with self.assertRaises(ValueError):
            patchify((), patch=2)

    def test_masked_indices_deterministic_and_ratio(self):
        idx = masked_indices(49, ratio=0.75, seed=0)
        self.assertEqual(len(idx), round(49 * 0.75))     # 37
        self.assertEqual(idx, masked_indices(49, 0.75, 0))
        self.assertNotEqual(idx, masked_indices(49, 0.75, 1))
        self.assertEqual(list(idx), sorted(idx))

    def test_mask_count_helper(self):
        self.assertEqual(mask_count(49, 0.75), 37)
        self.assertEqual(mask_count(4, 0.0), 0)
        self.assertEqual(mask_count(4, 1.0), 4)

    def test_masked_indices_rejects_bad_ratio(self):
        with self.assertRaises(ValueError):
            masked_indices(10, ratio=1.5)

    def test_apply_mask_hides_ratio_of_patches(self):
        grid = _grid(6)
        masked = apply_mask(grid, patch=3, ratio=0.75, seed=2, fill=0)
        self.assertIsInstance(masked, MaskedImage)
        self.assertEqual(len(masked.masked), 3)          # round(4*0.75)
        self.assertEqual(len(masked.visible), 1)
        # hidden patches are zero-filled; visible ones untouched.
        original = patchify(grid, 3)
        rebuilt = patchify(masked.grid, 3)
        for i, coord in enumerate(masked.order):
            if i in masked.masked:
                self.assertTrue(all(v == 0 for row in rebuilt[coord] for v in row))
            else:
                self.assertEqual(rebuilt[coord], original[coord])

    def test_mse_and_masked_mse(self):
        a = _grid(4)
        self.assertEqual(mse(a, a), 0.0)
        b = tuple(tuple(v + 1 for v in row) for row in a)
        self.assertEqual(mse(a, b), 1.0)
        masked = apply_mask(a, patch=2, ratio=0.5, seed=0, fill=0)
        # reconstruction differing only on a visible patch scores 0 on masked_mse.
        self.assertGreaterEqual(masked_mse(masked.grid, a, masked), 0.0)
        perfect = masked_mse(a, a, masked)
        self.assertEqual(perfect, 0.0)

    def test_mse_shape_mismatch(self):
        with self.assertRaises(ValueError):
            mse(_grid(4), _grid(3))


if __name__ == "__main__":
    unittest.main()
