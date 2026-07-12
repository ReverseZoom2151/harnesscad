"""Tests for editing.voxhammer_preserve."""
import unittest

from editing.voxhammer_preserve import (
    attention_allow_mask,
    blend_latents,
    blend_vectors,
    hard_replace,
    kv_replace,
    masked_kv_replace,
)


class TestBlendVectors(unittest.TestCase):
    def test_w_one_selects_new(self):
        self.assertEqual(blend_vectors((1.0, 2.0), (9.0, 9.0), 1.0), (1.0, 2.0))

    def test_w_zero_selects_preserved(self):
        self.assertEqual(blend_vectors((1.0, 2.0), (9.0, 8.0), 0.0), (9.0, 8.0))

    def test_half(self):
        self.assertEqual(blend_vectors((0.0, 0.0), (2.0, 4.0), 0.5), (1.0, 2.0))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            blend_vectors((1.0,), (1.0, 2.0), 0.5)


class TestBlendLatents(unittest.TestCase):
    def test_masked_blend(self):
        new = {(0, 0, 0): (1.0,), (1, 0, 0): (1.0,)}
        preserved = {(0, 0, 0): (5.0,), (1, 0, 0): (5.0,)}
        mask = {(0, 0, 0): 1.0, (1, 0, 0): 0.0}
        out = blend_latents(new, preserved, mask)
        self.assertEqual(out[(0, 0, 0)], (1.0,))  # edited -> new
        self.assertEqual(out[(1, 0, 0)], (5.0,))  # preserved -> cached

    def test_missing_mask_defaults_preserved(self):
        new = {(0, 0, 0): (1.0,)}
        preserved = {(0, 0, 0): (7.0,)}
        out = blend_latents(new, preserved, {})
        self.assertEqual(out[(0, 0, 0)], (7.0,))

    def test_coord_without_preserved_kept(self):
        new = {(2, 2, 2): (3.0,)}
        out = blend_latents(new, {}, {})
        self.assertEqual(out[(2, 2, 2)], (3.0,))


class TestHardReplace(unittest.TestCase):
    def test_keep_overwritten(self):
        latents = {(0, 0, 0): (1.0,), (1, 0, 0): (2.0,)}
        cached = {(0, 0, 0): (9.0,), (1, 0, 0): (8.0,)}
        out = hard_replace(latents, cached, {(0, 0, 0)})
        self.assertEqual(out[(0, 0, 0)], (9.0,))  # keep -> cached
        self.assertEqual(out[(1, 0, 0)], (2.0,))  # edit -> unchanged

    def test_does_not_mutate_input(self):
        latents = {(0, 0, 0): (1.0,)}
        cached = {(0, 0, 0): (9.0,)}
        hard_replace(latents, cached, {(0, 0, 0)})
        self.assertEqual(latents[(0, 0, 0)], (1.0,))

    def test_missing_cache_kept(self):
        latents = {(0, 0, 0): (1.0,)}
        out = hard_replace(latents, {}, {(0, 0, 0)})
        self.assertEqual(out[(0, 0, 0)], (1.0,))


class TestKVReplace(unittest.TestCase):
    def test_single_token(self):
        self.assertEqual(kv_replace((1.0, 1.0), (3.0, 3.0), 0.0), (3.0, 3.0))

    def test_masked_map(self):
        new = {"t0": (1.0,), "t1": (1.0,)}
        cache = {"t0": (5.0,), "t1": (5.0,)}
        w = {"t0": 1.0, "t1": 0.0}
        out = masked_kv_replace(new, cache, w)
        self.assertEqual(out["t0"], (1.0,))
        self.assertEqual(out["t1"], (5.0,))

    def test_uncached_token_kept(self):
        out = masked_kv_replace({"x": (2.0,)}, {}, {})
        self.assertEqual(out["x"], (2.0,))


class TestAttentionAllowMask(unittest.TestCase):
    def test_same_group_allowed(self):
        a = attention_allow_mask([True, True, False])
        self.assertTrue(a[0][1])   # edit-edit
        self.assertTrue(a[2][2])   # preserved-preserved
        self.assertFalse(a[0][2])  # edit-preserved blocked
        self.assertFalse(a[2][0])

    def test_symmetric(self):
        a = attention_allow_mask([True, False, True, False])
        n = len(a)
        for i in range(n):
            for j in range(n):
                self.assertEqual(a[i][j], a[j][i])

    def test_empty(self):
        self.assertEqual(attention_allow_mask([]), [])


if __name__ == "__main__":
    unittest.main()
