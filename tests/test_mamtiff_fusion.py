import math
import unittest

from harnesscad.domain.numeric.mamtiff_fusion import (
    window_mask, softmax, masked_attention, sigmoid, adaptive_fusion,
    sequence_aware_pe, multiscale_attention_flops,
)

NEG_INF = float("-inf")


class TestWindowMask(unittest.TestCase):
    def test_local_band(self):
        m = window_mask(4, 1)
        self.assertEqual(m[0][0], 0.0)
        self.assertEqual(m[0][1], 0.0)
        self.assertEqual(m[0][2], NEG_INF)
        self.assertEqual(m[0][3], NEG_INF)
        self.assertEqual(m[2][1], 0.0)
        self.assertEqual(m[2][3], 0.0)

    def test_global_all_zero(self):
        m = window_mask(3, None)
        for row in m:
            self.assertTrue(all(x == 0.0 for x in row))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            window_mask(0, 1)
        with self.assertRaises(ValueError):
            window_mask(4, -1)


class TestSoftmax(unittest.TestCase):
    def test_sums_to_one(self):
        s = softmax((1.0, 2.0, 3.0))
        self.assertAlmostEqual(sum(s), 1.0)
        self.assertTrue(s[2] > s[1] > s[0])

    def test_masked_entries_zero(self):
        s = softmax((0.0, NEG_INF, 0.0))
        self.assertAlmostEqual(s[1], 0.0)
        self.assertAlmostEqual(s[0], 0.5)
        self.assertAlmostEqual(s[2], 0.5)

    def test_all_masked_uniform(self):
        s = softmax((NEG_INF, NEG_INF))
        self.assertEqual(s, (0.5, 0.5))


class TestMaskedAttention(unittest.TestCase):
    def test_identity_values_recovered(self):
        # With uniform K (all equal) attention weights are uniform, so output
        # is the mean of V.
        q = ((1.0, 0.0), (0.0, 1.0))
        k = ((0.0, 0.0), (0.0, 0.0))
        v = ((2.0,), (4.0,))
        out = masked_attention(q, k, v)
        self.assertAlmostEqual(out[0][0], 3.0)
        self.assertAlmostEqual(out[1][0], 3.0)

    def test_window_restricts_attention(self):
        # A local window of 0 means each query attends only to itself.
        q = ((0.0,), (0.0,), (0.0,))
        k = ((0.0,), (0.0,), (0.0,))
        v = ((1.0,), (2.0,), (3.0,))
        m = window_mask(3, 0)
        out = masked_attention(q, k, v, m)
        self.assertAlmostEqual(out[0][0], 1.0)
        self.assertAlmostEqual(out[1][0], 2.0)
        self.assertAlmostEqual(out[2][0], 3.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            masked_attention(((1.0,),), ((1.0,), (2.0,)), ((1.0,),))

    def test_empty(self):
        self.assertEqual(masked_attention((), (), ()), ())


class TestAdaptiveFusion(unittest.TestCase):
    def test_default_identity_gate(self):
        hl = ((1.0,),)
        hm = ((2.0,),)
        hg = ((3.0,),)
        out = adaptive_fusion(hl, hm, hg)
        # concat = [1,2,3]; gate = sigmoid; gated = c*sigmoid(c)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 3)
        self.assertAlmostEqual(out[0][0], 1.0 * sigmoid(1.0))
        self.assertAlmostEqual(out[0][2], 3.0 * sigmoid(3.0))

    def test_projection(self):
        hl = ((1.0,),)
        hm = ((1.0,),)
        hg = ((1.0,),)
        # w_out sums the 3-dim gated vector into 1 dim
        w_out = ((1.0, 1.0, 1.0),)
        out = adaptive_fusion(hl, hm, hg, w_out=w_out)
        expected = 3.0 * (1.0 * sigmoid(1.0))
        self.assertAlmostEqual(out[0][0], expected)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            adaptive_fusion(((1.0,), (2.0,)), ((1.0,),), ((1.0,),))


class TestSequenceAwarePE(unittest.TestCase):
    def test_eta_zero_is_identity(self):
        seq = ((1.0, 2.0), (3.0, 4.0))
        out = sequence_aware_pe(seq, eta=0.0)
        self.assertEqual(out, seq)

    def test_position_zero_adds_sin0_cos0(self):
        seq = ((0.0, 0.0),)
        out = sequence_aware_pe(seq, eta=1.0)
        # pos 0: even channel sin(0)=0, odd channel cos(0)=1
        self.assertAlmostEqual(out[0][0], 0.0)
        self.assertAlmostEqual(out[0][1], 1.0)

    def test_eta_scales(self):
        seq = ((0.0, 0.0),)
        out = sequence_aware_pe(seq, eta=2.0)
        self.assertAlmostEqual(out[0][1], 2.0)

    def test_empty(self):
        self.assertEqual(sequence_aware_pe(()), ())


class TestFlops(unittest.TestCase):
    def test_local_cheaper_than_global(self):
        n, d = 256, 64
        local = multiscale_attention_flops(n, d, (64,))
        glob = multiscale_attention_flops(n, d, (None,))
        self.assertLess(local, glob)

    def test_combined_branches(self):
        n, d = 256, 64
        total = multiscale_attention_flops(n, d, (64, 128, None))
        self.assertGreater(total, 0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            multiscale_attention_flops(0, 1, (1,))


if __name__ == "__main__":
    unittest.main()
