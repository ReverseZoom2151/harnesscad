import unittest

from harnesscad.domain.numeric.geofusion_state_space import selective_scan
from harnesscad.domain.numeric.mambacad_bidirectional_scan import (
    reverse_seq, apply_order, invert_order, scan_direction,
    merge_directions, bidirectional_scan, multidirectional_scan,
)


def _const_kernels(length, d, aval=0.5, bval=1.0, cval=1.0, gval=0.0):
    a = tuple((aval,) * d for _ in range(length))
    b = tuple((bval,) * d for _ in range(length))
    c = tuple((cval,) * d for _ in range(length))
    g = tuple((gval,) * d for _ in range(length))
    return a, b, c, g


class TestOrdering(unittest.TestCase):
    def test_reverse(self):
        seq = ((1.0,), (2.0,), (3.0,))
        self.assertEqual(reverse_seq(seq), ((3.0,), (2.0,), (1.0,)))

    def test_apply_and_invert_roundtrip(self):
        seq = ((1.0,), (2.0,), (3.0,), (4.0,))
        order = (2, 0, 3, 1)
        permuted = apply_order(seq, order)
        self.assertEqual(permuted, ((3.0,), (1.0,), (4.0,), (2.0,)))
        self.assertEqual(invert_order(permuted, order), seq)

    def test_apply_order_bad_permutation(self):
        with self.assertRaises(ValueError):
            apply_order(((1.0,), (2.0,)), (0, 0))

    def test_invert_reverse_is_reverse(self):
        seq = ((1.0,), (2.0,), (3.0,))
        rev_order = (2, 1, 0)
        self.assertEqual(invert_order(seq, rev_order), reverse_seq(seq))


class TestScanDirection(unittest.TestCase):
    def test_forward_identity_matches_raw_scan(self):
        L, d = 5, 2
        z = tuple((float(k), float(k + 1)) for k in range(L))
        a, b, c, g = _const_kernels(L, d)
        order = tuple(range(L))
        got = scan_direction(z, a, b, c, g, order)
        expected, _ = selective_scan(z, a, b, c, g)
        self.assertEqual(got, expected)

    def test_backward_realigns_to_original_positions(self):
        # Reverse scan output at position k must equal the raw scan of the
        # reversed sequence at the mirrored index.
        L, d = 4, 1
        z = tuple((float(k + 1),) for k in range(L))
        a, b, c, g = _const_kernels(L, d)
        rev_order = tuple(reversed(range(L)))
        got = scan_direction(z, a, b, c, g, rev_order)
        raw_rev, _ = selective_scan(
            reverse_seq(z), reverse_seq(a), reverse_seq(b),
            reverse_seq(c), reverse_seq(g))
        # got[k] corresponds to original position k = raw_rev[L-1-k]
        for k in range(L):
            self.assertEqual(got[k], raw_rev[L - 1 - k])


class TestMerge(unittest.TestCase):
    def test_sum(self):
        a = ((1.0, 2.0), (3.0, 4.0))
        b = ((10.0, 20.0), (30.0, 40.0))
        self.assertEqual(merge_directions((a, b), "sum"),
                         ((11.0, 22.0), (33.0, 44.0)))

    def test_average(self):
        a = ((2.0,), (4.0,))
        b = ((4.0,), (8.0,))
        self.assertEqual(merge_directions((a, b), "average"),
                         ((3.0,), (6.0,)))

    def test_concat(self):
        a = ((1.0,), (2.0,))
        b = ((3.0,), (4.0,))
        self.assertEqual(merge_directions((a, b), "concat"),
                         ((1.0, 3.0), (2.0, 4.0)))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            merge_directions((((1.0,),), (((1.0,), (2.0,)))))

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            merge_directions((((1.0,),),), "median")

    def test_empty_directions(self):
        with self.assertRaises(ValueError):
            merge_directions(())


class TestBidirectional(unittest.TestCase):
    def test_sum_equals_forward_plus_reverse(self):
        L, d = 4, 2
        z = tuple((float(k), float(-k)) for k in range(L))
        a, b, c, g = _const_kernels(L, d, aval=0.3)
        merged = bidirectional_scan(z, a, b, c, g, mode="sum")
        fwd = scan_direction(z, a, b, c, g, tuple(range(L)))
        rev = scan_direction(z, a, b, c, g, tuple(reversed(range(L))))
        for k in range(L):
            for ch in range(d):
                self.assertAlmostEqual(merged[k][ch],
                                       fwd[k][ch] + rev[k][ch])

    def test_concat_doubles_width(self):
        L, d = 3, 2
        z = tuple((1.0, 1.0) for _ in range(L))
        a, b, c, g = _const_kernels(L, d)
        merged = bidirectional_scan(z, a, b, c, g, mode="concat")
        self.assertEqual(len(merged), L)
        self.assertEqual(len(merged[0]), 2 * d)

    def test_symmetric_input_symmetric_output(self):
        # A palindromic input with constant kernels and a passthrough (g=1,
        # c=0) merged by average should itself be palindromic.
        L, d = 5, 1
        z = ((1.0,), (2.0,), (3.0,), (2.0,), (1.0,))
        a = tuple((0.5,) for _ in range(L))
        b = tuple((1.0,) for _ in range(L))
        c = tuple((1.0,) for _ in range(L))
        g = tuple((0.0,) for _ in range(L))
        merged = bidirectional_scan(z, a, b, c, g, mode="average")
        for k in range(L):
            self.assertAlmostEqual(merged[k][0], merged[L - 1 - k][0])

    def test_empty_sequence(self):
        merged = bidirectional_scan((), (), (), (), (), mode="sum")
        self.assertEqual(merged, ())


class TestMultidirectional(unittest.TestCase):
    def test_two_orders_equals_bidirectional(self):
        L, d = 4, 1
        z = tuple((float(k + 1),) for k in range(L))
        a, b, c, g = _const_kernels(L, d, aval=0.4)
        fwd_order = tuple(range(L))
        rev_order = tuple(reversed(range(L)))
        multi = multidirectional_scan(z, a, b, c, g,
                                      (fwd_order, rev_order), mode="sum")
        bi = bidirectional_scan(z, a, b, c, g, mode="sum")
        self.assertEqual(multi, bi)

    def test_single_forward_order_equals_plain_scan(self):
        L, d = 3, 2
        z = tuple((float(k), 1.0) for k in range(L))
        a, b, c, g = _const_kernels(L, d)
        multi = multidirectional_scan(z, a, b, c, g, (tuple(range(L)),))
        expected, _ = selective_scan(z, a, b, c, g)
        self.assertEqual(multi, expected)

    def test_no_orders_raises(self):
        with self.assertRaises(ValueError):
            multidirectional_scan(((1.0,),), ((1.0,),), ((1.0,),),
                                  ((1.0,),), ((1.0,),), ())


if __name__ == "__main__":
    unittest.main()
