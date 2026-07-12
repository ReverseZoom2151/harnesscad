import unittest

from reconstruction.skexgen_code_layout import (
    CODE_LEN, EXTRUDE_CODEBOOK, TOPOLOGY_CODEBOOK, branch_histogram, branch_of,
    codebook_usage, code_model_vocab, filter_valid, is_valid_code, join_code,
    split_code, swap_branch, unique_codes,
)

CODE = [1, 2, 3, 4, 100, 200, 10, 20, 30, 40]


class TestLayout(unittest.TestCase):
    def test_lengths(self):
        self.assertEqual(CODE_LEN, 10)
        self.assertEqual(code_model_vocab(), 1000)

    def test_branch_of(self):
        self.assertEqual([branch_of(i) for i in range(10)],
                         ["topology"] * 4 + ["geometry"] * 2 + ["extrude"] * 4)
        self.assertRaises(ValueError, branch_of, 10)
        self.assertRaises(ValueError, branch_of, -1)

    def test_split(self):
        parts = split_code(CODE)
        self.assertEqual(parts["topology"], [1, 2, 3, 4])
        self.assertEqual(parts["geometry"], [100, 200])
        self.assertEqual(parts["extrude"], [10, 20, 30, 40])
        self.assertRaises(ValueError, split_code, [1, 2, 3])

    def test_join_roundtrip(self):
        parts = split_code(CODE)
        self.assertEqual(join_code(parts["topology"], parts["geometry"],
                                   parts["extrude"]), CODE)
        self.assertRaises(ValueError, join_code, [1], [1, 2], [1, 2, 3, 4])

    def test_swap_branch(self):
        out = swap_branch(CODE, "geometry", [7, 8])
        self.assertEqual(split_code(out)["geometry"], [7, 8])
        self.assertEqual(split_code(out)["topology"], [1, 2, 3, 4])
        self.assertRaises(ValueError, swap_branch, CODE, "colour", [1, 2])
        self.assertRaises(ValueError, swap_branch, CODE, "geometry", [1])


class TestValidity(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(is_valid_code(CODE))

    def test_topology_codebook_is_smaller(self):
        bad = list(CODE)
        bad[0] = TOPOLOGY_CODEBOOK        # 500 is out of range for topology
        self.assertFalse(is_valid_code(bad))
        ok = list(CODE)
        ok[4] = TOPOLOGY_CODEBOOK         # but fine for geometry
        self.assertTrue(is_valid_code(ok))

    def test_out_of_range(self):
        bad = list(CODE)
        bad[9] = EXTRUDE_CODEBOOK
        self.assertFalse(is_valid_code(bad))
        neg = list(CODE)
        neg[3] = -1
        self.assertFalse(is_valid_code(neg))

    def test_wrong_length(self):
        self.assertFalse(is_valid_code([1, 2, 3]))

    def test_filter_valid(self):
        bad = list(CODE)
        bad[0] = 999
        self.assertEqual(filter_valid([CODE, bad]), [CODE])


class TestStats(unittest.TestCase):
    def test_unique_codes(self):
        other = list(CODE)
        other[0] = 9
        self.assertEqual(unique_codes([CODE, list(CODE), other]), [CODE, other])

    def test_codebook_usage(self):
        usage = codebook_usage([CODE])
        self.assertAlmostEqual(usage["topology"], 4 / 500)
        self.assertAlmostEqual(usage["geometry"], 2 / 1000)
        self.assertAlmostEqual(usage["extrude"], 4 / 1000)

    def test_usage_dedups_repeated_codes(self):
        code = [1, 1, 1, 1, 5, 5, 2, 2, 2, 2]
        usage = codebook_usage([code])
        self.assertAlmostEqual(usage["topology"], 1 / 500)

    def test_histogram(self):
        hist = branch_histogram([CODE, CODE], "geometry")
        self.assertEqual(hist, {100: 2, 200: 2})
        self.assertRaises(ValueError, branch_histogram, [CODE], "nope")

    def test_empty(self):
        self.assertEqual(unique_codes([]), [])
        self.assertEqual(codebook_usage([])["topology"], 0.0)


if __name__ == "__main__":
    unittest.main()
