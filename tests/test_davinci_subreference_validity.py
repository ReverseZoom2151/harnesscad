import unittest

from harnesscad.io.ingest.constraint_validity import (
    END, ENTIRE, MIDDLE, START, canonical_pair, enumerate_candidates,
    filter_constraints, is_valid_pair, is_valid_subreference,
    valid_subreferences,
)


class TestValidSubrefs(unittest.TestCase):
    def test_per_type_sets(self):
        self.assertEqual(valid_subreferences("arc"), frozenset({1, 2, 3, 4}))
        self.assertEqual(valid_subreferences("circle"), frozenset({2, 3}))
        self.assertEqual(valid_subreferences("line"), frozenset({1, 2, 4}))
        self.assertEqual(valid_subreferences("point"), frozenset({4}))

    def test_line_has_no_middle(self):
        self.assertFalse(is_valid_subreference("line", MIDDLE))
        self.assertTrue(is_valid_subreference("line", START))

    def test_point_entire_only(self):
        self.assertTrue(is_valid_subreference("point", ENTIRE))
        self.assertFalse(is_valid_subreference("point", START))

    def test_unknown_type(self):
        with self.assertRaises(KeyError):
            valid_subreferences("bezier")


class TestPairValidity(unittest.TestCase):
    def test_invalid_circle_start_line_end(self):
        # <circle.start - line.end> must be rejected (paper's own example).
        self.assertFalse(is_valid_pair("circle", START, "line", END))

    def test_valid_line_line(self):
        self.assertTrue(is_valid_pair("line", START, "line", END))

    def test_canonical_pair_symmetry(self):
        self.assertEqual(canonical_pair(0, 1, 2, 4), canonical_pair(2, 4, 0, 1))


class TestEnumerate(unittest.TestCase):
    def test_none_slots_excluded(self):
        cands = enumerate_candidates(("line", "none"))
        for (a, b) in cands:
            self.assertNotEqual(a[0], 1)
            self.assertNotEqual(b[0], 1)

    def test_symmetric_dedup(self):
        cands = enumerate_candidates(("line", "point"))
        # keys are canonical; no reversed duplicates
        for k in cands:
            self.assertEqual(k, canonical_pair(k[0][0], k[0][1], k[1][0], k[1][1]))

    def test_self_pairs_toggle(self):
        with_self = enumerate_candidates(("arc",), include_self=True)
        without = enumerate_candidates(("arc",), include_self=False)
        self.assertTrue(with_self)
        self.assertEqual(without, frozenset())


class TestFilterConstraints(unittest.TestCase):
    def setUp(self):
        self.types = ("line", "circle", "point")

    def test_keeps_valid(self):
        res = filter_constraints([(0, START, 1, MIDDLE)], self.types)
        self.assertTrue(res.all_valid)
        self.assertEqual(len(res.valid), 1)

    def test_rejects_invalid_subref(self):
        # circle.start is illegal
        res = filter_constraints([(1, START, 2, ENTIRE)], self.types)
        self.assertFalse(res.all_valid)
        self.assertIn("invalid-subref", res.invalid[0][1])

    def test_rejects_out_of_range(self):
        res = filter_constraints([(0, START, 9, ENTIRE)], self.types)
        self.assertEqual(res.invalid[0][1], "index-out-of-range")

    def test_rejects_empty_slot(self):
        res = filter_constraints([(0, START, 1, ENTIRE)], ("line", "none"))
        self.assertEqual(res.invalid[0][1], "references-empty-slot")

    def test_rejects_degenerate_self_edge(self):
        res = filter_constraints([(0, START, 0, START)], self.types)
        self.assertEqual(res.invalid[0][1], "degenerate-self-edge")


if __name__ == "__main__":
    unittest.main()
