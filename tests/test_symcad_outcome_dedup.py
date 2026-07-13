import unittest

from harnesscad.data.dataengine.curation.outcome_dedup import (
    dedup_report,
    deduplicate,
    duplicate_groups,
    outcome_signature,
)


class TestSignature(unittest.TestCase):
    def test_rounding_absorbs_noise(self):
        a = outcome_signature([1.0000001, 2.0])
        b = outcome_signature([1.0000002, 2.0])
        self.assertEqual(a, b)

    def test_timeout_token(self):
        sig = outcome_signature([None, 3.0])
        self.assertEqual(sig[0], "T")
        self.assertEqual(sig[1], 3.0)

    def test_timeout_matches_timeout_not_finite(self):
        self.assertEqual(outcome_signature([None]), outcome_signature([None]))
        self.assertNotEqual(outcome_signature([None]), outcome_signature([0.0]))

    def test_decimals_control(self):
        a = outcome_signature([1.234567, 2.0], decimals=2)
        b = outcome_signature([1.234999, 2.0], decimals=2)
        self.assertEqual(a, b)


class TestDeduplicate(unittest.TestCase):
    def setUp(self):
        # rows 0 and 2 are identical outcome vectors; row 3 differs
        self.data = [
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 31.0],
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
            [5.0, 5.0, 5.0],
        ]

    def test_deduplicate_first_seen_wins(self):
        kept = deduplicate(self.data, return_indices=True)
        self.assertEqual(kept, [0, 1, 4])

    def test_deduplicate_returns_instances(self):
        out = deduplicate(self.data)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], [10.0, 20.0, 30.0])

    def test_duplicate_groups(self):
        groups = duplicate_groups(self.data)
        self.assertEqual(groups, [[0, 2, 3]])

    def test_no_duplicates(self):
        data = [[1.0], [2.0], [3.0]]
        self.assertEqual(deduplicate(data, return_indices=True), [0, 1, 2])
        self.assertEqual(duplicate_groups(data), [])

    def test_empty(self):
        self.assertEqual(deduplicate([]), [])
        self.assertEqual(duplicate_groups([]), [])

    def test_timeout_vectors_dedup(self):
        data = [[None, 5.0], [None, 5.0], [1.0, 5.0]]
        self.assertEqual(deduplicate(data, return_indices=True), [0, 2])


class TestReport(unittest.TestCase):
    def test_report_counts(self):
        data = [[1.0], [1.0], [1.0], [2.0]]
        rep = dedup_report(data)
        self.assertEqual(rep.n_before, 4)
        self.assertEqual(rep.n_after, 2)
        self.assertEqual(rep.n_removed, 2)
        self.assertEqual(rep.n_duplicate_groups, 1)
        self.assertAlmostEqual(rep.reduction_ratio, 0.5)

    def test_report_empty(self):
        rep = dedup_report([])
        self.assertEqual(rep.n_before, 0)
        self.assertEqual(rep.reduction_ratio, 0.0)

    def test_report_does_not_mutate(self):
        data = [[1.0], [1.0]]
        dedup_report(data)
        self.assertEqual(len(data), 2)


if __name__ == "__main__":
    unittest.main()
