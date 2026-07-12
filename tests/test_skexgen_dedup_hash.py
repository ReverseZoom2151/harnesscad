import unittest

from bench.skexgen_dedup_hash import (
    BRANCHES, branch_tokens, dedup_report, deduplicate, duplicate_groups,
    duplicate_percent, record_hash, token_hash, unique_percent,
)


def _rec(uid, pix, ext):
    return {"uid": uid, "se_pix": pix, "se_ext": ext}


A = _rec(0, [[10, 3, 2, 1, 0]], [[5, 0]])
A2 = _rec(1, [[10, 3, 2, 1, 0]], [[5, 0]])          # exact duplicate of A
SAME_SKETCH = _rec(2, [[10, 3, 2, 1, 0]], [[7, 0]])  # same sketch, other extrude
SAME_EXT = _rec(3, [[11, 3, 2, 1, 0]], [[5, 0]])     # other sketch, same extrude
EMPTY = _rec(4, [], [])


class TestTokens(unittest.TestCase):
    def test_shift_and_terminate(self):
        self.assertEqual(branch_tokens(A, "s"), [11, 4, 3, 2, 1, 0])
        self.assertEqual(branch_tokens(A, "e"), [6, 1, 0])
        self.assertEqual(branch_tokens(A, "se"), [11, 4, 3, 2, 1, 0, 6, 1, 0])

    def test_bad_branch(self):
        self.assertRaises(ValueError, branch_tokens, A, "x")

    def test_token_hash_deterministic(self):
        self.assertEqual(token_hash([1, 2, 3]), token_hash([1, 2, 3]))
        self.assertNotEqual(token_hash([1, 2, 3]), token_hash([3, 2, 1]))
        self.assertEqual(len(token_hash([1])), 64)


class TestRecordHash(unittest.TestCase):
    def test_identical(self):
        for branch in BRANCHES:
            self.assertEqual(record_hash(A, branch), record_hash(A2, branch))

    def test_sketch_branch_ignores_extrude(self):
        self.assertEqual(record_hash(A, "s"), record_hash(SAME_SKETCH, "s"))
        self.assertNotEqual(record_hash(A, "e"), record_hash(SAME_SKETCH, "e"))
        self.assertNotEqual(record_hash(A, "se"), record_hash(SAME_SKETCH, "se"))

    def test_extrude_branch_ignores_sketch(self):
        self.assertEqual(record_hash(A, "e"), record_hash(SAME_EXT, "e"))
        self.assertNotEqual(record_hash(A, "s"), record_hash(SAME_EXT, "s"))

    def test_empty_record(self):
        self.assertEqual(record_hash(EMPTY), "")


class TestGroups(unittest.TestCase):
    def test_groups(self):
        groups = duplicate_groups([A, A2, SAME_SKETCH, EMPTY], "se")
        self.assertEqual(len(groups), 2)
        self.assertIn([0, 1], list(groups.values()))

    def test_sketch_branch_groups(self):
        groups = duplicate_groups([A, A2, SAME_SKETCH, SAME_EXT], "s")
        self.assertEqual(len(groups), 2)
        sizes = sorted(len(g) for g in groups.values())
        self.assertEqual(sizes, [1, 3])


class TestPercentages(unittest.TestCase):
    def test_unique_percent(self):
        # 4 records; se-groups: {A,A2}, {SAME_SKETCH}, {SAME_EXT} -> 2 singletons
        self.assertAlmostEqual(unique_percent([A, A2, SAME_SKETCH, SAME_EXT], "se"), 50.0)

    def test_duplicate_percent(self):
        self.assertAlmostEqual(duplicate_percent([A, A2], "se"), 50.0)
        self.assertAlmostEqual(duplicate_percent([A, SAME_EXT], "se"), 0.0)

    def test_empty_inputs(self):
        self.assertEqual(unique_percent([]), 0.0)
        self.assertEqual(duplicate_percent([]), 0.0)


class TestDeduplicate(unittest.TestCase):
    def test_keeps_first(self):
        kept = deduplicate([A, A2, SAME_SKETCH], "se")
        self.assertEqual([r["uid"] for r in kept], [0, 2])

    def test_sketch_branch_drops_more(self):
        recs = [A, A2, SAME_SKETCH, SAME_EXT]
        self.assertEqual(len(deduplicate(recs, "s")), 2)
        self.assertEqual(len(deduplicate(recs, "e")), 2)
        self.assertEqual(len(deduplicate(recs, "se")), 3)

    def test_drops_empty(self):
        self.assertEqual(len(deduplicate([EMPTY])), 0)

    def test_report(self):
        report = dedup_report([A, A2, SAME_SKETCH, SAME_EXT])
        self.assertEqual(set(report), set(BRANCHES))
        self.assertEqual(report["se"]["kept"], 3.0)
        self.assertEqual(report["s"]["kept"], 2.0)
        self.assertAlmostEqual(report["se"]["duplicate_percent"], 25.0)


if __name__ == "__main__":
    unittest.main()
