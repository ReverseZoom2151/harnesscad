import unittest

from bench.symcad_grouped_folds import (
    augment_within_folds,
    group_leakage,
    grouped_kfold,
    train_test_folds,
)


def _rec(rid, family):
    return {"id": rid, "family": family}


class TestGroupedKFold(unittest.TestCase):
    def setUp(self):
        self.records = [
            _rec(i, f"fam{i % 7}") for i in range(40)
        ]

    def test_partition_is_complete(self):
        folds = grouped_kfold(self.records, k=5, group=lambda r: r["family"])
        total = sum(len(f) for f in folds)
        self.assertEqual(total, 40)
        self.assertEqual(len(folds), 5)

    def test_no_group_leakage(self):
        folds = grouped_kfold(self.records, k=5, group=lambda r: r["family"])
        self.assertEqual(group_leakage(folds, group=lambda r: r["family"]), [])

    def test_same_family_same_fold(self):
        folds = grouped_kfold(self.records, k=5, group=lambda r: r["family"])
        fam_to_fold = {}
        for fi, fold in enumerate(folds):
            for r in fold:
                fam_to_fold.setdefault(r["family"], fi)
                self.assertEqual(fam_to_fold[r["family"]], fi)

    def test_deterministic_regardless_of_order(self):
        folds_a = grouped_kfold(self.records, k=5, group=lambda r: r["family"])
        shuffled = list(reversed(self.records))
        folds_b = grouped_kfold(shuffled, k=5, group=lambda r: r["family"])
        # same family -> same fold index in both
        def fam_fold(folds):
            m = {}
            for fi, fold in enumerate(folds):
                for r in fold:
                    m[r["family"]] = fi
            return m
        self.assertEqual(fam_fold(folds_a), fam_fold(folds_b))

    def test_k_too_small(self):
        with self.assertRaises(ValueError):
            grouped_kfold(self.records, k=1)

    def test_default_group_is_identity(self):
        folds = grouped_kfold(["a", "b", "a"], k=3)
        # both "a" land together
        loc = {}
        for fi, fold in enumerate(folds):
            for r in fold:
                loc.setdefault(r, fi)
                self.assertEqual(loc[r], fi)


class TestAugmentWithinFolds(unittest.TestCase):
    def test_augment_expands_per_fold(self):
        folds = [[1, 2], [3]]
        out = augment_within_folds(folds, lambda x: (x, x * 10))
        self.assertEqual(out, [[1, 10, 2, 20], [3, 30]])

    def test_augment_keeps_copies_in_fold(self):
        # augmented copies carry the same family; verify none leak
        recs = [_rec(i, f"fam{i}") for i in range(10)]
        folds = grouped_kfold(recs, k=3, group=lambda r: r["family"])

        def aug(r):
            return [r, {"id": f"{r['id']}b", "family": r["family"]}]

        augmented = augment_within_folds(folds, aug)
        self.assertEqual(
            group_leakage(augmented, group=lambda r: r["family"]), []
        )


class TestTrainTestFolds(unittest.TestCase):
    def test_splits_cover_each_fold_as_test(self):
        folds = [["a"], ["b"], ["c"]]
        splits = train_test_folds(folds)
        self.assertEqual(len(splits), 3)
        self.assertEqual(splits[0].test, ("a",))
        self.assertEqual(set(splits[0].train), {"b", "c"})

    def test_augment_only_training(self):
        folds = [[1], [2], [3]]
        splits = train_test_folds(folds, augment=lambda x: (x, -x))
        s0 = splits[0]
        self.assertEqual(s0.test, (1,))  # test untouched
        self.assertEqual(set(s0.train), {2, -2, 3, -3})

    def test_test_never_augmented(self):
        folds = [[5], [6]]
        splits = train_test_folds(folds, augment=lambda x: (x, x + 100))
        for s in splits:
            self.assertEqual(len(s.test), 1)


if __name__ == "__main__":
    unittest.main()
