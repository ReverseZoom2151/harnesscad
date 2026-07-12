import unittest

from editing.paramdirect_model import FeatureTree, ParametricFeature
from editing.paramdirect_synctech import SyncPartition, from_tree


class TestSyncPartition(unittest.TestCase):
    def _p(self):
        return SyncPartition(["a", "b", "c", "d"], {"a"})

    def test_history_order_direct_first(self):
        p = SyncPartition(["a", "b", "c", "d"], {"c"})
        # direct-edit features first, each subgroup keeps creation order
        self.assertEqual(p.history_order(), ["c", "a", "b", "d"])

    def test_views(self):
        p = self._p()
        self.assertEqual(p.direct_edit(), ["a"])
        self.assertEqual(p.ordinary(), ["b", "c", "d"])

    def test_cascade_moves_prior_ordinaries(self):
        p = SyncPartition(["a", "b", "c", "d"], set())
        # direct-editing "c" forces a and b (created prior) into direct set
        collateral = p.move_to_direct_edit("c")
        self.assertEqual(collateral, ["a", "b"])
        self.assertEqual(sorted(p.direct), ["a", "b", "c"])

    def test_cascade_skips_already_direct(self):
        p = SyncPartition(["a", "b", "c", "d"], {"a"})
        collateral = p.move_to_direct_edit("c")
        # "a" already direct -> only "b" is collateral
        self.assertEqual(collateral, ["b"])

    def test_first_feature_no_collateral(self):
        p = SyncPartition(["a", "b", "c"], set())
        self.assertEqual(p.move_to_direct_edit("a"), [])

    def test_parametric_loss(self):
        p = SyncPartition(["a", "b", "c", "d"], set())
        p.move_to_direct_edit("c")  # a, b, c now direct
        self.assertEqual(p.parametric_loss(), 3)  # only 1 intended, 2 collateral

    def test_move_unknown(self):
        with self.assertRaises(KeyError):
            self._p().move_to_direct_edit("zz")

    def test_bad_construction(self):
        with self.assertRaises(ValueError):
            SyncPartition(["a", "b"], {"z"})

    def test_from_tree(self):
        t = FeatureTree([
            ParametricFeature("f0", "sketch_extrude", {}),
            ParametricFeature("f1", "hole", {}, direct_edit=True),
            ParametricFeature("f2", "fillet", {}),
        ])
        p = from_tree(t)
        self.assertEqual(p.creation_order, ["f0", "f1", "f2"])
        self.assertEqual(p.direct, {"f1"})
        self.assertEqual(p.history_order(), ["f1", "f0", "f2"])

    def test_copy_isolated(self):
        p = self._p()
        c = p.copy()
        c.move_to_direct_edit("d")
        self.assertNotIn("d", p.direct)


if __name__ == "__main__":
    unittest.main()
