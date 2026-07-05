"""Tests for the OpDAG branching layer: named branches + 3-way merge.

Branches are an additive layer over the shared content-hash chain: creating a
branch copies a prefix of the parent's nodes, so shared history keeps identical
hashes and divergent history is content-addressed. ``merge`` does a 3-way merge
from the common ancestor, replaying non-conflicting ops and *flagging* edits
where both branches touched the same feature/parameter.
"""

import unittest

from cisp.ops import (
    NewSketch, AddRectangle, AddCircle, Extrude, Boolean, SetParam,
)
from state.opdag import OpDAG, DEFAULT_BRANCH


def _base(dag):
    dag.append(NewSketch(plane="XY"))
    dag.append(AddRectangle(sketch="sk1", w=10.0, h=5.0))
    return dag


class TestBranching(unittest.TestCase):
    def test_default_branch_and_checkout(self):
        dag = OpDAG()
        self.assertEqual(dag.current_branch, DEFAULT_BRANCH)
        self.assertEqual(dag.branches(), [DEFAULT_BRANCH])
        _base(dag)
        dag.branch("feature")
        self.assertEqual(dag.branches(), ["feature", DEFAULT_BRANCH])  # sorted
        # Still on main until we checkout.
        self.assertEqual(dag.current_branch, DEFAULT_BRANCH)
        dag.checkout("feature")
        self.assertEqual(dag.current_branch, "feature")

    def test_branch_copies_prefix_and_shares_hashes(self):
        dag = _base(OpDAG())
        head_at_branch = dag.head_hash
        dag.branch("b", at=None)  # off current head
        dag.checkout("b")
        # The new branch has the same ops and head hash as the parent point.
        self.assertEqual(dag.branch_ops("b"), dag.branch_ops(DEFAULT_BRANCH))
        self.assertEqual(dag.head_hash, head_at_branch)

    def test_branch_at_checkpoint_label(self):
        dag = _base(OpDAG())
        dag.checkpoint("cp")
        dag.append(Extrude(sketch="sk1", distance=3.0))
        dag.branch("early", at="cp")
        self.assertEqual(len(dag.branch_ops("early")), dag.index_of("cp"))

    def test_appends_on_branch_do_not_touch_parent(self):
        dag = _base(OpDAG())
        dag.branch("b")
        dag.checkout("b")
        dag.append(Extrude(sketch="sk1", distance=7.0))
        self.assertEqual(len(dag.branch_ops("b")), 3)
        self.assertEqual(len(dag.branch_ops(DEFAULT_BRANCH)), 2)

    def test_identical_appends_on_two_branches_share_head_hash(self):
        dag = _base(OpDAG())
        dag.branch("b")
        dag.append(Extrude(sketch="sk1", distance=4.0))  # on main
        main_head = dag.head_hash
        dag.checkout("b")
        dag.append(Extrude(sketch="sk1", distance=4.0))  # identical, on b
        # Same ancestor + same op => identical content hash (determinism).
        self.assertEqual(dag.head_hash, main_head)


class TestMerge(unittest.TestCase):
    def _two_branches(self):
        dag = _base(OpDAG())
        dag.append(Extrude(sketch="sk1", distance=3.0))  # shared ancestor tail
        dag.branch("a")
        dag.branch("b")
        return dag

    def test_clean_merge_contains_both_sides(self):
        dag = self._two_branches()
        dag.checkout("a")
        dag.append(AddCircle(sketch="sk1", cx=1.0, r=2.0))   # a-only, additive
        dag.checkout("b")
        dag.append(NewSketch(plane="YZ"))                    # b-only, additive
        result = dag.merge("a", "b")
        self.assertTrue(result.clean)
        self.assertEqual(result.conflicts, [])
        # Base (3) + b's new (1) + a's new (1) all present.
        self.assertIn(AddCircle(sketch="sk1", cx=1.0, r=2.0), result.merged_ops)
        self.assertIn(NewSketch(plane="YZ"), result.merged_ops)
        self.assertEqual(len(result.merged_ops), 5)

    def test_conflicting_setparam_same_target_param(self):
        dag = self._two_branches()
        dag.checkout("a")
        dag.append(SetParam(target=1, param="w", value=20.0))  # edit rect width
        dag.checkout("b")
        dag.append(SetParam(target=1, param="w", value=99.0))  # same param, differ
        result = dag.merge("a", "b")
        self.assertFalse(result.clean)
        self.assertEqual(len(result.conflicts), 1)
        pair = result.conflicts[0]
        self.assertEqual(pair["a"].value, 20.0)
        self.assertEqual(pair["b"].value, 99.0)
        self.assertIn("param", pair["reason"])

    def test_conflicting_ops_same_feature(self):
        dag = self._two_branches()
        dag.checkout("a")
        dag.append(Boolean(kind="cut", target="body1", tool="t1"))
        dag.checkout("b")
        dag.append(Boolean(kind="union", target="body1", tool="t2"))
        result = dag.merge("a", "b")
        self.assertFalse(result.clean)
        self.assertEqual(len(result.conflicts), 1)
        self.assertIn("body1", result.conflicts[0]["reason"])

    def test_different_params_do_not_conflict(self):
        dag = self._two_branches()
        dag.checkout("a")
        dag.append(SetParam(target=1, param="w", value=20.0))
        dag.checkout("b")
        dag.append(SetParam(target=1, param="h", value=8.0))  # different param
        result = dag.merge("a", "b")
        self.assertTrue(result.clean)
        self.assertEqual(len(result.merged_ops), 5)  # base 3 + both edits

    def test_convergent_identical_edit_is_not_a_conflict(self):
        dag = self._two_branches()
        dag.checkout("a")
        dag.append(SetParam(target=1, param="w", value=42.0))
        dag.checkout("b")
        dag.append(SetParam(target=1, param="w", value=42.0))  # same value
        result = dag.merge("a", "b")
        self.assertTrue(result.clean)
        # Deduplicated: the identical edit appears once.
        self.assertEqual(
            sum(1 for o in result.merged_ops if isinstance(o, SetParam)), 1)

    def test_merge_default_target_is_current_branch(self):
        dag = self._two_branches()
        dag.checkout("a")
        dag.append(AddCircle(sketch="sk1", r=2.0))
        dag.checkout("b")
        dag.append(NewSketch(plane="YZ"))
        # target omitted -> merges into current branch "b".
        result = dag.merge("a")
        self.assertTrue(result.clean)
        self.assertIn(AddCircle(sketch="sk1", r=2.0), result.merged_ops)

    def test_merge_is_non_mutating(self):
        dag = self._two_branches()
        dag.checkout("a")
        dag.append(AddCircle(sketch="sk1", r=2.0))
        dag.checkout("b")
        before_a = list(dag.branch_ops("a"))
        before_b = list(dag.branch_ops("b"))
        dag.merge("a", "b")
        self.assertEqual(dag.branch_ops("a"), before_a)
        self.assertEqual(dag.branch_ops("b"), before_b)


if __name__ == "__main__":
    unittest.main()
