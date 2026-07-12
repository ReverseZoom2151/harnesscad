import unittest

from state.opencad_feature_rebuild import (
    BLOCKED,
    BUILT,
    FAILED,
    STALE,
    SUPPRESSED,
    CircularDependencyError,
    FeatureNode,
    FeatureTree,
    MissingDependencyError,
    add_feature,
    delete_feature,
    descendants,
    direct_dependents,
    edit_feature,
    rebuild,
    suppress_feature,
    topological_order,
)


def _tree():
    """box -> fillet -> shell, plus a cylinder tool consumed by a cut."""
    tree = FeatureTree()
    tree = add_feature(tree, FeatureNode(id="box", operation="create_box"))
    tree = add_feature(tree, FeatureNode(id="cyl", operation="create_cylinder"))
    tree = add_feature(
        tree,
        FeatureNode(id="cut", operation="boolean_cut", parent_id="box", tool_refs=["cyl"]),
    )
    tree = add_feature(
        tree, FeatureNode(id="fillet", operation="fillet_edges", parent_id="cut")
    )
    return tree


def _builder(node, tree):
    return "shape-%s" % node.id


class TestGraph(unittest.TestCase):
    def test_topological_order_is_deterministic(self):
        order = topological_order(_tree().nodes)
        self.assertEqual(order, ["box", "cyl", "cut", "fillet"])
        self.assertEqual(order, topological_order(_tree().nodes))

    def test_depends_on_includes_tools(self):
        node = _tree().nodes["cut"]
        self.assertEqual(node.depends_on, ["box", "cyl"])

    def test_descendants_and_dependents(self):
        tree = _tree()
        self.assertEqual(descendants(tree.nodes, "box"), {"cut", "fillet"})
        self.assertEqual(direct_dependents(tree.nodes, "cyl"), ["cut"])

    def test_missing_dependency(self):
        tree = FeatureTree()
        with self.assertRaises(MissingDependencyError):
            add_feature(tree, FeatureNode(id="a", operation="x", parent_id="ghost"))

    def test_cycle_detected(self):
        tree = _tree()
        tree.nodes["box"].parent_id = "fillet"
        with self.assertRaises(CircularDependencyError):
            topological_order(tree.nodes)

    def test_duplicate_id_rejected(self):
        tree = _tree()
        with self.assertRaises(ValueError):
            add_feature(tree, FeatureNode(id="box", operation="create_box"))

    def test_add_does_not_mutate_input(self):
        tree = FeatureTree()
        add_feature(tree, FeatureNode(id="a", operation="create_box"))
        self.assertEqual(tree.nodes, {})
        self.assertEqual(tree.revision, 0)


class TestInvalidation(unittest.TestCase):
    def test_edit_stales_node_and_descendants(self):
        tree = rebuild(_tree(), _builder).tree
        self.assertTrue(all(n.status == BUILT for n in tree.nodes.values()))
        edited = edit_feature(tree, "box", {"width": 20})
        self.assertEqual(edited.nodes["box"].status, STALE)
        self.assertEqual(edited.nodes["cut"].status, STALE)
        self.assertEqual(edited.nodes["fillet"].status, STALE)
        self.assertIsNone(edited.nodes["fillet"].shape_id)
        # An unrelated branch stays built.
        self.assertEqual(edited.nodes["cyl"].status, BUILT)
        self.assertEqual(edited.revision, tree.revision + 1)

    def test_edit_merges_parameters(self):
        tree = edit_feature(_tree(), "box", {"width": 10})
        tree = edit_feature(tree, "box", {"height": 5})
        self.assertEqual(tree.nodes["box"].parameters, {"width": 10, "height": 5})

    def test_edit_unknown_node(self):
        with self.assertRaises(ValueError):
            edit_feature(_tree(), "ghost", {})


class TestSuppression(unittest.TestCase):
    def test_suppressing_a_sole_parent_suppresses_descendants(self):
        tree = suppress_feature(_tree(), "cut")
        self.assertEqual(tree.nodes["cut"].status, SUPPRESSED)
        self.assertEqual(tree.nodes["fillet"].status, SUPPRESSED)
        self.assertTrue(tree.nodes["fillet"].suppressed)

    def test_partial_parent_suppression_only_stales(self):
        # 'cut' depends on box AND cyl; suppressing only cyl leaves a live parent.
        tree = suppress_feature(_tree(), "cyl")
        self.assertEqual(tree.nodes["cyl"].status, SUPPRESSED)
        self.assertEqual(tree.nodes["cut"].status, STALE)
        self.assertFalse(tree.nodes["cut"].suppressed)
        self.assertEqual(tree.nodes["fillet"].status, STALE)

    def test_unsuppress_stales_descendants(self):
        tree = suppress_feature(_tree(), "cut")
        tree = suppress_feature(tree, "cut", suppressed=False)
        self.assertEqual(tree.nodes["cut"].status, STALE)
        self.assertFalse(tree.nodes["fillet"].suppressed)
        self.assertEqual(tree.nodes["fillet"].status, STALE)

    def test_rebuild_skips_suppressed_and_blocks_dependents(self):
        tree = suppress_feature(_tree(), "cyl")
        report = rebuild(tree, _builder)
        self.assertEqual(report.skipped, ["cyl"])
        self.assertEqual(report.built, ["box"])
        self.assertEqual(report.blocked, ["cut", "fillet"])
        self.assertEqual(report.tree.nodes["cut"].status, BLOCKED)
        self.assertFalse(report.ok)


class TestDelete(unittest.TestCase):
    def test_delete_with_dependents_refused(self):
        with self.assertRaises(ValueError):
            delete_feature(_tree(), "cut")

    def test_cascade_delete(self):
        tree = delete_feature(_tree(), "cut", cascade=True)
        self.assertEqual(sorted(tree.nodes), ["box", "cyl"])

    def test_delete_leaf(self):
        tree = delete_feature(_tree(), "fillet")
        self.assertNotIn("fillet", tree.nodes)


class TestRebuild(unittest.TestCase):
    def test_full_rebuild(self):
        report = rebuild(_tree(), _builder)
        self.assertTrue(report.ok)
        self.assertEqual(report.built, ["box", "cyl", "cut", "fillet"])
        self.assertEqual(report.tree.nodes["fillet"].shape_id, "shape-fillet")

    def test_built_nodes_reused_unless_forced(self):
        first = rebuild(_tree(), _builder).tree
        calls = []

        def counting(node, tree):
            calls.append(node.id)
            return "shape-%s" % node.id

        rebuild(first, counting)
        self.assertEqual(calls, [])
        rebuild(first, counting, force=True)
        self.assertEqual(calls, ["box", "cyl", "cut", "fillet"])

    def test_failure_aborts_and_stales_remainder(self):
        def failing(node, tree):
            if node.id == "cut":
                raise RuntimeError("boolean produced zero volume")
            return "shape-%s" % node.id

        report = rebuild(_tree(), failing)
        self.assertEqual(report.failed, ["cut"])
        self.assertEqual(report.tree.nodes["cut"].status, FAILED)
        self.assertEqual(report.tree.nodes["fillet"].status, STALE)
        self.assertIn("zero volume", report.errors["cut"])
        self.assertFalse(report.ok)

    def test_continue_on_error_blocks_only_downstream(self):
        def failing(node, tree):
            if node.id == "cut":
                raise RuntimeError("boom")
            return "shape-%s" % node.id

        report = rebuild(_tree(), failing, continue_on_error=True)
        self.assertEqual(report.failed, ["cut"])
        self.assertEqual(report.blocked, ["fillet"])
        self.assertEqual(report.built, ["box", "cyl"])

    def test_rebuild_is_deterministic(self):
        a = rebuild(_tree(), _builder)
        b = rebuild(_tree(), _builder)
        self.assertEqual(a.order, b.order)
        self.assertEqual(a.tree.statuses(), b.tree.statuses())


if __name__ == "__main__":
    unittest.main()
