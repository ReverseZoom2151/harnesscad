"""Tests for agents.cadam_message_tree (conversation branch tree)."""

import unittest

from harnesscad.agents.agents.cadam_message_tree import MessageTree, MessageNode


class Row:
    def __init__(self, id, parent_message_id):
        self.id = id
        self.parent_message_id = parent_message_id


class TestBuild(unittest.TestCase):
    def test_linear_chain(self):
        tree = MessageTree([("a", None), ("b", "a"), ("c", "b")])
        self.assertEqual(len(tree), 3)
        self.assertEqual([r.id for r in tree.roots], ["a"])
        self.assertEqual([n.id for n in tree.path_to_root("c")], ["a", "b", "c"])

    def test_forward_reference(self):
        # Child listed before its parent must still wire up.
        tree = MessageTree([("b", "a"), ("a", None)])
        self.assertEqual([n.id for n in tree.path_to_root("b")], ["a", "b"])

    def test_branching_children(self):
        tree = MessageTree([("a", None), ("b", "a"), ("c", "a")])
        a = tree.get("a")
        self.assertEqual([c.id for c in a.children], ["b", "c"])

    def test_orphan_becomes_root(self):
        # parent_message_id points at a missing id -> treated as root.
        tree = MessageTree([("a", "missing"), ("b", None)])
        self.assertEqual(sorted(r.id for r in tree.roots), ["a", "b"])

    def test_from_objects(self):
        tree = MessageTree([Row("a", None), Row("b", "a")])
        self.assertEqual([n.id for n in tree.path_to_root("b")], ["a", "b"])
        # Payload is the object itself when built from attribute rows.
        self.assertIsInstance(tree.get("a").payload, Row)

    def test_payload_from_tuple(self):
        tree = MessageTree([("a", None, {"text": "hi"})])
        self.assertEqual(tree.get("a").payload, {"text": "hi"})


class TestSiblings(unittest.TestCase):
    def test_siblings_share_parent(self):
        tree = MessageTree([("a", None), ("b", "a"), ("c", "a")])
        b = tree.get("b")
        self.assertEqual([s.id for s in b.siblings], ["b", "c"])

    def test_root_siblings_are_roots(self):
        tree = MessageTree([("a", None), ("b", None)])
        self.assertEqual([s.id for s in tree.get("a").siblings], ["a", "b"])


class TestCycleSafety(unittest.TestCase):
    def test_self_cycle_truncates(self):
        tree = MessageTree([("a", "a")])
        # a points at itself -> orphan root, path is just [a].
        path = tree.path_to_root("a")
        self.assertEqual([n.id for n in path], ["a"])

    def test_two_node_cycle(self):
        # Manually construct a genuine parent cycle and ensure no infinite loop.
        tree = MessageTree([("a", None), ("b", "a")])
        a, b = tree.get("a"), tree.get("b")
        a.parent = b  # inject a cycle a<->b
        path = tree.path_to_root("a")
        ids = [n.id for n in path]
        self.assertLessEqual(len(ids), 2)
        self.assertEqual(ids[-1], "a")


class TestQueries(unittest.TestCase):
    def test_depth(self):
        tree = MessageTree([("a", None), ("b", "a"), ("c", "b")])
        self.assertEqual(tree.depth("a"), 0)
        self.assertEqual(tree.depth("b"), 1)
        self.assertEqual(tree.depth("c"), 2)
        self.assertEqual(tree.depth("missing"), -1)

    def test_leaves(self):
        tree = MessageTree([("a", None), ("b", "a"), ("c", "a"), ("d", "b")])
        self.assertEqual(sorted(n.id for n in tree.leaves()), ["c", "d"])

    def test_unknown_path_empty(self):
        tree = MessageTree([("a", None)])
        self.assertEqual(tree.path_to_root("zzz"), [])

    def test_contains_and_get(self):
        tree = MessageTree([("a", None)])
        self.assertIn("a", tree)
        self.assertNotIn("x", tree)
        self.assertIsNone(tree.get("x"))
        self.assertIsInstance(tree.get("a"), MessageNode)

    def test_deterministic_order(self):
        rows = [("a", None), ("b", "a"), ("c", "a"), ("d", "b")]
        t1 = MessageTree(rows)
        t2 = MessageTree(rows)
        self.assertEqual([n.id for n in t1.leaves()], [n.id for n in t2.leaves()])


if __name__ == "__main__":
    unittest.main()
