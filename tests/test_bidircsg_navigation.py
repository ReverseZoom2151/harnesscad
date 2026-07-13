"""Tests for editing.bidircsg_navigation."""

import unittest

from harnesscad.domain.programs.ast.bidirectional_csg import (
    Difference,
    Intersection,
    Primitive,
    Repeat,
    Translate,
    Union,
)
from harnesscad.domain.programs.runtime.csg_forward_eval import find_instance, get, leaves
from harnesscad.domain.editing.code_view_navigation import (
    consistency,
    forward_search,
    ghosts,
    locate_source,
    removed_operands,
    reverse_search,
)


class ReverseSearchTest(unittest.TestCase):
    def test_target_branch_root_to_leaf(self):
        prog = Translate((1, 0, 0), Primitive("sphere", (2.0,)))
        tree = get(prog)
        leaf = leaves(tree)[0]
        rev = reverse_search(tree, leaf)
        # branch: root Translate () -> primitive (0,)
        self.assertEqual(rev.target_source_paths, [(), (0,)])

    def test_impacted_from_repeat(self):
        prog = Repeat(3, (2, 0, 0), Primitive("cube", (1, 1, 1)))
        tree = get(prog)
        inst = find_instance(tree, (0,), (1,))
        rev = reverse_search(tree, inst)
        # two other instances share the same AST node -> impacted
        self.assertEqual(len(rev.impacted), 2)
        for g in rev.impacted:
            self.assertEqual(g.source_path, (0,))

    def test_call_order_of_selected(self):
        prog = Repeat(3, (2, 0, 0), Primitive("cube", (1, 1, 1)))
        tree = get(prog)
        inst = find_instance(tree, (0,), (2,))
        rev = reverse_search(tree, inst)
        # last element in the branch is the selected leaf; its call order = 2
        self.assertEqual(rev.call_order[-1], 2)

    def test_non_repeat_has_no_impacted(self):
        prog = Union((Primitive("sphere", (1.0,)), Primitive("cube", (1, 1, 1))))
        tree = get(prog)
        leaf = leaves(tree)[0]
        rev = reverse_search(tree, leaf)
        self.assertEqual(rev.impacted, [])

    def test_selected_not_in_tree_raises(self):
        prog = Primitive("sphere", (1.0,))
        tree = get(prog)
        other = get(Primitive("cube", (1, 1, 1)))
        with self.assertRaises(ValueError):
            reverse_search(tree, other)


class ForwardSearchTest(unittest.TestCase):
    def test_unique_is_target(self):
        prog = Union((Primitive("sphere", (1.0,)), Primitive("cube", (1, 1, 1))))
        tree = get(prog)
        fwd = forward_search(tree, (0,))
        self.assertEqual(len(fwd.target), 1)
        self.assertEqual(fwd.impacted, [])

    def test_multiple_is_impacted(self):
        prog = Repeat(3, (2, 0, 0), Primitive("cube", (1, 1, 1)))
        tree = get(prog)
        fwd = forward_search(tree, (0,))
        self.assertEqual(fwd.target, [])
        self.assertEqual(len(fwd.impacted), 3)


class GhostsTest(unittest.TestCase):
    def test_difference_ghosts_roles(self):
        prog = Difference((
            Primitive("sphere", (3.0,)),
            Primitive("cube", (1, 1, 1)),
            Primitive("cube", (2, 2, 2)),
        ))
        tree = get(prog)
        gs = ghosts(tree, tree)
        self.assertEqual([g.role for g in gs], ["target", "impacted", "impacted"])

    def test_ghosts_reject_non_operation(self):
        prog = Union((Primitive("sphere", (1.0,)), Primitive("cube", (1, 1, 1))))
        tree = get(prog)
        with self.assertRaises(ValueError):
            ghosts(tree, tree)

    def test_removed_operands_difference(self):
        prog = Difference((Primitive("sphere", (3.0,)), Primitive("cube", (1, 1, 1))))
        tree = get(prog)
        rem = removed_operands(tree)
        self.assertEqual(len(rem), 1)
        self.assertEqual(rem[0].kind, "cube")

    def test_removed_operands_intersection(self):
        prog = Intersection((Primitive("sphere", (3.0,)), Primitive("cube", (1, 1, 1))))
        tree = get(prog)
        self.assertEqual(len(removed_operands(tree)), 2)


class TraceabilityTest(unittest.TestCase):
    def test_locate_source(self):
        prog = Translate((1, 0, 0), Primitive("sphere", (2.0,)))
        tree = get(prog)
        leaf = leaves(tree)[0]
        node = locate_source(prog, leaf)
        self.assertEqual(node, Primitive("sphere", (2.0,)))

    def test_consistency(self):
        prog = Difference((
            Repeat(2, (1, 0, 0), Primitive("cube", (1, 1, 1))),
            Primitive("sphere", (1.0,)),
        ))
        self.assertTrue(consistency(prog))


if __name__ == "__main__":
    unittest.main()
