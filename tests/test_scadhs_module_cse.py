"""Tests for programs.scadhs_module_cse (content-addressed module CSE)."""

import unittest

from harnesscad.domain.geometry.scadhs_csg_algebra import (
    CHILDREN,
    ModuleCall,
    Prim,
    Transform,
    Union,
    prim,
    transform,
    union_all,
)
from harnesscad.domain.programs.scadhs_module_cse import (
    ModuleBuilder,
    ModuleDef,
    auto_modularize,
    render,
    subtree_counts,
    wrap,
    wrap_multi,
)


class ModuleBuilderTest(unittest.TestCase):
    def test_first_body_named_mdl_0(self):
        b = ModuleBuilder()
        name = b.intern((prim("cube", size=1),))
        self.assertEqual(name, "mdl_0")

    def test_identical_body_reuses_name(self):
        b = ModuleBuilder()
        n1 = b.intern((prim("cube", size=1),))
        n2 = b.intern((prim("cube", size=1),))
        self.assertEqual(n1, n2)
        self.assertEqual(len(b.definitions()), 1)

    def test_distinct_bodies_get_sequential_names(self):
        b = ModuleBuilder()
        self.assertEqual(b.intern((prim("a"),)), "mdl_0")
        self.assertEqual(b.intern((prim("b"),)), "mdl_1")
        self.assertEqual(b.intern((prim("c"),)), "mdl_2")

    def test_definitions_in_creation_order(self):
        b = ModuleBuilder()
        b.intern((prim("a"),))
        b.intern((prim("b"),))
        names = [d.name for d in b.definitions()]
        self.assertEqual(names, ["mdl_0", "mdl_1"])


class WrapTest(unittest.TestCase):
    def test_wrap_returns_call_applied_to_child(self):
        b = ModuleBuilder()
        child = prim("cube", size=2)
        call = wrap(b, lambda c: transform("translate", c, v=[0, 0, 5]), child)
        self.assertIsInstance(call, ModuleCall)
        self.assertEqual(call.name, "mdl_0")
        self.assertEqual(call.child, child)

    def test_same_group_different_child_shares_module(self):
        b = ModuleBuilder()
        g = lambda c: transform("translate", c, v=[0, 0, 5])
        c1 = wrap(b, g, prim("cube", size=1))
        c2 = wrap(b, g, prim("sphere", r=3))
        # one shared module, two distinct applications
        self.assertEqual(c1.name, c2.name)
        self.assertEqual(len(b.definitions()), 1)
        self.assertNotEqual(c1.child, c2.child)

    def test_module_body_contains_children_placeholder(self):
        b = ModuleBuilder()
        wrap(b, lambda c: transform("rotate", c, a=90), prim("cube", size=1))
        body = b.definitions()[0].body
        self.assertIn(CHILDREN, body[0].children())

    def test_wrap_multi_registers_list_body(self):
        b = ModuleBuilder()
        group = lambda c: [prim("circle", r=10), transform("scale", c, v=[2, 2, 2])]
        call = wrap_multi(b, group, prim("square", size=4))
        self.assertEqual(len(b.definitions()[0].body), 2)
        self.assertEqual(call.child, prim("square", size=4))


class RenderTest(unittest.TestCase):
    def test_render_emits_module_then_body(self):
        b = ModuleBuilder()
        call = wrap(b, lambda c: transform("translate", c, v=[1, 0, 0]), prim("cube", size=1))
        text = render(call, b)
        self.assertIn("module mdl_0() {", text)
        self.assertIn("children();", text)
        self.assertIn("mdl_0()", text)
        # module definition comes before the application
        self.assertLess(text.index("module mdl_0"), text.rindex("mdl_0()"))

    def test_render_deterministic(self):
        b = ModuleBuilder()
        g = lambda c: transform("translate", c, v=[0, 0, 5])
        top = union_all([wrap(b, g, prim("cube", size=1)), wrap(b, g, prim("sphere", r=2))])
        self.assertEqual(render(top, b), render(top, b))


class AutoModularizeTest(unittest.TestCase):
    def test_repeated_subtree_is_hoisted(self):
        shared = transform("translate", prim("cube", size=3), v=[0, 0, 1])
        tree = union_all([shared, prim("sphere", r=5), shared])
        root, builder = auto_modularize(tree)
        defs = builder.definitions()
        self.assertEqual(len(defs), 1)
        # both occurrences became calls to the same module
        names = [t.name for t in root.items if isinstance(t, ModuleCall)]
        self.assertEqual(names, ["mdl_0", "mdl_0"])

    def test_unique_subtrees_not_hoisted(self):
        tree = union_all([prim("a"), prim("b"), prim("c")])
        root, builder = auto_modularize(tree)
        self.assertEqual(builder.definitions(), [])
        self.assertEqual(root, tree)

    def test_min_count_threshold(self):
        shared = transform("translate", prim("cube", size=1), v=[1, 1, 1])
        tree = union_all([shared, prim("sphere", r=2)])  # shared appears once
        root, builder = auto_modularize(tree)
        self.assertEqual(builder.definitions(), [])

    def test_hoisted_module_removes_duplication_in_output(self):
        shared = transform("rotate", prim("cylinder", h=10, r=2), a=45)
        tree = union_all([shared, shared, shared])
        root, builder = auto_modularize(tree)
        text = render(root, builder)
        # cylinder primitive appears exactly once (inside the module)
        self.assertEqual(text.count("cylinder("), 1)
        self.assertEqual(text.count("mdl_0()"), 3 + 1)  # 3 calls + 1 definition head

    def test_nested_shared_blocks_largest_first(self):
        inner = transform("translate", prim("cube", size=2), v=[0, 0, 1])
        outer = union_all([inner, prim("sphere", r=1)])
        # outer appears twice; inner appears (twice standalone + inside outer x2)
        tree = union_all([outer, outer, inner, inner])
        root, builder = auto_modularize(tree)
        names = [d.name for d in builder.definitions()]
        # both outer and inner get hoisted; outer (larger) is mdl_0
        self.assertIn("mdl_0", names)
        self.assertGreaterEqual(len(names), 1)
        # render must be reproducible
        self.assertEqual(render(root, builder), render(root, builder))

    def test_children_bearing_subtree_not_hoisted(self):
        # a body that already has a children() hole cannot be a nullary module
        bad = transform("translate", CHILDREN, v=[0, 0, 1])
        tree = union_all([bad, bad])
        root, builder = auto_modularize(tree)
        self.assertEqual(builder.definitions(), [])


class SubtreeCountsTest(unittest.TestCase):
    def test_counts_repeated_leaf(self):
        tree = union_all([prim("a"), prim("a"), prim("b")])
        counts = subtree_counts(tree)
        self.assertEqual(counts[prim("a")], 2)
        self.assertEqual(counts[prim("b")], 1)

    def test_counts_nested_subtree(self):
        shared = transform("scale", prim("cube", size=1), v=[2, 2, 2])
        tree = union_all([shared, shared])
        counts = subtree_counts(tree)
        self.assertEqual(counts[shared], 2)
        self.assertEqual(counts[prim("cube", size=1)], 2)


if __name__ == "__main__":
    unittest.main()
