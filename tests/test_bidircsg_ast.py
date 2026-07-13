"""Tests for programs.bidircsg_ast."""

import unittest

from harnesscad.domain.programs.ast.bidircsg_ast import (
    Difference,
    Intersection,
    Primitive,
    Repeat,
    Rotate,
    Scale,
    Translate,
    Union,
    children,
    from_dict,
    iter_nodes,
    node_at,
    node_count,
    parent_path,
    replace_at,
    serialize,
    structural_signature,
    to_dict,
    with_children,
    wrap_at,
)


def sample():
    # difference( translate([1,0,0]) sphere(2) , cube([1,1,1]) )
    return Difference((
        Translate((1.0, 0.0, 0.0), Primitive("sphere", (2.0,))),
        Primitive("cube", (1.0, 1.0, 1.0)),
    ))


class ChildrenTest(unittest.TestCase):
    def test_primitive_has_no_children(self):
        self.assertEqual(children(Primitive("sphere", (1.0,))), ())

    def test_transform_has_one_child(self):
        t = Translate((0, 0, 0), Primitive("cube", (1, 1, 1)))
        self.assertEqual(len(children(t)), 1)

    def test_boolean_children(self):
        self.assertEqual(len(children(sample())), 2)

    def test_with_children_preserves_type(self):
        d = sample()
        prim = Primitive("cube", (2, 2, 2))
        d2 = with_children(d, (prim, prim))
        self.assertIsInstance(d2, Difference)
        self.assertEqual(children(d2), (prim, prim))

    def test_with_children_primitive_rejects(self):
        with self.assertRaises(ValueError):
            with_children(Primitive("sphere", (1.0,)), (sample(),))


class TraversalTest(unittest.TestCase):
    def test_iter_nodes_preorder_paths(self):
        paths = [p for p, _ in iter_nodes(sample())]
        self.assertEqual(paths, [(), (0,), (0, 0), (1,)])

    def test_node_at(self):
        prog = sample()
        self.assertIsInstance(node_at(prog, (0,)), Translate)
        self.assertEqual(node_at(prog, (0, 0)), Primitive("sphere", (2.0,)))

    def test_node_count(self):
        self.assertEqual(node_count(sample()), 4)

    def test_parent_path(self):
        self.assertEqual(parent_path((0, 0)), (0,))
        with self.assertRaises(ValueError):
            parent_path(())


class FunctionalUpdateTest(unittest.TestCase):
    def test_replace_at_root(self):
        prim = Primitive("cube", (1, 1, 1))
        self.assertEqual(replace_at(sample(), (), prim), prim)

    def test_replace_at_leaf_does_not_mutate(self):
        prog = sample()
        new = replace_at(prog, (0, 0), Primitive("sphere", (5.0,)))
        # original unchanged (frozen / functional)
        self.assertEqual(node_at(prog, (0, 0)), Primitive("sphere", (2.0,)))
        self.assertEqual(node_at(new, (0, 0)), Primitive("sphere", (5.0,)))

    def test_wrap_at_inserts_transform(self):
        prog = sample()
        new = wrap_at(prog, (0, 0), lambda old: Translate((9, 0, 0), old))
        wrapped = node_at(new, (0, 0))
        self.assertIsInstance(wrapped, Translate)
        self.assertEqual(wrapped.offset, (9, 0, 0))
        self.assertEqual(wrapped.child, Primitive("sphere", (2.0,)))


class SerialiseTest(unittest.TestCase):
    def test_serialize_deterministic(self):
        self.assertEqual(serialize(sample()), serialize(sample()))

    def test_serialize_contains_ops(self):
        text = serialize(sample())
        self.assertIn("difference()", text)
        self.assertIn("translate([1, 0, 0])", text)
        self.assertIn("sphere(2)", text)

    def test_serialize_repeat(self):
        r = Repeat(3, (2.0, 0.0, 0.0), Primitive("cube", (1, 1, 1)))
        self.assertIn("for(i=[0:2])", serialize(r))

    def test_dict_roundtrip(self):
        prog = Union((
            sample(),
            Intersection((
                Rotate((0, 0, 90), Primitive("cylinder", (1.0, 4.0))),
                Scale((2, 2, 2), Repeat(2, (1, 0, 0), Primitive("sphere", (1.0,)))),
            )),
        ))
        self.assertEqual(from_dict(to_dict(prog)), prog)

    def test_structural_signature_ignores_params(self):
        a = Translate((1, 2, 3), Primitive("sphere", (2.0,)))
        b = Translate((9, 9, 9), Primitive("sphere", (5.0,)))
        self.assertEqual(structural_signature(a), structural_signature(b))
        c = Translate((1, 2, 3), Primitive("cube", (1, 1, 1)))
        self.assertNotEqual(structural_signature(a), structural_signature(c))


if __name__ == "__main__":
    unittest.main()
