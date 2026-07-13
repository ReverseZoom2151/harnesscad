"""Tests for programs.bidircsg_forward."""

import math
import unittest

from harnesscad.domain.programs.bidircsg_ast import (
    Difference,
    Primitive,
    Repeat,
    Rotate,
    Scale,
    Translate,
    Union,
)
from harnesscad.domain.programs.bidircsg_forward import (
    Affine,
    IDENTITY,
    find_instance,
    find_instances,
    get,
    iter_geom,
    leaves,
    rotation,
    scaling,
    translation,
)


def approx(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


class AffineTest(unittest.TestCase):
    def test_identity_apply(self):
        self.assertEqual(IDENTITY.apply((1, 2, 3)), (1, 2, 3))

    def test_translation(self):
        self.assertEqual(translation((1, 2, 3)).apply((0, 0, 0)), (1, 2, 3))

    def test_scaling(self):
        self.assertEqual(scaling((2, 3, 4)).apply((1, 1, 1)), (2, 3, 4))

    def test_rotation_z_90(self):
        # rotate (1,0,0) by 90deg about z -> (0,1,0)
        self.assertTrue(approx(rotation((0, 0, 90)).apply((1, 0, 0)), (0, 1, 0)))

    def test_compose_order(self):
        # translate then scale: scale ∘ translate applies translate first
        a = scaling((2, 2, 2)).compose(translation((1, 0, 0)))
        self.assertTrue(approx(a.apply((0, 0, 0)), (2, 0, 0)))

    def test_inverse_linear_roundtrip(self):
        a = rotation((10, 20, 30)).compose(scaling((2, 3, 4)))
        v = (1.5, -2.0, 0.7)
        back = a.apply_inverse_linear(a.apply_linear(v))
        self.assertTrue(approx(back, v))

    def test_singular_raises(self):
        with self.assertRaises(ValueError):
            Affine((0, 0, 0, 0, 0, 0, 0, 0, 0)).inverse_linear()


class ForwardEvalTest(unittest.TestCase):
    def test_primitive_anchor(self):
        tree = get(Primitive("sphere", (1.0,)))
        self.assertEqual(tree.anchor, (0.0, 0.0, 0.0))
        self.assertTrue(tree.is_primitive())
        self.assertEqual(tree.params, (1.0,))

    def test_translate_moves_anchor(self):
        tree = get(Translate((5, 0, 0), Primitive("cube", (1, 1, 1))))
        leaf = leaves(tree)[0]
        self.assertTrue(approx(leaf.anchor, (5, 0, 0)))

    def test_nested_transform_accumulates(self):
        # rotate 90 about z, then translate x by 2 in that rotated frame
        prog = Rotate((0, 0, 90), Translate((2, 0, 0), Primitive("sphere", (1.0,))))
        leaf = leaves(get(prog))[0]
        self.assertTrue(approx(leaf.anchor, (0, 2, 0)))

    def test_source_path_reference(self):
        prog = Translate((1, 0, 0), Primitive("sphere", (1.0,)))
        leaf = leaves(get(prog))[0]
        self.assertEqual(leaf.source_path, (0,))

    def test_parent_transform_of_leaf_equals_world(self):
        prog = Translate((1, 2, 3), Primitive("sphere", (1.0,)))
        leaf = leaves(get(prog))[0]
        # primitive has no own transform
        self.assertEqual(leaf.parent_transform.t, leaf.world_transform.t)


class RepeatTest(unittest.TestCase):
    def test_repeat_produces_instances(self):
        prog = Repeat(3, (2, 0, 0), Primitive("cube", (1, 1, 1)))
        ls = leaves(get(prog))
        self.assertEqual(len(ls), 3)
        anchors = sorted(l.anchor[0] for l in ls)
        self.assertTrue(approx(anchors, (0, 2, 4)))

    def test_repeat_instances_share_source_path(self):
        prog = Repeat(3, (2, 0, 0), Primitive("cube", (1, 1, 1)))
        tree = get(prog)
        insts = find_instances(tree, (0,))  # the shared child AST node
        self.assertEqual(len(insts), 3)
        # distinct call stacks
        self.assertEqual(
            sorted(i.call_stack for i in insts), [(0,), (1,), (2,)]
        )

    def test_find_instance_by_call_stack(self):
        prog = Repeat(3, (2, 0, 0), Primitive("cube", (1, 1, 1)))
        tree = get(prog)
        g = find_instance(tree, (0,), (2,))
        self.assertIsNotNone(g)
        self.assertTrue(approx(g.anchor, (4, 0, 0)))


class StructureTest(unittest.TestCase):
    def test_boolean_children_traced(self):
        prog = Difference((Primitive("sphere", (2.0,)), Primitive("cube", (1, 1, 1))))
        tree = get(prog)
        self.assertEqual(tree.kind, "Difference")
        self.assertEqual(len(tree.children), 2)
        self.assertEqual(tree.children[0].source_path, (0,))
        self.assertEqual(tree.children[1].source_path, (1,))

    def test_deterministic(self):
        prog = Union((
            Scale((2, 2, 2), Primitive("sphere", (1.0,))),
            Primitive("cube", (1, 1, 1)),
        ))
        a = [g.anchor for g in iter_geom(get(prog))]
        b = [g.anchor for g in iter_geom(get(prog))]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
