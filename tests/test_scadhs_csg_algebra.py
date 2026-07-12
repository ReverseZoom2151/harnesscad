"""Tests for geometry.scadhs_csg_algebra (SetLike normalising CSG laws)."""

import unittest

from geometry.scadhs_csg_algebra import (
    CHILDREN,
    Children,
    Difference,
    Intersection,
    ModuleCall,
    Prim,
    Transform,
    Union,
    difference,
    emit,
    intersection,
    intersection_all,
    normalize,
    prim,
    transform,
    union,
    union_all,
)


class UnionFlatteningTest(unittest.TestCase):
    def setUp(self):
        self.a = prim("cube", size=1)
        self.b = prim("sphere", r=2)
        self.c = prim("circle", r=3)

    def test_union_of_two_prims(self):
        u = union(self.a, self.b)
        self.assertIsInstance(u, Union)
        self.assertEqual(u.items, (self.a, self.b))

    def test_union_of_union_flattens_left(self):
        u = union(union(self.a, self.b), self.c)
        self.assertEqual(u.items, (self.a, self.b, self.c))

    def test_union_of_union_flattens_right(self):
        u = union(self.a, union(self.b, self.c))
        self.assertEqual(u.items, (self.a, self.b, self.c))

    def test_union_associative(self):
        left = union(union(self.a, self.b), self.c)
        right = union(self.a, union(self.b, self.c))
        self.assertEqual(left, right)

    def test_union_all_is_single_flat_node(self):
        u = union_all([self.a, self.b, self.c])
        self.assertIsInstance(u, Union)
        self.assertEqual(u.items, (self.a, self.b, self.c))

    def test_union_all_empty_is_unit(self):
        self.assertEqual(union_all([]), Union(()))


class IntersectionFlatteningTest(unittest.TestCase):
    def test_intersection_flattens(self):
        a, b, c = prim("a"), prim("b"), prim("c")
        i = intersection(intersection(a, b), c)
        self.assertIsInstance(i, Intersection)
        self.assertEqual(i.items, (a, b, c))

    def test_intersection_all_fold(self):
        a, b = prim("a"), prim("b")
        self.assertEqual(intersection_all([a, b]).items, (a, b))

    def test_union_and_intersection_do_not_cross_flatten(self):
        a, b, c = prim("a"), prim("b"), prim("c")
        u = union(intersection(a, b), c)
        # the intersection must remain a nested child, not be flattened away
        self.assertEqual(u.items, (intersection(a, b), c))


class DifferenceAbsorptionTest(unittest.TestCase):
    def setUp(self):
        self.a = prim("a")
        self.b = prim("b")
        self.c = prim("c")
        self.d = prim("d")

    def test_single_difference(self):
        diff = difference(self.a, self.b)
        self.assertIsInstance(diff, Difference)
        self.assertEqual(diff.minuend, self.a)
        self.assertEqual(diff.subtrahends, (self.b,))

    def test_chained_difference_absorbs_into_one_node(self):
        # a - b - c - d  ==>  Difference(a, [b, c, d])
        diff = difference(difference(difference(self.a, self.b), self.c), self.d)
        self.assertIsInstance(diff, Difference)
        self.assertEqual(diff.minuend, self.a)
        self.assertEqual(diff.subtrahends, (self.b, self.c, self.d))

    def test_difference_emits_single_block(self):
        diff = difference(difference(self.a, self.b), self.c)
        text = emit(diff)
        self.assertEqual(text.count("difference()"), 1)
        self.assertIn("a();", text)
        self.assertIn("b();", text)
        self.assertIn("c();", text)


class NormalizeTest(unittest.TestCase):
    def test_normalize_flattens_nested_unions(self):
        a, b, c, d = prim("a"), prim("b"), prim("c"), prim("d")
        # deliberately construct a non-flat tree via raw constructors
        raw = Union((Union((a, b)), Union((c, d))))
        norm = normalize(raw)
        self.assertEqual(norm, Union((a, b, c, d)))

    def test_normalize_collapses_difference_chain(self):
        a, b, c = prim("a"), prim("b"), prim("c")
        raw = Difference(Difference(a, (b,)), (c,))
        self.assertEqual(normalize(raw), Difference(a, (b, c)))

    def test_normalize_single_element_union_simplifies(self):
        a = prim("a")
        self.assertEqual(normalize(Union((a,))), a)

    def test_normalize_recurses_through_transform(self):
        a, b = prim("a"), prim("b")
        raw = Transform("translate", Union((Union((a,)), b)), {"v": [1, 2, 3]})
        norm = normalize(raw)
        self.assertIsInstance(norm, Transform)
        self.assertEqual(norm.child, Union((a, b)))

    def test_normalize_idempotent(self):
        a, b, c = prim("a"), prim("b"), prim("c")
        raw = Union((Union((a, b)), c))
        once = normalize(raw)
        self.assertEqual(once, normalize(once))


class HashingAndEqualityTest(unittest.TestCase):
    def test_prims_hash_equal_by_value(self):
        self.assertEqual(prim("cube", size=1), prim("cube", size=1))
        self.assertEqual(hash(prim("cube", size=1)), hash(prim("cube", size=1)))

    def test_list_params_are_hashable(self):
        p = prim("translate", v=[1, 2, 3])
        self.assertEqual(hash(p), hash(prim("translate", v=[1, 2, 3])))

    def test_terms_usable_as_dict_keys(self):
        seen = {union(prim("a"), prim("b")): "x"}
        self.assertEqual(seen[union(prim("a"), prim("b"))], "x")

    def test_children_singleton_equality(self):
        self.assertEqual(CHILDREN, Children())
        self.assertEqual(hash(CHILDREN), hash(Children()))


class EmitTest(unittest.TestCase):
    def test_emit_prim(self):
        self.assertEqual(emit(prim("cube", size=2)), "cube(size = 2);")

    def test_emit_children_placeholder(self):
        self.assertEqual(emit(CHILDREN), "children();")

    def test_emit_module_call_with_child(self):
        call = ModuleCall("mdl_0", prim("cube", size=1))
        text = emit(call)
        self.assertIn("mdl_0()", text)
        self.assertIn("cube(size = 1);", text)

    def test_emit_float_formatting(self):
        self.assertEqual(emit(prim("circle", r=3.5)), "circle(r = 3.5);")
        self.assertEqual(emit(prim("circle", r=3.0)), "circle(r = 3);")

    def test_emit_deterministic(self):
        tree = transform(
            "translate", union_all([prim("a"), prim("b")]), v=[0, 0, 1]
        )
        self.assertEqual(emit(tree), emit(tree))


if __name__ == "__main__":
    unittest.main()
