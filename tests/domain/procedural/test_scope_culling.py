"""Tests for procedural.shapegramm_scope (scope visibility + culling)."""

import unittest

from harnesscad.domain.procedural.scope_culling import (
    OUTSIDE, SURROUNDING, INTERSECTS, WITHIN,
    make_aabb_frustum, classify_scope, is_visible, classify_hierarchy,
)


class ClassifyScopeTest(unittest.TestCase):
    def setUp(self):
        # frustum is the box [0,10]^3
        self.planes = make_aabb_frustum((0, 0, 0), (10, 10, 10))

    def test_within(self):
        v = classify_scope((2, 2, 2), (4, 4, 4), self.planes, camera_pos=(50, 50, 50))
        self.assertEqual(v, WITHIN)

    def test_outside(self):
        v = classify_scope((20, 20, 20), (25, 25, 25), self.planes, camera_pos=(50, 50, 50))
        self.assertEqual(v, OUTSIDE)

    def test_intersects(self):
        v = classify_scope((8, 8, 8), (12, 12, 12), self.planes, camera_pos=(50, 50, 50))
        self.assertEqual(v, INTERSECTS)

    def test_surrounding_when_camera_inside_box(self):
        v = classify_scope((0, 0, 0), (10, 10, 10), self.planes, camera_pos=(5, 5, 5))
        self.assertEqual(v, SURROUNDING)

    def test_is_visible(self):
        self.assertFalse(is_visible(OUTSIDE))
        self.assertTrue(is_visible(SURROUNDING))
        self.assertTrue(is_visible(INTERSECTS))
        self.assertTrue(is_visible(WITHIN))

    def test_enum_ordering(self):
        self.assertTrue(OUTSIDE < SURROUNDING < INTERSECTS < WITHIN)


class HierarchyTest(unittest.TestCase):
    def setUp(self):
        self.planes = make_aabb_frustum((0, 0, 0), (10, 10, 10))
        # tree: root -> [a, b]; a -> [a1]; b -> [b1]
        self.boxes = {
            "root": ((1, 1, 1), (9, 9, 9)),   # WITHIN
            "a": ((2, 2, 2), (3, 3, 3)),
            "a1": ((2, 2, 2), (2.5, 2.5, 2.5)),
            "b": ((4, 4, 4), (5, 5, 5)),
            "b1": ((4, 4, 4), (4.5, 4.5, 4.5)),
        }
        self.kids = {"root": ["a", "b"], "a": ["a1"], "b": ["b1"]}

    def _run(self, camera=(50, 50, 50)):
        return classify_hierarchy(
            "root",
            lambda n: self.boxes[n],
            lambda n: self.kids.get(n, []),
            self.planes,
            camera_pos=camera,
        )

    def test_within_root_propagates_to_all(self):
        values, stats = self._run()
        for node in self.boxes:
            self.assertEqual(values[node], WITHIN)
        # only the root ran a real frustum test; 4 descendants inherited it
        self.assertEqual(stats["computed"], 1)
        self.assertEqual(stats["propagated"], 4)

    def test_outside_subtree_is_pruned(self):
        # move a's box far outside; root no longer WITHIN so children tested
        self.boxes["root"] = ((1, 1, 1), (30, 30, 30))  # INTERSECTS
        self.boxes["a"] = ((20, 20, 20), (25, 25, 25))   # OUTSIDE
        values, stats = self._run()
        self.assertEqual(values["a"], OUTSIDE)
        self.assertNotIn("a1", values)  # pruned
        self.assertEqual(stats["culled"], 1)

    def test_all_nodes_classified_when_intersecting(self):
        self.boxes["root"] = ((1, 1, 1), (30, 30, 30))  # INTERSECTS
        values, stats = self._run()
        self.assertEqual(values["root"], INTERSECTS)
        # root intersects so a and b get their own test; a and b are WITHIN so
        # their leaves a1/b1 inherit without a test.
        self.assertEqual(values["a"], WITHIN)
        self.assertEqual(values["b"], WITHIN)
        self.assertEqual(stats["computed"], 3)   # root, a, b
        self.assertEqual(stats["propagated"], 2)  # a1, b1


if __name__ == "__main__":
    unittest.main()
