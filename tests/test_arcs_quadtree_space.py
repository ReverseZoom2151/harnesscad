"""Tests for geometry.arcs_quadtree_space."""

import unittest

from geometry.arcs_quadtree_space import (
    QuadTreeSpace,
    bbox_area,
    bbox_around_points,
    bbox_centre,
    bbox_contains_point,
    bbox_fully_contains,
    bbox_intersects,
    bbox_merge,
    bbox_new,
)


class TestBBoxHelpers(unittest.TestCase):
    def test_new_normalises_corners(self):
        self.assertEqual(bbox_new((10.0, 10.0), (0.0, 2.0)), (0.0, 2.0, 10.0, 10.0))

    def test_around_points(self):
        box = bbox_around_points([(1.0, 2.0), (-3.0, 5.0), (0.0, 0.0)])
        self.assertEqual(box, (-3.0, 0.0, 1.0, 5.0))
        self.assertIsNone(bbox_around_points([]))

    def test_merge_area_centre(self):
        merged = bbox_merge((0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 4.0))
        self.assertEqual(merged, (0.0, 0.0, 3.0, 4.0))
        self.assertAlmostEqual(bbox_area(merged), 12.0)
        self.assertEqual(bbox_centre(merged), (1.5, 2.0))

    def test_fully_contains(self):
        outer = (0.0, 0.0, 10.0, 10.0)
        self.assertTrue(bbox_fully_contains(outer, (1.0, 1.0, 2.0, 2.0)))
        self.assertFalse(bbox_fully_contains(outer, (5.0, 5.0, 15.0, 6.0)))

    def test_intersects_fixes_the_rust_stub(self):
        # partially overlapping boxes: neither contains the other, yet they
        # intersect (the Rust `intersects_with` returns false here)
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 15.0, 15.0)
        self.assertFalse(bbox_fully_contains(a, b))
        self.assertFalse(bbox_fully_contains(b, a))
        self.assertTrue(bbox_intersects(a, b))

    def test_disjoint_boxes_do_not_intersect(self):
        self.assertFalse(
            bbox_intersects((0.0, 0.0, 1.0, 1.0), (2.0, 0.0, 3.0, 1.0))
        )

    def test_touching_boxes_intersect(self):
        self.assertTrue(
            bbox_intersects((0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 2.0, 1.0))
        )

    def test_contains_point(self):
        box = (0.0, 0.0, 2.0, 2.0)
        self.assertTrue(bbox_contains_point(box, (1.0, 1.0)))
        self.assertTrue(bbox_contains_point(box, (0.0, 2.0)))
        self.assertFalse(bbox_contains_point(box, (3.0, 1.0)))


class TestQuadTreeSpace(unittest.TestCase):
    def _grid_space(self):
        space = QuadTreeSpace((0.0, 0.0, 100.0, 100.0), max_children=2, max_depth=4)
        for i in range(10):
            for j in range(10):
                x, y = i * 10.0, j * 10.0
                space.insert((i, j), (x, y, x + 4.0, y + 4.0))
        return space

    def test_insert_and_len(self):
        space = self._grid_space()
        self.assertEqual(len(space), 100)
        self.assertIn((3, 4), space)
        self.assertEqual(space.bounds_of((3, 4)), (30.0, 40.0, 34.0, 44.0))

    def test_query_region_matches_brute_force(self):
        space = self._grid_space()
        region = (22.0, 15.0, 55.0, 48.0)
        got = set(space.query_region(region))
        expected = {
            key
            for key, box in space.items()
            if bbox_intersects(box, region)
        }
        self.assertEqual(got, expected)
        self.assertTrue(expected)

    def test_query_results_are_in_insertion_order(self):
        space = self._grid_space()
        keys = space.query_region((0.0, 0.0, 100.0, 100.0))
        self.assertEqual(keys, space.keys())
        self.assertEqual(len(keys), 100)

    def test_query_point_with_radius(self):
        space = self._grid_space()
        self.assertEqual(space.query_point((31.0, 41.0)), [(3, 4)])
        # nothing sits exactly at the gap between the tiles
        self.assertEqual(space.query_point((36.0, 46.0)), [])
        # a radius of 5 reaches the four neighbouring tiles
        near = set(space.query_point((36.0, 46.0), 5.0))
        self.assertEqual(near, {(3, 4), (3, 5), (4, 4), (4, 5)})

    def test_negative_radius_raises(self):
        with self.assertRaises(ValueError):
            QuadTreeSpace().query_point((0.0, 0.0), -1.0)

    def test_modify_moves_an_entity(self):
        space = self._grid_space()
        space.modify((0, 0), (90.0, 90.0, 94.0, 94.0))
        self.assertEqual(len(space), 100)
        self.assertEqual(space.query_point((1.0, 1.0)), [])
        self.assertIn((0, 0), space.query_point((92.0, 92.0)))

    def test_remove(self):
        space = self._grid_space()
        self.assertTrue(space.remove((5, 5)))
        self.assertFalse(space.remove((5, 5)))
        self.assertEqual(len(space), 99)
        self.assertEqual(space.query_point((51.0, 51.0)), [])

    def test_clear(self):
        space = self._grid_space()
        space.clear()
        self.assertEqual(len(space), 0)
        self.assertEqual(space.query_region((0.0, 0.0, 100.0, 100.0)), [])

    def test_world_grows_when_an_item_falls_outside(self):
        space = QuadTreeSpace((0.0, 0.0, 10.0, 10.0), max_children=2)
        space.insert("a", (1.0, 1.0, 2.0, 2.0))
        space.insert("far", (500.0, 500.0, 510.0, 510.0))
        self.assertTrue(bbox_fully_contains(space.world, (500.0, 500.0, 510.0, 510.0)))
        self.assertEqual(space.query_point((505.0, 505.0)), ["far"])
        # the pre-existing item survives the rebuild
        self.assertEqual(space.query_point((1.5, 1.5)), ["a"])

    def test_straddling_items_are_still_found(self):
        space = QuadTreeSpace((0.0, 0.0, 100.0, 100.0), max_children=1, max_depth=6)
        space.insert("big", (10.0, 10.0, 90.0, 90.0))
        for i in range(20):
            space.insert(i, (float(i), float(i), float(i) + 0.5, float(i) + 0.5))
        self.assertIn("big", space.query_point((50.0, 50.0)))
        self.assertIn("big", space.query_region((85.0, 85.0, 95.0, 95.0)))

    def test_total_bounds(self):
        space = QuadTreeSpace((0.0, 0.0, 100.0, 100.0))
        self.assertIsNone(space.total_bounds())
        space.insert("a", (1.0, 2.0, 3.0, 4.0))
        space.insert("b", (-1.0, 0.0, 2.0, 9.0))
        self.assertEqual(space.total_bounds(), (-1.0, 0.0, 3.0, 9.0))

    def test_reinsert_same_key_replaces(self):
        space = QuadTreeSpace((0.0, 0.0, 100.0, 100.0))
        space.insert("a", (1.0, 1.0, 2.0, 2.0))
        space.insert("a", (50.0, 50.0, 51.0, 51.0))
        self.assertEqual(len(space), 1)
        self.assertEqual(space.query_point((1.5, 1.5)), [])
        self.assertEqual(space.query_point((50.5, 50.5)), ["a"])

    def test_invalid_config_and_boxes(self):
        with self.assertRaises(ValueError):
            QuadTreeSpace(max_children=0)
        with self.assertRaises(ValueError):
            QuadTreeSpace(max_depth=-1)
        with self.assertRaises(ValueError):
            QuadTreeSpace().insert("a", (5.0, 0.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            QuadTreeSpace().insert("a", (0.0, 0.0, 1.0))

    def test_deterministic_across_instances(self):
        first = self._grid_space().query_region((10.0, 10.0, 60.0, 60.0))
        second = self._grid_space().query_region((10.0, 10.0, 60.0, 60.0))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
