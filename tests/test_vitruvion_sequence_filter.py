"""Tests for dataengine.vitruvion_sequence_filter."""

import unittest

from harnesscad.data.dataengine.vitruvion_sequence_filter import (
    FilterConfig,
    filter_indices,
    filter_sketch,
    shard_range,
    sketch_is_renderable,
    unique_indices,
)
from harnesscad.domain.geometry.vitruvion_sketch_norm import VArc, VCircle, VPoint, entity_from_params


def _grid_sketch(count):
    """A sketch of ``count`` distinct, non-degenerate lines."""
    return [
        entity_from_params([0.0, float(i), 1.0, float(i) + 1.0]) for i in range(count)
    ]


class TestRenderability(unittest.TestCase):
    def test_good_sketch(self):
        self.assertTrue(sketch_is_renderable(_grid_sketch(6)))

    def test_zero_extent_sketch_is_rejected(self):
        self.assertFalse(sketch_is_renderable([VPoint(x=1.0, y=1.0)]))

    def test_zero_radius_circle_is_rejected(self):
        entities = _grid_sketch(3) + [VCircle(xCenter=0.5, yCenter=0.5, radius=0.0)]
        self.assertFalse(sketch_is_renderable(entities))

    def test_zero_length_line_is_rejected(self):
        entities = _grid_sketch(3) + [entity_from_params([0.5, 0.5, 0.5, 0.5])]
        self.assertFalse(sketch_is_renderable(entities))

    def test_zero_sweep_arc_is_rejected(self):
        entities = _grid_sketch(3) + [
            VArc(xCenter=0.5, yCenter=0.5, radius=0.2, startParam=1.0, endParam=1.0)
        ]
        self.assertFalse(sketch_is_renderable(entities))

    def test_healthy_arc_is_accepted(self):
        entities = _grid_sketch(3) + [
            VArc(xCenter=0.5, yCenter=0.5, radius=0.2, startParam=0.0, endParam=2.0)
        ]
        self.assertTrue(sketch_is_renderable(entities))

    def test_input_is_not_mutated(self):
        entities = _grid_sketch(6)
        before = entities[0].pntX
        sketch_is_renderable(entities)
        self.assertEqual(entities[0].pntX, before)


class TestFilterSketch(unittest.TestCase):
    def test_entity_count_bounds(self):
        self.assertFalse(filter_sketch(_grid_sketch(5)))
        self.assertTrue(filter_sketch(_grid_sketch(6)))
        self.assertTrue(filter_sketch(_grid_sketch(16)))
        self.assertFalse(filter_sketch(_grid_sketch(17)))

    def test_custom_bounds(self):
        config = FilterConfig(min_entities=1, max_entities=3)
        self.assertTrue(filter_sketch(_grid_sketch(2), config))
        self.assertFalse(filter_sketch(_grid_sketch(4), config))

    def test_filter_indices(self):
        sketches = [_grid_sketch(6), _grid_sketch(2), _grid_sketch(8)]
        self.assertEqual(filter_indices(sketches), [0, 2])


class TestUniqueIndices(unittest.TestCase):
    def test_keeps_first_occurrence(self):
        sequences = [[1, 2, 3], [4, 5], [1, 2, 3], [4, 5], [9]]
        self.assertEqual(unique_indices(sequences), [0, 1, 4])

    def test_equal_length_only_comparison(self):
        # A prefix of another stream is a different stream.
        self.assertEqual(unique_indices([[1, 2], [1, 2, 0]]), [0, 1])

    def test_indices_are_ascending(self):
        sequences = [[3], [1], [3], [2], [1]]
        self.assertEqual(unique_indices(sequences), [0, 1, 3])

    def test_empty(self):
        self.assertEqual(unique_indices([]), [])

    def test_all_unique(self):
        self.assertEqual(unique_indices([[1], [2], [3]]), [0, 1, 2])


class TestShardRange(unittest.TestCase):
    def test_tiles_the_range_exactly(self):
        n, k = 103, 8
        covered = []
        for shard in range(k):
            start, end = shard_range(shard, n, k)
            covered.extend(range(start, end))
        self.assertEqual(covered, list(range(n)))

    def test_sizes_differ_by_at_most_one(self):
        n, k = 103, 8
        sizes = [end - start for start, end in (shard_range(i, n, k) for i in range(k))]
        self.assertEqual(max(sizes) - min(sizes), 1)
        self.assertEqual(sizes[:7], [13] * 7)
        self.assertEqual(sizes[7], 12)

    def test_exact_division(self):
        self.assertEqual(shard_range(0, 10, 5), (0, 2))
        self.assertEqual(shard_range(4, 10, 5), (8, 10))

    def test_more_shards_than_items(self):
        self.assertEqual(shard_range(0, 2, 4), (0, 1))
        self.assertEqual(shard_range(2, 2, 4), (2, 2))

    def test_guards(self):
        with self.assertRaises(ValueError):
            shard_range(0, 10, 0)
        with self.assertRaises(ValueError):
            shard_range(5, 10, 5)
        with self.assertRaises(ValueError):
            shard_range(0, -1, 2)


if __name__ == "__main__":
    unittest.main()
