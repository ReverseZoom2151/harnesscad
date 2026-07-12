"""Tests for geometry.cascade_entity_selector."""

import unittest

from geometry.cascade_entity_selector import Entity, EntitySelector


def _box_edges():
    """The 12 edges of a unit box [0,1]^3 as selector rows.

    Each edge: midpoint position, tangent direction, length 1.
    """
    rows = []
    # 4 edges parallel to X (y,z in {0,1}), midpoint x=0.5
    for y in (0, 1):
        for z in (0, 1):
            rows.append({"position": [0.5, y, z], "direction": [1, 0, 0], "size": 1.0, "kind": "Line"})
    # 4 edges parallel to Y
    for x in (0, 1):
        for z in (0, 1):
            rows.append({"position": [x, 0.5, z], "direction": [0, 1, 0], "size": 1.0, "kind": "Line"})
    # 4 edges parallel to Z
    for x in (0, 1):
        for y in (0, 1):
            rows.append({"position": [x, y, 0.5], "direction": [0, 0, 1], "size": 1.0, "kind": "Line"})
    return EntitySelector.from_tuples(rows)


class TestOrientation(unittest.TestCase):
    def test_parallel_selects_four_z_edges(self):
        sel = _box_edges().parallel([0, 0, 1])
        self.assertEqual(sel.count(), 4)
        for e in sel.entities():
            self.assertEqual(e.direction, (0.0, 0.0, 1.0))

    def test_parallel_ignores_sign(self):
        sel = _box_edges().parallel([0, 0, -1])
        self.assertEqual(sel.count(), 4)

    def test_perpendicular_selects_eight(self):
        # edges perpendicular to Z = the 8 X- and Y-parallel edges
        sel = _box_edges().perpendicular([0, 0, 1])
        self.assertEqual(sel.count(), 8)

    def test_at_angle_45(self):
        rows = [
            {"position": [0, 0, 0], "direction": [1, 1, 0], "size": 1.0},
            {"position": [1, 0, 0], "direction": [1, 0, 0], "size": 1.0},
        ]
        sel = EntitySelector.from_tuples(rows).at_angle([1, 0, 0], 45.0)
        self.assertEqual(sel.count(), 1)
        self.assertEqual(sel.entities()[0].position, (0.0, 0.0, 0.0))

    def test_direction_none_is_skipped(self):
        rows = [{"position": [0, 0, 0], "direction": None, "size": 1.0}]
        self.assertEqual(EntitySelector.from_tuples(rows).parallel([1, 0, 0]).count(), 0)


class TestGrouping(unittest.TestCase):
    def test_max_returns_top_face_edge_set(self):
        # top edges (z=1): the 4 X/Y edges whose midpoint z == 1
        top = _box_edges().perpendicular([0, 0, 1]).max([0, 0, 1])
        self.assertEqual(top.count(), 4)
        for e in top.entities():
            self.assertAlmostEqual(e.position[2], 1.0)

    def test_min_returns_bottom_set(self):
        bottom = _box_edges().perpendicular([0, 0, 1]).min([0, 0, 1])
        self.assertEqual(bottom.count(), 4)
        for e in bottom.entities():
            self.assertAlmostEqual(e.position[2], 0.0)

    def test_group_by_counts(self):
        groups = _box_edges().perpendicular([0, 0, 1]).group_by([0, 0, 1])
        # two z-levels: 0 and 1, each with 4 edges
        self.assertEqual(len(groups), 2)
        self.assertEqual([len(g) for g in groups], [4, 4])

    def test_sort_by_is_ascending_projection(self):
        sel = _box_edges().parallel([0, 0, 1]).sort_by([1, 0, 0])
        xs = [e.position[0] for e in sel.entities()]
        self.assertEqual(xs, sorted(xs))


class TestSizeAndBox(unittest.TestCase):
    def test_longer_shorter(self):
        rows = [
            {"position": [0, 0, 0], "direction": [1, 0, 0], "size": 2.0},
            {"position": [0, 0, 0], "direction": [1, 0, 0], "size": 5.0},
        ]
        sel = EntitySelector.from_tuples(rows)
        self.assertEqual(sel.longer_than(3.0).count(), 1)
        self.assertEqual(sel.shorter_than(3.0).count(), 1)

    def test_within_box(self):
        sel = _box_edges().within_box([-0.1, -0.1, -0.1], [0.6, 1.1, 0.6])
        for e in sel.entities():
            self.assertTrue(e.position[0] <= 0.6 and e.position[2] <= 0.6)

    def test_of_type(self):
        rows = [
            {"position": [0, 0, 0], "direction": [1, 0, 0], "kind": "Line"},
            {"position": [0, 0, 0], "direction": [1, 0, 0], "kind": "Circle"},
        ]
        self.assertEqual(EntitySelector.from_tuples(rows).of_type("Circle").count(), 1)


class TestSlicingAndTerminals(unittest.TestCase):
    def test_indices_stable(self):
        sel = _box_edges().parallel([0, 0, 1])
        idx = sel.indices()
        self.assertEqual(len(idx), 4)
        self.assertEqual(idx, sorted(set(idx)))  # unique, ordered as built

    def test_first_last_at(self):
        sel = _box_edges().parallel([0, 0, 1]).sort_by([1, 0, 0])
        self.assertEqual(sel.first(2).count(), 2)
        self.assertEqual(sel.last(1).count(), 1)
        self.assertEqual(sel.at(0), sel.entities()[0].index)
        self.assertEqual(sel.at(999), -1)

    def test_immutability(self):
        base = _box_edges()
        _ = base.parallel([0, 0, 1])
        self.assertEqual(base.count(), 12)

    def test_zero_axis_raises(self):
        with self.assertRaises(ValueError):
            _box_edges().parallel([0, 0, 0])


if __name__ == "__main__":
    unittest.main()
