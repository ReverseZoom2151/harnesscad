import math
import unittest

from harnesscad.domain.geometry.assembly.placement import (
    Box,
    aggregate_box,
    apply_align,
    feature_point,
    parse_align,
    parse_align_chain,
    parse_offset,
    parse_polar,
    polar_position,
    resolve_placement,
    resolve_target,
)


class BoxTests(unittest.TestCase):
    def test_min_max(self):
        box = Box((0.0, 0.0, 0.0), (2.0, 4.0, 6.0))
        self.assertEqual(box.minimum(), (-1.0, -2.0, -3.0))
        self.assertEqual(box.maximum(), (1.0, 2.0, 3.0))

    def test_negative_size_rejected(self):
        with self.assertRaises(ValueError):
            Box((0.0, 0.0, 0.0), (-1.0, 1.0, 1.0))

    def test_translate(self):
        box = Box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)).translated((1.0, 2.0, 3.0))
        self.assertEqual(box.center, (1.0, 2.0, 3.0))


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.box = Box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))

    def test_faces(self):
        self.assertEqual(feature_point(self.box, "top_face"), (0.0, 0.0, 1.0))
        self.assertEqual(feature_point(self.box, "bottom_face"), (0.0, 0.0, -1.0))
        self.assertEqual(feature_point(self.box, "left_face"), (-1.0, 0.0, 0.0))
        self.assertEqual(feature_point(self.box, "right_face"), (1.0, 0.0, 0.0))
        self.assertEqual(feature_point(self.box, "front_face"), (0.0, -1.0, 0.0))
        self.assertEqual(feature_point(self.box, "back_face"), (0.0, 1.0, 0.0))

    def test_short_forms_and_center(self):
        self.assertEqual(feature_point(self.box, "top"), (0.0, 0.0, 1.0))
        self.assertEqual(feature_point(self.box, "center"), (0.0, 0.0, 0.0))

    def test_side_at(self):
        point = feature_point(self.box, "side_at(90)")
        self.assertAlmostEqual(point[0], 0.0)
        self.assertAlmostEqual(point[1], 1.0)
        self.assertAlmostEqual(point[2], 0.0)

    def test_unknown_feature(self):
        with self.assertRaises(ValueError):
            feature_point(self.box, "nose")

    def test_aggregate_box(self):
        aggregate = aggregate_box(
            [
                Box((-1.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
                Box((3.0, 0.0, 1.0), (2.0, 2.0, 2.0)),
            ]
        )
        self.assertEqual(aggregate.center, (1.0, 0.0, 0.5))
        self.assertEqual(aggregate.size, (6.0, 2.0, 3.0))

    def test_aggregate_empty(self):
        with self.assertRaises(ValueError):
            aggregate_box([])


class ParseTests(unittest.TestCase):
    def test_parse_align(self):
        clause = parse_align("Align(XY) shelf.left_face to side_L.right_face")
        self.assertEqual(clause.axes, (0, 1))
        self.assertEqual(clause.this_id, "shelf")
        self.assertEqual(clause.this_feature, "left_face")
        self.assertEqual(clause.target, "side_L.right_face")

    def test_parse_align_single_axis(self):
        self.assertEqual(parse_align("Align(Z) a.bottom to b.top").axes, (2,))

    def test_parse_align_rejects_other_text(self):
        with self.assertRaises(ValueError):
            parse_align("offset(1,2,3)")

    def test_parse_chain_skips_non_align(self):
        text = (
            "Align(XY) shelf.left_face to side_L.right_face; "
            "Align(Z) shelf.bottom_face to side_L.bottom_face; offset(0,0,0.01)"
        )
        clauses = parse_align_chain(text)
        self.assertEqual(len(clauses), 2)
        self.assertEqual(clauses[1].axes, (2,))

    def test_parse_offset(self):
        self.assertEqual(parse_offset("offset(0, 0, 0.010)"), (0.0, 0.0, 0.01))
        self.assertEqual(parse_offset("no offset here"), (0.0, 0.0, 0.0))

    def test_parse_polar(self):
        self.assertEqual(parse_polar("pos=polar(45)"), (45.0, 0.0))
        self.assertEqual(parse_polar("pos=polar(30; dr=0.02)"), (30.0, 0.02))
        with self.assertRaises(ValueError):
            parse_polar("offset(1,1,1)")


class ResolveTargetTests(unittest.TestCase):
    def setUp(self):
        self.boxes = {
            "base": Box((0.0, 0.0, 0.0), (4.0, 4.0, 1.0)),
            "leg_0": Box((-1.0, 0.0, 2.0), (1.0, 1.0, 2.0)),
            "leg_1": Box((1.0, 0.0, 2.0), (1.0, 1.0, 2.0)),
            "leg_2": Box((3.0, 0.0, 2.0), (1.0, 1.0, 2.0)),
        }

    def test_single_feature(self):
        self.assertEqual(resolve_target("base.top_face", self.boxes), (0.0, 0.0, 0.5))

    def test_indexed_instance(self):
        self.assertEqual(resolve_target("leg[1].center", self.boxes), (1.0, 0.0, 2.0))

    def test_index_out_of_range(self):
        with self.assertRaises(IndexError):
            resolve_target("leg[9].center", self.boxes)

    def test_aggregate_pattern(self):
        # Union of the three legs spans x in [-1.5, 3.5]; its top face is z = 3.
        point = resolve_target("leg[*].top_face", self.boxes)
        self.assertAlmostEqual(point[0], 1.0)
        self.assertAlmostEqual(point[2], 3.0)

    def test_avg_expression(self):
        point = resolve_target("Avg( leg_0.center , leg_2.center )", self.boxes)
        self.assertEqual(point, (1.0, 0.0, 2.0))

    def test_nested_avg_over_indexed(self):
        point = resolve_target("Avg(leg[0].center, leg[1].center)", self.boxes)
        self.assertEqual(point, (0.0, 0.0, 2.0))

    def test_unknown_reference(self):
        with self.assertRaises(KeyError):
            resolve_target("ghost.center", self.boxes)

    def test_unparseable(self):
        with self.assertRaises(ValueError):
            resolve_target("!!!", self.boxes)


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.boxes = {
            "side_L": Box((0.0, 0.0, 0.0), (0.02, 0.3, 1.0)),
        }
        self.shelf = Box((5.0, 5.0, 5.0), (0.6, 0.3, 0.02))

    def test_single_clause_locks_only_its_axes(self):
        clause = parse_align("Align(XY) shelf.left_face to side_L.right_face")
        moved = apply_align(self.shelf, clause, self.boxes)
        self.assertAlmostEqual(moved.minimum()[0], 0.01)  # touches the right face
        self.assertAlmostEqual(moved.center[1], 0.0)
        self.assertAlmostEqual(moved.center[2], 5.0)  # Z untouched

    def test_chain_then_offset(self):
        clauses = parse_align_chain(
            "Align(XY) shelf.left_face to side_L.right_face; "
            "Align(Z) shelf.bottom_face to side_L.bottom_face"
        )
        placed = resolve_placement(
            self.shelf, clauses, self.boxes, offset=parse_offset("offset(0,0,0.010)")
        )
        self.assertAlmostEqual(placed.minimum()[0], 0.01)
        self.assertAlmostEqual(placed.minimum()[2], -0.5 + 0.010)

    def test_clause_order_is_idempotent_when_reapplied(self):
        clauses = parse_align_chain("Align(XYZ) shelf.center to side_L.center")
        once = resolve_placement(self.shelf, clauses, self.boxes)
        twice = resolve_placement(once, clauses, self.boxes)
        self.assertEqual(once.center, twice.center)

    def test_no_clauses_leaves_box_alone(self):
        self.assertEqual(resolve_placement(self.shelf, (), self.boxes), self.shelf)

    def test_avg_target_centres_between_legs(self):
        boxes = {
            "leg_LF": Box((-1.0, -1.0, 0.0), (0.2, 0.2, 2.0)),
            "leg_RF": Box((1.0, -1.0, 0.0), (0.2, 0.2, 2.0)),
        }
        clauses = parse_align_chain("Align(XY) top.center to Avg(leg_LF.center, leg_RF.center)")
        placed = resolve_placement(Box((9.0, 9.0, 1.0), (2.0, 2.0, 0.1)), clauses, boxes)
        self.assertEqual(placed.center[:2], (0.0, -1.0))


class PolarTests(unittest.TestCase):
    def test_child_touches_parent(self):
        parent = Box((0.0, 0.0, 0.0), (2.0, 2.0, 1.0))
        child = Box((0.0, 0.0, 0.5), (0.4, 0.4, 0.4))
        placed = polar_position(child, parent, 0.0)
        self.assertAlmostEqual(placed.center[0], 1.2)  # 1.0 + 0.2
        self.assertAlmostEqual(placed.center[1], 0.0)
        self.assertAlmostEqual(placed.center[2], 0.5)  # z preserved

    def test_angle_and_radial_shift(self):
        parent = Box((0.0, 0.0, 0.0), (2.0, 2.0, 1.0))
        child = Box((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        placed = polar_position(child, parent, 90.0, dr=0.5)
        self.assertAlmostEqual(placed.center[0], 0.0)
        self.assertAlmostEqual(placed.center[1], 1.5)

    def test_matches_manual_trigonometry(self):
        parent = Box((1.0, 2.0, 0.0), (4.0, 4.0, 1.0))
        child = Box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        placed = polar_position(child, parent, 30.0)
        radius = 2.0 + 0.5
        self.assertAlmostEqual(placed.center[0], 1.0 + radius * math.cos(math.radians(30)))
        self.assertAlmostEqual(placed.center[1], 2.0 + radius * math.sin(math.radians(30)))


if __name__ == "__main__":
    unittest.main()
