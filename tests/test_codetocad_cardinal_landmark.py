import unittest

from harnesscad.domain.geometry.topology.landmarks import (
    BoundaryAxis,
    BoundaryBox,
    CardinalError,
    Landmark,
    LandmarkRegistry,
    cardinal_point,
    nearest,
    resolve_cardinal,
)


def unit_box():
    # 2 x 4 x 6 box centred at the origin.
    return BoundaryBox.from_extents((0.0, 0.0, 0.0), (2.0, 4.0, 6.0))


class TestBoundaryAxis(unittest.TestCase):
    def test_center_and_length(self):
        axis = BoundaryAxis(-1.0, 3.0)
        self.assertAlmostEqual(axis.center, 1.0)
        self.assertAlmostEqual(axis.length, 4.0)

    def test_select(self):
        axis = BoundaryAxis(0.0, 10.0)
        self.assertAlmostEqual(axis.select("min"), 0.0)
        self.assertAlmostEqual(axis.select("max"), 10.0)
        self.assertAlmostEqual(axis.select("center"), 5.0)

    def test_invalid(self):
        with self.assertRaises(CardinalError):
            BoundaryAxis(5.0, 1.0)
        with self.assertRaises(CardinalError):
            BoundaryAxis(0.0, 1.0).select("middle")


class TestBoundaryBox(unittest.TestCase):
    def test_from_points(self):
        box = BoundaryBox.from_points([(0, 0, 0), (1, 2, 3), (-1, 0, 1)])
        self.assertEqual(box.size, (2.0, 2.0, 3.0))
        self.assertAlmostEqual(box.center[0], 0.0)

    def test_from_points_empty(self):
        with self.assertRaises(CardinalError):
            BoundaryBox.from_points([])

    def test_from_points_bad_dim(self):
        with self.assertRaises(CardinalError):
            BoundaryBox.from_points([(0, 0)])

    def test_extents(self):
        box = unit_box()
        self.assertEqual(box.size, (2.0, 4.0, 6.0))
        self.assertAlmostEqual(box.x.min, -1.0)
        self.assertAlmostEqual(box.z.max, 3.0)

    def test_negative_size(self):
        with self.assertRaises(CardinalError):
            BoundaryBox.from_extents((0, 0, 0), (1, -1, 1))

    def test_contains(self):
        box = unit_box()
        self.assertTrue(box.contains((0.0, 0.0, 0.0)))
        self.assertTrue(box.contains((1.0, 2.0, 3.0)))
        self.assertFalse(box.contains((1.1, 0.0, 0.0)))
        self.assertTrue(box.contains((1.1, 0.0, 0.0), tolerance=0.2))

    def test_expand_union_translate(self):
        box = unit_box().expand(1.0)
        self.assertEqual(box.size, (4.0, 6.0, 8.0))
        merged = unit_box().union(BoundaryBox.from_points([(10, 0, 0)]))
        self.assertAlmostEqual(merged.x.max, 10.0)
        moved = unit_box().translated((1.0, 0.0, -1.0))
        self.assertAlmostEqual(moved.x.min, 0.0)
        self.assertAlmostEqual(moved.z.max, 2.0)

    def test_diagonal(self):
        box = BoundaryBox.from_extents((0, 0, 0), (2, 2, 1))
        self.assertAlmostEqual(box.diagonal, 3.0)


class TestResolveCardinal(unittest.TestCase):
    def test_center(self):
        self.assertEqual(resolve_cardinal("center"), ("center", "center", "center"))

    def test_faces(self):
        self.assertEqual(resolve_cardinal("top"), ("center", "center", "max"))
        self.assertEqual(resolve_cardinal("left"), ("min", "center", "center"))
        self.assertEqual(resolve_cardinal("front"), ("center", "min", "center"))

    def test_corner(self):
        self.assertEqual(resolve_cardinal("top_front_left"), ("min", "min", "max"))
        self.assertEqual(resolve_cardinal("BOTTOM_BACK_RIGHT"), ("max", "max", "min"))

    def test_format_insensitivity(self):
        expected = ("min", "min", "max")
        for text in (
            "top_front_left",
            "TOP-FRONT-LEFT",
            "TopFrontLeft",
            "top front left",
            "left front top",
        ):
            self.assertEqual(resolve_cardinal(text), expected)

    def test_center_suffix(self):
        self.assertEqual(resolve_cardinal("top_center"), ("center", "center", "max"))

    def test_contradiction(self):
        with self.assertRaises(CardinalError):
            resolve_cardinal("top_bottom")

    def test_unknown(self):
        with self.assertRaises(CardinalError):
            resolve_cardinal("northwest")
        with self.assertRaises(CardinalError):
            resolve_cardinal("")
        with self.assertRaises(CardinalError):
            resolve_cardinal(7)


class TestCardinalPoint(unittest.TestCase):
    def test_anchors(self):
        box = unit_box()
        self.assertEqual(cardinal_point(box, "center"), (0.0, 0.0, 0.0))
        self.assertEqual(cardinal_point(box, "top_center"), (0.0, 0.0, 3.0))
        self.assertEqual(
            cardinal_point(box, "bottom_front_left"), (-1.0, -2.0, -3.0)
        )
        self.assertEqual(
            cardinal_point(box, "top_back_right"), (1.0, 2.0, 3.0)
        )

    def test_offset_expression(self):
        box = unit_box()
        point = cardinal_point(box, "top_left", offset=("5mm", 0, "-1cm"))
        self.assertAlmostEqual(point[0], -1.0 + 0.005)
        self.assertAlmostEqual(point[1], 0.0)
        self.assertAlmostEqual(point[2], 3.0 - 0.01)


class TestLandmarks(unittest.TestCase):
    def test_landmark_position(self):
        mark = Landmark("hole", "top_front_left", ("5mm", "5mm", 0))
        position = mark.position(unit_box())
        self.assertAlmostEqual(position[0], -0.995)
        self.assertAlmostEqual(position[1], -1.995)
        self.assertAlmostEqual(position[2], 3.0)

    def test_registry(self):
        registry = LandmarkRegistry(unit_box())
        registry.add("top", "top_center")
        registry.add("nub", "right_center", offset=(0, 0, "10mm"))
        self.assertEqual(registry.position("top"), (0.0, 0.0, 3.0))
        self.assertAlmostEqual(registry.position("nub")[2], 0.01)
        self.assertEqual(sorted(registry.positions()), ["nub", "top"])

    def test_registry_follows_box(self):
        registry = LandmarkRegistry(unit_box())
        registry.add("top", "top_center")
        registry.set_box(unit_box().translated((0.0, 0.0, 5.0)))
        self.assertAlmostEqual(registry.position("top")[2], 8.0)

    def test_registry_errors(self):
        registry = LandmarkRegistry(unit_box())
        registry.add("a", "top")
        with self.assertRaises(CardinalError):
            registry.add("a", "top")
        with self.assertRaises(CardinalError):
            registry.add("b", "sideways")
        with self.assertRaises(CardinalError):
            registry.get("missing")


class TestNearest(unittest.TestCase):
    def test_sorted_by_distance(self):
        candidates = [
            ("far", (10.0, 0.0, 0.0)),
            ("near", (1.0, 0.0, 0.0)),
            ("mid", (3.0, 0.0, 0.0)),
        ]
        result = nearest(candidates, (0.0, 0.0, 0.0))
        self.assertEqual([key for key, _ in result], ["near", "mid", "far"])
        self.assertAlmostEqual(result[0][1], 1.0)

    def test_search_radius(self):
        candidates = [("a", (0.0, 0.0, 0.0)), ("b", (5.0, 0.0, 0.0))]
        result = nearest(candidates, (0.0, 0.0, 0.0), search_radius=1.0)
        self.assertEqual([key for key, _ in result], ["a"])

    def test_tie_break_is_stable(self):
        candidates = [("b", (1.0, 0.0, 0.0)), ("a", (0.0, 1.0, 0.0))]
        result = nearest(candidates, (0.0, 0.0, 0.0))
        self.assertEqual([key for key, _ in result], ["b", "a"])

    def test_empty(self):
        self.assertEqual(nearest([], (0.0, 0.0, 0.0)), [])

    def test_vertex_selection_on_a_box(self):
        box = unit_box()
        corners = []
        for x in (box.x.min, box.x.max):
            for y in (box.y.min, box.y.max):
                for z in (box.z.min, box.z.max):
                    corners.append(((x, y, z), (x, y, z)))
        target = cardinal_point(box, "top_front_left")
        result = nearest(corners, target, search_radius=1e-9)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], (-1.0, -2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
