"""Tests for reconstruction.worldcraft_layout_spec."""

import math
import unittest

from harnesscad.domain.reconstruction.scene.layout_spec import (
    LayoutSpec,
    ObjectPlacement,
    Pose,
)


class TestPose(unittest.TestCase):
    def test_defaults(self):
        p = Pose()
        self.assertEqual(p.position, (0.0, 0.0, 0.0))
        self.assertEqual(p.orientation, (0.0, 0.0, 0.0))
        self.assertEqual(p.scale, (1.0, 1.0, 1.0))

    def test_angle_wrapped_into_range(self):
        p = Pose(orientation=(_two_pi := 2.0 * math.pi, -math.pi / 2.0, 3.0 * math.pi))
        self.assertAlmostEqual(p.orientation[0], 0.0)
        self.assertAlmostEqual(p.orientation[1], 1.5 * math.pi)
        self.assertAlmostEqual(p.orientation[2], math.pi)
        for a in p.orientation:
            self.assertGreaterEqual(a, 0.0)
            self.assertLess(a, 2.0 * math.pi)

    def test_scale_must_be_positive(self):
        with self.assertRaises(ValueError):
            Pose(scale=(1.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            Pose(scale=(-1.0, 1.0, 1.0))

    def test_bad_length(self):
        with self.assertRaises(ValueError):
            Pose(position=(1.0, 2.0))

    def test_constructors_and_transforms(self):
        p = Pose.at(1.0, 2.0, 3.0).translated(1.0, 0.0, -1.0)
        self.assertEqual(p.position, (2.0, 2.0, 2.0))
        r = Pose.uniform_scale(2.0)
        self.assertEqual(r.scale, (2.0, 2.0, 2.0))
        y = Pose().rotated_z(math.pi / 2.0)
        self.assertAlmostEqual(y.yaw, math.pi / 2.0)

    def test_yaw_wraps_when_rotated(self):
        p = Pose(orientation=(0.0, 0.0, 1.5 * math.pi)).rotated_z(math.pi)
        self.assertAlmostEqual(p.yaw, 0.5 * math.pi)

    def test_roundtrip(self):
        p = Pose(position=(1.0, 2.0, 3.0), orientation=(0.1, 0.2, 0.3), scale=(1.0, 2.0, 3.0))
        self.assertEqual(Pose.from_dict(p.to_dict()), p)


class TestObjectPlacement(unittest.TestCase):
    def test_world_bounds_identity(self):
        o = ObjectPlacement("a", "chair", (0.5, 0.5, 1.0), Pose.at(2.0, 3.0, 1.0))
        lo, hi = o.world_bounds()
        self.assertEqual(lo, (1.5, 2.5, 0.0))
        self.assertEqual(hi, (2.5, 3.5, 2.0))

    def test_scale_grows_footprint(self):
        o = ObjectPlacement("a", "chair", (0.5, 0.5, 1.0), Pose(scale=(2.0, 2.0, 1.0)))
        self.assertEqual(o.scaled_half_extent(), (1.0, 1.0, 1.0))

    def test_quarter_turn_swaps_footprint(self):
        o = ObjectPlacement("a", "sofa", (2.0, 0.5, 1.0), Pose().rotated_z(math.pi / 2.0))
        wx, wy, wz = o.world_half_extent()
        self.assertAlmostEqual(wx, 0.5)
        self.assertAlmostEqual(wy, 2.0)
        self.assertAlmostEqual(wz, 1.0)

    def test_negative_extent_rejected(self):
        with self.assertRaises(ValueError):
            ObjectPlacement("a", "c", (-1.0, 1.0, 1.0))

    def test_roundtrip(self):
        o = ObjectPlacement("a", "chair", (0.5, 0.5, 1.0), Pose.at(1.0, 2.0, 3.0),
                            parent_id=None, attributes={"color": "red"})
        o2 = ObjectPlacement.from_dict(o.to_dict())
        self.assertEqual(o2.object_id, "a")
        self.assertEqual(o2.attributes["color"], "red")
        self.assertEqual(o2.pose.position, (1.0, 2.0, 3.0))


class TestLayoutSpec(unittest.TestCase):
    def _spec(self):
        s = LayoutSpec(room_bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 3.0)))
        s.add(ObjectPlacement("shelf", "shelf", (1.0, 0.3, 1.5), Pose.at(5.0, 1.0, 1.5)))
        s.add(ObjectPlacement("book1", "book", (0.1, 0.2, 0.3), Pose.at(4.5, 1.0, 1.0),
                              parent_id="shelf"))
        s.add(ObjectPlacement("book2", "book", (0.1, 0.2, 0.3), Pose.at(5.5, 1.0, 1.0),
                              parent_id="shelf"))
        s.add(ObjectPlacement("table", "table", (1.0, 1.0, 0.8), Pose.at(2.0, 2.0, 0.8)))
        return s

    def test_add_and_order(self):
        s = self._spec()
        self.assertEqual(len(s), 4)
        self.assertEqual(s.object_ids, ["shelf", "book1", "book2", "table"])

    def test_duplicate_rejected(self):
        s = LayoutSpec()
        s.add(ObjectPlacement("a", "c", (1.0, 1.0, 1.0)))
        with self.assertRaises(ValueError):
            s.add(ObjectPlacement("a", "c", (1.0, 1.0, 1.0)))

    def test_unknown_parent_rejected(self):
        s = LayoutSpec()
        with self.assertRaises(KeyError):
            s.add(ObjectPlacement("a", "c", (1.0, 1.0, 1.0), parent_id="missing"))

    def test_object_tree(self):
        s = self._spec()
        self.assertEqual([r.object_id for r in s.roots()], ["shelf", "table"])
        self.assertEqual([c.object_id for c in s.children("shelf")], ["book1", "book2"])
        self.assertEqual([d.object_id for d in s.descendants("shelf")], ["book1", "book2"])
        self.assertEqual([a.object_id for a in s.ancestors("book1")], ["shelf"])

    def test_topological_order_parents_first(self):
        s = self._spec()
        order = [p.object_id for p in s.topological_order()]
        self.assertLess(order.index("shelf"), order.index("book1"))
        self.assertLess(order.index("shelf"), order.index("book2"))
        self.assertEqual(set(order), {"shelf", "book1", "book2", "table"})

    def test_bad_room_bounds(self):
        with self.assertRaises(ValueError):
            LayoutSpec(room_bounds=((0.0, 0.0, 0.0), (-1.0, 1.0, 1.0)))

    def test_roundtrip(self):
        s = self._spec()
        s2 = LayoutSpec.from_dict(s.to_dict())
        self.assertEqual(s2.object_ids, s.object_ids)
        self.assertEqual(s2.get("book1").parent_id, "shelf")
        self.assertEqual(s2.room_bounds, ((0.0, 0.0, 0.0), (10.0, 10.0, 3.0)))


if __name__ == "__main__":
    unittest.main()
