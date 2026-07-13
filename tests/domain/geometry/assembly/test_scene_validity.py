"""Tests for geometry.worldcraft_scene_collision."""

import unittest

from harnesscad.domain.reconstruction.scene.layout_spec import LayoutSpec, ObjectPlacement, Pose
from harnesscad.domain.geometry.assembly.scene_validity import (
    SceneReport,
    check_child_containment,
    check_floating,
    check_object_collisions,
    check_out_of_bounds,
    check_scene,
    check_stacking,
)


def _obj(oid, x, y, z, hx=0.5, hy=0.5, hz=0.5, parent=None, attrs=None):
    return ObjectPlacement(oid, "box", (hx, hy, hz), Pose.at(x, y, z),
                           parent_id=parent, attributes=attrs or {})


class TestCollisions(unittest.TestCase):
    def test_no_collision_when_apart(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0, 0.5))
        s.add(_obj("b", 5.0, 0.0, 0.5))
        self.assertEqual(check_object_collisions(s), [])

    def test_collision_detected(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0, 0.5))
        s.add(_obj("b", 0.5, 0.0, 0.5))
        issues = check_object_collisions(s)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "object_collision")

    def test_touching_is_not_collision(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0, 0.5))
        s.add(_obj("b", 1.0, 0.0, 0.5))  # faces touch exactly
        self.assertEqual(check_object_collisions(s), [])

    def test_parent_child_exempt(self):
        s = LayoutSpec()
        s.add(_obj("shelf", 0.0, 0.0, 1.0, hx=1.0, hy=1.0, hz=1.0))
        s.add(_obj("book", 0.0, 0.0, 1.0, hx=0.2, hy=0.2, hz=0.2, parent="shelf"))
        # book fully inside shelf, but they are family -> no collision error
        self.assertEqual(check_object_collisions(s), [])


class TestOutOfBounds(unittest.TestCase):
    def test_inside_ok(self):
        s = LayoutSpec(room_bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 3.0)))
        s.add(_obj("a", 5.0, 5.0, 0.5))
        self.assertEqual(check_out_of_bounds(s), [])

    def test_escapes_flagged(self):
        s = LayoutSpec(room_bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 3.0)))
        s.add(_obj("a", 9.8, 5.0, 0.5))  # extends to x=10.3
        issues = check_out_of_bounds(s)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "out_of_bounds")

    def test_no_room_no_check(self):
        s = LayoutSpec()
        s.add(_obj("a", 1000.0, 0.0, 0.5))
        self.assertEqual(check_out_of_bounds(s), [])


class TestContainmentAndStacking(unittest.TestCase):
    def test_child_contained_ok(self):
        s = LayoutSpec()
        s.add(_obj("shelf", 0.0, 0.0, 1.0, hx=1.0, hy=1.0, hz=1.0))  # top at z=2
        s.add(_obj("book", 0.0, 0.0, 2.25, hx=0.2, hy=0.2, hz=0.25, parent="shelf"))
        self.assertEqual(check_child_containment(s), [])
        self.assertEqual(check_stacking(s), [])

    def test_child_escapes_footprint(self):
        s = LayoutSpec()
        s.add(_obj("shelf", 0.0, 0.0, 1.0, hx=1.0, hy=1.0, hz=1.0))
        s.add(_obj("book", 3.0, 0.0, 2.25, hx=0.2, hy=0.2, hz=0.25, parent="shelf"))
        issues = check_child_containment(s)
        self.assertEqual(issues[0].code, "child_escapes_host")

    def test_stacking_inversion(self):
        s = LayoutSpec()
        s.add(_obj("shelf", 0.0, 0.0, 1.0, hx=1.0, hy=1.0, hz=1.0))  # top z=2
        s.add(_obj("book", 0.0, 0.0, 0.5, hx=0.2, hy=0.2, hz=0.25, parent="shelf"))  # base z=0.25
        issues = check_stacking(s)
        self.assertEqual(issues[0].code, "stacking_inversion")


class TestFloating(unittest.TestCase):
    def test_on_floor_ok(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0, 0.5, attrs={"needs_support": True}))  # base at 0
        self.assertEqual(check_floating(s), [])

    def test_supported_by_other_ok(self):
        s = LayoutSpec()
        s.add(_obj("table", 0.0, 0.0, 0.5, hx=1.0, hy=1.0, hz=0.5))  # top z=1
        s.add(_obj("cup", 0.0, 0.0, 1.25, hx=0.2, hy=0.2, hz=0.25,
                   attrs={"needs_support": True}))  # base z=1
        self.assertEqual(check_floating(s), [])

    def test_floating_flagged(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0, 5.0, attrs={"needs_support": True}))  # base at 4.5
        issues = check_floating(s)
        self.assertEqual(issues[0].code, "floating_object")

    def test_no_flag_without_needs_support(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0, 5.0))
        self.assertEqual(check_floating(s), [])


class TestCheckScene(unittest.TestCase):
    def test_clean_scene_ok(self):
        s = LayoutSpec(room_bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 3.0)))
        s.add(_obj("a", 2.0, 2.0, 0.5))
        s.add(_obj("b", 6.0, 6.0, 0.5))
        report = check_scene(s)
        self.assertIsInstance(report, SceneReport)
        self.assertTrue(report.ok)
        self.assertEqual(report.errors, [])

    def test_dirty_scene_not_ok(self):
        s = LayoutSpec(room_bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 3.0)))
        s.add(_obj("a", 0.0, 0.0, 0.5))
        s.add(_obj("b", 0.3, 0.0, 0.5))  # collides with a
        report = check_scene(s)
        self.assertFalse(report.ok)
        self.assertIn("object_collision", report.codes())

    def test_selective_checks(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0, 0.5))
        s.add(_obj("b", 0.3, 0.0, 0.5))
        report = check_scene(s, check_collisions=False)
        self.assertTrue(report.ok)


if __name__ == "__main__":
    unittest.main()
