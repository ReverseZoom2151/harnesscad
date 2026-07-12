"""Tests for generation.worldcraft_instantiation."""

import math
import unittest

from reconstruction.worldcraft_layout_spec import LayoutSpec, ObjectPlacement, Pose
from generation.worldcraft_instantiation import (
    AssetDef,
    AssetLibrary,
    SceneInstance,
    Transform,
    instances_bounds,
    instantiate_layout,
)


class TestTransform(unittest.TestCase):
    def test_apply_translation(self):
        t = Transform(translation=(1.0, 2.0, 3.0))
        self.assertEqual(t.apply((0.0, 0.0, 0.0)), (1.0, 2.0, 3.0))

    def test_apply_scale(self):
        t = Transform(scale=(2.0, 3.0, 4.0))
        self.assertEqual(t.apply((1.0, 1.0, 1.0)), (2.0, 3.0, 4.0))

    def test_apply_yaw_quarter_turn(self):
        t = Transform(yaw=math.pi / 2.0)
        x, y, z = t.apply((1.0, 0.0, 0.0))
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1.0)
        self.assertAlmostEqual(z, 0.0)

    def test_from_pose(self):
        t = Transform.from_pose(Pose(position=(1.0, 0.0, 0.0), scale=(2.0, 2.0, 2.0)))
        self.assertEqual(t.translation, (1.0, 0.0, 0.0))
        self.assertEqual(t.scale, (2.0, 2.0, 2.0))

    def test_compose_translation_chains(self):
        parent = Transform(translation=(10.0, 0.0, 0.0))
        child = Transform(translation=(1.0, 0.0, 0.0))
        c = parent.compose(child)
        self.assertEqual(c.translation, (11.0, 0.0, 0.0))

    def test_compose_scale_and_yaw(self):
        parent = Transform(scale=(2.0, 2.0, 2.0), yaw=math.pi / 2.0)
        child = Transform(translation=(1.0, 0.0, 0.0))
        c = parent.compose(child)
        # child origin (1,0,0) scaled by 2 -> (2,0,0), yawed 90deg -> (0,2,0)
        self.assertAlmostEqual(c.translation[0], 0.0)
        self.assertAlmostEqual(c.translation[1], 2.0)
        self.assertEqual(c.scale, (2.0, 2.0, 2.0))
        self.assertAlmostEqual(c.yaw, math.pi / 2.0)


class TestAssetLibrary(unittest.TestCase):
    def test_register_and_get(self):
        lib = AssetLibrary()
        lib.register(AssetDef("chair", (0.5, 0.5, 1.0), {"material": "wood"}))
        self.assertTrue(lib.has("chair"))
        self.assertEqual(lib.get("chair").half_extent, (0.5, 0.5, 1.0))
        self.assertEqual(lib.names, ["chair"])

    def test_duplicate_rejected(self):
        lib = AssetLibrary()
        lib.register(AssetDef("chair", (0.5, 0.5, 1.0)))
        with self.assertRaises(ValueError):
            lib.register(AssetDef("chair", (1.0, 1.0, 1.0)))


class TestInstantiate(unittest.TestCase):
    def test_simple_flat_scene(self):
        s = LayoutSpec()
        s.add(ObjectPlacement("a", "box", (0.5, 0.5, 0.5), Pose.at(2.0, 0.0, 0.5)))
        insts = instantiate_layout(s)
        self.assertEqual(len(insts), 1)
        self.assertEqual(insts[0].world_min, (1.5, -0.5, 0.0))
        self.assertEqual(insts[0].world_max, (2.5, 0.5, 1.0))

    def test_parent_composition(self):
        s = LayoutSpec()
        s.add(ObjectPlacement("shelf", "shelf", (1.0, 0.3, 1.5), Pose.at(10.0, 0.0, 1.5)))
        # book pose relative to shelf frame: 0.5 in local +x
        s.add(ObjectPlacement("book", "book", (0.1, 0.1, 0.2), Pose.at(0.5, 0.0, 0.0),
                              parent_id="shelf"))
        insts = instantiate_layout(s, compose_parents=True)
        by_id = {i.object_id: i for i in insts}
        # book world center x = shelf(10) + local(0.5) = 10.5
        self.assertAlmostEqual(by_id["book"].world_center[0], 10.5)

    def test_no_parent_composition_treats_global(self):
        s = LayoutSpec()
        s.add(ObjectPlacement("shelf", "shelf", (1.0, 0.3, 1.5), Pose.at(10.0, 0.0, 1.5)))
        s.add(ObjectPlacement("book", "book", (0.1, 0.1, 0.2), Pose.at(0.5, 0.0, 0.0),
                              parent_id="shelf"))
        insts = instantiate_layout(s, compose_parents=False)
        by_id = {i.object_id: i for i in insts}
        self.assertAlmostEqual(by_id["book"].world_center[0], 0.5)

    def test_library_supplies_extent_and_attrs(self):
        lib = AssetLibrary()
        lib.register(AssetDef("chair", (0.6, 0.6, 1.0), {"material": "oak", "weight": 5}))
        s = LayoutSpec()
        s.add(ObjectPlacement("c1", "chair", (0.1, 0.1, 0.1), Pose.at(0.0, 0.0, 1.0),
                              attributes={"weight": 9}))
        insts = instantiate_layout(s, lib)
        inst = insts[0]
        # library half-extent used, not placement's
        self.assertEqual(inst.world_max, (0.6, 0.6, 2.0))
        # library default present, placement override wins
        self.assertEqual(inst.attributes["material"], "oak")
        self.assertEqual(inst.attributes["weight"], 9)

    def test_yaw_swaps_footprint_bounds(self):
        s = LayoutSpec()
        s.add(ObjectPlacement("a", "sofa", (2.0, 0.5, 1.0),
                              Pose(position=(0.0, 0.0, 1.0)).rotated_z(math.pi / 2.0)))
        inst = instantiate_layout(s)[0]
        self.assertAlmostEqual(inst.world_max[0] - inst.world_min[0], 1.0)
        self.assertAlmostEqual(inst.world_max[1] - inst.world_min[1], 4.0)

    def test_bounds_of_scene(self):
        s = LayoutSpec()
        s.add(ObjectPlacement("a", "box", (0.5, 0.5, 0.5), Pose.at(0.0, 0.0, 0.5)))
        s.add(ObjectPlacement("b", "box", (0.5, 0.5, 0.5), Pose.at(5.0, 5.0, 0.5)))
        insts = instantiate_layout(s)
        lo, hi = instances_bounds(insts)
        self.assertEqual(lo, (-0.5, -0.5, 0.0))
        self.assertEqual(hi, (5.5, 5.5, 1.0))

    def test_empty_bounds_none(self):
        self.assertIsNone(instances_bounds([]))

    def test_deterministic(self):
        s = LayoutSpec()
        s.add(ObjectPlacement("a", "box", (0.5, 0.5, 0.5), Pose.at(1.0, 2.0, 0.5)))
        a = [i.__dict__ for i in instantiate_layout(s)]
        b = [i.__dict__ for i in instantiate_layout(s)]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
