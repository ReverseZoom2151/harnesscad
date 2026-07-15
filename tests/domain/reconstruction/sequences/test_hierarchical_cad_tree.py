"""Tests for domain.reconstruction.sequences.hierarchical_cad_tree."""

import unittest

from harnesscad.domain.reconstruction.sequences.hierarchical_cad_tree import (
    build_tree,
    object_level_trajectory,
    part_level_topology,
    primitive_type,
    structure_preserving_perturbation,
)

CAD = {
    "part_1": {
        "face_1": {
            "loop_1": {"line_1": {"x": 0}, "arc_1": {"x": 1}, "line_2": {"x": 2},
                       "line_3": {"x": 3}},
            "loop_2": {"circle_1": {"r": 5}},
        },
        "operation": "NewBodyFeatureOperation",
    },
    "part_2": {
        "face_1": {"loop_1": {"line_1": {"x": 0}, "line_2": {"x": 1},
                              "line_3": {"x": 2}, "line_4": {"x": 3}}},
        "operation": "CutFeatureOperation",
    },
}


class PrimitiveTypeTest(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(primitive_type("line_1"), "line")
        self.assertEqual(primitive_type("arc_12"), "arc")
        self.assertEqual(primitive_type("circle"), "circle")


class BuildTreeTest(unittest.TestCase):
    def test_counts(self):
        t = build_tree(CAD)
        self.assertEqual(t.num_parts(), 2)
        self.assertEqual(t.num_faces(), 2)
        self.assertEqual(t.num_loops(), 3)

    def test_operations_abbreviated(self):
        t = build_tree(CAD)
        self.assertEqual(t.parts[0].operation, "NewBody")
        self.assertEqual(t.parts[1].operation, "Cut")


class TrajectoryTest(unittest.TestCase):
    def test_object_level(self):
        traj = object_level_trajectory(build_tree(CAD))
        self.assertEqual(traj, ["part_1: NewBody", "part_2: Cut"])

    def test_part_level_topology(self):
        topo = part_level_topology(build_tree(CAD))
        self.assertEqual(topo[0], "loop_1: line|arc|line|line")
        self.assertEqual(topo[1], "loop_2: circle")
        self.assertEqual(topo[2], "loop_1: line|line|line|line")


class PerturbationTest(unittest.TestCase):
    def test_shape_preserved_one_value_changed(self):
        out = structure_preserving_perturbation(CAD, delta=10.0, index=0)
        # tree shape identical
        self.assertEqual(build_tree(out).num_loops(), build_tree(CAD).num_loops())
        # exactly one numeric leaf differs
        self.assertNotEqual(out["part_1"]["face_1"]["loop_1"]["line_1"]["x"],
                            CAD["part_1"]["face_1"]["loop_1"]["line_1"]["x"])

    def test_original_unmutated(self):
        structure_preserving_perturbation(CAD, delta=99.0, index=1)
        self.assertEqual(CAD["part_1"]["face_1"]["loop_1"]["arc_1"]["x"], 1)

    def test_index_out_of_range(self):
        with self.assertRaises(IndexError):
            structure_preserving_perturbation(CAD, index=9999)


if __name__ == "__main__":
    unittest.main()
