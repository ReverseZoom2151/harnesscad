"""Tests for mesh label propagation and connected-component segmentation."""

import unittest

from geometry.jointsdf_mesh_segments import (
    majority_face_labels,
    connected_components,
    part_count,
    component_count,
)


class MajorityFaceTest(unittest.TestCase):
    def test_clear_majority(self):
        faces = [(0, 1, 2)]
        vlab = [5, 5, 8]
        self.assertEqual(majority_face_labels(faces, vlab), [5])

    def test_unanimous(self):
        faces = [(0, 1, 2)]
        self.assertEqual(majority_face_labels(faces, [3, 3, 3]), [3])

    def test_three_way_tie_smallest_repr(self):
        faces = [(0, 1, 2)]
        vlab = [3, 1, 2]
        # all distinct -> smallest repr ("1") chosen
        self.assertEqual(majority_face_labels(faces, vlab), [1])

    def test_non_triangle_raises(self):
        with self.assertRaises(ValueError):
            majority_face_labels([(0, 1)], [0, 1])


class ConnectedComponentTest(unittest.TestCase):
    def test_two_separate_squares(self):
        # square A: faces 0,1 share edge (1,2); square B: faces 2,3 share edge.
        faces = [
            (0, 1, 2), (0, 2, 3),   # part 0
            (4, 5, 6), (4, 6, 7),   # part 1, disjoint vertices
        ]
        labels = [0, 0, 1, 1]
        comp = connected_components(faces, labels)
        self.assertEqual(comp[0], comp[1])
        self.assertEqual(comp[2], comp[3])
        self.assertNotEqual(comp[0], comp[2])
        self.assertEqual(component_count(faces, labels), 2)

    def test_same_label_but_disconnected(self):
        # both parts labelled 0 but no shared edge -> two components
        faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 5, 6), (4, 6, 7),
        ]
        labels = [0, 0, 0, 0]
        self.assertEqual(part_count(labels), 1)
        self.assertEqual(component_count(faces, labels), 2)

    def test_edge_shared_but_different_label(self):
        # faces 0 and 1 share edge (1,2) but different labels -> not merged
        faces = [(0, 1, 2), (1, 2, 3)]
        labels = [0, 1]
        comp = connected_components(faces, labels)
        self.assertNotEqual(comp[0], comp[1])

    def test_deterministic_ids_ascending(self):
        faces = [(0, 1, 2), (3, 4, 5)]
        labels = [9, 9]
        comp = connected_components(faces, labels)
        self.assertEqual(comp[0], 0)
        self.assertEqual(comp[1], 1)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            connected_components([(0, 1, 2)], [0, 1])


if __name__ == "__main__":
    unittest.main()
