"""Tests for HistCAD symmetric-difference flat sketch representation."""

import unittest

from harnesscad.domain.geometry.sketch import flat_sketch as fs
from harnesscad.domain.geometry.sketch.flat_sketch import SubPrimitive as SP


class TestFlattenEdgeKeys(unittest.TestCase):
    def test_shared_interior_edge_cancels(self):
        # Two faces sharing edge "m": face1={a,b,m}, face2={c,d,m}
        face1 = ["a", "b", "m"]
        face2 = ["c", "d", "m"]
        flat = fs.flatten_edge_keys([face1, face2])
        self.assertEqual(flat, ["a", "b", "c", "d"])
        self.assertNotIn("m", flat)

    def test_single_face_all_retained(self):
        flat = fs.flatten_edge_keys([["e1", "e2", "e3"]])
        self.assertEqual(flat, ["e1", "e2", "e3"])

    def test_odd_count_retained(self):
        # edge appears 3 times -> odd -> retained
        flat = fs.flatten_edge_keys([["x"], ["x"], ["x"]])
        self.assertEqual(flat, ["x"])

    def test_duplicate_within_face_ignored(self):
        flat = fs.flatten_edge_keys([["x", "x", "y"]])
        self.assertEqual(flat, ["x", "y"])


class TestFlattenFaces(unittest.TestCase):
    def test_two_squares_share_edge(self):
        # Square A: (0,0)-(1,0)-(1,1)-(0,1); Square B to its right shares edge x=1.
        shared = SP((1.0, 0.0), (1.0, 1.0))
        faceA = [
            SP((0.0, 0.0), (1.0, 0.0)),
            shared,
            SP((1.0, 1.0), (0.0, 1.0)),
            SP((0.0, 1.0), (0.0, 0.0)),
        ]
        faceB = [
            SP((1.0, 0.0), (2.0, 0.0)),
            SP((2.0, 0.0), (2.0, 1.0)),
            SP((2.0, 1.0), (1.0, 1.0)),
            SP((1.0, 1.0), (1.0, 0.0)),  # same geometry as shared, reversed
        ]
        flat = fs.flatten_faces([faceA, faceB])
        keys = {sp.canonical_key() for sp in flat}
        # The shared vertical edge (1,0)-(1,1) must cancel.
        self.assertNotIn(shared.canonical_key(), keys)
        self.assertEqual(len(flat), 6)  # outer rectangle has 6 edges

    def test_hole_boundary_retained(self):
        # A single face's edges are all retained (outer + hole appear once).
        face = [SP((0, 0), (1, 0)), SP((1, 0), (1, 1)), SP((1, 1), (0, 0))]
        flat = fs.flatten_faces([face])
        self.assertEqual(len(flat), 3)


class TestBoundaryValid(unittest.TestCase):
    def test_valid_when_max_two(self):
        m = SP((0, 0), (1, 0), key="m")
        faces = [[SP((0, 0), (0, 1), key="a"), m], [SP((1, 0), (1, 1), key="b"), m]]
        self.assertTrue(fs.is_boundary_valid(faces))

    def test_invalid_when_three(self):
        faces = [[SP((0, 0), (1, 0), key="m")],
                 [SP((0, 0), (1, 0), key="m")],
                 [SP((0, 0), (1, 0), key="m")]]
        self.assertFalse(fs.is_boundary_valid(faces))


class TestRecoverLoops(unittest.TestCase):
    def test_recover_single_triangle(self):
        prims = [SP((0, 0), (1, 0)), SP((1, 0), (0, 1)), SP((0, 1), (0, 0))]
        loops = fs.recover_loops(prims)
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(loops[0]), 3)

    def test_recover_two_separate_loops(self):
        tri = [SP((0, 0), (1, 0)), SP((1, 0), (0, 1)), SP((0, 1), (0, 0))]
        sq = [SP((5, 5), (6, 5)), SP((6, 5), (6, 6)), SP((6, 6), (5, 6)), SP((5, 6), (5, 5))]
        loops = fs.recover_loops(tri + sq)
        self.assertEqual(len(loops), 2)
        self.assertEqual(sorted(len(l) for l in loops), [3, 4])


class TestMergeCollinear(unittest.TestCase):
    def test_two_collinear_lines_merge(self):
        loop = [SP((0, 0), (1, 0)), SP((1, 0), (2, 0)), SP((2, 0), (0, 0))]
        merged = fs.merge_collinear(loop)
        # first two are collinear along y=0 -> merge into (0,0)-(2,0)
        self.assertEqual(len(merged), 2)

    def test_non_collinear_not_merged(self):
        loop = [SP((0, 0), (1, 0)), SP((1, 0), (1, 1))]
        merged = fs.merge_collinear(loop)
        self.assertEqual(len(merged), 2)

    def test_arc_not_merged(self):
        loop = [SP((0, 0), (1, 0), key=("arc", 1)), SP((1, 0), (2, 0), key=("arc", 2))]
        merged = fs.merge_collinear(loop)
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
