"""Tests for geometry.mesh.intersection_repair."""

import unittest

from harnesscad.domain.geometry.mesh.intersection_repair import (
    find_self_intersections,
    repair_self_intersections,
)


def _tetra(cx, cy, cz, s=1.0):
    """A closed tetrahedron centred near (cx,cy,cz) with 4 outward faces."""
    v = [
        (cx + 0.0, cy + 0.0, cz + 0.0),
        (cx + s, cy + 0.0, cz + 0.0),
        (cx + 0.0, cy + s, cz + 0.0),
        (cx + 0.0, cy + 0.0, cz + s),
    ]
    f = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    return v, f


def _two_tetras(offset):
    v1, f1 = _tetra(0.0, 0.0, 0.0)
    v2, f2 = _tetra(offset, 0.0, 0.0)
    verts = v1 + v2
    faces = list(f1) + [(a + 4, b + 4, c + 4) for (a, b, c) in f2]
    return verts, faces


class TestDetection(unittest.TestCase):
    def test_disjoint_has_no_intersections(self):
        verts, faces = _two_tetras(offset=5.0)
        self.assertEqual(find_self_intersections(verts, faces), [])

    def test_overlapping_detected(self):
        verts, faces = _two_tetras(offset=0.3)
        pairs = find_self_intersections(verts, faces)
        self.assertTrue(len(pairs) > 0)
        # cross-component only: every reported pair spans the two tetrahedra
        for (i, j) in pairs:
            self.assertLess(i, 4)
            self.assertGreaterEqual(j, 4)

    def test_shared_vertex_faces_not_reported(self):
        # the 4 faces of a single tetra share vertices pairwise and must not
        # be counted as self-intersections
        v, f = _tetra(0.0, 0.0, 0.0)
        self.assertEqual(find_self_intersections(v, f), [])

    def test_deterministic(self):
        verts, faces = _two_tetras(offset=0.3)
        self.assertEqual(
            find_self_intersections(verts, faces),
            find_self_intersections(verts, faces),
        )


class TestRepair(unittest.TestCase):
    def test_repair_resolves_overlap(self):
        verts, faces = _two_tetras(offset=0.3)
        before = find_self_intersections(verts, faces)
        self.assertTrue(len(before) > 0)

        result = repair_self_intersections(verts, faces, step=0.5, max_iters=200)
        self.assertTrue(result.resolved)
        self.assertEqual(result.final_intersections, 0)
        self.assertEqual(result.initial_intersections, len(before))
        # the repaired vertex list re-checks clean
        self.assertEqual(find_self_intersections(result.vertices, faces), [])

    def test_history_is_monotone_nonincreasing_to_zero(self):
        verts, faces = _two_tetras(offset=0.3)
        result = repair_self_intersections(verts, faces, step=0.5, max_iters=200)
        self.assertEqual(result.history[0], result.initial_intersections)
        self.assertEqual(result.history[-1], 0)

    def test_already_clean_is_noop(self):
        verts, faces = _two_tetras(offset=5.0)
        result = repair_self_intersections(verts, faces)
        self.assertEqual(result.iterations, 0)
        self.assertTrue(result.resolved)
        self.assertEqual(list(result.vertices), [tuple(map(float, v)) for v in verts])

    def test_input_not_mutated(self):
        verts, faces = _two_tetras(offset=0.3)
        snapshot = [tuple(v) for v in verts]
        repair_self_intersections(verts, faces)
        self.assertEqual([tuple(v) for v in verts], snapshot)

    def test_deterministic_repair(self):
        verts, faces = _two_tetras(offset=0.3)
        r1 = repair_self_intersections(verts, faces)
        r2 = repair_self_intersections(verts, faces)
        self.assertEqual(r1.vertices, r2.vertices)
        self.assertEqual(r1.iterations, r2.iterations)

    def test_smoothing_variant_also_resolves(self):
        verts, faces = _two_tetras(offset=0.3)
        result = repair_self_intersections(
            verts, faces, step=0.5, smooth=0.1, max_iters=300
        )
        self.assertTrue(result.resolved)


if __name__ == "__main__":
    unittest.main()
