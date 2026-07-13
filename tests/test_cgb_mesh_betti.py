"""Tests for mesh-based Betti numbers and the CADGenBench topology-match score."""
import unittest

from harnesscad.eval.bench.geometry.cgb_mesh_betti import (
    BETTI_SHARPNESS,
    BettiResult,
    MeshGateError,
    MeshSurface,
    betti_axis_score,
    compute_betti,
    euler_characteristic,
    mesh_gate_errors,
    topo_match,
    topo_match_score,
    triangle_components,
)


def _box(x0, y0, z0, x1, y1, z1, base=0, invert=False):
    verts = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    tris = [
        (0, 3, 2), (0, 2, 1),      # bottom (-z)
        (4, 5, 6), (4, 6, 7),      # top (+z)
        (0, 1, 5), (0, 5, 4),      # front (-y)
        (1, 2, 6), (1, 6, 5),      # right (+x)
        (2, 3, 7), (2, 7, 6),      # back (+y)
        (3, 0, 4), (3, 4, 7),      # left (-x)
    ]
    if invert:
        tris = [(c, b, a) for a, b, c in tris]
    return verts, [(a + base, b + base, c + base) for a, b, c in tris]


def _cube():
    verts, tris = _box(0, 0, 0, 1, 1, 1)
    return MeshSurface(verts, tris)


def _two_cubes():
    v1, t1 = _box(0, 0, 0, 1, 1, 1)
    v2, t2 = _box(5, 0, 0, 6, 1, 1, base=len(v1))
    return MeshSurface(v1 + v2, t1 + t2)


def _hollow_cube():
    """Outer cube with an inner (inverted) cube: one void."""
    v1, t1 = _box(0, 0, 0, 4, 4, 4)
    v2, t2 = _box(1, 1, 1, 3, 3, 3, base=len(v1), invert=True)
    return MeshSurface(v1 + v2, t1 + t2)


def _plate_with_square_hole():
    """Annular prism: one through-hole, b = (1, 1, 0)."""
    outer = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)]
    inner = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]
    verts = (
        [(x, y, 0.0) for x, y in outer]        # 0-3   outer bottom
        + [(x, y, 1.0) for x, y in outer]      # 4-7   outer top
        + [(x, y, 0.0) for x, y in inner]      # 8-11  inner bottom
        + [(x, y, 1.0) for x, y in inner]      # 12-15 inner top
    )
    tris = []

    def quad(a, b, c, d):
        tris.append((a, b, c))
        tris.append((a, c, d))

    for i in range(4):
        j = (i + 1) % 4
        quad(i, j, 4 + j, 4 + i)               # outer walls
        quad(8 + i, 12 + i, 12 + j, 8 + j)     # hole walls
        quad(4 + i, 4 + j, 12 + j, 12 + i)     # top ring (+z)
        quad(i, 8 + i, 8 + j, j)               # bottom ring (-z)
    return MeshSurface(verts, tris)


class TestMeshGate(unittest.TestCase):
    def test_closed_cube_passes(self):
        self.assertEqual(mesh_gate_errors(_cube()), [])

    def test_open_mesh_is_not_closed(self):
        verts, tris = _box(0, 0, 0, 1, 1, 1)
        open_mesh = MeshSurface(verts, tris[:-2])
        errors = mesh_gate_errors(open_mesh)
        self.assertTrue(any("not closed" in e for e in errors))

    def test_flipped_triangle_breaks_orientation(self):
        verts, tris = _box(0, 0, 0, 1, 1, 1)
        a, b, c = tris[0]
        bad = MeshSurface(verts, [(c, b, a)] + list(tris[1:]))
        errors = mesh_gate_errors(bad)
        self.assertTrue(any("orientation inconsistent" in e for e in errors))

    def test_compute_betti_raises_on_bad_mesh(self):
        verts, tris = _box(0, 0, 0, 1, 1, 1)
        with self.assertRaises(MeshGateError):
            compute_betti(MeshSurface(verts, tris[:-2]))


class TestBetti(unittest.TestCase):
    def test_cube(self):
        result = compute_betti(_cube())
        self.assertEqual(result.as_vector(), (1, 0, 0))
        self.assertEqual(result.chi_surface, 2)
        self.assertEqual(result.n_components, 1)

    def test_euler_characteristic_of_cube(self):
        self.assertEqual(euler_characteristic(_cube()), 2)

    def test_two_cubes(self):
        result = compute_betti(_two_cubes())
        self.assertEqual(result.as_vector(), (2, 0, 0))
        self.assertEqual(len(triangle_components(_two_cubes())), 2)

    def test_hollow_cube_has_one_void(self):
        result = compute_betti(_hollow_cube())
        self.assertEqual(result.as_vector(), (1, 0, 1))
        self.assertEqual(result.chi_surface, 4)

    def test_plate_with_through_hole(self):
        result = compute_betti(_plate_with_square_hole())
        self.assertEqual(result.as_vector(), (1, 1, 0))
        self.assertEqual(result.chi_surface, 0)


class TestScore(unittest.TestCase):
    def test_perfect_match_is_one(self):
        gt = BettiResult(1, 2, 0, 0, 1, 0, 0)
        score, per_axis = topo_match_score(gt, gt)
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(set(per_axis), {"b0", "b1", "b2"})

    def test_doubled_hole_count_matches_doc(self):
        # GT (1, 2, 0) vs candidate (1, 4, 0): s1 = (3/5) ** 2 = 0.36.
        gt = BettiResult(1, 2, 0, 0, 1, 0, 0)
        cand = BettiResult(1, 4, 0, 0, 1, 0, 0)
        score, per_axis = topo_match_score(cand, gt)
        self.assertAlmostEqual(per_axis["b1"], 0.36, places=6)
        self.assertAlmostEqual(score, 0.36, places=6)

    def test_split_part_matches_doc(self):
        # GT (1, 0, 0) vs candidate (2, 0, 0): s0 = (2/3) ** 2 = 0.4444.
        gt = BettiResult(1, 0, 0, 2, 1, 0, 0)
        cand = BettiResult(2, 0, 0, 4, 2, 0, 0)
        score, per_axis = topo_match_score(cand, gt)
        self.assertAlmostEqual(per_axis["b0"], 4.0 / 9.0, places=6)
        self.assertAlmostEqual(score, 4.0 / 9.0, places=6)

    def test_axis_score_is_symmetric_and_min_over_max(self):
        self.assertAlmostEqual(betti_axis_score(4, 2), betti_axis_score(2, 4))
        self.assertAlmostEqual(
            betti_axis_score(4, 2), (3.0 / 5.0) ** BETTI_SHARPNESS, places=9
        )

    def test_negative_count_scores_zero(self):
        self.assertEqual(betti_axis_score(-1, 0), 0.0)

    def test_product_not_mean(self):
        # Two axes right, one badly wrong -> the aggregate collapses.
        gt = BettiResult(1, 0, 0, 2, 1, 0, 0)
        cand = BettiResult(1, 5, 0, 2, 1, 0, 0)
        score, _ = topo_match_score(cand, gt)
        self.assertLess(score, 0.2)

    def test_end_to_end_cube_vs_plate(self):
        result = topo_match(_plate_with_square_hole(), _cube())
        self.assertEqual(result.candidate.as_vector(), (1, 1, 0))
        self.assertEqual(result.gt.as_vector(), (1, 0, 0))
        self.assertAlmostEqual(result.score, 0.25, places=6)  # (1/2) ** 2
        self.assertIn("per_axis_scores", result.to_dict())


if __name__ == "__main__":
    unittest.main()
