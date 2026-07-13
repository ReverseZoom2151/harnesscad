"""Tests for CraftsMan relative Laplacian smoothing operators."""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.mesh.craftsman_relative_laplacian import (
    laplacian_smooth,
    mean_displacement,
    relative_laplacian_smooth,
    taubin_smooth,
    uniform_laplacian,
    vertex_adjacency,
)


def _grid(nx, ny, z=0.0):
    """A flat (nx+1)x(ny+1) triangulated grid on the z=const plane."""
    verts = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            verts.append((float(i), float(j), z))
    faces = []
    def idx(i, j):
        return j * (nx + 1) + i
    for j in range(ny):
        for i in range(nx):
            faces.append((idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)))
            faces.append((idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)))
    return verts, faces


class VertexAdjacencyTest(unittest.TestCase):
    def test_single_triangle_full_adjacency(self):
        adj = vertex_adjacency(3, [(0, 1, 2)])
        self.assertEqual(adj, [[1, 2], [0, 2], [0, 1]])

    def test_quad_face_uses_ring_edges_not_diagonal(self):
        adj = vertex_adjacency(4, [(0, 1, 2, 3)])
        # ring 0-1-2-3-0; no 0-2 or 1-3 diagonal
        self.assertEqual(adj, [[1, 3], [0, 2], [1, 3], [0, 2]])

    def test_isolated_vertex_has_no_neighbours(self):
        adj = vertex_adjacency(4, [(0, 1, 2)])
        self.assertEqual(adj[3], [])

    def test_sorted_and_deduplicated_across_shared_edges(self):
        adj = vertex_adjacency(4, [(0, 1, 2), (0, 2, 3)])
        self.assertEqual(adj[0], [1, 2, 3])
        self.assertEqual(adj[2], [0, 1, 3])

    def test_out_of_range_face_raises(self):
        with self.assertRaises(IndexError):
            vertex_adjacency(3, [(0, 1, 5)])

    def test_negative_count_raises(self):
        with self.assertRaises(ValueError):
            vertex_adjacency(-1, [])


class UniformLaplacianTest(unittest.TestCase):
    def test_center_of_regular_neighbourhood_is_zero(self):
        # central vertex at origin with 4 symmetric neighbours -> zero delta
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
               (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)]
        adj = [[1, 2, 3, 4], [0], [0], [0], [0]]
        deltas = uniform_laplacian(pts, adj)
        for component in deltas[0]:
            self.assertAlmostEqual(component, 0.0)

    def test_delta_points_toward_neighbour_mean(self):
        pts = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
        adj = [[1], [0, 2], [1]]
        deltas = uniform_laplacian(pts, adj)
        # vertex 0 neighbour mean is (2,0,0) -> delta (2,0,0)
        self.assertAlmostEqual(deltas[0][0], 2.0)
        # vertex 1 is already the mean of 0 and 2 -> zero
        self.assertAlmostEqual(deltas[1][0], 0.0)

    def test_isolated_vertex_zero(self):
        deltas = uniform_laplacian([(3.0, 3.0, 3.0)], [[]])
        self.assertEqual(deltas[0], (0.0, 0.0, 0.0))

    def test_mismatched_adjacency_raises(self):
        with self.assertRaises(ValueError):
            uniform_laplacian([(0.0, 0.0, 0.0)], [[], []])


class LaplacianSmoothTest(unittest.TestCase):
    def test_smoothing_pulls_spike_down(self):
        verts, faces = _grid(2, 2)
        # raise the central vertex (index 4) into a spike
        verts[4] = (1.0, 1.0, 1.0)
        out = laplacian_smooth(verts, faces, iterations=1, lam=0.5)
        # spike height should decrease toward the flat neighbourhood (z=0)
        self.assertLess(out[4][2], 1.0)
        self.assertGreaterEqual(out[4][2], 0.0)

    def test_flat_mesh_is_fixed_point(self):
        verts, faces = _grid(3, 3)
        out = laplacian_smooth(verts, faces, iterations=5, lam=0.5)
        for original, smoothed in zip(verts, out):
            self.assertAlmostEqual(original[2], smoothed[2])

    def test_zero_iterations_identity(self):
        verts, faces = _grid(1, 1)
        out = laplacian_smooth(verts, faces, iterations=0, lam=0.5)
        self.assertEqual([tuple(v) for v in verts], out)

    def test_invalid_lam_raises(self):
        verts, faces = _grid(1, 1)
        with self.assertRaises(ValueError):
            laplacian_smooth(verts, faces, lam=1.5)

    def test_determinism(self):
        verts, faces = _grid(3, 3)
        verts[5] = (1.0, 1.0, 0.7)
        a = laplacian_smooth(verts, faces, iterations=3, lam=0.4)
        b = laplacian_smooth(verts, faces, iterations=3, lam=0.4)
        self.assertEqual(a, b)


class RelativeLaplacianSmoothTest(unittest.TestCase):
    def test_no_displacement_is_fixed_point(self):
        # when vertices equal the anchor, displacement is zero everywhere
        verts, faces = _grid(3, 3)
        out = relative_laplacian_smooth(verts, verts, faces,
                                        iterations=5, lam=0.5)
        for original, smoothed in zip(verts, out):
            for c in range(3):
                self.assertAlmostEqual(original[c], smoothed[c])

    def test_relative_resists_thin_feature_collapse(self):
        # Build a mesh whose true shape is a raised ridge along the middle row.
        verts, faces = _grid(4, 2)
        anchor = [tuple(v) for v in verts]
        # raise the whole middle row (j == 1) to z = 1 -> a thin ridge feature
        ridge = []
        for k, (x, y, z) in enumerate(verts):
            if abs(y - 1.0) < 1e-9:
                ridge.append((x, y, 1.0))
            else:
                ridge.append((x, y, 0.0))
        anchor = ridge
        current = [tuple(v) for v in ridge]

        std = laplacian_smooth(current, faces, iterations=8, lam=0.5)
        rel = relative_laplacian_smooth(current, anchor, faces,
                                        iterations=8, lam=0.5)

        # measure how much the ridge (middle row) sagged from its anchor height
        def ridge_sag(result):
            total = 0.0
            n = 0
            for (x, y, _), (_, _, rz) in zip(anchor, result):
                if abs(y - 1.0) < 1e-9:
                    total += (1.0 - rz)
                    n += 1
            return total / n

        # standard smoothing collapses the ridge toward the flat neighbours;
        # relative smoothing keeps it near the anchor -> far less sag.
        self.assertGreater(ridge_sag(std), ridge_sag(rel) + 0.1)
        self.assertLess(mean_displacement(rel, anchor),
                        mean_displacement(std, anchor))

    def test_per_vertex_speed_scales_update(self):
        verts, faces = _grid(2, 2)
        anchor = [tuple(v) for v in verts]
        verts[4] = (1.0, 1.0, 1.0)
        speeds = [0.0] * len(verts)
        out = relative_laplacian_smooth(verts, anchor, faces,
                                        iterations=1, lam=0.5, speed=speeds)
        # zero speed everywhere pins result to the anchor
        for anchor_pt, result in zip(anchor, out):
            for c in range(3):
                self.assertAlmostEqual(anchor_pt[c], result[c])

    def test_length_mismatch_raises(self):
        verts, faces = _grid(1, 1)
        with self.assertRaises(ValueError):
            relative_laplacian_smooth(verts, verts[:-1], faces)

    def test_bad_speed_length_raises(self):
        verts, faces = _grid(1, 1)
        with self.assertRaises(ValueError):
            relative_laplacian_smooth(verts, verts, faces, speed=[1.0, 2.0])

    def test_determinism(self):
        verts, faces = _grid(3, 3)
        anchor = [tuple(v) for v in verts]
        verts[5] = (1.0, 1.0, 0.8)
        a = relative_laplacian_smooth(verts, anchor, faces, iterations=4, lam=0.3)
        b = relative_laplacian_smooth(verts, anchor, faces, iterations=4, lam=0.3)
        self.assertEqual(a, b)


class TaubinSmoothTest(unittest.TestCase):
    def test_removes_noise_from_spike(self):
        verts, faces = _grid(3, 3)
        verts[5] = (verts[5][0], verts[5][1], 1.0)
        out = taubin_smooth(verts, faces, iterations=3, lam=0.5, mu=-0.53)
        self.assertLess(out[5][2], 1.0)

    def test_less_shrinkage_than_laplacian(self):
        # Taubin's inflating pass should preserve scale better than plain
        # Laplacian on a closed-ish bumpy grid. Compare bounding-box z-extent.
        verts, faces = _grid(4, 4)
        for k in range(len(verts)):
            x, y, _ = verts[k]
            verts[k] = (x, y, 0.5 * math.sin(x) * math.cos(y))
        lap = laplacian_smooth(verts, faces, iterations=6, lam=0.5)
        tau = taubin_smooth(verts, faces, iterations=3, lam=0.5, mu=-0.53)

        def z_extent(mesh):
            zs = [p[2] for p in mesh]
            return max(zs) - min(zs)

        self.assertGreater(z_extent(tau), z_extent(lap))

    def test_invalid_mu_raises(self):
        verts, faces = _grid(1, 1)
        with self.assertRaises(ValueError):
            taubin_smooth(verts, faces, lam=0.5, mu=-0.4)

    def test_invalid_lam_raises(self):
        verts, faces = _grid(1, 1)
        with self.assertRaises(ValueError):
            taubin_smooth(verts, faces, lam=0.0, mu=-0.5)

    def test_determinism(self):
        verts, faces = _grid(3, 3)
        verts[4] = (1.0, 1.0, 0.9)
        a = taubin_smooth(verts, faces, iterations=2)
        b = taubin_smooth(verts, faces, iterations=2)
        self.assertEqual(a, b)


class MeanDisplacementTest(unittest.TestCase):
    def test_zero_for_identical(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]
        self.assertEqual(mean_displacement(pts, pts), 0.0)

    def test_known_distance(self):
        a = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        b = [(3.0, 4.0, 0.0), (0.0, 0.0, 0.0)]
        self.assertAlmostEqual(mean_displacement(a, b), 2.5)

    def test_empty_is_zero(self):
        self.assertEqual(mean_displacement([], []), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mean_displacement([(0.0, 0.0, 0.0)], [])


if __name__ == "__main__":
    unittest.main()
