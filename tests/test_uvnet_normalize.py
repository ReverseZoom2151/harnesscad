"""Tests for UV-Net grid normalisation (mask-aware box, unit-box scale, quarter turns)."""

import math
import unittest

from harnesscad.domain.geometry import uvnet_normalize as nz
from harnesscad.domain.geometry import uvnet_u_grid as ug
from harnesscad.domain.geometry import uvnet_uv_grid as uvg


def _plane_grid(origin=(0.0, 0.0, 0.0), u=(0.0, 4.0), v=(0.0, 2.0), loops=None):
    plane = uvg.Plane(origin=origin, axis=(0.0, 0.0, 1.0),
                      ref_dir=(1.0, 0.0, 0.0), u_range=u, v_range=v)
    return uvg.face_feature_grid(plane, 5, 5, trim_loops=loops)


class BoundingBoxTest(unittest.TestCase):
    def test_box_of_untrimmed_plane(self):
        box = nz.bounding_box_uvgrid(_plane_grid())
        self.assertEqual(box, ((0.0, 0.0, 0.0), (4.0, 2.0, 0.0)))
        self.assertEqual(nz.box_center(box), (2.0, 1.0, 0.0))
        self.assertEqual(nz.box_diagonal(box), (4.0, 2.0, 0.0))

    def test_mask_excludes_trimmed_nodes(self):
        # Trim to the left half of the [0,4]x[0,2] domain: box must shrink in u.
        loops = [[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]]
        grid = _plane_grid(loops=loops)
        masked = nz.bounding_box_uvgrid(grid, masked=True)
        unmasked = nz.bounding_box_uvgrid(grid, masked=False)
        self.assertAlmostEqual(masked[1][0], 2.0, places=12)
        self.assertAlmostEqual(unmasked[1][0], 4.0, places=12)

    def test_empty_inputs_raise(self):
        with self.assertRaises(ValueError):
            nz.bounding_box([])


class CenterAndScaleTest(unittest.TestCase):
    def test_unit_box_scaling(self):
        grid, center, scale = nz.center_and_scale_uvgrid(_plane_grid())
        self.assertEqual(center, (2.0, 1.0, 0.0))
        self.assertAlmostEqual(scale, 2.0 / 4.0, places=12)
        box = nz.bounding_box_uvgrid(grid)
        self.assertAlmostEqual(box[0][0], -1.0, places=12)
        self.assertAlmostEqual(box[1][0], 1.0, places=12)
        self.assertAlmostEqual(box[1][1], 0.5, places=12)   # aspect preserved
        self.assertTrue(nz.grid_bounds_check([grid]))

    def test_normals_are_untouched_by_scaling(self):
        grid = _plane_grid()
        scaled, _, _ = nz.center_and_scale_uvgrid(grid)
        for row_a, row_b in zip(grid, scaled):
            for a, b in zip(row_a, row_b):
                self.assertEqual(a[3:], b[3:])

    def test_solid_uses_one_transform_for_faces_and_edges(self):
        faces = [_plane_grid(), _plane_grid(origin=(0.0, 0.0, 4.0))]
        edges = [ug.edge_feature_grid(
            ug.Line((0.0, 0.0, 0.0), (0.0, 0.0, 4.0)), 5)]
        nf, ne, center, scale = nz.center_and_scale_solid(faces, edges)
        self.assertEqual(center, (2.0, 1.0, 2.0))
        self.assertAlmostEqual(scale, 0.5, places=12)
        self.assertTrue(nz.grid_bounds_check(nf))
        self.assertTrue(nz.grid_bounds_check(ne))
        # the edge endpoints land on the box's z extremes
        self.assertAlmostEqual(ne[0][0][2], -1.0, places=12)
        self.assertAlmostEqual(ne[0][-1][2], 1.0, places=12)
        # edge tangents survive intact
        self.assertAlmostEqual(ne[0][0][5], 1.0, places=12)

    def test_invert_round_trip(self):
        grid, center, scale = nz.center_and_scale_uvgrid(_plane_grid())
        p = (grid[0][0][0], grid[0][0][1], grid[0][0][2])
        back = nz.invert_point(p, center, scale)
        for a, b in zip(back, (0.0, 0.0, 0.0)):
            self.assertAlmostEqual(a, b, places=10)

    def test_degenerate_box_raises(self):
        line = ug.edge_feature_grid(ug.Line((1, 1, 1), (1, 0, 0)), 3)
        flat = [(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)] * 3
        with self.assertRaises(ValueError):
            nz.center_scale_from_box(nz.bounding_box_uvgrid(flat))
        self.assertIsNotNone(nz.bounding_box_uvgrid(line))

    def test_bounds_check_rejects_unnormalised(self):
        self.assertFalse(nz.grid_bounds_check([_plane_grid()]))


class RotationTest(unittest.TestCase):
    def test_identity(self):
        m = nz.rotation_matrix(2, 0)
        self.assertEqual(m, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))

    def test_quarter_turn_about_z(self):
        m = nz.rotation_matrix(2, 1)
        self.assertEqual(nz.rotate_vector((1, 0, 0), m), (0, 1, 0))
        self.assertEqual(nz.rotate_vector((0, 1, 0), m), (-1, 0, 0))
        self.assertEqual(nz.rotate_vector((0, 0, 1), m), (0, 0, 1))

    def test_all_matrices_orthonormal_and_integer(self):
        for axis, k in nz.quarter_turns():
            m = nz.rotation_matrix(axis, k)
            self.assertTrue(nz.matrix_is_orthonormal(m))
            for row in m:
                for c in row:
                    self.assertIn(c, (-1, 0, 1))
        self.assertEqual(len(nz.quarter_turns()), 12)

    def test_four_turns_is_identity(self):
        grid = _plane_grid()
        m = nz.rotation_matrix(1, 1)
        out = grid
        for _ in range(4):
            out = nz.rotate_grid(out, m)
        for row_a, row_b in zip(grid, out):
            for a, b in zip(row_a, row_b):
                for ca, cb in zip(a, b):
                    self.assertAlmostEqual(ca, cb, places=12)

    def test_rotation_moves_points_and_normals(self):
        grid = _plane_grid()                       # normals along +z
        m = nz.rotation_matrix(0, 1)               # 90 deg about x: z -> -y... check
        out = nz.rotate_grid(grid, m)
        for row in out:
            for c in row:
                n = (c[3], c[4], c[5])
                self.assertAlmostEqual(nz.vector_length(n), 1.0, places=12)
                self.assertAlmostEqual(n[1], -1.0, places=12)
                self.assertAlmostEqual(c[6], 1.0, places=12)   # mask preserved

    def test_rotation_preserves_edge_tangent_length(self):
        edge = ug.edge_feature_grid(
            ug.Circle((0, 0, 0), (0, 0, 1), 2.0), 8)
        out = nz.rotate_grid(edge, nz.rotation_matrix(1, 3))
        self.assertEqual(len(out), 8)
        for c in out:
            self.assertAlmostEqual(nz.vector_length((c[3], c[4], c[5])), 1.0,
                                   places=12)
            self.assertAlmostEqual(
                math.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2), 2.0, places=10)

    def test_bad_axis(self):
        with self.assertRaises(ValueError):
            nz.rotation_matrix(3, 1)

    def test_rotation_commutes_with_scaling_determinism(self):
        grid = _plane_grid()
        m = nz.rotation_matrix(2, 2)
        a, _, _ = nz.center_and_scale_uvgrid(nz.rotate_grid(grid, m))
        b, _, _ = nz.center_and_scale_uvgrid(nz.rotate_grid(grid, m))
        self.assertEqual(a, b)
        self.assertTrue(nz.grid_bounds_check([a]))


if __name__ == "__main__":
    unittest.main()
