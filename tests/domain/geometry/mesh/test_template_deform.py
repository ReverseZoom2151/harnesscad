"""Tests for geometry.magic3d_template_deform (Magic3DSketch sphere backbone)."""

import math
import unittest

from harnesscad.domain.geometry.mesh.template_deform import (
    icosphere,
    apply_offsets,
    vertex_normals,
    apply_normal_displacement,
    flatten_loss,
)


def _norm(v):
    return math.sqrt(sum(c * c for c in v))


class IcosphereTest(unittest.TestCase):
    def test_base_counts(self):
        v, f = icosphere(0)
        self.assertEqual(len(v), 12)
        self.assertEqual(len(f), 20)

    def test_subdivision_counts(self):
        v, f = icosphere(1)
        # 20*4 faces; Euler: V - E + F = 2, F=80 -> E=120 -> V=42
        self.assertEqual(len(f), 80)
        self.assertEqual(len(v), 42)

    def test_all_on_unit_sphere(self):
        v, _ = icosphere(2)
        for p in v:
            self.assertAlmostEqual(_norm(p), 1.0, places=9)

    def test_deterministic(self):
        self.assertEqual(icosphere(1), icosphere(1))

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            icosphere(-1)

    def test_faces_index_valid(self):
        v, f = icosphere(1)
        for face in f:
            for idx in face:
                self.assertTrue(0 <= idx < len(v))


class OffsetTest(unittest.TestCase):
    def test_apply_offsets(self):
        verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        offs = [(0.0, 1.0, 0.0), (0.0, 0.0, 2.0)]
        self.assertEqual(
            apply_offsets(verts, offs), [(0.0, 1.0, 0.0), (1.0, 0.0, 2.0)]
        )

    def test_offset_length_mismatch(self):
        with self.assertRaises(ValueError):
            apply_offsets([(0, 0, 0)], [(0, 0, 0), (1, 1, 1)])

    def test_zero_offsets_identity(self):
        v, _ = icosphere(0)
        z = [(0.0, 0.0, 0.0)] * len(v)
        self.assertEqual(apply_offsets(v, z), list(v))


class NormalTest(unittest.TestCase):
    def test_sphere_normals_point_outward(self):
        v, f = icosphere(1)
        normals = vertex_normals(v, f)
        # On a unit sphere the outward normal at p is ~ p itself.
        for p, n in zip(v, normals):
            self.assertGreater(sum(a * b for a, b in zip(p, n)), 0.9)

    def test_normals_unit_length(self):
        v, f = icosphere(0)
        for n in vertex_normals(v, f):
            self.assertAlmostEqual(_norm(n), 1.0, places=9)

    def test_normal_displacement_inflates_sphere(self):
        v, f = icosphere(1)
        disp = [0.5] * len(v)
        out = apply_normal_displacement(v, f, disp)
        for p in out:
            self.assertAlmostEqual(_norm(p), 1.5, places=6)

    def test_displacement_length_mismatch(self):
        v, f = icosphere(0)
        with self.assertRaises(ValueError):
            apply_normal_displacement(v, f, [0.1])


class FlattenLossTest(unittest.TestCase):
    def test_coplanar_is_zero(self):
        # Two triangles forming a flat square in the z=0 plane.
        verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        faces = [(0, 1, 2), (0, 2, 3)]
        self.assertAlmostEqual(flatten_loss(verts, faces), 0.0)

    def test_folded_edge_positive(self):
        # Two triangles sharing edge (0,1), folded at 90 degrees.
        verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        faces = [(0, 1, 2), (1, 0, 3)]
        loss = flatten_loss(verts, faces)
        self.assertGreater(loss, 0.0)

    def test_sphere_positive_small(self):
        v, f = icosphere(2)
        loss = flatten_loss(v, f)
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 0.2)  # smooth sphere -> small dihedral folds

    def test_no_interior_edges_zero(self):
        # single triangle: all edges are boundary
        self.assertAlmostEqual(
            flatten_loss([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)]), 0.0
        )

    def test_subdivision_reduces_flatten_loss(self):
        # Finer tessellation of the sphere has flatter local neighbourhoods.
        v0, f0 = icosphere(1)
        v1, f1 = icosphere(3)
        self.assertLess(flatten_loss(v1, f1), flatten_loss(v0, f0))


if __name__ == "__main__":
    unittest.main()
