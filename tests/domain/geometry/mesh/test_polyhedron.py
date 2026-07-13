"""Tests for geometry.angelcad_polyhedron."""

import math
import unittest

from harnesscad.domain.geometry.mesh.polyhedron import (
    Polyhedron,
    PolyhedronError,
    tetrahedron,
    unit_cube,
    verify,
)


class TestFaceGeometry(unittest.TestCase):
    def test_face_area_triangle_and_quad(self):
        tet = tetrahedron(2.0)
        # face (0,2,1) lies in z=0 with legs 2 and 2 -> area 2
        self.assertAlmostEqual(tet.face_area(0), 2.0)
        cube = unit_cube(3.0)
        self.assertAlmostEqual(cube.face_area(0), 9.0)
        self.assertAlmostEqual(cube.surface_area(), 6 * 9.0)

    def test_outward_unit_normals(self):
        cube = unit_cube(2.0)
        self.assertEqual(cube.face_unit_normal(0), (0.0, 0.0, -1.0))
        self.assertEqual(cube.face_unit_normal(1), (0.0, 0.0, 1.0))
        self.assertEqual(cube.face_unit_normal(3), (1.0, 0.0, 0.0))

    def test_face_centroid_and_planarity(self):
        cube = unit_cube(2.0)
        self.assertEqual(cube.face_centroid(1), (1.0, 1.0, 2.0))
        self.assertAlmostEqual(cube.face_planarity(1), 0.0)

    def test_nonplanar_quad_detected(self):
        p = Polyhedron(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0.5), (0, 1, 0)],
            [(0, 1, 2, 3)],
        )
        self.assertGreater(p.face_planarity(0), 0.05)

    def test_fan_triangulation(self):
        cube = unit_cube()
        tris = cube.triangles()
        self.assertEqual(len(tris), 12)
        self.assertEqual(tris[0], (0, 3, 2))


class TestMassProperties(unittest.TestCase):
    def test_cube_volume_and_centroid(self):
        cube = unit_cube(2.0)
        self.assertAlmostEqual(cube.volume(), 8.0)
        c = cube.centroid()
        for v in c:
            self.assertAlmostEqual(v, 1.0)

    def test_tetra_volume(self):
        # corner tetra of leg s has volume s^3/6
        self.assertAlmostEqual(tetrahedron(3.0).volume(), 27.0 / 6.0)

    def test_flipped_faces_give_negative_volume(self):
        cube = unit_cube(2.0)
        cube.flip_faces()
        self.assertAlmostEqual(cube.volume(), -8.0)
        self.assertAlmostEqual(cube.oriented_outward().volume(), 8.0)

    def test_flip_single_face(self):
        cube = unit_cube()
        before = cube.faces[0]
        cube.flip_face(0)
        self.assertEqual(cube.faces[0], tuple(reversed(before)))

    def test_cube_inertia_matches_closed_form(self):
        s = 2.0
        cube = unit_cube(s)
        t = cube.inertia_tensor()
        expected = (s ** 5) / 6.0  # m*(a^2+b^2)/12 with m=s^3
        for i in range(3):
            self.assertAlmostEqual(t[i][i], expected, places=9)
            for j in range(3):
                if i != j:
                    self.assertAlmostEqual(t[i][j], 0.0, places=9)

    def test_inertia_is_translation_invariant_about_centroid(self):
        a = unit_cube(2.0)
        b = Polyhedron([(x + 7, y - 3, z + 11) for (x, y, z) in a.vertices], a.faces)
        ta, tb = a.inertia_tensor(), b.inertia_tensor()
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(ta[i][j], tb[i][j], places=6)

    def test_inertia_about_origin_differs(self):
        cube = unit_cube(2.0)
        t0 = cube.inertia_tensor(about_centroid=False)
        tc = cube.inertia_tensor(about_centroid=True)
        # parallel axis: I0_xx = Ic_xx + m*(cy^2+cz^2) = Ic + 8*2
        self.assertAlmostEqual(t0[0][0] - tc[0][0], 8.0 * 2.0, places=9)

    def test_bounds(self):
        lo, hi = unit_cube(4.0).bounds()
        self.assertEqual(lo, (0.0, 0.0, 0.0))
        self.assertEqual(hi, (4.0, 4.0, 4.0))


class TestVerify(unittest.TestCase):
    def test_valid_solids(self):
        self.assertEqual(verify(unit_cube()), [])
        self.assertEqual(verify(tetrahedron()), [])
        unit_cube().check()

    def test_inward_orientation(self):
        cube = unit_cube()
        cube.flip_faces()
        codes = [i.code for i in verify(cube)]
        self.assertEqual(codes, ["orientation-inward"])

    def test_one_face_flipped_breaks_orientation(self):
        cube = unit_cube()
        cube.flip_face(0)
        codes = {i.code for i in verify(cube)}
        self.assertIn("orientation", codes)

    def test_open_surface(self):
        cube = unit_cube()
        del cube.faces[0]
        codes = {i.code for i in verify(cube)}
        self.assertIn("edge-boundary", codes)

    def test_index_out_of_range(self):
        p = Polyhedron([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 5)])
        codes = [i.code for i in verify(p)]
        self.assertEqual(codes, ["index-range"])

    def test_repeated_index(self):
        p = Polyhedron([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 1)])
        self.assertEqual([i.code for i in verify(p)], ["face-degenerate"])

    def test_short_face(self):
        p = Polyhedron([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1)])
        self.assertEqual([i.code for i in verify(p)], ["face-degenerate"])

    def test_zero_area_face(self):
        p = Polyhedron(
            [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0)],
            [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)],
        )
        codes = {i.code for i in verify(p)}
        self.assertIn("face-degenerate", codes)

    def test_unused_vertex(self):
        cube = unit_cube()
        cube.vertices.append((9.0, 9.0, 9.0))
        codes = [i.code for i in verify(cube)]
        self.assertEqual(codes, ["vertex-unused"])

    def test_nonmanifold_edge(self):
        # two tetrahedra glued along a shared face edge, sharing one triangle 3 times
        p = Polyhedron(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)],
            [(0, 1, 2), (0, 1, 3), (0, 1, 4), (1, 2, 3)],
        )
        codes = {i.code for i in verify(p)}
        self.assertIn("edge-nonmanifold", codes)

    def test_empty(self):
        self.assertEqual([i.code for i in verify(Polyhedron([], []))], ["empty", "empty"])

    def test_check_raises(self):
        cube = unit_cube()
        cube.flip_faces()
        with self.assertRaises(PolyhedronError) as ctx:
            cube.check()
        self.assertEqual(len(ctx.exception.issues), 1)

    def test_issues_are_deterministic(self):
        cube = unit_cube()
        cube.flip_face(2)
        self.assertEqual([i.key() for i in verify(cube)], [i.key() for i in verify(cube)])

    def test_nonplanar_face_reported(self):
        p = Polyhedron(
            [
                (0, 0, 0),
                (2, 0, 0),
                (2, 2, 0.4),
                (0, 2, 0),
                (0, 0, 2),
                (2, 0, 2),
                (2, 2, 2),
                (0, 2, 2),
            ],
            [
                (0, 3, 2, 1),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
            ],
        )
        codes = {i.code for i in verify(p, planarity_tol=1e-6)}
        self.assertIn("face-nonplanar", codes)


if __name__ == "__main__":
    unittest.main()
