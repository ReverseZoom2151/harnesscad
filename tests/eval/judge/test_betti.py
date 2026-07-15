import unittest

from harnesscad.eval.judge.betti import (
    betti_from_mesh,
    euler_characteristic,
    surface_components,
)

# A tetrahedron: 4 vertices, 4 faces, 6 edges, chi = 2 (a sphere topologically).
TETRA = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def _shifted(faces, offset):
    return [(a + offset, b + offset, c + offset) for a, b, c in faces]


def _torus_grid(m=3, n=3):
    """Triangulated m*n torus grid (chi = 0, genus 1)."""
    def v(i, j):
        return (i % m) * n + (j % n)

    faces = []
    for i in range(m):
        for j in range(n):
            a, b = v(i, j), v(i, j + 1)
            c, d = v(i + 1, j), v(i + 1, j + 1)
            faces.append((a, b, d))
            faces.append((a, d, c))
    return faces


class BettiTests(unittest.TestCase):
    def test_euler_of_tetrahedron(self):
        self.assertEqual(euler_characteristic(TETRA), 2)

    def test_surface_components_single(self):
        self.assertEqual(surface_components(TETRA), 1)

    def test_betti_of_tetrahedron(self):
        self.assertEqual(betti_from_mesh(TETRA), (1, 0, 0))

    def test_two_disjoint_solids(self):
        faces = TETRA + _shifted(TETRA, 4)
        self.assertEqual(surface_components(faces), 2)
        self.assertEqual(euler_characteristic(faces), 4)
        self.assertEqual(betti_from_mesh(faces), (2, 0, 0))

    def test_hollow_ball_one_void(self):
        # Two nested shells: outer + inner. shells=2, n_voids=1 -> (1,0,1).
        faces = TETRA + _shifted(TETRA, 4)
        self.assertEqual(betti_from_mesh(faces, n_voids=1), (1, 0, 1))

    def test_torus_has_one_handle(self):
        faces = _torus_grid(3, 3)
        self.assertEqual(euler_characteristic(faces), 0)
        self.assertEqual(surface_components(faces), 1)
        self.assertEqual(betti_from_mesh(faces), (1, 1, 0))

    def test_empty_mesh(self):
        self.assertEqual(euler_characteristic([]), 0)
        self.assertEqual(surface_components([]), 0)

    def test_non_triangle_rejected(self):
        with self.assertRaises(ValueError):
            euler_characteristic([(0, 1, 2, 3)])

    def test_n_voids_out_of_range(self):
        with self.assertRaises(ValueError):
            betti_from_mesh(TETRA, n_voids=2)

    def test_determinism(self):
        self.assertEqual(betti_from_mesh(TETRA), betti_from_mesh(TETRA))


if __name__ == "__main__":
    unittest.main()
