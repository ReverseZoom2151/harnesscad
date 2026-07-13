"""Tests for cubical-complex Betti numbers of voxel shapes."""
import unittest

from harnesscad.domain.geometry.volumes.topodiff_betti_voxel import (
    BettiNumbers,
    betti_numbers,
    cavity_count,
    connected_components,
    cubical_cell_counts,
    cubical_euler_characteristic,
    genus,
    voxels_from_grid,
)


def _solid_box(nx, ny, nz):
    return {(x, y, z) for x in range(nx) for y in range(ny) for z in range(nz)}


def _square_ring():
    """A 3x3 plane of voxels (thickness 1) with the centre removed -> a loop."""
    ring = set()
    for x in range(3):
        for y in range(3):
            if (x, y) != (1, 1):
                ring.add((x, y, 0))
    return ring


def _hollow_shell():
    """3x3x3 solid cube with the single centre voxel removed -> one cavity."""
    cube = _solid_box(3, 3, 3)
    cube.discard((1, 1, 1))
    return cube


class TestCellCounts(unittest.TestCase):
    def test_single_voxel_cells(self):
        c0, c1, c2, c3 = cubical_cell_counts({(0, 0, 0)})
        self.assertEqual((c0, c1, c2, c3), (8, 12, 6, 1))

    def test_single_voxel_euler_is_one(self):
        self.assertEqual(cubical_euler_characteristic({(0, 0, 0)}), 1)

    def test_shared_faces_counted_once(self):
        # Two adjacent cubes share a 2-face (4 verts, 4 edges, 1 face).
        c0, c1, c2, c3 = cubical_cell_counts({(0, 0, 0), (1, 0, 0)})
        self.assertEqual(c0, 12)   # 8 + 8 - 4 shared
        self.assertEqual(c3, 2)
        # Union of two contractible cubes is still contractible -> chi = 1.
        self.assertEqual(c0 - c1 + c2 - c3, 1)


class TestBetti(unittest.TestCase):
    def test_empty(self):
        b = betti_numbers(set())
        self.assertEqual(b.vector, (0, 0, 0))
        self.assertEqual(b.euler, 0)

    def test_single_voxel_is_a_ball(self):
        b = betti_numbers({(0, 0, 0)})
        self.assertEqual(b.vector, (1, 0, 0))
        self.assertEqual(b.genus, 0)

    def test_solid_ball_genus_zero(self):
        b = betti_numbers(_solid_box(3, 3, 3))
        self.assertEqual(b.vector, (1, 0, 0))
        self.assertEqual(b.euler, 1)
        self.assertEqual(b.genus, 0)

    def test_two_components(self):
        shape = {(0, 0, 0), (5, 5, 5)}
        b = betti_numbers(shape)
        self.assertEqual(b.b0, 2)
        self.assertEqual(connected_components(shape), 2)
        self.assertEqual(b.euler, 2)

    def test_torus_ring_genus_one(self):
        b = betti_numbers(_square_ring())
        self.assertEqual(b.b0, 1)
        self.assertEqual(b.b1, 1)   # one loop
        self.assertEqual(b.b2, 0)
        self.assertEqual(b.genus, 1)
        self.assertEqual(b.euler, 0)

    def test_hollow_shell_has_one_cavity(self):
        shell = _hollow_shell()
        self.assertEqual(cavity_count(shell), 1)
        b = betti_numbers(shell)
        self.assertEqual(b.b0, 1)
        self.assertEqual(b.b1, 0)
        self.assertEqual(b.b2, 1)
        # Sphere-like shell -> chi = 2.
        self.assertEqual(b.euler, 2)

    def test_solid_cube_has_no_cavity(self):
        self.assertEqual(cavity_count(_solid_box(3, 3, 3)), 0)

    def test_genus_helper_matches(self):
        self.assertEqual(genus(_square_ring()), 1)
        self.assertEqual(genus(_solid_box(2, 2, 2)), 0)


class TestGridHelper(unittest.TestCase):
    def test_voxels_from_grid(self):
        grid = [[[1, 0], [0, 0]], [[0, 0], [0, 1]]]
        vox = voxels_from_grid(grid)
        self.assertEqual(vox, {(0, 0, 0), (1, 1, 1)})
        self.assertEqual(betti_numbers(vox).b0, 2)

    def test_dataclass_frozen(self):
        b = BettiNumbers(1, 0, 0, 1)
        with self.assertRaises(Exception):
            b.b0 = 5  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
