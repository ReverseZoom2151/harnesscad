"""Tests for geometry.brick_structure (BRICKGPT brick representation)."""

import unittest

from harnesscad.domain.geometry.brick_structure import (
    STANDARD_BRICKS,
    Brick,
    BrickStructure,
    bricks_overlap,
)


class TestBrick(unittest.TestCase):
    def test_construction_and_validation(self):
        b = Brick(2, 4, 1, 3, 0)
        self.assertEqual(b.dims, (2, 4))
        self.assertEqual(b.stud_count, 8)
        with self.assertRaises(ValueError):
            Brick(0, 1, 0, 0, 0)
        with self.assertRaises(ValueError):
            Brick(1, 1, -1, 0, 0)
        with self.assertRaises(TypeError):
            Brick(1, 1, 0, 0, 1.5)  # type: ignore[arg-type]

    def test_cells_and_voxels(self):
        b = Brick(2, 3, 1, 1, 2)
        cells = sorted(b.cells())
        self.assertEqual(len(cells), 6)
        self.assertIn((1, 1), cells)
        self.assertIn((2, 3), cells)
        self.assertNotIn((3, 3), cells)
        voxels = b.voxel_set()
        self.assertEqual(len(voxels), 6)
        self.assertTrue(all(z == 2 for _, _, z in voxels))

    def test_center_and_orientation(self):
        self.assertEqual(Brick(2, 4, 0, 0, 0).center, (1.0, 2.0, 0.5))
        self.assertEqual(Brick(1, 2, 0, 0, 0).orientation, 0)
        self.assertEqual(Brick(2, 1, 0, 0, 0).orientation, 1)
        self.assertEqual(Brick(2, 2, 0, 0, 0).orientation, 0)

    def test_library_and_bounds(self):
        self.assertTrue(Brick(2, 4, 0, 0, 0).in_library())
        self.assertFalse(Brick(3, 3, 0, 0, 0).in_library())
        self.assertIn((1, 8), STANDARD_BRICKS)
        self.assertIn((8, 1), STANDARD_BRICKS)
        self.assertTrue(Brick(2, 2, 18, 18, 19).in_bounds(20, 20, 20))
        self.assertFalse(Brick(2, 2, 19, 19, 0).in_bounds(20, 20, 20))
        self.assertFalse(Brick(1, 1, 0, 0, 20).in_bounds(20, 20, 20))


class TestTextFormat(unittest.TestCase):
    def test_roundtrip(self):
        b = Brick(2, 4, 1, 3, 5)
        self.assertEqual(b.to_text(), "2x4 (1,3,5)")
        self.assertEqual(Brick.from_text("2x4 (1,3,5)"), b)
        self.assertEqual(Brick.from_text("2×4 (1,3,5)"), b)  # unicode times
        self.assertEqual(Brick.from_text("  1x1 ( 0 , 0 , 0 ) "), Brick(1, 1, 0, 0, 0))

    def test_malformed(self):
        for bad in ["2x4 1,3,5", "2x4x1 (1,3,5)", "2x4 (1,3)"]:
            with self.assertRaises(ValueError):
                Brick.from_text(bad)

    def test_structure_roundtrip(self):
        text = "2x2 (0,0,0)\n2x2 (0,0,1)"
        s = BrickStructure.from_text(text)
        self.assertEqual(len(s), 2)
        self.assertEqual(s.to_text(), text)
        # comments and blank lines ignored
        s2 = BrickStructure.from_text("# header\n\n1x1 (0,0,0)\n")
        self.assertEqual(len(s2), 1)


class TestCollision(unittest.TestCase):
    def test_bricks_overlap(self):
        self.assertTrue(bricks_overlap(Brick(2, 2, 0, 0, 0), Brick(2, 2, 1, 1, 0)))
        self.assertFalse(bricks_overlap(Brick(2, 2, 0, 0, 0), Brick(2, 2, 2, 0, 0)))
        # touching edge (adjacent, not overlapping)
        self.assertFalse(bricks_overlap(Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 2, 0)))
        # different layers never collide
        self.assertFalse(bricks_overlap(Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1)))

    def test_structure_collision(self):
        good = BrickStructure.from_bricks([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1)])
        self.assertFalse(good.has_collision())
        self.assertEqual(good.colliding_pairs(), [])
        bad = BrickStructure.from_bricks(
            [Brick(2, 2, 0, 0, 0), Brick(2, 2, 1, 1, 0), Brick(1, 1, 5, 5, 0)]
        )
        self.assertTrue(bad.has_collision())
        self.assertEqual(bad.colliding_pairs(), [(0, 1)])

    def test_collides_with_existing(self):
        s = BrickStructure.from_bricks([Brick(2, 2, 0, 0, 0)])
        self.assertTrue(s.collides_with_existing(Brick(1, 1, 1, 1, 0)))
        self.assertFalse(s.collides_with_existing(Brick(1, 1, 5, 5, 0)))
        self.assertFalse(s.collides_with_existing(Brick(2, 2, 0, 0, 1)))

    def test_prefix_and_validity(self):
        s = BrickStructure.from_bricks(
            [Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1), Brick(1, 1, 0, 0, 2)]
        )
        self.assertEqual(len(s.prefix(2)), 2)
        self.assertTrue(s.all_in_bounds())
        self.assertTrue(s.all_in_library())
        odd = BrickStructure.from_bricks([Brick(3, 3, 0, 0, 0)])
        self.assertFalse(odd.all_in_library())


if __name__ == "__main__":
    unittest.main()
