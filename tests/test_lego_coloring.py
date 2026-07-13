import unittest

from harnesscad.domain.fabrication.lego_brick_library import Brick
from harnesscad.domain.fabrication.lego_coloring import (
    LEGO_PALETTE,
    assign_brick_colors,
    brick_color,
    nearest_lego_color,
    voxel_color,
)


class TestColorMath(unittest.TestCase):
    def test_voxel_color_mean(self):
        self.assertEqual(voxel_color([(0, 0, 0), (100, 100, 100)]), (50, 50, 50))

    def test_brick_color_mean(self):
        self.assertEqual(brick_color([(10, 20, 30), (30, 40, 50)]), (20, 30, 40))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            voxel_color([])


class TestNearest(unittest.TestCase):
    def test_exact_palette_hit(self):
        self.assertEqual(nearest_lego_color((196, 40, 27)), "bright_red")
        self.assertEqual(nearest_lego_color((13, 105, 172)), "bright_blue")

    def test_near_black(self):
        self.assertEqual(nearest_lego_color((10, 10, 10)), "black")

    def test_deterministic(self):
        c = (120, 120, 120)
        self.assertEqual(nearest_lego_color(c), nearest_lego_color(c))
        self.assertIn(nearest_lego_color(c), LEGO_PALETTE)


class TestAssign(unittest.TestCase):
    def test_assign_uses_visible_faces(self):
        b = Brick(1, 2, 0, 0, 0)
        faces = {
            (0, 0, 0): [(196, 40, 27)],
            (0, 1, 0): [(196, 40, 27)],
        }
        self.assertEqual(assign_brick_colors([b], faces), ["bright_red"])

    def test_occluded_brick_falls_back_to_grey(self):
        b = Brick(1, 1, 5, 5, 5)
        names = assign_brick_colors([b], {})
        self.assertEqual(len(names), 1)
        self.assertIn(names[0], LEGO_PALETTE)

    def test_multiple_bricks(self):
        bricks = [Brick(1, 1, 0, 0, 0), Brick(1, 1, 1, 0, 0)]
        faces = {
            (0, 0, 0): [(13, 105, 172)],
            (1, 0, 0): [(245, 205, 47)],
        }
        self.assertEqual(
            assign_brick_colors(bricks, faces), ["bright_blue", "bright_yellow"]
        )


if __name__ == "__main__":
    unittest.main()
