"""Tests for cadrille 2x2 multi-view grid composition."""

import unittest

from harnesscad.domain.vision.cadrille_multiview_grid import (
    compose_grid,
    compose_named,
    grid_layout,
    VIEW_ORDER,
    NUM_VIEWS,
    VIEW_SIZE,
    COMBINED_SIZE,
)


def _img(fill, h=2, w=2):
    return [[fill for _ in range(w)] for _ in range(h)]


class ComposeGridTest(unittest.TestCase):
    def test_layout(self):
        views = [_img("a"), _img("b"), _img("c"), _img("d")]
        grid = compose_grid(views)
        self.assertEqual(len(grid), 4)       # 2H
        self.assertEqual(len(grid[0]), 4)    # 2W
        # top-left cell -> a, top-right -> b
        self.assertEqual(grid[0][0], "a")
        self.assertEqual(grid[0][3], "b")
        # bottom-left -> c, bottom-right -> d
        self.assertEqual(grid[3][0], "c")
        self.assertEqual(grid[3][3], "d")

    def test_wrong_view_count(self):
        with self.assertRaises(ValueError):
            compose_grid([_img("a"), _img("b")])

    def test_mismatched_dims(self):
        with self.assertRaises(ValueError):
            compose_grid([_img("a", 2, 2), _img("b", 3, 3),
                          _img("c"), _img("d")])

    def test_rgb_pixels(self):
        views = [_img((i, i, i)) for i in range(NUM_VIEWS)]
        grid = compose_grid(views)
        self.assertEqual(grid[0][0], (0, 0, 0))


class NamedTest(unittest.TestCase):
    def test_compose_named(self):
        view_map = {name: _img(name) for name in VIEW_ORDER}
        grid = compose_named(view_map)
        self.assertEqual(grid[0][0], VIEW_ORDER[0])

    def test_missing_view(self):
        view_map = {name: _img(name) for name in VIEW_ORDER[:3]}
        with self.assertRaises(ValueError):
            compose_named(view_map)


class LayoutTest(unittest.TestCase):
    def test_grid_layout(self):
        layout = grid_layout()
        self.assertEqual(set(layout), set(VIEW_ORDER))
        self.assertEqual(layout[VIEW_ORDER[0]], {"row": 0, "col": 0, "size": VIEW_SIZE})
        self.assertEqual(layout[VIEW_ORDER[3]]["row"], VIEW_SIZE)
        self.assertEqual(VIEW_SIZE * 2, COMBINED_SIZE)


if __name__ == "__main__":
    unittest.main()
