import unittest

from harnesscad.domain.reconstruction.cad2program_shape_program import ShapeProgram, make_instance
from harnesscad.domain.drawings.cad2program_canvas_layout import (
    layout_program, PixelRect, CanvasLayout,
)
from harnesscad.domain.drawings.cad2program_view_lifting import FRONT, TOP, SIDE


def _prog():
    return ShapeProgram([
        make_instance("a", (0, 0, 0, 100, 60, 200, 0)),
        make_instance("b", (0, 0, 210, 100, 60, 20, 0)),
    ])


class LayoutTest(unittest.TestCase):
    def test_canvas_size(self):
        lay = layout_program(_prog(), canvas=512)
        self.assertEqual(lay.width, 512)
        self.assertEqual(lay.height, 512)
        self.assertIsInstance(lay, CanvasLayout)

    def test_three_views_present(self):
        lay = layout_program(_prog())
        self.assertEqual(set(lay.rects), {FRONT, TOP, SIDE})
        for view in (FRONT, TOP, SIDE):
            self.assertEqual(len(lay.rects[view]), 2)

    def test_rects_within_canvas(self):
        lay = layout_program(_prog(), canvas=512)
        for view in lay.rects:
            for r in lay.rects[view]:
                self.assertGreaterEqual(r.left, 0)
                self.assertGreaterEqual(r.top, 0)
                self.assertLessEqual(r.right, 512)
                self.assertLessEqual(r.bottom, 512)

    def test_quadrant_placement(self):
        lay = layout_program(_prog(), canvas=512, margin=8, gap=8)
        mid = 512 // 2
        # TOP is top-left, FRONT bottom-left, SIDE bottom-right.
        self.assertLess(lay.rects[TOP][0].top, mid)
        self.assertGreater(lay.rects[FRONT][0].top, mid)
        self.assertGreater(lay.rects[SIDE][0].left, mid)

    def test_uniform_scale_shared_width(self):
        # Both primitives share X width 100; in FRONT and TOP their pixel widths
        # must match (uniform scale, shared X axis).
        lay = layout_program(_prog())
        self.assertEqual(lay.rects[FRONT][0].width, lay.rects[TOP][0].width)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            layout_program(ShapeProgram())

    def test_deterministic(self):
        a = layout_program(_prog())
        b = layout_program(_prog())
        self.assertEqual(a.rects[FRONT][0], b.rects[FRONT][0])
        self.assertEqual(a.scale, b.scale)


if __name__ == "__main__":
    unittest.main()
