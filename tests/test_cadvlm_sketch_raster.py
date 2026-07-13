import unittest

from harnesscad.domain.vision.sketch_raster import (
    RasterImage, rasterize_entity, rasterize_sketch,
)


class CadVLMSketchRasterTests(unittest.TestCase):
    def test_horizontal_line_is_connected_row(self):
        img = rasterize_sketch(
            ({"type": "line", "start": (1, 32), "end": (64, 32)},),
            resolution=64, coord_range=(1, 64))
        ys = {y for _, y in img.pixels}
        xs = sorted(x for x, _ in img.pixels)
        self.assertEqual(ys, {31})                     # (32-1)/63*63 = 31
        self.assertEqual(xs[0], 0)
        self.assertEqual(xs[-1], 63)
        # 8-connected: no horizontal gaps.
        self.assertEqual(xs, list(range(0, 64)))

    def test_determinism_and_grid_shape(self):
        entities = ({"type": "line", "start": (1, 1), "end": (64, 64)},)
        a = rasterize_sketch(entities, resolution=32)
        b = rasterize_sketch(entities, resolution=32)
        self.assertEqual(a.pixels, b.pixels)
        grid = a.to_grid()
        self.assertEqual(len(grid), 32)
        self.assertTrue(all(len(row) == 32 for row in grid))
        self.assertEqual(sum(sum(row) for row in grid), a.occupancy)

    def test_circle_from_four_points_is_ring(self):
        entity = {"type": "circle",
                  "points": ((48, 32), (32, 48), (16, 32), (32, 16))}
        pix = rasterize_entity(entity, resolution=64, coord_range=(1, 64))
        # centre pixel is not lit (it is a ring, not a disc).
        center = (round((32 - 1) / 63 * 63), round((32 - 1) / 63 * 63))
        self.assertNotIn(center, pix)
        # roughly circular: radius ~16 in coord units -> ~16 px.
        rs = [((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5 for x, y in pix]
        self.assertTrue(14 <= sum(rs) / len(rs) <= 18)

    def test_circle_center_radius_form(self):
        pts = rasterize_entity(
            {"type": "circle", "points": ((48, 32), (32, 48), (16, 32), (32, 16))},
            resolution=64)
        cr = rasterize_entity(
            {"type": "circle", "center": (32, 32), "radius": 16.0},
            resolution=64)
        self.assertEqual(pts, cr)

    def test_arc_endpoints_present_and_connected(self):
        entity = {"type": "arc", "start": (16, 32), "mid": (32, 48), "end": (48, 32)}
        pix = rasterize_entity(entity, resolution=64, coord_range=(1, 64))
        self.assertIn((round(15 / 63 * 63), round(31 / 63 * 63)), pix)
        self.assertIn((round(47 / 63 * 63), round(31 / 63 * 63)), pix)
        # arc bulges upward through the mid point (higher y).
        self.assertTrue(any(y >= 46 for _, y in pix))

    def test_collinear_arc_falls_back_to_stroke(self):
        entity = {"type": "arc", "start": (1, 32), "mid": (32, 32), "end": (64, 32)}
        pix = rasterize_entity(entity, resolution=64, coord_range=(1, 64))
        self.assertEqual({y for _, y in pix}, {31})

    def test_clamping_keeps_pixels_in_bounds(self):
        img = rasterize_sketch(
            ({"type": "line", "start": (-100, -100), "end": (500, 500)},),
            resolution=16, coord_range=(1, 64))
        self.assertTrue(all(0 <= x < 16 and 0 <= y < 16 for x, y in img.pixels))

    def test_rejects_bad_resolution_and_type(self):
        with self.assertRaises(ValueError):
            rasterize_sketch((), resolution=0)
        with self.assertRaises(ValueError):
            rasterize_entity({"type": "spline", "start": (1, 1)}, resolution=8)


if __name__ == "__main__":
    unittest.main()
