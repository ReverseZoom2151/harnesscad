"""Tests for drawings.arcs_viewport_transform."""

import unittest

from drawings.arcs_viewport_transform import (
    IDENTITY,
    Dimension,
    Viewport,
    apply_affine,
    compose_affine,
    invert_affine,
    to_canvas_coordinates,
    to_drawing_coordinates,
    transform_to_canvas_space,
    transform_to_drawing_space,
    visible_bounds,
    zoom_to_fit,
)

# The worked example from arcs/src/window/utils.rs
KNOWN_VIEWPORT = Viewport((300.0, 150.0), 4.0)
KNOWN_WINDOW = (800.0, 400.0)
KNOWN_POINTS = [
    ((300.0, 150.0), (400.0, 200.0)),  # viewport centre
    ((200.0, 200.0), (0.0, 0.0)),  # top-left
    ((200.0, 100.0), (0.0, 400.0)),  # bottom-left
    ((400.0, 100.0), (800.0, 400.0)),  # bottom-right
    ((400.0, 200.0), (800.0, 0.0)),  # top-right
]


class TestAffine(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(apply_affine(IDENTITY, (3.0, 4.0)), (3.0, 4.0))

    def test_invert_round_trip(self):
        transform = (2.0, 0.0, 0.0, -0.5, 10.0, -3.0)
        inverse = invert_affine(transform)
        p = (7.0, -2.0)
        back = apply_affine(inverse, apply_affine(transform, p))
        self.assertAlmostEqual(back[0], p[0])
        self.assertAlmostEqual(back[1], p[1])

    def test_singular_affine_raises(self):
        with self.assertRaises(ValueError):
            invert_affine((0.0, 0.0, 0.0, 0.0, 1.0, 1.0))

    def test_compose_matches_sequential_application(self):
        first = (1.0, 0.0, 0.0, 1.0, 5.0, 6.0)  # translate
        then = (2.0, 0.0, 0.0, 3.0, 0.0, 0.0)  # scale
        combined = compose_affine(first, then)
        p = (1.0, 1.0)
        self.assertEqual(
            apply_affine(combined, p),
            apply_affine(then, apply_affine(first, p)),
        )


class TestKnownExample(unittest.TestCase):
    def test_drawing_to_canvas(self):
        for drawing, canvas in KNOWN_POINTS:
            got = to_canvas_coordinates(drawing, KNOWN_VIEWPORT, KNOWN_WINDOW)
            self.assertAlmostEqual(got[0], canvas[0])
            self.assertAlmostEqual(got[1], canvas[1])

    def test_canvas_to_drawing(self):
        for drawing, canvas in KNOWN_POINTS:
            got = to_drawing_coordinates(canvas, KNOWN_VIEWPORT, KNOWN_WINDOW)
            self.assertAlmostEqual(got[0], drawing[0])
            self.assertAlmostEqual(got[1], drawing[1])

    def test_known_matrices(self):
        drawing = transform_to_drawing_space(KNOWN_VIEWPORT, KNOWN_WINDOW)
        canvas = transform_to_canvas_space(KNOWN_VIEWPORT, KNOWN_WINDOW)
        for got, expected in zip(drawing, (0.25, 0.0, 0.0, -0.25, 200.0, 200.0)):
            self.assertAlmostEqual(got, expected)
        for got, expected in zip(canvas, (4.0, 0.0, 0.0, -4.0, -800.0, 800.0)):
            self.assertAlmostEqual(got, expected)

    def test_visible_bounds(self):
        bounds = visible_bounds(KNOWN_VIEWPORT, KNOWN_WINDOW)
        for got, expected in zip(bounds, (200.0, 100.0, 400.0, 200.0)):
            self.assertAlmostEqual(got, expected)


class TestViewport(unittest.TestCase):
    def test_bad_scale(self):
        with self.assertRaises(ValueError):
            Viewport((0.0, 0.0), 0.0)

    def test_zoom_out_doubles_visible_area(self):
        zoomed = KNOWN_VIEWPORT.zoomed(2.0)
        self.assertAlmostEqual(zoomed.pixels_per_drawing_unit, 2.0)
        min_x, min_y, max_x, max_y = visible_bounds(zoomed, KNOWN_WINDOW)
        self.assertAlmostEqual(max_x - min_x, 400.0)
        self.assertAlmostEqual(max_y - min_y, 200.0)
        self.assertEqual(zoomed.centre, KNOWN_VIEWPORT.centre)

    def test_zoom_by_zero_raises(self):
        with self.assertRaises(ValueError):
            KNOWN_VIEWPORT.zoomed(0.0)

    def test_pan_moves_the_centre(self):
        panned = KNOWN_VIEWPORT.translated(10.0, -5.0)
        self.assertEqual(panned.centre, (310.0, 145.0))
        # the centre still maps to the middle of the window
        got = to_canvas_coordinates(panned.centre, panned, KNOWN_WINDOW)
        self.assertAlmostEqual(got[0], 400.0)
        self.assertAlmostEqual(got[1], 200.0)


class TestDimension(unittest.TestCase):
    def test_pixels_are_zoom_invariant(self):
        d = Dimension.pixels(3.0)
        self.assertEqual(d.in_pixels(4.0), 3.0)
        self.assertEqual(d.in_pixels(100.0), 3.0)
        self.assertAlmostEqual(d.in_drawing_units(4.0), 0.75)

    def test_drawing_units_scale_with_zoom(self):
        d = Dimension.drawing_units(5.0)
        self.assertEqual(d.in_pixels(4.0), 20.0)
        self.assertEqual(d.in_drawing_units(4.0), 5.0)

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            Dimension("furlongs", 1.0).in_pixels(1.0)

    def test_bad_scale(self):
        with self.assertRaises(ValueError):
            Dimension.pixels(1.0).in_drawing_units(0.0)


class TestZoomToFit(unittest.TestCase):
    def test_fits_wide_bounds(self):
        viewport = zoom_to_fit((0.0, 0.0, 200.0, 50.0), (800.0, 400.0))
        self.assertAlmostEqual(viewport.pixels_per_drawing_unit, 4.0)
        self.assertEqual(viewport.centre, (100.0, 25.0))
        min_x, min_y, max_x, max_y = visible_bounds(viewport, (800.0, 400.0))
        self.assertLessEqual(min_x, 0.0 + 1e-9)
        self.assertGreaterEqual(max_x, 200.0 - 1e-9)
        self.assertLessEqual(min_y, 0.0 + 1e-9)
        self.assertGreaterEqual(max_y, 50.0 - 1e-9)

    def test_margin_shrinks_the_scale(self):
        plain = zoom_to_fit((0.0, 0.0, 100.0, 100.0), (400.0, 400.0))
        padded = zoom_to_fit((0.0, 0.0, 100.0, 100.0), (400.0, 400.0), 0.1)
        self.assertAlmostEqual(plain.pixels_per_drawing_unit, 4.0)
        self.assertAlmostEqual(padded.pixels_per_drawing_unit, 3.2)

    def test_degenerate_bounds_fall_back_to_unit_scale(self):
        viewport = zoom_to_fit((5.0, 5.0, 5.0, 5.0), (400.0, 400.0))
        self.assertEqual(viewport.pixels_per_drawing_unit, 1.0)
        self.assertEqual(viewport.centre, (5.0, 5.0))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            zoom_to_fit((10.0, 0.0, 0.0, 10.0), (400.0, 400.0))
        with self.assertRaises(ValueError):
            zoom_to_fit((0.0, 0.0, 1.0, 1.0), (0.0, 400.0))
        with self.assertRaises(ValueError):
            zoom_to_fit((0.0, 0.0, 1.0, 1.0), (400.0, 400.0), 0.6)


if __name__ == "__main__":
    unittest.main()
