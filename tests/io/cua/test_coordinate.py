"""Tests for the atlas-derived coordinate discipline utilities."""

import unittest

from harnesscad.io.cua.coordinate import (
    CoordinateError, CoordinateMapper, Denormalizer, NormalizedAction,
    ScreenInfo, extract_safety_decision, normalize_function_call,
)


class TestScreenInfo(unittest.TestCase):
    def test_physical_is_scaled(self):
        s = ScreenInfo(1920, 1080, 2.0)
        self.assertEqual(s.physical_width, 3840)
        self.assertEqual(s.physical_height, 2160)

    def test_resolution_string(self):
        self.assertEqual(ScreenInfo(1920, 1080).resolution_string(), "1920x1080")

    def test_bad_size_rejected(self):
        with self.assertRaises(CoordinateError):
            ScreenInfo(0, 100)

    def test_bad_scale_rejected(self):
        with self.assertRaises(CoordinateError):
            ScreenInfo(100, 100, 0)


class TestCoordinateMapper(unittest.TestCase):
    def test_physical_to_logical_halves_on_2x(self):
        m = CoordinateMapper(ScreenInfo(1920, 1080, 2.0))
        self.assertEqual(m.to_logical(3840, 2160), (1920, 1080))

    def test_logical_to_physical_doubles_on_2x(self):
        m = CoordinateMapper(ScreenInfo(1920, 1080, 2.0))
        self.assertEqual(m.to_physical(960, 540), (1920, 1080))

    def test_roundtrip_1x_is_identity(self):
        m = CoordinateMapper(ScreenInfo(1000, 800, 1.0))
        self.assertEqual(m.to_logical(*m.to_physical(123, 456)), (123, 456))


class TestDenormalizer(unittest.TestCase):
    def test_zero_maps_to_zero(self):
        self.assertEqual(Denormalizer().denorm(0, 1000), 0)

    def test_max_maps_to_last_pixel_not_past_it(self):
        # THE off-by-one: 999 must land on pixel 999 for a 1000-wide frame, not 1000.
        d = Denormalizer(grid=999)
        self.assertEqual(d.denorm(999, 1000), 999)

    def test_midpoint(self):
        d = Denormalizer(grid=1000)
        self.assertEqual(d.denorm(500, 1001), 500)

    def test_denorm_into_rect_offsets_and_scales(self):
        d = Denormalizer(grid=999)
        # A viewport rect at (100, 50) sized 200x100. Grid 0 -> top-left corner.
        self.assertEqual(d.denorm_into_rect(0, 0, (100, 50, 200, 100)), (100, 50))
        # Grid max -> bottom-right pixel of the rect.
        self.assertEqual(d.denorm_into_rect(999, 999, (100, 50, 200, 100)),
                         (100 + 199, 50 + 99))

    def test_normalize_is_inverse_at_endpoints(self):
        d = Denormalizer(grid=999)
        self.assertEqual(d.normalize(0, 0, 1000, 800), (0, 0))
        self.assertEqual(d.normalize(999, 799, 1000, 800), (999, 999))

    def test_clamps_out_of_range(self):
        d = Denormalizer(grid=999)
        self.assertEqual(d.denorm(2000, 500), 499)

    def test_bad_grid(self):
        with self.assertRaises(CoordinateError):
            Denormalizer(grid=0)


class TestNormalizeFunctionCall(unittest.TestCase):
    def test_click_at(self):
        a = normalize_function_call("click_at", {"x": 999, "y": 0}, 1000, 500)
        self.assertEqual(a.verb, "click")
        self.assertEqual(a.coords, (999, 0))

    def test_type_without_coords_does_not_click_origin(self):
        # The key nuance: no coords -> type into focus, coords stay None.
        a = normalize_function_call("type_text_at", {"text": "37.5"}, 1000, 500)
        self.assertEqual(a.verb, "type")
        self.assertIsNone(a.coords)
        self.assertEqual(a.text, "37.5")

    def test_type_with_zero_coords_is_focus_type(self):
        a = normalize_function_call("type_text_at", {"x": 0, "y": 0, "text": "hi"},
                                    1000, 500)
        self.assertIsNone(a.coords)

    def test_type_with_coords_resolves(self):
        a = normalize_function_call("type_text_at",
                                    {"x": 999, "y": 999, "text": "hi", "press_enter": True},
                                    1000, 500)
        self.assertEqual(a.coords, (999, 499))
        self.assertTrue(a.press_enter)

    def test_key_combination_splits(self):
        a = normalize_function_call("key_combination", {"keys": "Ctrl+Shift+S"}, 100, 100)
        self.assertEqual(a.keys, ("ctrl", "shift", "s"))

    def test_unknown_function_is_dropped(self):
        self.assertIsNone(normalize_function_call("teleport", {}, 100, 100))

    def test_safety_decision_extracted(self):
        args = {"x": 1, "y": 1, "safety_decision": {"decision": "require_confirmation"}}
        a = normalize_function_call("click_at", args, 100, 100)
        self.assertTrue(a.requires_confirmation)

    def test_extract_safety_none_when_absent(self):
        self.assertIsNone(extract_safety_decision({}))

    def test_to_dict(self):
        a = NormalizedAction(verb="click", coords=(1, 2))
        self.assertEqual(a.to_dict()["coords"], [1, 2])


if __name__ == "__main__":
    unittest.main()
