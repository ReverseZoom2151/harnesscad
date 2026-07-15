"""primitives — the OS-control action-primitive spec + the NCC template match.

All pure: no OS, no numpy required (numpy is only an optional accelerator inside
the matcher, and the reference path exercised here is pure Python).
"""

import unittest

from harnesscad.io.cua import primitives as P
from harnesscad.io.cua.frames import Frame


class ActionSpecTest(unittest.TestCase):
    def test_every_primitive_has_a_param_contract(self):
        for kind in P.PrimitiveKind:
            self.assertIn(kind, P.PRIMITIVE_PARAMS,
                          "%s has no param contract" % kind)

    def test_wellformed_action_validates_clean(self):
        a = P.Action(P.PrimitiveKind.MOUSE_MOVE, {"x": 10, "y": 20})
        self.assertEqual(P.validate(a), [])

    def test_missing_required_param_is_reported(self):
        a = P.Action(P.PrimitiveKind.MOUSE_MOVE, {"x": 10})
        probs = P.validate(a)
        self.assertTrue(any("missing required" in p and "'y'" in p for p in probs))

    def test_unexpected_param_is_reported(self):
        a = P.Action(P.PrimitiveKind.SCREEN_SIZE, {"x": 1})
        self.assertTrue(any("unexpected" in p for p in P.validate(a)))

    def test_bad_button_is_reported(self):
        a = P.Action(P.PrimitiveKind.MOUSE_CLICK, {"button": "nope"})
        self.assertTrue(any("MouseButton" in p for p in P.validate(a)))

    def test_bad_modifier_is_reported(self):
        a = P.Action(P.PrimitiveKind.KEY_TAP, {"key": "a", "modifiers": ["hyper"]})
        self.assertTrue(any("Modifier" in p for p in P.validate(a)))


class SplitVerbTest(unittest.TestCase):
    def test_click_decomposes_to_move_down_up(self):
        seq = P.click_sequence(5, 6)
        kinds = [a.kind for a in seq]
        self.assertEqual(kinds, [P.PrimitiveKind.MOUSE_MOVE,
                                 P.PrimitiveKind.MOUSE_DOWN,
                                 P.PrimitiveKind.MOUSE_UP])
        self.assertEqual([P.validate(a) for a in seq], [[], [], []])

    def test_drag_is_move_down_move_up(self):
        seq = P.drag_sequence(0, 0, 9, 9)
        self.assertEqual([a.kind for a in seq],
                         [P.PrimitiveKind.MOUSE_MOVE, P.PrimitiveKind.MOUSE_DOWN,
                          P.PrimitiveKind.MOUSE_MOVE, P.PrimitiveKind.MOUSE_UP])
        # start then end coordinate
        self.assertEqual(seq[0].args, {"x": 0, "y": 0})
        self.assertEqual(seq[2].args, {"x": 9, "y": 9})

    def test_chord_holds_modifier_across_key(self):
        seq = P.chord_sequence("a", [P.Modifier.CTRL, P.Modifier.SHIFT])
        kinds = [a.kind for a in seq]
        # down ctrl, down shift, tap a, up shift, up ctrl  -> reverse-nested
        self.assertEqual(kinds[0], P.PrimitiveKind.KEY_DOWN)
        self.assertEqual(kinds[-1], P.PrimitiveKind.KEY_UP)
        self.assertEqual(seq[0].args["key"], "ctrl")
        self.assertEqual(seq[-1].args["key"], "ctrl")   # closes what opened first
        self.assertEqual(seq[1].args["key"], "shift")
        self.assertEqual(seq[-2].args["key"], "shift")


class ColourTest(unittest.TestCase):
    def test_rgb_hex_roundtrip(self):
        self.assertEqual(P.rgb_to_hex(255, 0, 128), "ff0080")
        self.assertEqual(P.hex_to_rgb("#ff0080"), (255, 0, 128))
        self.assertEqual(P.hex_to_rgb("00aa10"), (0, 170, 16))

    def test_rgb_range_checked(self):
        with self.assertRaises(ValueError):
            P.rgb_to_hex(256, 0, 0)


def _blank(h, w, v=0.0):
    return [[float(v)] * w for _ in range(h)]


def _stamp(img, tpl, oy, ox):
    for j, row in enumerate(tpl):
        for i, v in enumerate(row):
            img[oy + j][ox + i] = float(v)


class TemplateMatchTest(unittest.TestCase):
    def setUp(self):
        # A distinctive 3x3 "icon" placed into a larger flat-ish field.
        self.icon = [[9, 1, 9], [1, 9, 1], [9, 1, 9]]

    def test_exact_hit_at_known_position(self):
        hay = _blank(20, 30, v=4.0)
        _stamp(hay, self.icon, oy=7, ox=11)
        m = P.best_match(hay, self.icon)
        self.assertIsNotNone(m)
        self.assertEqual((m.x, m.y), (11, 7))
        self.assertAlmostEqual(m.score, 1.0, places=6)
        self.assertEqual(m.center, (11 + 1, 7 + 1))

    def test_threshold_filters(self):
        hay = _blank(20, 30, v=4.0)
        _stamp(hay, self.icon, oy=3, ox=5)
        hits = P.match_template(hay, self.icon, threshold=0.99)
        self.assertEqual(len(hits), 1)
        self.assertEqual((hits[0].x, hits[0].y), (5, 3))

    def test_two_icons_two_hits_nonmax_suppressed(self):
        hay = _blank(24, 40, v=2.0)
        _stamp(hay, self.icon, oy=4, ox=6)
        _stamp(hay, self.icon, oy=15, ox=25)
        hits = P.match_template(hay, self.icon, threshold=0.99)
        self.assertEqual(len(hits), 2)
        coords = sorted((h.x, h.y) for h in hits)
        self.assertEqual(coords, [(6, 4), (25, 15)])

    def test_no_match_returns_empty(self):
        hay = _blank(10, 10, v=5.0)   # perfectly flat -> no correlation structure
        self.assertEqual(P.match_template(hay, self.icon, threshold=0.9), [])

    def test_template_larger_than_image_is_empty_not_error(self):
        self.assertEqual(P.match_template(_blank(2, 2), self.icon), [])

    def test_contrast_invariance(self):
        # NCC is invariant to affine intensity change: scale + offset the icon and
        # it still matches perfectly (this is why a theme's exact brightness does
        # not break a fixed-icon match).
        hay = _blank(16, 16, v=10.0)
        bright = [[v * 3 + 20 for v in row] for row in self.icon]
        _stamp(hay, bright, oy=5, ox=5)
        m = P.best_match(hay, self.icon)
        self.assertEqual((m.x, m.y), (5, 5))
        self.assertGreater(m.score, 0.999)

    def test_rgb_input_reduced_to_luma(self):
        # A 3-channel icon and a grayscale search agree via the luma reduction.
        rgb_icon = [[(200, 200, 200), (0, 0, 0)],
                    [(0, 0, 0), (200, 200, 200)]]
        hay = _blank(8, 8, v=50.0)
        # stamp the luma of the rgb icon
        luma = [[0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2] for px in row]
                for row in rgb_icon]
        _stamp(hay, luma, oy=2, ox=3)
        m = P.best_match(hay, rgb_icon)
        self.assertEqual((m.x, m.y), (3, 2))
        self.assertGreater(m.score, 0.999)

    def test_locate_icon_maps_through_frame(self):
        hay = _blank(40, 60, v=1.0)
        _stamp(hay, self.icon, oy=10, ox=20)
        # A 1:1 frame anchored at screen origin (100, 200): image px -> +offset.
        frame = Frame.identity(100, 200, 60, 40)
        screen = P.locate_icon(hay, self.icon, frame, threshold=0.9)
        self.assertEqual(screen, (100 + 21, 200 + 11))

    def test_locate_icon_none_below_threshold(self):
        hay = _blank(20, 20, v=3.0)
        self.assertIsNone(P.locate_icon(hay, self.icon, threshold=0.95))

    def test_step_subsampling_still_finds_icon(self):
        hay = _blank(30, 30, v=2.0)
        _stamp(hay, self.icon, oy=8, ox=8)  # aligned to step grid
        hits = P.match_template(hay, self.icon, threshold=0.99, step=2)
        self.assertTrue(any((h.x, h.y) == (8, 8) for h in hits))


if __name__ == "__main__":
    unittest.main()
