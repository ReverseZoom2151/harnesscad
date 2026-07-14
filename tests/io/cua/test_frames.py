"""Coordinate discipline: letterbox never stretches, and a frame is never inferred."""

import sys
import unittest

from harnesscad.io.cua.frames import (
    Frame, FrameError, assert_frames_agree, ensure_dpi_aware, monitors,
    primary_monitor, system_metrics,
)


class TestLetterbox(unittest.TestCase):
    def test_uniform_scale_never_distorts_aspect(self):
        """The bug in TWO reference repos: a 16:9 screen squashed into 16:10, so a
        circle becomes an ellipse. Indefensible for a CAD agent."""
        f = Frame.letterbox(0, 0, 2560, 1440, 1280, 800)
        self.assertAlmostEqual(f.scale, 0.5)
        self.assertTrue(f.aspect_preserved)
        # A square on screen must stay square in the image.
        a = f.to_image(100, 100)
        b = f.to_image(200, 200)
        self.assertEqual(b[0] - a[0], b[1] - a[1])

    def test_pads_rather_than_stretching(self):
        f = Frame.letterbox(0, 0, 2560, 1440, 1280, 800)
        self.assertEqual((f.dst_w, f.dst_h), (1280, 800))
        self.assertEqual(f.pad_x, 0)
        self.assertEqual(f.pad_y, (800 - 720) // 2)

    def test_round_trip_through_both_maps(self):
        f = Frame.letterbox(0, 0, 2560, 1440, 1280, 800)
        for pt in ((0, 0), (2559, 1439), (1000, 700)):
            back = f.to_screen(*f.to_image(*pt))
            self.assertLessEqual(abs(back[0] - pt[0]), 2)
            self.assertLessEqual(abs(back[1] - pt[1]), 2)

    def test_negative_origin_multi_monitor(self):
        """A left-hand secondary monitor has NEGATIVE x in virtual-desktop space."""
        f = Frame.letterbox(-1920, 0, 1920, 1080, 960, 540)
        self.assertEqual(f.to_screen(*f.to_image(-1000, 500)), (-1000, 500))

    def test_padding_is_not_screen(self):
        f = Frame.letterbox(0, 0, 2560, 1440, 1280, 800)
        with self.assertRaises(FrameError):
            f.to_screen(10, 2)          # inside the letterbox bar, not the content

    def test_degenerate_frames_are_refused(self):
        with self.assertRaises(FrameError):
            Frame.letterbox(0, 0, 0, 100, 100, 100)
        with self.assertRaises(FrameError):
            Frame.identity(0, 0, 10, 0)


class TestNoFrameIsEverInferred(unittest.TestCase):
    def test_normalised_space_has_no_magnitude_heuristic(self):
        """TuriX guesses whether the model meant 0-1 or 0-1000 from the MAGNITUDE
        of the number. Here 0.5 means half way, always, and out-of-range raises."""
        f = Frame.identity(0, 0, 1000, 1000)
        self.assertEqual(f.normalized_to_screen(0.5, 0.5), (500, 500))
        with self.assertRaises(FrameError):
            f.normalized_to_screen(500, 500)

    def test_subframe_cannot_escape_its_parent(self):
        f = Frame.identity(100, 100, 800, 600)
        vp = f.subframe(200, 200, 100, 100, label="viewport")
        self.assertEqual(vp.scale, 1.0)          # full-res zoom, no downscale
        self.assertEqual(vp.screen_rect, (200, 200, 300, 300))
        with self.assertRaises(FrameError):
            f.subframe(0, 0, 100, 100)


@unittest.skipUnless(sys.platform.startswith("win"), "Windows-only DPI probe")
class TestDpi(unittest.TestCase):
    def test_process_is_dpi_aware_at_import(self):
        self.assertTrue(ensure_dpi_aware())

    def test_frames_agree_or_we_refuse_to_run(self):
        size = assert_frames_agree()
        self.assertEqual(size, system_metrics())
        mon = primary_monitor()
        self.assertEqual((mon.width, mon.height), size)
        self.assertTrue(monitors())


if __name__ == "__main__":
    unittest.main()
