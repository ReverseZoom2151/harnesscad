"""The observation must not show the agent's own UI, and stolen focus is a halt."""

import unittest

from harnesscad.io.cua import frames
from harnesscad.io.cua.windowing import (
    FocusDiscipline, OcclusionRegistry, Overlay, WindowState, frame_for_window,
    monitor_for_rect, raise_plan, rects_overlap,
)


class TestOcclusion(unittest.TestCase):
    def test_overlay_over_content_contaminates(self):
        reg = OcclusionRegistry()
        reg.register(Overlay("hud", (0, 0, 200, 40), hideable=True))
        reg.register(Overlay("orb", (1000, 700, 1080, 780), hideable=False))
        content = (0, 0, 1280, 800)
        contam = {o.name for o in reg.contaminates(content)}
        self.assertEqual(contam, {"hud", "orb"})

    def test_hideable_are_hidden_but_blocking_remain(self):
        reg = OcclusionRegistry()
        reg.register(Overlay("hud", (0, 0, 200, 40), hideable=True))
        reg.register(Overlay("stuck", (10, 10, 50, 50), hideable=False))
        content = (0, 0, 1280, 800)
        self.assertEqual([o.name for o in reg.hideable()], ["hud"])
        self.assertEqual([o.name for o in reg.blocking(content)], ["stuck"])

    def test_overlay_outside_content_is_fine(self):
        reg = OcclusionRegistry()
        reg.register(Overlay("far", (5000, 5000, 5100, 5100)))
        self.assertEqual(reg.contaminates((0, 0, 1280, 800)), [])


class TestFocusDiscipline(unittest.TestCase):
    def test_focus_theft_is_detected(self):
        disc = FocusDiscipline(target_app="FreeCAD")
        disc.snapshot(WindowState("FreeCAD", "part.FCStd", "Pad", owned=True))
        after = WindowState("Slack", "general", owned=False)
        msg = disc.detect_theft(after)
        self.assertIsNotNone(msg)
        self.assertIn("Slack", msg)

    def test_still_ours_is_no_theft(self):
        disc = FocusDiscipline(target_app="FreeCAD")
        after = WindowState("FreeCAD", "part.FCStd", "Length", owned=True)
        self.assertIsNone(disc.detect_theft(after))
        self.assertTrue(disc.is_foreground(after))

    def test_owned_but_wrong_app_is_not_foreground(self):
        disc = FocusDiscipline(target_app="FreeCAD")
        self.assertFalse(disc.is_foreground(WindowState("Onshape", owned=True)))


class TestRaisePlan(unittest.TestCase):
    def test_no_raise_when_already_foreground(self):
        cur = WindowState("FreeCAD", owned=True)
        self.assertEqual(raise_plan(cur, "FreeCAD"), [])

    def test_raise_sequence_when_not_foreground(self):
        cur = WindowState("Chrome", owned=False)
        self.assertEqual(raise_plan(cur, "FreeCAD"),
                         ["activate:FreeCAD", "raise", "verify_foreground"])


class TestMonitorTargeting(unittest.TestCase):
    def _mons(self):
        return [frames.Monitor(0, 0, 0, 1920, 1080, primary=True),
                frames.Monitor(1, -1920, 0, 1920, 1080)]

    def test_window_on_left_negative_monitor(self):
        mon = monitor_for_rect((-1800, 100, -800, 700), self._mons())
        self.assertIsNotNone(mon)
        self.assertEqual(mon.index, 1)

    def test_window_mostly_on_primary(self):
        mon = monitor_for_rect((100, 100, 900, 700), self._mons())
        self.assertEqual(mon.index, 0)

    def test_offscreen_window_matches_nothing(self):
        self.assertIsNone(monitor_for_rect((5000, 5000, 5100, 5100), self._mons()))

    def test_frame_for_window_refuses_offscreen(self):
        with self.assertRaises(frames.FrameError):
            frame_for_window((5000, 5000, 5100, 5100), self._mons())

    def test_frame_for_window_tags_correct_monitor(self):
        f = frame_for_window((-1800, 100, -800, 700), self._mons())
        self.assertEqual(f.monitor, 1)

    def test_rects_overlap_helper(self):
        self.assertTrue(rects_overlap((0, 0, 10, 10), (5, 5, 15, 15)))
        self.assertFalse(rects_overlap((0, 0, 10, 10), (10, 10, 20, 20)))


if __name__ == "__main__":
    unittest.main()
