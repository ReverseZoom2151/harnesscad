"""Tests for geometry.shapeit_keyframe."""

import unittest

from geometry.shapeit_heightfield import HeightField
from geometry import shapeit_keyframe as kf


class TestEasing(unittest.TestCase):
    def test_endpoints(self):
        for fn in (kf.ease_linear, kf.ease_in, kf.ease_out, kf.ease_in_out):
            self.assertAlmostEqual(fn(0.0), 0.0)
            self.assertAlmostEqual(fn(1.0), 1.0)

    def test_in_out_midpoint(self):
        self.assertAlmostEqual(kf.ease_in_out(0.5), 0.5)

    def test_ease_in_below_linear(self):
        self.assertLess(kf.ease_in(0.5), 0.5)

    def test_ease_out_above_linear(self):
        self.assertGreater(kf.ease_out(0.5), 0.5)


class TestLerp(unittest.TestCase):
    def test_endpoints(self):
        a = HeightField.from_rows([[0.0, 0.0]])
        b = HeightField.from_rows([[1.0, 0.4]])
        self.assertEqual(kf.lerp_field(a, b, 0.0).to_rows(), a.to_rows())
        self.assertEqual(kf.lerp_field(a, b, 1.0).to_rows(), b.to_rows())

    def test_midpoint(self):
        a = HeightField.from_rows([[0.0, 0.2]])
        b = HeightField.from_rows([[1.0, 0.6]])
        mid = kf.lerp_field(a, b, 0.5)
        self.assertAlmostEqual(mid.get(0, 0), 0.5)
        self.assertAlmostEqual(mid.get(0, 1), 0.4)

    def test_shape_mismatch(self):
        a = HeightField(2, 2)
        b = HeightField(2, 3)
        with self.assertRaises(ValueError):
            kf.lerp_field(a, b, 0.5)


class TestTween(unittest.TestCase):
    def test_frame_count_and_ends(self):
        a = HeightField.from_rows([[0.0]])
        b = HeightField.from_rows([[1.0]])
        seq = kf.tween(a, b, frames=5)
        self.assertEqual(len(seq), 5)
        self.assertEqual(seq[0].get(0, 0), 0.0)
        self.assertEqual(seq[-1].get(0, 0), 1.0)
        self.assertAlmostEqual(seq[2].get(0, 0), 0.5)

    def test_monotone(self):
        a = HeightField.from_rows([[0.0]])
        b = HeightField.from_rows([[1.0]])
        seq = kf.tween(a, b, 6)
        vals = [f.get(0, 0) for f in seq]
        self.assertEqual(vals, sorted(vals))

    def test_easing_applied(self):
        a = HeightField.from_rows([[0.0]])
        b = HeightField.from_rows([[1.0]])
        lin = kf.tween(a, b, 5, kf.ease_linear)
        ein = kf.tween(a, b, 5, kf.ease_in)
        # ease-in lags the linear curve in the middle
        self.assertLess(ein[2].get(0, 0), lin[2].get(0, 0))

    def test_too_few_frames(self):
        a = HeightField(1, 1)
        b = HeightField(1, 1)
        with self.assertRaises(ValueError):
            kf.tween(a, b, 1)


class TestKeyframeSequence(unittest.TestCase):
    def test_length_no_duplicate_boundaries(self):
        k0 = HeightField.from_rows([[0.0]])
        k1 = HeightField.from_rows([[1.0]])
        k2 = HeightField.from_rows([[0.0]])
        seq = kf.keyframe_sequence([k0, k1, k2], frames_per_segment=3)
        # 3 + (3-1) = 5 frames
        self.assertEqual(len(seq), 5)
        self.assertEqual(seq[0].get(0, 0), 0.0)
        self.assertEqual(seq[2].get(0, 0), 1.0)
        self.assertEqual(seq[-1].get(0, 0), 0.0)

    def test_needs_two_keyframes(self):
        with self.assertRaises(ValueError):
            kf.keyframe_sequence([HeightField(1, 1)], 3)


class TestPulse(unittest.TestCase):
    def test_starts_at_max_gain(self):
        base = HeightField.from_rows([[0.0, 1.0]])
        seq = kf.pulse(base, frames=8, min_gain=0.0, max_gain=1.0, cycles=1.0)
        self.assertEqual(len(seq), 8)
        # frame 0 at max gain 1.0 -> unchanged
        self.assertAlmostEqual(seq[0].get(0, 1), 1.0)

    def test_dips_to_min(self):
        base = HeightField.from_rows([[1.0]])
        seq = kf.pulse(base, frames=8, min_gain=0.0, max_gain=1.0, cycles=1.0)
        # half way through one cycle -> min gain -> flattened to floor
        self.assertAlmostEqual(seq[4].get(0, 0), 0.0)

    def test_gain_bounds(self):
        base = HeightField(1, 1)
        with self.assertRaises(ValueError):
            kf.pulse(base, 4, min_gain=1.0, max_gain=0.5)
        with self.assertRaises(ValueError):
            kf.pulse(base, 0)


class TestLoopPingPong(unittest.TestCase):
    def test_loop_repeats(self):
        frames = [HeightField.from_rows([[0.0]]), HeightField.from_rows([[1.0]])]
        looped = kf.loop(frames, 3)
        self.assertEqual(len(looped), 6)
        self.assertEqual([f.get(0, 0) for f in looped],
                         [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    def test_loop_copies_independent(self):
        frames = [HeightField.from_rows([[0.0]])]
        looped = kf.loop(frames, 2)
        looped[0].set(0, 0, 1.0)
        self.assertEqual(looped[1].get(0, 0), 0.0)

    def test_loop_bad_repeats(self):
        with self.assertRaises(ValueError):
            kf.loop([HeightField(1, 1)], 0)

    def test_ping_pong(self):
        frames = [
            HeightField.from_rows([[0.0]]),
            HeightField.from_rows([[0.5]]),
            HeightField.from_rows([[1.0]]),
        ]
        pp = kf.ping_pong(frames)
        # forward 3 + middle-only reverse (0.5) = 4
        self.assertEqual([f.get(0, 0) for f in pp], [0.0, 0.5, 1.0, 0.5])

    def test_ping_pong_short(self):
        frames = [HeightField.from_rows([[0.0]])]
        self.assertEqual(len(kf.ping_pong(frames)), 1)


if __name__ == "__main__":
    unittest.main()
