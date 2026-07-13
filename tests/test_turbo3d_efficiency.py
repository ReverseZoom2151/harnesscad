"""Tests for bench.turbo3d_efficiency."""

import unittest

from harnesscad.eval.bench.harness.turbo3d_efficiency import (
    PipelineLatency,
    compare_methods_by_time,
    latent_speedup_fraction,
    sequence_length_ratio,
    speedup_percent,
    step_speedup,
    token_count,
    under_one_second,
)


class StepSpeedupTest(unittest.TestCase):
    def test_ratio(self):
        self.assertAlmostEqual(step_speedup(200, 4), 50.0)

    def test_bad(self):
        with self.assertRaises(ValueError):
            step_speedup(4, 0)


class TokenCountTest(unittest.TestCase):
    def test_pixel(self):
        # 256 / 16 = 16 -> 256 tokens
        self.assertEqual(token_count(256, 16, downsample=1), 256)

    def test_latent_downsample(self):
        # 256 / 8 / 16 = 2 -> 4 tokens
        self.assertEqual(token_count(256, 16, downsample=8), 4)

    def test_not_divisible_raises(self):
        with self.assertRaises(ValueError):
            token_count(100, 16, downsample=1)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            token_count(0, 16)


class SequenceLengthRatioTest(unittest.TestCase):
    def test_halving_via_downsample(self):
        # downsample sqrt(2) not integer; use downsample where ratio=2 not possible.
        # ratio equals downsample^2
        self.assertAlmostEqual(sequence_length_ratio(256, 16, 8), 64.0)

    def test_ratio_equals_downsample_squared(self):
        self.assertAlmostEqual(sequence_length_ratio(256, 8, 2), 4.0)

    def test_no_downsample_is_one(self):
        self.assertAlmostEqual(sequence_length_ratio(256, 16, 1), 1.0)


class PipelineLatencyTest(unittest.TestCase):
    def test_latent_skips_decode(self):
        lat = PipelineLatency(generate=0.25, decode=0.10, reconstruct=0.10)
        self.assertAlmostEqual(lat.total(latent=True), 0.35)
        self.assertAlmostEqual(lat.total(latent=False), 0.45)

    def test_negative_raises(self):
        lat = PipelineLatency(generate=-1.0, decode=0.0, reconstruct=0.0)
        with self.assertRaises(ValueError):
            lat.total(latent=True)


class SpeedupTest(unittest.TestCase):
    def test_res256_fraction_matches_paper(self):
        # pixel 0.45 -> latent 0.35, ~22% speedup
        lat = PipelineLatency(generate=0.25, decode=0.10, reconstruct=0.10)
        self.assertAlmostEqual(latent_speedup_fraction(lat), 0.10 / 0.45)
        self.assertAlmostEqual(speedup_percent(lat), 100.0 * 0.10 / 0.45, places=5)
        # rounds to ~22%
        self.assertEqual(round(speedup_percent(lat)), 22)

    def test_res512_fraction_matches_paper(self):
        # pixel 1.62 -> latent 1.28, decode 0.34, ~21%
        lat = PipelineLatency(generate=0.94, decode=0.34, reconstruct=0.34)
        self.assertAlmostEqual(lat.total(latent=False), 1.62)
        self.assertAlmostEqual(lat.total(latent=True), 1.28)
        self.assertEqual(round(speedup_percent(lat)), 21)

    def test_zero_pixel_raises(self):
        lat = PipelineLatency(generate=0.0, decode=0.0, reconstruct=0.0)
        with self.assertRaises(ValueError):
            latent_speedup_fraction(lat)


class UnderOneSecondTest(unittest.TestCase):
    def test_turbo3d_qualifies(self):
        self.assertTrue(under_one_second(0.35))

    def test_slow_method_fails(self):
        self.assertFalse(under_one_second(6.56))


class CompareMethodsTest(unittest.TestCase):
    def test_turbo3d_is_fastest(self):
        # Table 1 inference times
        latencies = {
            "TripoSR": 1.19,
            "SV3D": 12.52,
            "Instant3D": 15.02,
            "LGM": 6.56,
            "Turbo3D": 0.35,
        }
        order = compare_methods_by_time(latencies)
        self.assertEqual(order[0], "Turbo3D")
        self.assertEqual(order[-1], "Instant3D")

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            compare_methods_by_time({"x": -1.0})


if __name__ == "__main__":
    unittest.main()
