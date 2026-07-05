"""Tests for generation.flexcad2_appendix (FlexCAD App. A.1 / A.4)."""

from __future__ import annotations

import math
import unittest

from generation.flexcad2_appendix import (
    CIRCLE_VARIANTS,
    DEFAULT_PV_FLOOR,
    SamplingConfig,
    circle_points_on_circumference,
    circle_repr_token_count,
    decode_circle,
    diversity_score,
    encode_circle,
    pv_frontier,
    select_sampling_config,
)


class TestCircleRepr(unittest.TestCase):
    def test_token_counts(self):
        self.assertEqual(circle_repr_token_count("center_radius"), 3)
        self.assertEqual(circle_repr_token_count("diameter"), 4)
        self.assertEqual(circle_repr_token_count("four_points"), 8)

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            circle_repr_token_count("nope")
        with self.assertRaises(ValueError):
            encode_circle((0.0, 0.0), 1.0, "nope")

    def test_roundtrip_all_variants(self):
        center, radius = (3.0, -2.0), 5.0
        for variant in CIRCLE_VARIANTS:
            enc = encode_circle(center, radius, variant)
            self.assertEqual(len(enc), circle_repr_token_count(variant))
            (cx, cy), r = decode_circle(enc, variant)
            self.assertAlmostEqual(cx, center[0], places=9)
            self.assertAlmostEqual(cy, center[1], places=9)
            self.assertAlmostEqual(r, radius, places=9)

    def test_four_points_layout(self):
        enc = encode_circle((0.0, 0.0), 2.0, "four_points")
        self.assertEqual(
            enc, [2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, -2.0]
        )

    def test_diameter_opposed(self):
        enc = encode_circle((1.0, 1.0), 3.0, "diameter")
        self.assertEqual(enc, [4.0, 1.0, -2.0, 1.0])

    def test_negative_radius_raises(self):
        with self.assertRaises(ValueError):
            encode_circle((0.0, 0.0), -1.0, "four_points")

    def test_decode_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            decode_circle([1.0, 2.0], "four_points")

    def test_points_on_circumference_uniform(self):
        pts = circle_points_on_circumference((0.0, 0.0), 1.0, 4)
        self.assertEqual(len(pts), 4)
        for x, y in pts:
            self.assertAlmostEqual(math.hypot(x, y), 1.0, places=9)
        # Consecutive points are 90 degrees apart.
        self.assertAlmostEqual(pts[0][0], 1.0, places=9)
        self.assertAlmostEqual(pts[1][1], 1.0, places=9)

    def test_points_on_circumference_bad_n(self):
        with self.assertRaises(ValueError):
            circle_points_on_circumference((0.0, 0.0), 1.0, 0)


def _cfg(tau, top_p, pv, cov=0.6, mmd=1.2, jsd=0.9, novel=0.9, unique=0.9):
    return SamplingConfig(
        tau, top_p,
        {"cov": cov, "mmd": mmd, "jsd": jsd, "novel": novel, "unique": unique},
        pv,
    )


class TestDiversityScore(unittest.TestCase):
    def test_direction_signs(self):
        # Higher COV improves the score; higher MMD/JSD worsen it.
        base = diversity_score({"cov": 0.5, "mmd": 1.0, "jsd": 1.0})
        better_cov = diversity_score({"cov": 0.6, "mmd": 1.0, "jsd": 1.0})
        worse_mmd = diversity_score({"cov": 0.5, "mmd": 2.0, "jsd": 1.0})
        self.assertGreater(better_cov, base)
        self.assertLess(worse_mmd, base)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            diversity_score({"bogus": 1.0})

    def test_weights_applied(self):
        m = {"cov": 1.0}
        self.assertAlmostEqual(diversity_score(m, {"cov": 0.5}), 0.5, places=9)
        self.assertAlmostEqual(diversity_score(m, {}), 0.0, places=9)


class TestSelectSamplingConfig(unittest.TestCase):
    def test_default_floor(self):
        self.assertEqual(DEFAULT_PV_FLOOR, 0.90)

    def test_respects_pv_floor(self):
        # High-diversity config violates PV floor; must not be chosen.
        high_div_low_pv = _cfg(1.5, 0.99, pv=0.70, cov=0.9, mmd=0.5, jsd=0.4)
        ok = _cfg(1.1, 0.9, pv=0.934, cov=0.656, mmd=1.19, jsd=0.82)
        chosen = select_sampling_config([high_div_low_pv, ok])
        self.assertIs(chosen, ok)

    def test_picks_highest_diversity_above_floor(self):
        a = _cfg(0.9, 0.8, pv=0.967, cov=0.60, mmd=1.30, jsd=0.90)
        b = _cfg(1.1, 0.9, pv=0.934, cov=0.70, mmd=1.10, jsd=0.80)
        chosen = select_sampling_config([a, b])
        self.assertIs(chosen, b)

    def test_no_candidate_meets_floor_raises(self):
        low = _cfg(1.3, 0.9, pv=0.50)
        with self.assertRaises(ValueError):
            select_sampling_config([low], pv_floor=0.90)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            select_sampling_config([])

    def test_tie_breaks_toward_higher_pv(self):
        metrics = {"cov": 0.6, "mmd": 1.2, "jsd": 0.9, "novel": 0.9, "unique": 0.9}
        a = SamplingConfig(1.0, 0.9, dict(metrics), 0.92)
        b = SamplingConfig(1.2, 0.9, dict(metrics), 0.95)
        chosen = select_sampling_config([a, b])
        self.assertIs(chosen, b)


class TestPvFrontier(unittest.TestCase):
    def test_dominated_config_excluded(self):
        strong = _cfg(1.1, 0.9, pv=0.95, cov=0.9, mmd=0.5, jsd=0.4)
        weak = _cfg(1.0, 0.8, pv=0.90, cov=0.5, mmd=1.5, jsd=1.2)  # dominated
        front = pv_frontier([strong, weak])
        self.assertIn(strong, front)
        self.assertNotIn(weak, front)

    def test_tradeoff_pair_both_on_frontier(self):
        # Classic trade-off: one high PV/low diversity, one low PV/high diversity.
        high_pv = _cfg(0.9, 0.8, pv=0.97, cov=0.60, mmd=1.30, jsd=0.90)
        high_div = _cfg(1.3, 1.0, pv=0.89, cov=0.90, mmd=0.50, jsd=0.40)
        front = pv_frontier([high_pv, high_div])
        self.assertIn(high_pv, front)
        self.assertIn(high_div, front)
        # Sorted by descending PV.
        self.assertEqual(front[0], high_pv)


if __name__ == "__main__":
    unittest.main()
