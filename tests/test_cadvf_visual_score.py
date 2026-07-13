import unittest

from harnesscad.data.dataengine.cadvf_visual_score import (
    Component, collision_penalty, distribution_score, shape_quality_score,
    shape_quantity_score, spacing_penalty, visual_score, MAX_SCORE,
)


def box(cx, cy, cz, sx=1.0, sy=1.0, sz=1.0):
    """Component centred at (cx,cy,cz) with the given full extents."""
    return Component(lo=(cx - sx / 2, cy - sy / 2, cz - sz / 2),
                     hi=(cx + sx / 2, cy + sy / 2, cz + sz / 2))


class TestComponent(unittest.TestCase):
    def test_rejects_bad_box(self):
        with self.assertRaises(ValueError):
            Component(lo=(1, 0, 0), hi=(0, 0, 0))

    def test_rejects_wrong_arity(self):
        with self.assertRaises(ValueError):
            Component(lo=(0, 0), hi=(1, 1))

    def test_extents_center_volume(self):
        c = box(0, 0, 0, 2, 4, 6)
        self.assertEqual(c.extents, (2.0, 4.0, 6.0))
        self.assertEqual(c.center, (0.0, 0.0, 0.0))
        self.assertEqual(c.volume, 48.0)

    def test_degenerate_flag(self):
        flat = Component(lo=(0, 0, 0), hi=(1, 1, 0))
        self.assertTrue(flat.is_degenerate)
        self.assertFalse(box(0, 0, 0).is_degenerate)


class TestShapeQuality(unittest.TestCase):
    def test_empty_scores_zero(self):
        self.assertEqual(shape_quality_score([]), 0.0)

    def test_all_well_formed(self):
        self.assertEqual(shape_quality_score([box(0, 0, 0), box(5, 0, 0)]), 1.0)

    def test_degenerate_penalised(self):
        flat = Component(lo=(0, 0, 0), hi=(1, 1, 0))
        self.assertEqual(shape_quality_score([box(0, 0, 0), flat]), 0.5)

    def test_extreme_sliver_penalised(self):
        sliver = Component(lo=(0, 0, 0), hi=(100, 1, 1))  # ratio 100 > limit
        self.assertEqual(shape_quality_score([sliver]), 0.0)


class TestShapeQuantity(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(shape_quantity_score(3, 3), 1.0)

    def test_linear_decay(self):
        self.assertAlmostEqual(shape_quantity_score(2, 4), 0.5)
        self.assertAlmostEqual(shape_quantity_score(6, 4), 0.5)

    def test_floor_at_zero(self):
        self.assertEqual(shape_quantity_score(20, 2), 0.0)

    def test_expected_zero(self):
        self.assertEqual(shape_quantity_score(0, 0), 1.0)
        self.assertEqual(shape_quantity_score(1, 0), 0.0)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            shape_quantity_score(-1, 3)


class TestDistribution(unittest.TestCase):
    def test_single_component_is_ideal(self):
        self.assertEqual(distribution_score([box(0, 0, 0)]), 1.0)

    def test_empty_scores_zero(self):
        self.assertEqual(distribution_score([]), 0.0)

    def test_well_spaced_no_penalty(self):
        # two unit boxes 2 units apart -> gap ~2 size < limit 3, no collision.
        comps = [box(0, 0, 0), box(2, 0, 0)]
        self.assertEqual(collision_penalty(comps), 0.0)
        self.assertEqual(spacing_penalty(comps), 0.0)
        self.assertEqual(distribution_score(comps), 1.0)

    def test_full_overlap_max_collision(self):
        comps = [box(0, 0, 0), box(0, 0, 0)]
        self.assertAlmostEqual(collision_penalty(comps), 1.0)
        self.assertEqual(distribution_score(comps), 0.0)

    def test_excessive_spacing_penalised(self):
        # unit boxes 10 units apart: gap 10 >> limit 3 -> spacing penalty.
        comps = [box(0, 0, 0), box(10, 0, 0)]
        self.assertGreater(spacing_penalty(comps), 0.0)
        self.assertLess(distribution_score(comps), 1.0)


class TestVisualScore(unittest.TestCase):
    def test_perfect_object(self):
        comps = [box(0, 0, 0), box(3, 0, 0)]
        out = visual_score(comps, expected_count=2)
        self.assertEqual(out["shape_quality"], 1.0)
        self.assertEqual(out["shape_quantity"], 1.0)
        self.assertEqual(out["distribution"], 1.0)
        self.assertAlmostEqual(out["score"], MAX_SCORE)

    def test_score_in_range(self):
        comps = [box(0, 0, 0), box(0, 0, 0), Component(lo=(0, 0, 0), hi=(1, 1, 0))]
        out = visual_score(comps, expected_count=5)
        self.assertGreaterEqual(out["score"], 0.0)
        self.assertLessEqual(out["score"], MAX_SCORE)
        self.assertLess(out["score"], MAX_SCORE)

    def test_deterministic(self):
        comps = [box(0, 0, 0), box(4, 0, 0)]
        self.assertEqual(visual_score(comps, 2), visual_score(comps, 2))

    def test_weight_renormalisation(self):
        comps = [box(0, 0, 0), box(3, 0, 0)]
        a = visual_score(comps, 2, weights=(1, 1, 1))
        b = visual_score(comps, 2, weights=(2, 2, 2))
        self.assertEqual(a["combined"], b["combined"])

    def test_bad_weights_rejected(self):
        with self.assertRaises(ValueError):
            visual_score([box(0, 0, 0)], 1, weights=(0, 0, 0))
        with self.assertRaises(ValueError):
            visual_score([box(0, 0, 0)], 1, weights=(1, 1))

    def test_count_mismatch_lowers_score(self):
        comps = [box(0, 0, 0), box(3, 0, 0)]
        good = visual_score(comps, expected_count=2)
        bad = visual_score(comps, expected_count=6)
        self.assertLess(bad["score"], good["score"])


if __name__ == "__main__":
    unittest.main()
