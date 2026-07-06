import unittest

from bench.llm3dmodel_complexity_scale import (
    REFERENCE_LEVELS, BANDS, band_of, score_description, band_histogram,
    ComplexityEstimate)


class ReferenceTableTests(unittest.TestCase):
    def test_ten_levels(self):
        self.assertEqual(len(REFERENCE_LEVELS), 10)
        self.assertEqual([lv for lv, _, _ in REFERENCE_LEVELS], list(range(1, 11)))

    def test_band_of_ranges(self):
        self.assertEqual(band_of(1), "primitive")
        self.assertEqual(band_of(3), "primitive")
        self.assertEqual(band_of(4), "compositional")
        self.assertEqual(band_of(7), "compositional")
        self.assertEqual(band_of(8), "specialised")
        self.assertEqual(band_of(10), "specialised")

    def test_band_of_out_of_range(self):
        with self.assertRaises(ValueError):
            band_of(0)
        with self.assertRaises(ValueError):
            band_of(11)


class ScoreTests(unittest.TestCase):
    def test_plain_primitive_is_level_1(self):
        est = score_description("Create a cube with side 50mm at origin")
        self.assertEqual(est.level, 1)
        self.assertEqual(est.band, "primitive")

    def test_fillet_raises_level(self):
        est = score_description("Create a cuboid and apply a fillet to all edges")
        self.assertGreater(est.level, 1)
        self.assertIn("fillet", est.features)

    def test_boolean_is_compositional(self):
        est = score_description(
            "Create a box and cylinder and perform a boolean union to merge them")
        self.assertIn(est.band, ("compositional", "specialised"))
        self.assertIn("boolean_union", est.features)

    def test_gear_is_specialised(self):
        est = score_description(
            "Create a parametric gear with an involute profile and 24 teeth")
        self.assertEqual(est.band, "specialised")
        self.assertIn("specialised", est.features)

    def test_frame_with_ribs_high(self):
        est = score_description(
            "Fully constrained parametric frame with reinforcement rib and "
            "mounting holes at each corner")
        self.assertGreaterEqual(est.level, 8)

    def test_clamped_to_10(self):
        est = score_description(
            "parametric hinge gear involute rib chamfer fillet union subtract "
            "cutout multiple constraint assembly")
        self.assertLessEqual(est.level, 10)
        self.assertGreaterEqual(est.level, 1)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            score_description("")

    def test_returns_estimate(self):
        self.assertIsInstance(score_description("cube"), ComplexityEstimate)

    def test_case_insensitive(self):
        self.assertEqual(score_description("GEAR involute").level,
                         score_description("gear INVOLUTE").level)


class HistogramTests(unittest.TestCase):
    def test_histogram_counts(self):
        h = band_histogram([1, 2, 3, 5, 9])
        self.assertEqual(h["primitive"], 3)
        self.assertEqual(h["compositional"], 1)
        self.assertEqual(h["specialised"], 1)

    def test_all_bands_present(self):
        h = band_histogram([1])
        self.assertEqual(set(h), {name for name, _, _ in BANDS})


if __name__ == "__main__":
    unittest.main()
