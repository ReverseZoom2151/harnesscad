import unittest

from datagen.capture_augment import (
    CURRICULUM,
    CaptureProfile,
    augment_capture,
    augment_render_style,
    augment_skill_geometry,
)


class CaptureAugmentTests(unittest.TestCase):
    sketch = (
        ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
        ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1)),
    )

    def test_seed_is_deterministic(self):
        self.assertEqual(
            augment_capture(self.sketch, seed=42, stage="moderate"),
            augment_capture(self.sketch, seed=42, stage="moderate"),
        )

    def test_clean_stage_preserves_geometry(self):
        result = augment_capture(self.sketch, seed=1, stage="clean")
        self.assertEqual(result.strokes, self.sketch)

    def test_partial_view_reduces_each_stroke(self):
        profile = CaptureProfile(partial_view=0.5)
        result = augment_capture(self.sketch, seed=1, profile=profile)
        self.assertEqual(tuple(map(len, result.strokes)), (2, 2))

    def test_quantization_is_applied(self):
        profile = CaptureProfile(quantization=0.5)
        result = augment_capture((((0.24, 0.26),),), seed=1, profile=profile)
        self.assertEqual(result.strokes, (((0.0, 0.5),),))

    def test_severity_stages_are_ordered(self):
        self.assertLess(CURRICULUM["mild"].jitter, CURRICULUM["moderate"].jitter)
        self.assertLess(
            CURRICULUM["moderate"].missing_segments,
            CURRICULUM["severe"].missing_segments,
        )

    def test_capture_provenance_records_seed_profile_and_source(self):
        event = augment_capture(self.sketch, seed=9, stage="mild").provenance[0]
        self.assertEqual(event.family, "capture")
        self.assertEqual(event.seed, 9)
        self.assertEqual(event.parameters["source_points"], 10)
        self.assertIn("profile", event.to_dict()["parameters"])

    def test_geometry_skill_does_not_add_render_style(self):
        result = augment_skill_geometry(self.sketch, seed=5, level="novice")
        self.assertEqual(result.style, {})
        self.assertEqual(result.provenance[0].family, "geometry_skill")

    def test_render_style_never_changes_geometry(self):
        result = augment_render_style(self.sketch, seed=5, level="novice")
        self.assertEqual(result.strokes, self.sketch)
        self.assertEqual(len(result.style["stroke_widths"]), 2)
        self.assertEqual(result.provenance[0].family, "render_style")

    def test_skill_levels_produce_distinct_style_envelopes(self):
        expert = augment_render_style(self.sketch, seed=3, level="expert")
        novice = augment_render_style(self.sketch, seed=3, level="novice")
        self.assertNotEqual(expert.style, novice.style)

    def test_invalid_profiles_and_names_are_rejected(self):
        with self.assertRaises(ValueError):
            CaptureProfile(occlusion=1.1)
        with self.assertRaises(ValueError):
            augment_capture(self.sketch, seed=1, stage="impossible")
        with self.assertRaises(ValueError):
            augment_skill_geometry(self.sketch, seed=1, level="unknown")

    def test_high_outlier_profile_changes_some_points_deterministically(self):
        result = augment_capture(
            self.sketch,
            seed=2,
            profile=CaptureProfile(outlier_rate=1.0),
        )
        self.assertNotEqual(result.strokes, self.sketch)


if __name__ == "__main__":
    unittest.main()
