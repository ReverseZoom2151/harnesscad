"""Tests for fabrication.mfgfeat2_difficulty (paper 122 supplementary tooling)."""

import unittest

from harnesscad.domain.fabrication.feature_difficulty import (
    CONFUSABLE_PAIRS,
    EASY_MAX_FEATURE_COUNT,
    HARD_COUNT_THRESHOLD,
    DifficultyResult,
    PairConfusion,
    classify_difficulty,
    confusion_report,
    distinct_feature_count,
    normalise_counts,
    stratify_dataset,
    total_feature_quantity,
    total_swap_confusion,
)


class NormalisationTests(unittest.TestCase):
    def test_key_canonicalisation_and_merge(self):
        counts = {"Pipe/Tube": 2, "pipe tube": 1, "Gear Teeth": 3}
        norm = normalise_counts(counts)
        self.assertEqual(norm, {"pipe_tube": 3, "gear_teeth": 3})

    def test_zero_dropped(self):
        self.assertEqual(normalise_counts({"hole": 0, "slot": 2}), {"slot": 2})

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            normalise_counts({"hole": -1})

    def test_totals(self):
        counts = {"hole": 4, "slot": 2, "chamfer": 0}
        self.assertEqual(total_feature_quantity(counts), 6)
        self.assertEqual(distinct_feature_count(counts), 2)


class DifficultyEasyTests(unittest.TestCase):
    def test_simple_part_is_easy(self):
        res = classify_difficulty({"hole": 2, "slot": 1, "chamfer": 1})
        self.assertIsInstance(res, DifficultyResult)
        self.assertEqual(res.level, "easy")
        self.assertEqual(res.total_quantity, 4)
        self.assertEqual(res.excluded_present, ())
        self.assertEqual(res.hard_present, ())

    def test_boundary_count_is_medium_not_easy(self):
        # exactly EASY_MAX_FEATURE_COUNT features -> no longer "fewer than six"
        res = classify_difficulty({"hole": EASY_MAX_FEATURE_COUNT})
        self.assertEqual(res.level, "medium")

    def test_just_below_boundary_is_easy(self):
        res = classify_difficulty({"hole": EASY_MAX_FEATURE_COUNT - 1})
        self.assertEqual(res.level, "easy")


class DifficultyMediumTests(unittest.TestCase):
    def test_excluded_feature_forces_medium(self):
        res = classify_difficulty({"hole": 1, "thread": 1})
        self.assertEqual(res.level, "medium")
        self.assertIn("thread", res.excluded_present)

    def test_sheet_metal_bend_forces_medium(self):
        res = classify_difficulty({"sheet metal bend": 2, "hole": 1})
        self.assertEqual(res.level, "medium")
        self.assertIn("sheet_metal_bend", res.excluded_present)


class DifficultyHardTests(unittest.TestCase):
    def test_casting_draft_forces_hard(self):
        res = classify_difficulty({"hole": 1, "draft": 1})
        self.assertEqual(res.level, "hard")
        self.assertEqual(res.hard_present, ("draft",))

    def test_freeform_forces_hard(self):
        res = classify_difficulty({"depression": 1, "protrusion": 2})
        self.assertEqual(res.level, "hard")
        self.assertEqual(res.hard_present, ("depression", "protrusion"))

    def test_numerous_features_forces_hard(self):
        res = classify_difficulty({"hole": HARD_COUNT_THRESHOLD})
        self.assertEqual(res.level, "hard")

    def test_hard_indicator_beats_excluded(self):
        # both a casting feature and a thread -> hard wins
        res = classify_difficulty({"thread": 1, "draft": 1})
        self.assertEqual(res.level, "hard")


class DifficultyParamTests(unittest.TestCase):
    def test_invalid_easy_max(self):
        with self.assertRaises(ValueError):
            classify_difficulty({"hole": 1}, easy_max_count=0)

    def test_invalid_hard_threshold(self):
        with self.assertRaises(ValueError):
            classify_difficulty({"hole": 1}, easy_max_count=6, hard_count_threshold=3)

    def test_custom_thresholds(self):
        res = classify_difficulty({"hole": 3}, easy_max_count=3)
        self.assertEqual(res.level, "medium")


class StratifyTests(unittest.TestCase):
    def test_bucketing_and_sorting(self):
        parts = {
            "p_easy": {"hole": 2},
            "p_med": {"rib": 1},
            "p_hard": {"draft": 1},
            "a_easy": {"slot": 1},
        }
        buckets = stratify_dataset(parts)
        self.assertEqual(buckets["easy"], ["a_easy", "p_easy"])
        self.assertEqual(buckets["medium"], ["p_med"])
        self.assertEqual(buckets["hard"], ["p_hard"])

    def test_all_levels_present_even_if_empty(self):
        buckets = stratify_dataset({"x": {"hole": 1}})
        self.assertEqual(set(buckets), {"easy", "medium", "hard"})
        self.assertEqual(buckets["medium"], [])
        self.assertEqual(buckets["hard"], [])


class ConfusionTests(unittest.TestCase):
    def test_perfect_prediction_no_swap(self):
        gt = {"chamfer": 3, "fillet": 2}
        rep = confusion_report(gt, gt)
        pc = rep[("chamfer", "fillet")]
        self.assertIsInstance(pc, PairConfusion)
        self.assertEqual(pc.swap_magnitude, 0)

    def test_pure_swap_detected(self):
        # model calls chamfers "fillet": under chamfer by 3, over fillet by 3
        gt = {"chamfer": 3, "fillet": 0}
        pr = {"chamfer": 0, "fillet": 3}
        pc = confusion_report(gt, pr)[("chamfer", "fillet")]
        self.assertEqual(pc.under["chamfer"], 3)
        self.assertEqual(pc.over["fillet"], 3)
        self.assertEqual(pc.swap_magnitude, 3)

    def test_partial_swap(self):
        gt = {"pipe_tube": 4, "boss": 1}
        pr = {"pipe_tube": 2, "boss": 2}
        # under pipe_tube by 2, over boss by 1 -> swap = min(1,2) = 1
        pc = confusion_report(gt, pr)[("boss", "pipe_tube")]
        self.assertEqual(pc.swap_magnitude, 1)

    def test_no_swap_when_both_over(self):
        gt = {"chamfer": 1, "fillet": 1}
        pr = {"chamfer": 3, "fillet": 3}
        pc = confusion_report(gt, pr)[("chamfer", "fillet")]
        self.assertEqual(pc.swap_magnitude, 0)

    def test_total_swap_confusion(self):
        gt = {"chamfer": 2, "fillet": 0, "pipe_tube": 2, "boss": 0}
        pr = {"chamfer": 0, "fillet": 2, "pipe_tube": 0, "boss": 2}
        self.assertEqual(total_swap_confusion(gt, pr), 4)

    def test_report_keys_are_sorted_pairs(self):
        rep = confusion_report({"chamfer": 1}, {"fillet": 1})
        for key in rep:
            self.assertEqual(key, tuple(sorted(key)))
        self.assertEqual(len(rep), len(CONFUSABLE_PAIRS))


if __name__ == "__main__":
    unittest.main()
