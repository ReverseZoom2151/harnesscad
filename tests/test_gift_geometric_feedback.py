import unittest

from harnesscad.data.dataengine.gift_geometric_feedback import (
    Candidate, augment_example, build_augmented_dataset, build_fda_dataset,
    build_srs_dataset, feedback_category, fda_indicator, geometric_agreement,
    geometric_feedback, partition_candidates, srs_indicator,
    TAU_LOW, TAU_MATCH, TAU_VALID,
)


class TestIndicators(unittest.TestCase):
    def test_srs_band_is_half_open(self):
        self.assertEqual(srs_indicator(0.90), 1)
        self.assertEqual(srs_indicator(0.985), 1)
        self.assertEqual(srs_indicator(0.99), 0)   # match, excluded
        self.assertEqual(srs_indicator(0.89), 0)   # near-miss

    def test_fda_band_is_half_open(self):
        self.assertEqual(fda_indicator(0.50), 1)
        self.assertEqual(fda_indicator(0.89), 1)
        self.assertEqual(fda_indicator(0.90), 0)   # valid, not fda
        self.assertEqual(fda_indicator(0.49), 0)   # reject

    def test_bands_are_disjoint(self):
        for iou in (0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0):
            self.assertLessEqual(srs_indicator(iou) + fda_indicator(iou), 1)


class TestCategory(unittest.TestCase):
    def test_all_four_bands(self):
        self.assertEqual(feedback_category(0.3), "reject")
        self.assertEqual(feedback_category(0.6), "near_miss")
        self.assertEqual(feedback_category(0.95), "valid")
        self.assertEqual(feedback_category(0.999), "match")

    def test_bad_thresholds_raise(self):
        with self.assertRaises(ValueError):
            feedback_category(0.5, tau_low=0.9, tau_valid=0.5, tau_match=0.99)


class TestGeometricFeedback(unittest.TestCase):
    def test_discrepancy_is_complement(self):
        fb = geometric_feedback(0.7)
        self.assertAlmostEqual(fb["agreement"], 0.7)
        self.assertAlmostEqual(fb["discrepancy"], 0.3)
        self.assertEqual(fb["category"], "near_miss")
        self.assertTrue(fb["feeds_fda"])
        self.assertFalse(fb["feeds_srs"])

    def test_agreement_uses_injected_kernel(self):
        score = geometric_agreement("gen", "gt", lambda a, b: 0.5)
        self.assertEqual(score, 0.5)

    def test_agreement_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            geometric_agreement("g", "t", lambda a, b: 1.5)


class TestPartition(unittest.TestCase):
    def setUp(self):
        self.cands = [
            Candidate("a", 0.2),
            Candidate("b", 0.6),
            Candidate("c", 0.95),
            Candidate("d", 0.995),
            Candidate("e", 0.92),
        ]

    def test_partition_counts(self):
        b = partition_candidates(self.cands)
        self.assertEqual(len(b["reject"]), 1)
        self.assertEqual(len(b["near_miss"]), 1)
        self.assertEqual(len(b["valid"]), 2)
        self.assertEqual(len(b["match"]), 1)

    def test_valid_sorted_descending(self):
        b = partition_candidates(self.cands)
        ious = [c.iou for c in b["valid"]]
        self.assertEqual(ious, sorted(ious, reverse=True))

    def test_candidate_iou_range(self):
        with self.assertRaises(ValueError):
            Candidate("z", 1.5)


class TestSRS(unittest.TestCase):
    def test_excludes_ground_truth_and_dedups(self):
        cands = [
            Candidate("gt_code", 0.93),   # equals GT -> excluded
            Candidate("alt1", 0.95),
            Candidate("alt1", 0.91),      # duplicate -> collapsed
            Candidate("alt2", 0.92),
            Candidate("bad", 0.6),        # near-miss -> not SRS
            Candidate("exact", 0.995),    # match -> not SRS
        ]
        pairs = build_srs_dataset("img7", "gt_code", cands)
        progs = [p[1] for p in pairs]
        self.assertEqual(progs, ["alt1", "alt2"])   # sorted by desc IoU
        self.assertTrue(all(p[0] == "img7" for p in pairs))

    def test_dedup_keeps_highest_iou(self):
        cands = [Candidate("alt", 0.91), Candidate("alt", 0.97)]
        pairs = build_srs_dataset("i", "gt", cands)
        self.assertEqual(len(pairs), 1)


class TestFDA(unittest.TestCase):
    def test_renders_near_miss_paired_with_gt(self):
        cands = [
            Candidate("miss1", 0.6),
            Candidate("miss2", 0.8),
            Candidate("good", 0.95),      # valid -> not FDA
            Candidate("junk", 0.2),       # reject -> not FDA
        ]
        pairs = build_fda_dataset("img", "GTCODE", cands,
                                  render_fn=lambda p: "render::" + p)
        self.assertEqual(len(pairs), 2)
        for synth, code in pairs:
            self.assertTrue(synth.startswith("render::"))
            self.assertEqual(code, "GTCODE")

    def test_fda_input_is_synthetic_not_original(self):
        cands = [Candidate("m", 0.7)]
        pairs = build_fda_dataset("orig_img", "gt", cands,
                                  render_fn=lambda p: ("synth", p))
        self.assertEqual(pairs[0][0], ("synth", "m"))


class TestAugmentExample(unittest.TestCase):
    def test_counts_and_no_fda_without_renderer(self):
        cands = [Candidate("v", 0.93), Candidate("m", 0.7)]
        rec = augment_example("img", "gt", cands)
        self.assertEqual(rec.counts["srs"], 1)
        self.assertEqual(rec.counts["fda"], 0)
        rec2 = augment_example("img", "gt", cands, render_fn=lambda p: p)
        self.assertEqual(rec2.counts["fda"], 1)


class TestBuildAugmentedDataset(unittest.TestCase):
    def test_combines_base_srs_fda(self):
        base = [("i1", "gt1"), ("i2", "gt2")]
        sampled = {
            "i1": [Candidate("alt", 0.93), Candidate("near", 0.7)],
            "i2": [Candidate("gt2", 0.995)],   # only exact match -> nothing
        }
        out = build_augmented_dataset(base, sampled, render_fn=lambda p: "r:" + p)
        self.assertEqual(out["counts"]["base"], 2)
        self.assertEqual(out["counts"]["srs"], 1)
        self.assertEqual(out["counts"]["fda"], 1)
        self.assertEqual(out["counts"]["total"], 4)
        # base pairs come first
        self.assertEqual(out["pairs"][:2], base)


if __name__ == "__main__":
    unittest.main()
