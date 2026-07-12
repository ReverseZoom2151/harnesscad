"""Tests for bench.sldprtnet_statistics."""

import unittest

from bench.sldprtnet_statistics import SldprtNetStats, compute_statistics
from dataengine.sldprtnet_record import (
    STANDARD_VIEWS,
    CompositeImage,
    SldprtNetRecord,
)
from reconstruction.sldprtnet_feature_tree import (
    FEATURE_TYPES,
    FeatureNode,
    FeatureTree,
)


def _encoder_txt(*specs):
    """Build an encoder txt from (name, type, parent) specs."""
    nodes = [FeatureNode(n, t, parent=p) for n, t, p in specs]
    return FeatureTree(nodes).to_text()


def _complete_image():
    return CompositeImage(
        view_digests={v: f"d{v}" for v in STANDARD_VIEWS},
        composite_digest="m",
    )


class TestComputeStatistics(unittest.TestCase):
    def test_empty(self):
        stats = compute_statistics([])
        self.assertEqual(stats.num_samples, 0)
        self.assertEqual(stats.mean_features_per_part, 0.0)
        self.assertEqual(stats.fully_aligned_rate, 0.0)
        self.assertEqual(set(stats.feature_frequency), set(FEATURE_TYPES))
        self.assertTrue(all(v == 0 for v in stats.feature_frequency.values()))
        self.assertEqual(stats.most_common_feature, "")

    def test_feature_frequency_from_encoder_txt(self):
        r = SldprtNetRecord(
            id="a",
            encoder_txt=_encoder_txt(
                ("Plane", "RefPlane", None),
                ("S1", "ProfileFeature", "Plane"),
                ("E1", "Extrusion", "S1"),
                ("E2", "Extrusion", "E1"),
            ),
        )
        stats = compute_statistics([r])
        self.assertEqual(stats.feature_frequency["Extrusion"], 2)
        self.assertEqual(stats.feature_frequency["RefPlane"], 1)
        self.assertEqual(stats.feature_frequency["Fillet"], 0)
        self.assertEqual(stats.most_common_feature, "Extrusion")
        self.assertEqual(stats.mean_features_per_part, 4.0)

    def test_complexity_histogram(self):
        # 6 features -> level 2.
        specs = [("F%d" % i, "Extrusion", None) for i in range(6)]
        # make them a valid chain (unique names, parent precedes)
        chain = [("F0", "Extrusion", None)]
        for i in range(1, 6):
            chain.append(("F%d" % i, "Extrusion", "F%d" % (i - 1)))
        r = SldprtNetRecord(id="a", encoder_txt=_encoder_txt(*chain))
        stats = compute_statistics([r])
        self.assertEqual(stats.complexity_histogram[2], 1)
        self.assertEqual(stats.complexity_histogram[1], 0)

    def test_fallback_to_feature_count_field(self):
        r = SldprtNetRecord(id="a", feature_count=3)
        stats = compute_statistics([r])
        self.assertEqual(stats.mean_features_per_part, 3.0)
        self.assertEqual(stats.complexity_histogram[1], 1)

    def test_modality_coverage_and_alignment(self):
        aligned = SldprtNetRecord(
            id="a",
            sldprt_digest="s",
            step_digest="t",
            image=_complete_image(),
            encoder_txt=_encoder_txt(("E", "Extrusion", None)),
            description="d",
        )
        partial = SldprtNetRecord(id="b", sldprt_digest="s", feature_count=1)
        stats = compute_statistics([aligned, partial])
        self.assertEqual(stats.num_samples, 2)
        self.assertEqual(stats.fully_aligned_rate, 0.5)
        self.assertEqual(stats.modality_coverage["sldprt"], 1.0)
        self.assertEqual(stats.modality_coverage["description"], 0.5)

    def test_proportions_sum(self):
        r = SldprtNetRecord(id="a", feature_count=4)
        stats = compute_statistics([r])
        self.assertAlmostEqual(sum(stats.complexity_proportions.values()), 1.0)

    def test_deterministic(self):
        r = SldprtNetRecord(id="a", feature_count=2)
        s1 = compute_statistics([r])
        s2 = compute_statistics([r])
        self.assertEqual(s1.feature_frequency, s2.feature_frequency)
        self.assertEqual(s1.complexity_histogram, s2.complexity_histogram)


class TestStatsObject(unittest.TestCase):
    def test_most_common_empty(self):
        self.assertEqual(SldprtNetStats().most_common_feature, "")


if __name__ == "__main__":
    unittest.main()
