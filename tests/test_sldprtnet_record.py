"""Tests for dataengine.sldprtnet_record."""

import unittest

from harnesscad.data.dataengine.schemas.sldprtnet_record import (
    MODALITIES,
    STANDARD_VIEWS,
    CompositeImage,
    SldprtNetRecord,
    fully_aligned_rate,
    modality_coverage,
    multimodal_completeness,
)


def _complete_image() -> CompositeImage:
    return CompositeImage(
        view_digests={v: f"d{v}" for v in STANDARD_VIEWS},
        composite_digest="merged",
    )


def _full_record(rid="p1") -> SldprtNetRecord:
    return SldprtNetRecord(
        id=rid,
        sldprt_digest="s",
        step_digest="t",
        image=_complete_image(),
        encoder_txt="Feature Tree:\n",
        description="A bracket.",
        feature_count=4,
    )


class TestCompositeImage(unittest.TestCase):
    def test_seven_views(self):
        self.assertEqual(len(STANDARD_VIEWS), 7)

    def test_complete(self):
        img = _complete_image()
        self.assertTrue(img.is_complete)
        self.assertEqual(img.view_coverage, 1.0)
        self.assertEqual(img.covered_views, STANDARD_VIEWS)

    def test_missing_view_incomplete(self):
        img = CompositeImage(
            view_digests={v: "d" for v in STANDARD_VIEWS[:-1]},
            composite_digest="m",
        )
        self.assertFalse(img.is_complete)
        self.assertAlmostEqual(img.view_coverage, 6 / 7)

    def test_missing_composite_incomplete(self):
        img = CompositeImage(view_digests={v: "d" for v in STANDARD_VIEWS})
        self.assertFalse(img.is_complete)

    def test_unknown_view_rejected(self):
        with self.assertRaises(ValueError):
            CompositeImage(view_digests={"diagonal": "d"})


class TestRecord(unittest.TestCase):
    def test_five_modalities(self):
        self.assertEqual(len(MODALITIES), 5)

    def test_id_required(self):
        with self.assertRaises(ValueError):
            SldprtNetRecord(id="")

    def test_negative_feature_count(self):
        with self.assertRaises(ValueError):
            SldprtNetRecord(id="p", feature_count=-1)

    def test_full_record_aligned(self):
        r = _full_record()
        self.assertTrue(r.is_fully_aligned)
        self.assertEqual(r.completeness, 1.0)
        self.assertEqual(r.present_modalities, frozenset(MODALITIES))
        self.assertEqual(multimodal_completeness(r), 1.0)

    def test_partial_record(self):
        r = SldprtNetRecord(id="p", sldprt_digest="s", encoder_txt="x")
        self.assertFalse(r.is_fully_aligned)
        self.assertEqual(r.completeness, 2 / 5)
        self.assertEqual(r.present_modalities, frozenset({"sldprt", "encoder_txt"}))

    def test_modality_present_keys(self):
        r = SldprtNetRecord(id="p")
        self.assertEqual(set(r.modality_present().keys()), set(MODALITIES))
        self.assertFalse(any(r.modality_present().values()))


class TestAggregates(unittest.TestCase):
    def test_modality_coverage(self):
        recs = [
            _full_record("a"),
            SldprtNetRecord(id="b", sldprt_digest="s", description="d"),
        ]
        cov = modality_coverage(recs)
        self.assertEqual(cov["sldprt"], 1.0)      # both
        self.assertEqual(cov["description"], 1.0)  # both
        self.assertEqual(cov["step"], 0.5)         # only a
        self.assertEqual(cov["image"], 0.5)
        self.assertEqual(cov["encoder_txt"], 0.5)

    def test_coverage_empty(self):
        cov = modality_coverage([])
        self.assertTrue(all(v == 0.0 for v in cov.values()))
        self.assertEqual(set(cov), set(MODALITIES))

    def test_fully_aligned_rate(self):
        recs = [_full_record("a"), SldprtNetRecord(id="b")]
        self.assertEqual(fully_aligned_rate(recs), 0.5)

    def test_fully_aligned_rate_empty(self):
        self.assertEqual(fully_aligned_rate([]), 0.0)


if __name__ == "__main__":
    unittest.main()
