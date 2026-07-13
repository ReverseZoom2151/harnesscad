"""Tests for dataengine.sldprtnet_cleaning."""

import unittest

from harnesscad.data.dataengine.sldprtnet_cleaning import (
    REASON_DUPLICATE,
    REASON_NO_SUPPORTED,
    CleaningReport,
    RawPart,
    clean,
    unsupported_feature_ratio,
)


class TestRawPart(unittest.TestCase):
    def test_supported_features(self):
        p = RawPart("a", ("ProfileFeature", "Weldment", "Extrusion"))
        self.assertEqual(p.supported_features, ("ProfileFeature", "Extrusion"))
        self.assertTrue(p.has_supported_feature)

    def test_no_supported(self):
        p = RawPart("a", ("Weldment", "Cosmetic"))
        self.assertFalse(p.has_supported_feature)

    def test_empty_has_none(self):
        self.assertFalse(RawPart("a", ()).has_supported_feature)

    def test_signature_order_independent(self):
        p1 = RawPart("a", ("Extrusion", "Fillet", "Extrusion"))
        p2 = RawPart("b", ("Fillet", "Extrusion", "Extrusion"))
        self.assertEqual(p1.signature(), p2.signature())

    def test_signature_ignores_unsupported(self):
        p1 = RawPart("a", ("Extrusion",))
        p2 = RawPart("b", ("Extrusion", "Weldment"))
        self.assertEqual(p1.signature(), p2.signature())


class TestClean(unittest.TestCase):
    def test_drops_no_supported(self):
        parts = [
            RawPart("a", ("Extrusion",)),
            RawPart("b", ("Weldment",)),
            RawPart("c", ()),
        ]
        rep = clean(parts)
        self.assertEqual(rep.retained_count, 1)
        self.assertEqual(rep.retained[0].id, "a")
        self.assertEqual(rep.dropped[REASON_NO_SUPPORTED], 2)

    def test_dedup_first_wins(self):
        parts = [
            RawPart("a", ("Extrusion", "Fillet")),
            RawPart("b", ("Fillet", "Extrusion")),  # same signature -> dup
            RawPart("c", ("Extrusion",)),           # different -> kept
        ]
        rep = clean(parts)
        self.assertEqual([p.id for p in rep.retained], ["a", "c"])
        self.assertEqual(rep.dropped[REASON_DUPLICATE], 1)

    def test_retention_yield(self):
        parts = [
            RawPart("a", ("Extrusion",)),
            RawPart("b", ()),
            RawPart("c", ("Fillet",)),
            RawPart("d", ()),
        ]
        rep = clean(parts)
        self.assertEqual(rep.input_count, 4)
        self.assertEqual(rep.retained_count, 2)
        self.assertEqual(rep.dropped_count, 2)
        self.assertEqual(rep.retention_yield, 0.5)

    def test_empty_input(self):
        rep = clean([])
        self.assertEqual(rep.retention_yield, 0.0)
        self.assertEqual(rep.retained_count, 0)

    def test_report_default(self):
        rep = CleaningReport()
        self.assertEqual(rep.dropped[REASON_NO_SUPPORTED], 0)
        self.assertEqual(rep.dropped[REASON_DUPLICATE], 0)


class TestUnsupportedRatio(unittest.TestCase):
    def test_ratio(self):
        parts = [
            RawPart("a", ("Extrusion", "Weldment")),
            RawPart("b", ("Fillet",)),
        ]
        # 1 unsupported (Weldment) of 3 total.
        self.assertAlmostEqual(unsupported_feature_ratio(parts), 1 / 3)

    def test_empty(self):
        self.assertEqual(unsupported_feature_ratio([]), 0.0)
        self.assertEqual(unsupported_feature_ratio([RawPart("a", ())]), 0.0)


if __name__ == "__main__":
    unittest.main()
