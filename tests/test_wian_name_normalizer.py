import unittest

from harnesscad.domain.library.name_normalizer import (
    CleanDocument,
    clean_corpus,
    clean_document,
    dedupe_names,
    document_features,
    is_default_name,
    is_user_name,
    name_key,
    name_statistics,
    normalize_name,
    strip_instance_suffix,
    tokenize_name,
)


class TestDefaultNameDetection(unittest.TestCase):
    def test_host_defaults(self):
        for name in [
            "Part 1",
            "Part1",
            "part_12",
            "Body3",
            "Extrude 2",
            "Boss-Extrude1",
            "Cut-Extrude12",
            "Sketch 4",
            "Fillet",
            "Pad",
            "Pocket001",
            "Untitled",
            "Front Plane",
            "Solid <3>",
            "Assembly (2)",
        ]:
            self.assertTrue(is_default_name(name), name)

    def test_user_names(self):
        for name in [
            "bracket",
            "M6 bolt head",
            "left_wheel_hub",
            "motorMount",
            "extruder carriage",
            "part of the frame",
        ]:
            self.assertTrue(is_user_name(name), name)

    def test_empty_and_numeric_are_default(self):
        self.assertTrue(is_default_name(""))
        self.assertTrue(is_default_name("   "))
        self.assertTrue(is_default_name(None))
        self.assertTrue(is_default_name("12"))
        self.assertTrue(is_default_name("---"))

    def test_strip_instance_suffix(self):
        self.assertEqual(strip_instance_suffix("bracket <3>"), "bracket")
        self.assertEqual(strip_instance_suffix("bracket (2) <1>"), "bracket")
        self.assertEqual(strip_instance_suffix("bracket:1"), "bracket")
        self.assertEqual(strip_instance_suffix("bracket"), "bracket")


class TestNormalisation(unittest.TestCase):
    def test_normalize_splits_case_and_separators(self):
        self.assertEqual(normalize_name("M6_boltHead <2>"), "m 6 bolt head")
        self.assertEqual(normalize_name("left-wheel/hub"), "left wheel hub")
        self.assertEqual(normalize_name("MotorMount"), "motor mount")

    def test_normalize_is_idempotent(self):
        once = normalize_name("Left_Wheel-Hub")
        self.assertEqual(normalize_name(once), once)

    def test_tokenize(self):
        self.assertEqual(tokenize_name("bolt_head M4"), ["bolt", "head", "m", "4"])
        self.assertEqual(tokenize_name("  "), [])

    def test_name_key_is_order_insensitive(self):
        self.assertEqual(name_key("wheel hub"), name_key("hub_wheel"))
        self.assertNotEqual(name_key("wheel hub"), name_key("wheel axle"))

    def test_dedupe_names_preserves_first_occurrence(self):
        got = dedupe_names(["Wheel Hub", "wheel_hub", "Axle", "", "axle"])
        self.assertEqual(got, ["Wheel Hub", "Axle"])


class TestCorpusCleaning(unittest.TestCase):
    def setUp(self):
        self.corpus = {
            "d2": {
                "body_names": ["Part 1", "bracket", "Bracket", "wheel hub"],
                "feature_names": ["Extrude 1", "chamfer top edge"],
                "document_name": "Skateboard truck",
                "document_description": "a truck",
            },
            "d1": {
                "body_names": ["Part 1", "Part 2"],
                "feature_names": [],
                "document_name": "Untitled",
                "document_description": "",
            },
        }

    def test_clean_document_filters_and_dedupes(self):
        doc = clean_document("d2", self.corpus["d2"])
        self.assertEqual(doc.body_names, ("bracket", "wheel hub"))
        self.assertEqual(doc.feature_names, ("chamfer top edge",))
        self.assertEqual(doc.document_name, "Skateboard truck")
        self.assertEqual(doc.num_parts, 2)

    def test_default_document_name_cleared(self):
        doc = clean_document("d1", self.corpus["d1"])
        self.assertEqual(doc.document_name, "")
        self.assertEqual(doc.body_names, ())

    def test_clean_corpus_sorted_and_deterministic(self):
        cleaned = clean_corpus(self.corpus)
        self.assertEqual(list(cleaned), ["d1", "d2"])
        again = clean_corpus(self.corpus)
        self.assertEqual(
            {k: v.as_dict() for k, v in cleaned.items()},
            {k: v.as_dict() for k, v in again.items()},
        )

    def test_document_features(self):
        doc = clean_document("d2", self.corpus["d2"])
        self.assertEqual(document_features(doc), (1, 1, 1, 1, 1))
        empty = CleanDocument(document_id="x", document_name="", document_description="")
        self.assertEqual(document_features(empty), (0, 0, 0, 0, 0))

    def test_as_dict_roundtrip_schema(self):
        doc = clean_document("d2", self.corpus["d2"])
        d = doc.as_dict()
        self.assertEqual(
            sorted(d),
            ["body_names", "document_description", "document_name", "feature_names"],
        )


class TestNameStatistics(unittest.TestCase):
    def test_statistics(self):
        stats = name_statistics(["Part 1", "wheel hub", "bracket"])
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["default"], 1)
        self.assertEqual(stats["user"], 2)
        self.assertAlmostEqual(stats["default_ratio"], 1 / 3)
        self.assertEqual(stats["vocabulary"], ["1", "bracket", "hub", "part", "wheel"])

    def test_empty(self):
        stats = name_statistics([])
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["default_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
