import unittest

from harnesscad.data.dataengine.audit.bias_audit import DIMENSIONS, audit_bias


class BiasAuditTests(unittest.TestCase):
    def corpus(self):
        return [
            {"metadata": {
                "source": "synthetic", "geography": "eu",
                "process": "cnc", "geometry_family": "bracket",
            }},
            {"metadata": {
                "source": "synthetic", "geography": "eu",
                "process": "cnc", "geometry_family": "bracket",
            }},
            {"metadata": {
                "source": "real", "geography": "us",
                "process": "casting", "geometry_family": "housing",
            }},
            {"metadata": {
                "source": "synthetic", "geography": "eu",
                "process": "cnc",
            }},
        ]

    def test_collects_all_dimensions_and_missing_values(self):
        report = audit_bias(self.corpus(), minimum_share=0)
        self.assertEqual(set(report.distributions), set(DIMENSIONS))
        self.assertEqual(report.distributions["source"]["synthetic"], 3)
        self.assertEqual(report.missing["geometry_family"], 1)

    def test_flags_excessive_missing_metadata(self):
        report = audit_bias(self.corpus(), minimum_share=0, maximum_missing_share=0.2)
        codes = {(w.dimension, w.code) for w in report.warnings}
        self.assertIn(("geometry_family", "missing-metadata"), codes)

    def test_target_skew_is_reported(self):
        report = audit_bias(
            self.corpus(),
            minimum_share=0,
            maximum_missing_share=1,
            targets={"source": {"real": 0.75, "synthetic": 0.25}},
            target_tolerance=0.1,
        )
        by_value = {(w.value, w.code) for w in report.warnings}
        self.assertIn(("real", "below-target"), by_value)
        self.assertIn(("synthetic", "above-target"), by_value)

    def test_balanced_complete_corpus_is_ok(self):
        rows = [
            {key: value for key, value in zip(DIMENSIONS, values)}
            for values in [
                ("a", "eu", "cnc", "plate"),
                ("b", "us", "cast", "housing"),
            ]
        ]
        report = audit_bias(rows, minimum_share=0.4, maximum_missing_share=0)
        self.assertTrue(report.ok)
        self.assertEqual(report.to_dict()["n_items"], 2)

    def test_invalid_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            audit_bias([], minimum_share=2)


if __name__ == "__main__":
    unittest.main()
