import unittest

from harnesscad.io.surfaces.confidence import build_overlays, confidence_level


class ConfidenceOverlayTests(unittest.TestCase):
    def test_levels_and_normalization(self):
        overlays = build_overlays([
            {"where": "f2", "code": "thin-wall", "confidence": 0.4,
             "message": "wall may be thin"},
            {"target": "f1", "label": "valid", "confidence": 0.9,
             "source": "brep"},
        ])
        self.assertEqual([item.target for item in overlays], ["f1", "f2"])
        self.assertEqual(overlays[1].level, "low")
        self.assertEqual(overlays[0].to_dict()["source"], "brep")

    def test_invalid_confidence_rejected(self):
        with self.assertRaises(ValueError):
            confidence_level(1.1)
