"""Tests for suggest_cots.py — the advisory COTS standard-part matcher.

Deterministic, stdlib-only, no network.
"""

from __future__ import annotations

import unittest

from backends.stub import StubBackend
from loop import HarnessSession

from library.catalog import build_default_catalog
from quality.suggest_cots import suggest_cots, Suggestion, FASTENERS, BEARINGS


def session_factory() -> HarnessSession:
    return HarnessSession(StubBackend())


class TestFastenerSuggestion(unittest.TestCase):
    def test_matching_hole_diameter_suggests_fastener(self):
        # 5.5 mm is the M5 medium clearance hole.
        sug = suggest_cots([{"kind": "hole", "diameter": 5.5}])
        fasteners = [s for s in sug if s.kind == "fastener"]
        self.assertTrue(fasteners)
        self.assertEqual(fasteners[0].standard_part, "ISO 4762 M5")
        self.assertAlmostEqual(fasteners[0].confidence, 1.0)

    def test_near_match_within_tolerance(self):
        sug = suggest_cots([{"kind": "hole", "diameter": 6.7}])  # ~M6 (6.6)
        fasteners = [s for s in sug if s.kind == "fastener"]
        self.assertTrue(fasteners)
        self.assertEqual(fasteners[0].standard_part, "ISO 4762 M6")
        self.assertLess(fasteners[0].confidence, 1.0)

    def test_no_match_for_odd_hole(self):
        sug = suggest_cots([{"kind": "hole", "diameter": 100.0}])
        self.assertEqual([s for s in sug if s.kind == "fastener"], [])

    def test_rationale_mentions_standard(self):
        sug = suggest_cots([{"kind": "hole", "diameter": 3.4}])
        self.assertTrue(any("M3" in s.rationale for s in sug))


class TestBearingSuggestion(unittest.TestCase):
    def test_shaft_diameter_suggests_bearing(self):
        sug = suggest_cots([{"kind": "shaft", "diameter": 8.0}])
        bearings = [s for s in sug if s.kind == "bearing"]
        self.assertTrue(bearings)
        self.assertEqual(bearings[0].standard_part, "bearing 608")

    def test_bore_suggests_bearing(self):
        sug = suggest_cots([{"kind": "bore", "diameter": 10.0}])
        bearings = [s for s in sug if s.kind == "bearing"]
        self.assertTrue(bearings)
        self.assertEqual(bearings[0].standard_part, "bearing 6000")


class TestBackendInput(unittest.TestCase):
    def test_reads_holes_from_backend(self):
        # Build a flange (which cuts a 6.6 mm bolt clearance hole) and scan it.
        from library.parts import flange_card
        backend = StubBackend()
        session = HarnessSession(backend)
        ops = flange_card().instantiate()
        result = session.apply_ops(ops)
        self.assertTrue(result.ok)
        sug = suggest_cots(backend)
        # The 6.6 mm bolt hole -> M6 fastener suggestion.
        self.assertTrue(any(s.standard_part == "ISO 4762 M6"
                            for s in sug if s.kind == "fastener"))

    def test_single_dict_input(self):
        sug = suggest_cots({"kind": "hole", "diameter": 4.5})
        self.assertTrue(any(s.standard_part == "ISO 4762 M4" for s in sug))


class TestCatalogAugmentation(unittest.TestCase):
    def test_catalog_part_suggested_for_bore(self):
        cat = build_default_catalog(session_factory)
        sug = suggest_cots([{"kind": "bore", "diameter": 8.0}], cat)
        catalog_parts = [s for s in sug if s.source == "catalog"]
        self.assertTrue(catalog_parts)
        self.assertEqual(catalog_parts[0].standard_part, "bearing_seat")


class TestSuggestionShape(unittest.TestCase):
    def test_to_dict_and_sorting(self):
        sug = suggest_cots([
            {"kind": "hole", "diameter": 5.5},   # exact M5 -> conf 1.0
            {"kind": "hole", "diameter": 6.7},   # near M6 -> conf < 1.0
        ])
        self.assertTrue(all(isinstance(s, Suggestion) for s in sug))
        confidences = [s.confidence for s in sug]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        d = sug[0].to_dict()
        self.assertIn("standard_part", d)
        self.assertIn("rationale", d)

    def test_tables_present(self):
        self.assertTrue(FASTENERS and BEARINGS)


if __name__ == "__main__":
    unittest.main()
