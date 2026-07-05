"""Tests for the parametric parts library (library/parts.py + library/catalog.py).

Deterministic, stdlib-only, no network. Every Model Card must build cleanly on a
StubBackend HarnessSession; the catalog's range validation, function-tag
retrieval and Voyager admission gate are exercised directly.
"""

from __future__ import annotations

import unittest

from backends.stub import StubBackend
from cisp.ops import NewSketch, Extrude
from loop import HarnessSession

from library.parts import (
    ModelCard, default_cards, flange_card, spur_gear_blank_card,
)
from library.catalog import PartCatalog, build_default_catalog


def session_factory() -> HarnessSession:
    return HarnessSession(StubBackend())


class TestModelCardsBuild(unittest.TestCase):
    def test_every_card_builds_ok_with_defaults(self):
        for card in default_cards():
            ops = card.instantiate()  # defaults, range-validated
            self.assertTrue(ops, f"{card.name} produced no ops")
            result = session_factory().apply_ops(ops)
            self.assertTrue(
                result.ok,
                f"{card.name} failed to build: "
                f"{[d.to_dict() for d in result.diagnostics]}")
            self.assertEqual(result.applied, len(ops))

    def test_cards_build_with_custom_in_range_params(self):
        cat = build_default_catalog(session_factory)
        cases = {
            "flange": dict(diameter=120.0, bolt_circle=90.0, n_holes=8, thickness=10.0),
            "bracket": dict(width=60.0, height=45.0, thickness=6.0, hole_r=5.0),
            "spur_gear_blank": dict(module=3.0, teeth=40, thickness=12.0, bore=12.0),
            "shaft": dict(diameter=25.0, length=200.0),
            "bearing_seat": dict(bore=10.0, outer_diameter=26.0, width=8.0, wall=4.0),
        }
        for name, params in cases.items():
            ops = cat.instantiate(name, **params)
            result = session_factory().apply_ops(ops)
            self.assertTrue(result.ok, f"{name} custom build failed")

    def test_gear_blank_documents_limitation(self):
        self.assertIn("involute", spur_gear_blank_card().notes.lower())


class TestRangeValidation(unittest.TestCase):
    def test_rejects_below_minimum(self):
        card = flange_card()
        with self.assertRaises(ValueError):
            card.instantiate(diameter=1.0)  # min is 5.0

    def test_rejects_above_maximum(self):
        with self.assertRaises(ValueError):
            spur_gear_blank_card().instantiate(teeth=10000)

    def test_rejects_unknown_parameter(self):
        with self.assertRaises(ValueError):
            flange_card().instantiate(nonsense=1.0)

    def test_rejects_non_integer_for_int_param(self):
        with self.assertRaises(ValueError):
            flange_card().instantiate(n_holes=4.5)

    def test_accepts_in_range(self):
        ops = flange_card().instantiate(diameter=60.0, n_holes=6)
        self.assertTrue(ops)

    def test_catalog_instantiate_unknown_part(self):
        cat = build_default_catalog(session_factory)
        with self.assertRaises(KeyError):
            cat.instantiate("no_such_part")


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.cat = build_default_catalog(session_factory)

    def test_find_flange_by_name_tag(self):
        results = self.cat.find("flange")
        self.assertTrue(results)
        self.assertEqual(results[0].name, "flange")

    def test_find_flange_by_function_tag_mounting(self):
        names = [c.name for c in self.cat.find("mounting", k=5)]
        self.assertIn("flange", names)

    def test_find_bearing_tag(self):
        results = self.cat.find("bearing")
        self.assertTrue(results)
        self.assertEqual(results[0].name, "bearing_seat")

    def test_find_gear(self):
        names = [c.name for c in self.cat.find("gear")]
        self.assertIn("spur_gear_blank", names)


class TestAddVerified(unittest.TestCase):
    def test_admits_good_card(self):
        cat = PartCatalog()
        admitted = cat.add_verified(flange_card(), session_factory)
        self.assertTrue(admitted)
        self.assertIn("flange", cat)
        self.assertTrue(cat.get("flange").verified)

    def test_rejects_broken_card(self):
        # Build ops that cannot verify: extrude a sketch that has no profile.
        def broken_build(**_params):
            return [NewSketch(plane="XY"), Extrude(sketch="sk1", distance=5.0)]

        broken = ModelCard(
            name="broken_part",
            function_tags=["broken"],
            description="intentionally unbuildable",
            build=broken_build,
            param_schema={},
        )
        cat = PartCatalog()
        admitted = cat.add_verified(broken, session_factory)
        self.assertFalse(admitted)
        self.assertNotIn("broken_part", cat)

    def test_default_catalog_all_verified(self):
        cat = build_default_catalog(session_factory)
        self.assertEqual(len(cat.names()), len(default_cards()))
        for card in cat.cards():
            self.assertTrue(card.verified, f"{card.name} not verified")


if __name__ == "__main__":
    unittest.main()
