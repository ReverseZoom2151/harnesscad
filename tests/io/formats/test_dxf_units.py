"""Tests for formats.dxf_units -- the DXF $INSUNITS code table.

The point of the table is the refusals as much as the factors: unitless is not
millimetres, and absent is not unitless.
"""

import unittest

from harnesscad.io.formats.dxf import DxfDocument
from harnesscad.io.formats.dxf_units import (
    INSUNITS,
    INSUNITS_GROUP_CODE,
    MEASUREMENT,
    DxfUnits,
    parse_insunits,
    resolve_insunits,
    units_from_dxf_text,
)


def _dxf(insunits=None, extra=""):
    body = "  0\nSECTION\n  2\nHEADER\n  9\n$ACADVER\n  1\nAC1027\n"
    if insunits is not None:
        body += f"  9\n$INSUNITS\n {INSUNITS_GROUP_CODE}\n{insunits:6d}\n"
    body += "  0\nENDSEC\n" + extra + "  0\nEOF\n"
    return body


class CodeTableTest(unittest.TestCase):
    def test_table_covers_codes_0_through_20(self):
        self.assertEqual(sorted(INSUNITS), list(range(21)))

    def test_every_code_resolves(self):
        for code in INSUNITS:
            with self.subTest(code=code):
                u = resolve_insunits(code)
                self.assertEqual(u.code, code)
                self.assertEqual(u.name, INSUNITS[code][0])

    def test_every_resolvable_factor_is_positive(self):
        for code, (_, scale) in INSUNITS.items():
            if scale is not None:
                with self.subTest(code=code):
                    self.assertGreater(scale, 0.0)

    def test_names_are_unique(self):
        names = [name for name, _ in INSUNITS.values()]
        self.assertEqual(len(names), len(set(names)))

    def test_millimetre_is_identity(self):
        self.assertEqual(resolve_insunits(4).scale_to_mm, 1.0)

    def test_inch_is_exactly_25_4(self):
        self.assertEqual(resolve_insunits(1).scale_to_mm, 25.4)

    def test_metre_is_the_silent_1000x_code(self):
        self.assertEqual(resolve_insunits(6).scale_to_mm, 1000.0)

    def test_imperial_chain_derives_from_one_inch(self):
        inch = resolve_insunits(1).scale_to_mm
        self.assertAlmostEqual(resolve_insunits(2).scale_to_mm, 12 * inch)
        self.assertAlmostEqual(resolve_insunits(10).scale_to_mm, 36 * inch)
        self.assertAlmostEqual(resolve_insunits(3).scale_to_mm, 63360 * inch,
                               places=3)
        self.assertAlmostEqual(resolve_insunits(9).scale_to_mm, inch / 1000.0)
        self.assertAlmostEqual(resolve_insunits(8).scale_to_mm, inch / 1e6)

    def test_metric_chain_is_decimal(self):
        self.assertAlmostEqual(resolve_insunits(5).scale_to_mm, 10.0)
        self.assertAlmostEqual(resolve_insunits(7).scale_to_mm, 1e6)
        self.assertAlmostEqual(resolve_insunits(13).scale_to_mm, 1e-3)
        self.assertAlmostEqual(resolve_insunits(14).scale_to_mm, 100.0)
        self.assertAlmostEqual(resolve_insunits(15).scale_to_mm, 1e4)
        self.assertAlmostEqual(resolve_insunits(16).scale_to_mm, 1e5)

    def test_factors_increase_with_the_metric_ladder(self):
        ladder = [11, 12, 13, 4, 5, 14, 6, 15, 16, 7, 17]
        scales = [resolve_insunits(c).scale_to_mm for c in ladder]
        self.assertEqual(scales, sorted(scales))

    def test_measurement_is_not_a_unit_table(self):
        # $MEASUREMENT picks a pattern file; it must never be read as a scale.
        self.assertEqual(set(MEASUREMENT), {0, 1})
        for value in MEASUREMENT.values():
            self.assertIsInstance(value, str)


class UnresolvedTest(unittest.TestCase):
    """Unitless, absent, and unknown: three different "I don't know"s."""

    def test_unitless_has_no_scale(self):
        u = resolve_insunits(0)
        self.assertIsNone(u.scale_to_mm)
        self.assertFalse(u.resolved)

    def test_unitless_is_declared(self):
        # Code 0 IS a declaration -- of nothing.
        self.assertTrue(resolve_insunits(0).declared)

    def test_unitless_warns_about_the_mm_assumption(self):
        self.assertTrue(any("silent-scale" in n
                            for n in resolve_insunits(0).notes))

    def test_absent_is_not_declared(self):
        u = resolve_insunits(None)
        self.assertFalse(u.declared)
        self.assertIsNone(u.code)
        self.assertIsNone(u.scale_to_mm)

    def test_absent_and_unitless_are_distinguishable(self):
        self.assertNotEqual(resolve_insunits(None).declared,
                            resolve_insunits(0).declared)

    def test_unknown_code_is_unresolved_but_remembered(self):
        u = resolve_insunits(99)
        self.assertEqual(u.code, 99)
        self.assertEqual(u.name, "unknown")
        self.assertIsNone(u.scale_to_mm)
        self.assertTrue(u.notes)

    def test_negative_code_is_unresolved(self):
        self.assertIsNone(resolve_insunits(-1).scale_to_mm)

    def test_non_integer_value_is_unresolved(self):
        u = resolve_insunits("banana")
        self.assertIsNone(u.scale_to_mm)
        self.assertFalse(u.resolved)

    def test_nothing_ever_defaults_to_millimetres(self):
        for value in (None, 0, 99, -1, "banana"):
            with self.subTest(value=value):
                self.assertNotEqual(resolve_insunits(value).scale_to_mm, 1.0)

    def test_resolve_never_raises(self):
        for value in (None, 0, 4, 99, -1, "x", 3.7, [], object()):
            with self.subTest(value=value):
                self.assertIsInstance(resolve_insunits(value), DxfUnits)


class HeaderScanTest(unittest.TestCase):
    def test_reads_declared_code(self):
        self.assertEqual(parse_insunits(_dxf(1)), 1)
        self.assertEqual(parse_insunits(_dxf(6)), 6)
        self.assertEqual(parse_insunits(_dxf(0)), 0)

    def test_absent_variable_returns_none(self):
        self.assertIsNone(parse_insunits(_dxf(None)))

    def test_resolves_straight_from_text(self):
        self.assertEqual(units_from_dxf_text(_dxf(1)).scale_to_mm, 25.4)
        self.assertEqual(units_from_dxf_text(_dxf(6)).scale_to_mm, 1000.0)

    def test_absent_text_yields_undeclared(self):
        self.assertFalse(units_from_dxf_text(_dxf(None)).declared)

    def test_entity_text_is_not_mistaken_for_the_header_variable(self):
        # Only group code 9 declares a header variable.
        decoy = _dxf(None, extra=("  0\nSECTION\n  2\nENTITIES\n"
                                  "  0\nTEXT\n  1\n$INSUNITS\n 70\n     1\n"
                                  "  0\nENDSEC\n"))
        self.assertIsNone(parse_insunits(decoy))

    def test_non_numeric_value_is_none(self):
        bad = ("  0\nSECTION\n  2\nHEADER\n"
               "  9\n$INSUNITS\n 70\nmm\n  0\nENDSEC\n  0\nEOF\n")
        self.assertIsNone(parse_insunits(bad))

    def test_wrong_group_code_is_not_the_value(self):
        wrong = ("  0\nSECTION\n  2\nHEADER\n"
                 "  9\n$INSUNITS\n  1\n4\n  0\nENDSEC\n  0\nEOF\n")
        self.assertIsNone(parse_insunits(wrong))

    def test_empty_and_garbage_text_are_safe(self):
        for text in ("", "\n", "not a dxf at all", "  0\nEOF\n"):
            with self.subTest(text=text):
                self.assertIsNone(parse_insunits(text))


class DocumentBridgeTest(unittest.TestCase):
    def test_maps_onto_the_dxfdocument_vocabulary(self):
        self.assertEqual(resolve_insunits(4).document_units, "mm")
        self.assertEqual(resolve_insunits(5).document_units, "cm")
        self.assertEqual(resolve_insunits(6).document_units, "m")
        self.assertEqual(resolve_insunits(1).document_units, "in")
        self.assertEqual(resolve_insunits(2).document_units, "ft")

    def test_inexpressible_units_return_none_rather_than_a_lie(self):
        for code in (0, 3, 8, 9, 10, 11, 20):
            with self.subTest(code=code):
                self.assertIsNone(resolve_insunits(code).document_units)

    def test_document_units_are_accepted_by_the_contract(self):
        for code in (1, 2, 4, 5, 6):
            with self.subTest(code=code):
                doc = DxfDocument(units=resolve_insunits(code).document_units,
                                  layers=(), entities={})
                self.assertEqual(doc.units,
                                 resolve_insunits(code).document_units)

    def test_to_dict_round_trips_the_facts(self):
        d = resolve_insunits(1).to_dict()
        self.assertEqual(d["code"], 1)
        self.assertEqual(d["scale_to_mm"], 25.4)
        self.assertTrue(d["resolved"])
        self.assertEqual(d["document_units"], "in")


class SelfcheckTest(unittest.TestCase):
    def test_selfcheck_exits_zero(self):
        from harnesscad.io.formats.dxf_units import main
        self.assertEqual(main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
