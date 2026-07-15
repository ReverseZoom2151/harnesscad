"""Tests for the text-to-cad design-brief IR."""

import unittest

from harnesscad.domain.spec import design_brief as db


BRIEF_TEXT = """
CAD brief:
- Model: mounting plate
- Task type: create
- Overall dimensions: 60 x 40 x 10 mm
- Functional features: four M3 corner holes
- Positioning/mating requirements:
  - seats on a rail
- STEP target path: out/plate.step
"""


class ParseTest(unittest.TestCase):
    def test_parses_fields(self):
        brief = db.parse_brief(BRIEF_TEXT)
        self.assertEqual(brief.model, "mounting plate")
        self.assertEqual(brief.task_type, "create")
        self.assertEqual(brief.overall_dimensions, "60 x 40 x 10 mm")
        self.assertEqual(brief.step_target_path, "out/plate.step")

    def test_units_unstated(self):
        brief = db.parse_brief(BRIEF_TEXT)
        self.assertIsNone(brief.units)

    def test_sub_bullet_appended(self):
        brief = db.parse_brief(BRIEF_TEXT)
        self.assertIn("seats on a rail", brief.positioning_requirements or "")

    def test_placeholder_ignored(self):
        brief = db.parse_brief("- Model: -\n- Units: TBD\n")
        self.assertIsNone(brief.model)
        self.assertIsNone(brief.units)


class DefaultsTest(unittest.TestCase):
    def test_units_inferred(self):
        brief = db.parse_brief(BRIEF_TEXT)
        resolved, prov = db.resolve_defaults(brief)
        self.assertEqual(resolved.units, "millimeters")
        by_name = {p.field_name: p for p in prov}
        self.assertEqual(by_name["units"].status, "inferred")

    def test_stated_field_kept(self):
        brief = db.parse_brief(BRIEF_TEXT)
        resolved, prov = db.resolve_defaults(brief)
        self.assertEqual(resolved.model, "mounting plate")
        by_name = {p.field_name: p for p in prov}
        self.assertEqual(by_name["model"].status, "stated")

    def test_coordinate_convention_default(self):
        brief = db.parse_brief(BRIEF_TEXT)
        resolved, _ = db.resolve_defaults(brief)
        self.assertIn("XY", resolved.coordinate_convention)
        self.assertIn("+Z", resolved.coordinate_convention)


class ClearanceTest(unittest.TestCase):
    def test_m3(self):
        self.assertAlmostEqual(db.clearance_for("M3"), 3.4)

    def test_case_insensitive(self):
        self.assertAlmostEqual(db.clearance_for("m5"), 5.5)

    def test_unknown_raises(self):
        with self.assertRaises(KeyError):
            db.clearance_for("M99")


class CompletenessTest(unittest.TestCase):
    def test_buildable_with_defaults(self):
        brief = db.parse_brief(BRIEF_TEXT)
        report = db.completeness(brief)
        self.assertTrue(report.buildable)
        self.assertIn("model", report.stated)
        self.assertIn("units", report.inferred)

    def test_not_buildable_when_model_missing(self):
        brief = db.parse_brief("- Overall dimensions: 10 x 10 x 10 mm\n")
        report = db.completeness(brief)
        self.assertFalse(report.buildable)
        self.assertIn("model", report.missing)

    def test_overall_dimensions_required(self):
        brief = db.parse_brief("- Model: widget\n")
        report = db.completeness(brief)
        self.assertIn("overall_dimensions", report.missing)


if __name__ == "__main__":
    unittest.main()
