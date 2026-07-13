import unittest

from harnesscad.domain.drawings.gdt import (
    FORM,
    LOCATION,
    category_of,
    validate_frame,
    validate_frames,
    worst_severity,
)
from harnesscad.domain.drawings.annotation_parser import parse_gdt_frame


class CategoryTests(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(category_of("flatness"), FORM)
        self.assertEqual(category_of("position"), LOCATION)
        self.assertIsNone(category_of("bogus"))


class ValidateTests(unittest.TestCase):
    def test_valid_position(self):
        f = parse_gdt_frame("POSITION Ø0.2 M A B C")
        v = validate_frame(f)
        self.assertTrue(v.ok)
        self.assertEqual(v.errors, [])

    def test_valid_flatness_datumless(self):
        f = parse_gdt_frame("FLATNESS 0.05")
        v = validate_frame(f)
        self.assertTrue(v.ok)

    def test_flatness_with_datum_error(self):
        f = parse_gdt_frame("FLATNESS 0.05 A")
        v = validate_frame(f)
        self.assertFalse(v.ok)
        self.assertTrue(any(e.code == "datum_on_form_control" for e in v.errors))

    def test_orientation_needs_datum(self):
        f = parse_gdt_frame("PERPENDICULARITY 0.1")
        v = validate_frame(f)
        self.assertFalse(v.ok)
        self.assertTrue(any(e.code == "missing_datum" for e in v.errors))

    def test_nonpositive_tolerance(self):
        v = validate_frame({"symbol": "position", "tolerance": 0.0,
                            "diametral_zone": False, "modifier": None,
                            "datums": ["A"]})
        self.assertFalse(v.ok)
        self.assertTrue(any(e.code == "nonpositive_tolerance" for e in v.errors))

    def test_missing_tolerance(self):
        v = validate_frame({"symbol": "position", "tolerance": None,
                            "diametral_zone": False, "modifier": None,
                            "datums": ["A"]})
        self.assertFalse(v.ok)

    def test_duplicate_datum(self):
        v = validate_frame({"symbol": "position", "tolerance": 0.2,
                            "diametral_zone": True, "modifier": None,
                            "datums": ["A", "A"]})
        self.assertFalse(v.ok)
        self.assertTrue(any(e.code == "duplicate_datum" for e in v.errors))

    def test_modifier_not_applicable(self):
        # Flatness is a form control; MMC modifier is illegal. But flatness with a
        # modifier + datum also triggers datum error; use a plane-ish control.
        v = validate_frame({"symbol": "straightness", "tolerance": 0.1,
                            "diametral_zone": False, "modifier": "MMC",
                            "datums": []})
        self.assertFalse(v.ok)
        self.assertTrue(any(e.code == "modifier_not_applicable" for e in v.errors))

    def test_diametral_zone_warning(self):
        v = validate_frame({"symbol": "flatness", "tolerance": 0.1,
                            "diametral_zone": True, "modifier": None,
                            "datums": []})
        # flatness is datumless & form; diametral zone unconventional -> warning
        self.assertTrue(v.ok)  # warning only, no error
        self.assertTrue(any(w.code == "unexpected_diametral_zone"
                            for w in v.warnings))

    def test_unknown_symbol(self):
        v = validate_frame({"symbol": "unicorn", "tolerance": 0.1,
                            "diametral_zone": False, "modifier": None,
                            "datums": []})
        self.assertFalse(v.ok)

    def test_no_symbol(self):
        v = validate_frame({})
        self.assertFalse(v.ok)

    def test_to_dict(self):
        f = parse_gdt_frame("FLATNESS 0.05")
        v = validate_frame(f)
        d = v.to_dict()
        self.assertIn("findings", d)
        self.assertEqual(d["symbol"], "flatness")


class BatchTests(unittest.TestCase):
    def test_worst_severity(self):
        frames = [parse_gdt_frame("FLATNESS 0.05"),
                  parse_gdt_frame("PERPENDICULARITY 0.1")]  # missing datum error
        vals = validate_frames(frames)
        self.assertEqual(worst_severity(vals), "ERROR")

    def test_all_ok(self):
        frames = [parse_gdt_frame("FLATNESS 0.05"),
                  parse_gdt_frame("POSITION 0.2 A")]
        vals = validate_frames(frames)
        self.assertEqual(worst_severity(vals), "OK")


if __name__ == "__main__":
    unittest.main()
