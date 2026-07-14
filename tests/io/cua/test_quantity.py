"""The 375mm bug, made impossible.

Measured on this machine, FreeCAD 1.1.1: a field showing '10,00 mm' was focused,
select-all'd, and the perfectly ordinary string '37.5' was typed into it. It read
back '375,00 mm'. The dot was SWALLOWED. Thirty-seven point five became three
hundred and seventy-five: a silent 10x dimensional error, no exception, and the
geometry regenerates happily. A vision agent cannot see it -- the screenshot says
'375,00 mm' and a model that asked for 37.5 will read what it expects.

:class:`CommaLocaleWidget` below is a faithful simulation of that widget, so the
whole defence is tested with no GUI running.
"""

import unittest

from harnesscad.io.cua.quantity import (
    COMMA_LOCALE, DOT_LOCALE, Locale, QuantityError, QuantityMismatch,
    detect_locale, format_quantity, parse_quantity, write_quantity,
)


class CommaLocaleWidget:
    """A Qt quantity spinbox on a comma-decimal locale. Behaves as measured.

    Typing a string containing '.' drops the dot (the validator rejects the
    keystroke), so '37.5' becomes the integer 375 and commits as '375,00 mm'.
    """

    def __init__(self, value=10.0, unit="mm"):
        self.value = float(value)
        self.unit = unit

    def type(self, text):
        raw = str(text).replace(".", "")          # <- THE BUG, verbatim
        raw = raw.replace(",", ".")
        try:
            self.value = float(raw)
        except ValueError:
            pass

    def read(self):
        return ("%.2f" % self.value).replace(".", ",") + " " + self.unit


class DotLocaleWidget(CommaLocaleWidget):
    def type(self, text):
        try:
            self.value = float(str(text).replace(",", ""))
        except ValueError:
            pass

    def read(self):
        return ("%.2f" % self.value) + " " + self.unit


class TestParseAndFormat(unittest.TestCase):
    def test_parse_comma_locale(self):
        self.assertEqual(parse_quantity("375,00 mm", COMMA_LOCALE), (375.0, "mm"))
        self.assertEqual(parse_quantity("37,50 mm", COMMA_LOCALE), (37.5, "mm"))
        self.assertEqual(parse_quantity("1.234,56 mm", COMMA_LOCALE),
                         (1234.56, "mm"))
        self.assertEqual(parse_quantity("0,00", COMMA_LOCALE), (0.0, ""))

    def test_parse_dot_locale(self):
        self.assertEqual(parse_quantity("37.50 mm", DOT_LOCALE), (37.5, "mm"))

    def test_ambiguity_is_refused_not_guessed(self):
        with self.assertRaises(QuantityError):
            parse_quantity("37.5", COMMA_LOCALE)   # a dot in a comma locale
        with self.assertRaises(QuantityError):
            parse_quantity("", COMMA_LOCALE)

    def test_format_uses_the_apps_separator_never_str_float(self):
        self.assertEqual(format_quantity(37.5, COMMA_LOCALE), "37,50")
        self.assertEqual(format_quantity(37.5, DOT_LOCALE), "37.50")
        self.assertNotIn(".", format_quantity(37.5, COMMA_LOCALE))

    def test_detect_locale_from_what_the_app_rendered(self):
        self.assertEqual(detect_locale(["10,00 mm", "0,00 mm"]).decimal, ",")
        self.assertEqual(detect_locale(["10.00 mm", "0.00 mm"]).decimal, ".")


class TestWriteReadBack(unittest.TestCase):
    """Every numeric field write goes through this or it does not happen."""

    def test_naive_str_float_would_have_shipped_a_10x_error(self):
        """Proof the simulated widget really has the bug we are defending against."""
        w = CommaLocaleWidget()
        w.type(str(37.5))                       # what any normal code would type
        self.assertEqual(w.read(), "375,00 mm")  # 10x, silently

    def test_locale_correct_write_verifies(self):
        w = CommaLocaleWidget()
        report = write_quantity("boxLength", 37.5, w.type, w.read,
                                locale=COMMA_LOCALE)
        self.assertTrue(report.verified)
        self.assertEqual(report.typed, "37,50")
        self.assertEqual(report.read_text, "37,50 mm")
        self.assertEqual(report.read_value, 37.5)

    def test_locale_is_detected_from_the_field_itself(self):
        w = CommaLocaleWidget()
        report = write_quantity("boxLength", 37.5, w.type, w.read)  # no locale given
        self.assertEqual(report.locale_decimal, ",")
        self.assertEqual(report.read_value, 37.5)

    def test_a_write_that_does_not_read_back_RAISES(self):
        """The failure path. A wrong value must be a hard, loud failure -- never a
        warning, never a return code the caller can ignore."""
        w = CommaLocaleWidget()
        with self.assertRaises(QuantityMismatch) as ctx:
            # Formatting with the WRONG locale is exactly the original bug.
            write_quantity("boxLength", 37.5, w.type, w.read, locale=DOT_LOCALE,
                           retries=0)
        exc = ctx.exception
        self.assertEqual(exc.typed, "37.50")           # dots, into a comma widget
        self.assertEqual(exc.read_text, "3750,00 mm")  # the dot was swallowed
        self.assertIn("refusing to guess", str(exc))   # and the read-back is not
        self.assertIn("3750", str(exc))                # silently re-interpreted

    def test_a_dead_field_raises_rather_than_reporting_success(self):
        class Dead(CommaLocaleWidget):
            def type(self, text):
                pass                     # the write silently no-ops, as SetValue does

        w = Dead(value=10.0)
        with self.assertRaises(QuantityMismatch):
            write_quantity("boxLength", 37.5, w.type, w.read, locale=COMMA_LOCALE,
                           retries=0)

    def test_the_mismatch_names_the_power_of_ten(self):
        """The mirror image of the measured bug: a comma typed into a dot-locale
        widget. Here the read-back DOES parse -- to 375 -- and the error names the
        10x factor in as many words, which is exactly what a human needs to see."""
        w = DotLocaleWidget()
        with self.assertRaises(QuantityMismatch) as ctx:
            write_quantity("boxLength", 37.5, lambda _t: w.type("37,5"), w.read,
                           locale=DOT_LOCALE, retries=0)
        exc = ctx.exception
        self.assertEqual(exc.read_value, 375.0)
        self.assertAlmostEqual(exc.factor, 10.0)
        self.assertIn("SWALLOWED DECIMAL SEPARATOR", str(exc))

    def test_dot_locale_app_still_works(self):
        w = DotLocaleWidget()
        report = write_quantity("boxLength", 37.5, w.type, w.read,
                                locale=DOT_LOCALE)
        self.assertTrue(report.verified)

    def test_tolerance_admits_display_rounding_but_not_a_wrong_part(self):
        w = CommaLocaleWidget()
        report = write_quantity("boxLength", 12.344999, w.type, w.read,
                                locale=COMMA_LOCALE)   # UI shows 2dp
        self.assertTrue(report.verified)
        with self.assertRaises(QuantityMismatch):
            write_quantity("boxLength", 37.5, lambda t: w.type("37,6"), w.read,
                           locale=COMMA_LOCALE, retries=0)

    def test_bad_locale_definition_is_refused(self):
        with self.assertRaises(QuantityError):
            Locale(decimal=";")
        with self.assertRaises(QuantityError):
            Locale(decimal=",", group=",")


if __name__ == "__main__":
    unittest.main()
