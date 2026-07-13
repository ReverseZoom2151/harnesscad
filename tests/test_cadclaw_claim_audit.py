"""Tests for quality.cadclaw_claim_audit.

Deterministic, stdlib-only. Exercises forbidden-absolute, stale-term and
untagged-numeric detection plus negation- and context-awareness.
"""
import unittest

from harnesscad.eval.quality.report.claim_audit import (
    audit_text, ClaimReport, ClaimFinding, DEFAULT_FORBIDDEN_ABSOLUTES,
)


class ForbiddenAbsoluteTest(unittest.TestCase):

    def test_flags_forbidden(self):
        r = audit_text("This harness is production-ready and guaranteed.")
        rules = {f.match for f in r.findings if f.rule == "forbidden_absolute"}
        self.assertIn("production-ready", rules)
        self.assertIn("guaranteed", rules)
        self.assertTrue(r.failed)

    def test_case_insensitive(self):
        r = audit_text("Totally Bulletproof design.")
        self.assertTrue(any(f.match == "bulletproof" for f in r.findings))

    def test_negation_suppresses(self):
        r = audit_text("This is not production-ready yet.")
        self.assertFalse(any(f.rule == "forbidden_absolute" for f in r.findings))

    def test_negation_bounded_by_sentence(self):
        # negation in the previous sentence must NOT suppress
        r = audit_text("We do not ship. This is production-ready.")
        self.assertTrue(any(f.rule == "forbidden_absolute" for f in r.findings))

    def test_clean_text_passes(self):
        r = audit_text("Tested against the v0.6 fixture suite.")
        self.assertFalse(r.failed)
        self.assertEqual(r.n_fail, 0)


class StaleTermTest(unittest.TestCase):

    def test_flags_stale_term(self):
        r = audit_text("Bond it with JB Weld.", stale_terms=["JB Weld"])
        self.assertTrue(any(f.rule == "stale_term" for f in r.findings))

    def test_negated_stale_ok(self):
        r = audit_text("We no longer use JB Weld.", stale_terms=["JB Weld"])
        self.assertFalse(any(f.rule == "stale_term" for f in r.findings))

    def test_no_stale_terms_configured(self):
        r = audit_text("Uses JB Weld everywhere.")
        self.assertFalse(any(f.rule == "stale_term" for f in r.findings))


class NumericClaimTest(unittest.TestCase):

    def test_untagged_numeric_warns(self):
        r = audit_text("The frame has a safety factor of 3.")
        warns = [f for f in r.findings if f.rule == "untagged_numeric"]
        self.assertEqual(len(warns), 1)
        self.assertEqual(warns[0].severity, "warn")
        self.assertFalse(r.failed)  # warn only

    def test_tagged_numeric_ok(self):
        r = audit_text("The frame has a safety factor of 3 [analysis].")
        self.assertFalse(any(f.rule == "untagged_numeric" for f in r.findings))

    def test_deflection_pattern(self):
        r = audit_text("Deflection under 0.5 mm at mid-span.")
        self.assertTrue(any(f.rule == "untagged_numeric" for f in r.findings))

    def test_one_warning_per_line(self):
        r = audit_text("safety factor of 3 and deflection of 0.2 mm")
        warns = [f for f in r.findings if f.rule == "untagged_numeric"]
        self.assertEqual(len(warns), 1)


class ContextStrippingTest(unittest.TestCase):

    def test_code_fence_ignored(self):
        text = "Intro.\n```\nproduction-ready = True\n```\nOutro."
        r = audit_text(text, is_markdown=True)
        self.assertFalse(any(f.rule == "forbidden_absolute" for f in r.findings))

    def test_code_fence_scanned_when_not_markdown(self):
        text = "Intro.\n```\nproduction-ready\n```\n"
        r = audit_text(text, is_markdown=False)
        self.assertTrue(any(f.rule == "forbidden_absolute" for f in r.findings))

    def test_license_line_not_flagged_for_stale(self):
        text = "Licensed under the MIT License by OpenBuilds."
        r = audit_text(text, stale_terms=["OpenBuilds"])
        self.assertFalse(any(f.rule == "stale_term" for f in r.findings))

    def test_numeric_still_scanned_in_license_block(self):
        # numeric scan ignores license stripping
        text = "Copyright (c) 2026. safety factor of 5 stated here."
        r = audit_text(text)
        self.assertTrue(any(f.rule == "untagged_numeric" for f in r.findings))


class ReportTest(unittest.TestCase):

    def test_line_numbers(self):
        text = "clean line\nthis is bulletproof\n"
        r = audit_text(text)
        f = next(x for x in r.findings if x.rule == "forbidden_absolute")
        self.assertEqual(f.line, 2)

    def test_counts(self):
        r = audit_text("guaranteed and bulletproof; safety factor of 2")
        self.assertEqual(r.n_fail, 2)
        self.assertEqual(r.n_warn, 1)

    def test_lines_scanned(self):
        r = audit_text("a\nb\nc")
        self.assertEqual(r.lines_scanned, 3)

    def test_default_list_nonempty(self):
        self.assertIn("production-ready", DEFAULT_FORBIDDEN_ABSOLUTES)


if __name__ == "__main__":
    unittest.main()
