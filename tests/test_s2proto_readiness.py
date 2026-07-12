import unittest

from fabrication.s2proto_readiness import (
    OK, WARNING, ERROR,
    check_prompt_safety, check_source_image_text, check_fragmentation,
    check_watertight, check_surface_smoothness, check_manufacturable_volume,
    evaluate, worst_severity, is_ready, readiness_report, ALL_CHECKS,
)


CLEAN = {
    "prompt_flagged_unsafe": False,
    "image_contains_text": False,
    "component_count": 1,
    "hole_count": 0,
    "surface_smooth": True,
    "volume": 12.5,
}


class TestIndividualChecks(unittest.TestCase):
    def test_prompt_unsafe_error(self):
        self.assertEqual(check_prompt_safety({"prompt_flagged_unsafe": True}).severity, ERROR)
        self.assertEqual(check_prompt_safety(CLEAN).severity, OK)

    def test_image_text_warning(self):
        self.assertEqual(check_source_image_text({"image_contains_text": True}).severity, WARNING)
        self.assertEqual(check_source_image_text(CLEAN).severity, OK)

    def test_fragmentation_error(self):
        self.assertEqual(check_fragmentation({"component_count": 3}).severity, ERROR)
        self.assertEqual(check_fragmentation(CLEAN).severity, OK)

    def test_fragmentation_bad_value(self):
        with self.assertRaises(ValueError):
            check_fragmentation({"component_count": 0})

    def test_watertight_warning(self):
        self.assertEqual(check_watertight({"hole_count": 2}).severity, WARNING)
        self.assertEqual(check_watertight(CLEAN).severity, OK)

    def test_watertight_bad_value(self):
        with self.assertRaises(ValueError):
            check_watertight({"hole_count": -1})

    def test_smoothness_warning(self):
        self.assertEqual(check_surface_smoothness({"surface_smooth": False}).severity, WARNING)
        self.assertEqual(check_surface_smoothness(CLEAN).severity, OK)

    def test_volume_error(self):
        self.assertEqual(check_manufacturable_volume({"volume": 0.0}).severity, ERROR)
        self.assertEqual(check_manufacturable_volume({"volume": -3.0}).severity, ERROR)
        self.assertEqual(check_manufacturable_volume(CLEAN).severity, OK)

    def test_volume_missing_skipped(self):
        self.assertEqual(check_manufacturable_volume({}).severity, OK)


class TestEvaluate(unittest.TestCase):
    def test_clean_all_ok(self):
        findings = evaluate(CLEAN)
        self.assertEqual(len(findings), len(ALL_CHECKS))
        self.assertTrue(all(f.severity == OK for f in findings))

    def test_deterministic_order(self):
        order1 = [f.check for f in evaluate(CLEAN)]
        messy = dict(CLEAN, component_count=4, hole_count=1)
        order2 = [f.check for f in evaluate(messy)]
        self.assertEqual(order1, order2)

    def test_worst_severity(self):
        self.assertEqual(worst_severity(evaluate(CLEAN)), OK)
        self.assertEqual(worst_severity(evaluate(dict(CLEAN, hole_count=1))), WARNING)
        self.assertEqual(worst_severity(evaluate(dict(CLEAN, component_count=2))), ERROR)


class TestReadinessGate(unittest.TestCase):
    def test_clean_ready(self):
        self.assertTrue(is_ready(CLEAN))
        self.assertTrue(is_ready(CLEAN, allow_warnings=False))

    def test_warning_ready_only_when_allowed(self):
        d = dict(CLEAN, hole_count=1)
        self.assertTrue(is_ready(d, allow_warnings=True))
        self.assertFalse(is_ready(d, allow_warnings=False))

    def test_error_never_ready(self):
        d = dict(CLEAN, component_count=5)
        self.assertFalse(is_ready(d, allow_warnings=True))
        self.assertFalse(is_ready(d, allow_warnings=False))


class TestReport(unittest.TestCase):
    def test_report_clean(self):
        r = readiness_report(CLEAN)
        self.assertTrue(r["ready"])
        self.assertEqual(r["worst_severity"], OK)
        self.assertEqual(r["error_count"], 0)
        self.assertEqual(r["warning_count"], 0)
        self.assertEqual(len(r["findings"]), len(ALL_CHECKS))

    def test_report_counts(self):
        d = dict(CLEAN, hole_count=1, surface_smooth=False, component_count=2)
        r = readiness_report(d)
        self.assertFalse(r["ready"])
        self.assertEqual(r["worst_severity"], ERROR)
        self.assertEqual(r["error_count"], 1)
        self.assertEqual(r["warning_count"], 2)


if __name__ == "__main__":
    unittest.main()
