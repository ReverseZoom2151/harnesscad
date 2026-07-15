"""Tests for the Windows-UIA known-hazards checklist and driver audit."""

import unittest

from harnesscad.io.cua import hazards as H


class ChecklistTest(unittest.TestCase):
    def test_all_three_hazards_present(self):
        kinds = {h.kind for h in H.HAZARDS}
        self.assertEqual(kinds, {H.HazardKind.ASYNC_KEY_STATE,
                                 H.HazardKind.APPEND_NOT_REPLACE,
                                 H.HazardKind.DISABLED_NOOP})

    def test_by_kind_roundtrips(self):
        for h in H.HAZARDS:
            self.assertIs(H.by_kind(h.kind), h)

    def test_every_hazard_names_a_correct_primitive(self):
        for h in H.HAZARDS:
            self.assertTrue(h.correct_primitive.strip())
            self.assertTrue(h.verification.strip())
            self.assertTrue(h.wrong_api.strip())

    def test_checklist_is_serialisable(self):
        rows = H.checklist()
        self.assertEqual(len(rows), 3)
        self.assertIn("correct_primitive", rows[0])


class AuditPlanTest(unittest.TestCase):
    def test_safe_plan_trips_nothing(self):
        self.assertEqual(H.audit_plan(H.SAFE_PLAN), [])
        self.assertTrue(H.is_hazard_safe_plan(H.SAFE_PLAN))

    def test_empty_plan_trips_all_three_hazards(self):
        probs = H.audit_plan({})
        # one message per hazard family: async-key, append(clear), append(readback),
        # disabled-outcome -> at least the three kinds are all named.
        joined = " ".join(probs)
        self.assertIn(H.HazardKind.ASYNC_KEY_STATE.value, joined)
        self.assertIn(H.HazardKind.APPEND_NOT_REPLACE.value, joined)
        self.assertIn(H.HazardKind.DISABLED_NOOP.value, joined)
        self.assertFalse(H.is_hazard_safe_plan({}))

    def test_postmessage_is_flagged(self):
        plan = dict(H.SAFE_PLAN, key_dispatch="postmessage")
        probs = H.audit_plan(plan)
        self.assertTrue(any(H.HazardKind.ASYNC_KEY_STATE.value in p for p in probs))

    def test_missing_readback_is_flagged(self):
        plan = dict(H.SAFE_PLAN, reads_back=False)
        self.assertFalse(H.is_hazard_safe_plan(plan))

    def test_absence_is_unsafe_default(self):
        # Only opting into SendInput, leaving the rest unspecified, is still unsafe.
        self.assertFalse(H.is_hazard_safe_plan({"key_dispatch": "sendinput"}))


class _SafeDriver:
    """A stand-in with the required correct primitives and none of the forbidden."""

    def send_text(self, text): ...
    def send_key(self, vk, ctrl=False): ...
    def select_all(self): ...
    def read_value(self, element): ...


class _UnsafeDriver:
    """Missing the read-back and using the forbidden PostMessage path."""

    def send_text(self, text): ...
    def post_message(self, hwnd, msg): ...   # forbidden


class AuditDriverTest(unittest.TestCase):
    def test_safe_driver_passes(self):
        self.assertEqual(H.audit_driver(_SafeDriver), [])

    def test_unsafe_driver_flags_forbidden_and_missing(self):
        probs = H.audit_driver(_UnsafeDriver)
        self.assertTrue(any("forbidden" in p for p in probs))
        self.assertTrue(any("missing correct primitive" in p for p in probs))

    def test_real_uia_module_is_hazard_safe(self):
        # The whole point: our own driver is asserted against the checklist. uia is
        # import-safe without Windows deps (it degrades to available()==False), so
        # this import and attribute scan run everywhere.
        try:
            from harnesscad.io.cua import uia
            from harnesscad.io.cua.uia import UiaDriver
        except Exception as exc:  # pragma: no cover - import-safety regression
            self.skipTest("uia import failed: %s" % exc)
        # Module-level send_text/send_key/select_all + UiaDriver.read_value together
        # cover every required primitive; neither surface exposes PostMessage.
        module_names = H._visible_names(uia)
        driver_names = H._visible_names(UiaDriver)
        for req in H.REQUIRED_PRIMITIVES:
            self.assertTrue(req in module_names or req in driver_names,
                            "uia is missing required primitive %r" % req)
        # Neither the module nor the driver may expose a PostMessage-style path.
        for surface in (uia, UiaDriver):
            self.assertFalse(any("forbidden" in p for p in H.audit_driver(surface)),
                             "uia exposes a forbidden PostMessage primitive")


if __name__ == "__main__":
    unittest.main()
