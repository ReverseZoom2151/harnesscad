"""A CAD CUA trial started on a dirty app is worthless: catch the leak first."""

import unittest

from harnesscad.io.cua.reset import (
    DirtyEnvironment, RESET_CHECKLIST, SessionState, StateLeakDetector,
    checklist_keys, clean_baseline, requires_fresh_process,
)


class TestChecklist(unittest.TestCase):
    def test_every_step_has_a_key_in_the_baseline(self):
        base = clean_baseline()
        for step in RESET_CHECKLIST:
            self.assertTrue(base.has(step.key), step.key)
            self.assertEqual(base.get(step.key), step.clean)

    def test_keys_are_unique(self):
        keys = checklist_keys()
        self.assertEqual(len(keys), len(set(keys)))

    def test_the_uncleanable_categories_need_a_fresh_process(self):
        """The E2B finding: some state has no sound in-process reset -- only a new
        process. Sticky tool defaults, the undo stack and preferences are among
        them, and that is the whole reason for fresh-VM-per-run."""
        cats = {s.category for s in requires_fresh_process()}
        self.assertIn("tool-defaults", cats)
        self.assertIn("undo-stack", cats)
        self.assertIn("preferences", cats)
        for step in requires_fresh_process():
            self.assertIsNone(step.in_process_remedy)


class TestLeakDetector(unittest.TestCase):
    def test_clean_snapshot_passes(self):
        det = StateLeakDetector()
        det.assert_clean(clean_baseline())  # must not raise
        self.assertTrue(det.is_clean(clean_baseline()))

    def test_sticky_pad_default_is_caught(self):
        """The killer: a Pad dialog remembering 37.5 from the prior trial."""
        det = StateLeakDetector()
        dirty = SessionState({**clean_baseline().to_dict(),
                              "tool_defaults": {"pad_length": 37.5}})
        leaks = det.leaks(dirty)
        self.assertEqual([l.category for l in leaks], ["tool-defaults"])
        self.assertTrue(leaks[0].fresh_process_only)
        with self.assertRaises(DirtyEnvironment):
            det.assert_clean(dirty)

    def test_leftover_undo_and_selection_both_reported(self):
        det = StateLeakDetector()
        dirty = SessionState({**clean_baseline().to_dict(),
                              "undo_depth": 4, "selection": ("Face6",)})
        cats = {l.category for l in det.leaks(dirty)}
        self.assertEqual(cats, {"undo-stack", "selection"})

    def test_view_orientation_leak_is_soft(self):
        det = StateLeakDetector()
        dirty = SessionState({**clean_baseline().to_dict(), "view": "top"})
        leaks = det.leaks(dirty)
        self.assertEqual(len(leaks), 1)
        self.assertFalse(leaks[0].fresh_process_only)
        self.assertIsNotNone(leaks[0].remedy)

    def test_unprobed_category_is_not_assumed_clean(self):
        """Silence is never a pass: a snapshot that never observed a category is
        reported as unverified, not clean."""
        det = StateLeakDetector()
        partial = SessionState({"undo_depth": 0})  # only one field observed
        self.assertIn("tool-defaults", det.unverified(partial))
        # and no false leak is raised for the unobserved fields
        self.assertTrue(det.is_clean(partial))

    def test_report_shape(self):
        det = StateLeakDetector()
        dirty = SessionState({**clean_baseline().to_dict(),
                              "prefs_dirty": True})
        rep = det.report(dirty)
        self.assertFalse(rep["clean"])
        self.assertTrue(rep["fresh_process_required"])
        self.assertEqual(rep["leaks"][0]["category"], "preferences")

    def test_list_and_set_values_normalise_for_comparison(self):
        det = StateLeakDetector()
        # selection given as a list must still compare equal to the tuple baseline
        s = SessionState({**clean_baseline().to_dict(), "selection": []})
        self.assertTrue(det.is_clean(s))


if __name__ == "__main__":
    unittest.main()
