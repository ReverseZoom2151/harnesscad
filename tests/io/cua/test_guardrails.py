"""Resolve-before-click, so we can REFUSE. The agent NEVER saves."""

import os
import tempfile
import unittest

from harnesscad.io.cua.guardrails import (
    DENY_NAMES, Guardrails, GuardrailViolation, Scratch, dirty_tripwire,
    is_confirm_in_dangerous_dialog, is_denied, normalize_name,
)


class FakeElement:
    def __init__(self, name, rect=(10, 10, 50, 30), enabled=True):
        self.name = name
        self.rect = rect
        self.enabled = enabled
        self.key = "ButtonControl|aid|" + name


WINDOW = (0, 0, 1000, 800)


class TestDenyList(unittest.TestCase):
    def test_mnemonics_and_ellipses_are_not_a_disguise(self):
        self.assertEqual(normalize_name("&Save As..."), "save as")
        self.assertTrue(is_denied("&Save As..."))

    def test_every_destructive_name_is_refused(self):
        for name in ("Save", "Save As", "Save All", "Don't Save", "Discard",
                     "Overwrite", "Replace", "Delete", "Exit", "Quit",
                     "Close Without Saving"):
            with self.subTest(name=name):
                self.assertIsNotNone(is_denied(name), name)

    def test_the_ops_we_actually_need_are_allowed(self):
        for name in ("Pad", "Pocket", "Fillet", "New Document", "New Body",
                     "Additive Primitive", "Recompute", "OK", "Cancel"):
            with self.subTest(name=name):
                self.assertIsNone(is_denied(name), name)

    def test_yes_is_only_dangerous_in_a_dangerous_dialog(self):
        self.assertTrue(is_confirm_in_dangerous_dialog(
            "Yes", "The document has unsaved changes. Save?"))
        self.assertFalse(is_confirm_in_dangerous_dialog("Yes", "Recompute now?"))

    def test_the_agent_can_never_reach_save(self):
        guards = Guardrails(window_rect=WINDOW)
        with self.assertRaises(GuardrailViolation) as ctx:
            guards.enforce(FakeElement("Save"))
        self.assertIn("harness owns all file I/O", str(ctx.exception))


class TestScopeAndModals(unittest.TestCase):
    def test_a_click_outside_the_target_window_is_refused(self):
        guards = Guardrails(window_rect=WINDOW)
        self.assertIsNone(guards.check_scope(FakeElement("Pad")))
        far = FakeElement("Pad", rect=(1500, 900, 1600, 950))
        self.assertEqual(guards.check_scope(far).rule, "out-of-scope")
        self.assertEqual(guards.check_point(5000, 5000).rule, "out-of-scope")

    def test_a_point_without_a_frame_is_never_guessed(self):
        self.assertEqual(Guardrails().check_point(10, 10).rule, "no-frame")

    def test_a_disabled_control_is_refused(self):
        guards = Guardrails(window_rect=WINDOW)
        r = guards.check_element(FakeElement("Pad", enabled=False))
        self.assertEqual(r.rule, "disabled")

    def test_an_unexpected_modal_halts(self):
        guards = Guardrails(window_rect=WINDOW)
        self.assertIsNone(guards.check_modals(["Gui::MainWindow::* Unnamed"]))
        r = guards.check_modals(["Gui::MainWindow::x", "QDialog::Save changes?"])
        self.assertEqual(r.rule, "unexpected-modal")
        self.assertIn("HALT", r.detail)

    def test_dirty_title_tripwire(self):
        self.assertTrue(dirty_tripwire("* Unnamed - FreeCAD 1.1.1")["dirty"])
        self.assertFalse(dirty_tripwire("FreeCAD 1.1.1")["dirty"])


class TestScratch(unittest.TestCase):
    def test_a_user_file_is_copied_never_opened(self):
        with tempfile.TemporaryDirectory() as user_dir:
            user_file = os.path.join(user_dir, "precious.FCStd")
            with open(user_file, "w", encoding="utf-8") as fh:
                fh.write("the user's irreplaceable part")
            with Scratch() as scratch:
                copy = scratch.sacrificial_copy(user_file)
                self.assertTrue(scratch.owns(copy))
                self.assertNotEqual(os.path.realpath(copy),
                                    os.path.realpath(user_file))
                with open(copy, "w", encoding="utf-8") as fh:
                    fh.write("destroyed")
            with open(user_file, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "the user's irreplaceable part")

    def test_exports_cannot_escape_the_scratch_dir(self):
        with Scratch() as scratch:
            self.assertTrue(scratch.owns(scratch.export_path("model.step")))
            with self.assertRaises(GuardrailViolation):
                scratch.path("..", "..", "escaped.step")

    def test_cleanup_removes_only_what_we_own(self):
        scratch = Scratch()
        root = scratch.root
        scratch.cleanup()
        self.assertFalse(os.path.isdir(root))


if __name__ == "__main__":
    unittest.main()
