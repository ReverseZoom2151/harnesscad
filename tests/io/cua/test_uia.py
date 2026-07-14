"""The UIA driver: an unverified action is not an action.

The live half needs Windows + `uiautomation` + FreeCAD, and SKIPs cleanly when
they are absent (never hangs, never fails on absence). The pure half -- change
detection, element identity, the SendInput-only rule -- runs everywhere.
"""

import os
import unittest

from harnesscad.io.cua import uia
from harnesscad.io.cua.uia import Element, Outcome, Snapshot

LIVE = os.environ.get("HARNESSCAD_CUA_LIVE") == "1"


def _element(name="Pad", aid="a.b.Pad", ct="ButtonControl"):
    return Element(name=name, automation_id=aid, control_type=ct,
                   class_name="QToolButton", rect=(10, 10, 50, 40), enabled=True,
                   depth=3)


class TestElement(unittest.TestCase):
    def test_identity_is_type_id_name_not_a_coordinate(self):
        self.assertEqual(_element().key, "ButtonControl|a.b.Pad|Pad")
        self.assertEqual(_element().center, (30, 25))


class TestSnapshotIsTheEvidence(unittest.TestCase):
    def test_a_tree_that_grew_is_evidence_an_action_landed(self):
        a = Snapshot("FreeCAD", 156, ("x",), ("Gui::MainWindow::FreeCAD",))
        b = Snapshot("* Unnamed - FreeCAD", 234, ("x", "y"),
                     ("Gui::MainWindow::* Unnamed - FreeCAD",))
        diff = a.diff(b)
        self.assertIn("element_count", diff)
        self.assertIn("title", diff)
        self.assertEqual(diff["appeared"], ["y"])

    def test_no_change_is_no_evidence(self):
        a = Snapshot("FreeCAD", 156, ("x",), ())
        self.assertEqual(a.diff(a), {})

    def test_the_dirty_star_is_a_free_tripwire(self):
        self.assertTrue(Snapshot("* Unnamed - FreeCAD", 1, (), ()).dirty)
        self.assertFalse(Snapshot("FreeCAD 1.1.1", 1, (), ()).dirty)


class TestOutcome(unittest.TestCase):
    def test_a_failed_outcome_carries_why(self):
        o = Outcome(False, "invoke", "ButtonControl|x|Pad",
                    error="Invoke() returned but NOTHING CHANGED")
        self.assertFalse(o.to_dict()["ok"])
        self.assertIn("NOTHING CHANGED", o.to_dict()["error"])


class TestNoPostMessageAnywhere(unittest.TestCase):
    """Two APIs are banned OUTRIGHT, and the ban is checked structurally (the AST,
    not the text -- the module's own docstrings name them as the hazards they are)."""

    def _called_attributes(self):
        import ast

        with open(uia.__file__, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
        return names

    def test_PostMessage_is_never_called(self):
        """PostMessage returns TRUE for a disabled control, a window behind a modal,
        or a DirectX viewport, and does nothing -- and it never updates the async
        key state, so a modifier-aware app sees Ctrl as up."""
        self.assertNotIn("PostMessage", self._called_attributes())
        self.assertIn("SendInput", self._called_attributes())

    def test_ValuePattern_SetValue_is_never_the_write_path(self):
        """It silently no-ops on Qt spinboxes: IsReadOnly says False, nothing
        raises, the value does not change. The only write path is focus ->
        select-all -> SendInput -> READ BACK."""
        self.assertNotIn("SetValue", self._called_attributes())
        self.assertIn("GetValuePattern", self._called_attributes())  # reads are fine


class TestAvailability(unittest.TestCase):
    def test_absence_is_reported_never_hung_on(self):
        self.assertIsInstance(uia.available(), bool)
        if not uia.available():
            with self.assertRaises(uia.UiaUnavailable):
                uia.UiaDriver(pid=1)


@unittest.skipUnless(LIVE and uia.available(),
                     "live UIA run (set HARNESSCAD_CUA_LIVE=1)")
class TestLiveTree(unittest.TestCase):
    """Drives the real FreeCAD GUI. Killed on teardown; no user file is touched."""

    @classmethod
    def setUpClass(cls):
        from harnesscad.io.cua import environment_freecad as E
        ok, why = E.available()
        if not ok:
            raise unittest.SkipTest(why)
        cls.env = E.FreeCADGuiEnvironment()
        cls.env.reset()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        cls.env.scratch.cleanup()

    def test_the_tree_is_enumerable_and_the_chrome_is_coordinate_free(self):
        elems = self.env.driver.tree()
        self.assertGreater(len(elems), 100)
        names = {e.name for e in elems}
        for expected in ("Pad", "Pocket", "Fillet", "Chamfer", "New Body",
                         "Additive Primitive"):
            self.assertIn(expected, names)

    def test_the_viewport_rect_comes_from_the_tree_not_a_guess(self):
        frame = self.env.driver.viewport_frame()
        left, top, right, bottom = frame.screen_rect
        self.assertGreater(right - left, 200)
        self.assertGreater(bottom - top, 200)

    def test_the_apps_locale_is_detected_from_what_it_rendered(self):
        self.assertIn(self.env.driver.locale().decimal, (".", ","))


if __name__ == "__main__":
    unittest.main()
