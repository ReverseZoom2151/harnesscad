"""Tests for OpenECAD editability operations."""

import unittest

from harnesscad.domain.programs.ast import openecad_script as oe
from harnesscad.domain.programs.params import openecad_edit as ed


CODE = "\n".join([
    "SketchPlane0 = add_sketchplane(origin=[0.0, 0.0, 0.0], "
    "normal=[0.0, 0.0, 1.0], x_axis=[1.0, 0.0, 0.0])",
    "Loops0, Curves0_0 = [], []",
    "Line0_0_0 = add_line(start=[-1000.0, -750.0], end=[1000.0, -750.0])",
    "Loop0_0 = add_loop(Curves0_0)",
    "Profile0 = add_profile(Loops0)",
    "Sketch0 = add_sketch(SketchPlane0, Profile0, position=[0.0, 0.0], size=1.0)",
    "Extrude0 = add_extrude(Sketch0, operation=1, type=0, extent_one=1000.0, "
    "extent_two=0.0)",
    "SketchPlane1 = add_sketchplane_ref(Extrude0, origin=[0.0, 0.0], "
    'type="sameplan", reverse=True)',
])


class TestRename(unittest.TestCase):
    def test_rename_updates_definition_and_references(self):
        prog = oe.parse(CODE)
        renamed = ed.rename_variable(prog, "Extrude0", "TableLegExtrude")
        text = oe.emit(renamed)
        self.assertIn("TableLegExtrude = add_extrude", text)
        self.assertIn("add_sketchplane_ref(TableLegExtrude,", text)
        self.assertNotIn("Extrude0", text)

    def test_rename_preserves_reference_count(self):
        prog = oe.parse(CODE)
        before = ed.count_references(prog, "Extrude0")
        renamed = ed.rename_variable(prog, "Extrude0", "E9")
        self.assertEqual(ed.count_references(renamed, "E9"), before)
        self.assertEqual(before, 1)

    def test_rename_does_not_mutate_original(self):
        prog = oe.parse(CODE)
        ed.rename_variable(prog, "Sketch0", "S9")
        self.assertIn("Sketch0", oe.emit(prog))

    def test_rename_undefined_raises(self):
        prog = oe.parse(CODE)
        with self.assertRaises(ValueError):
            ed.rename_variable(prog, "Nope0", "X")

    def test_rename_collision_raises(self):
        prog = oe.parse(CODE)
        with self.assertRaises(ValueError):
            ed.rename_variable(prog, "Extrude0", "Sketch0")

    def test_rename_tuple_target(self):
        prog = oe.parse(CODE)
        renamed = ed.rename_variable(prog, "Curves0_0", "C0")
        text = oe.emit(renamed)
        self.assertIn("Loops0, C0 = [], []", text)
        self.assertIn("add_loop(C0)", text)


class TestSetKeyword(unittest.TestCase):
    def test_paper_example_change_extent(self):
        # Sec. 6.5: change extent_one from 1000 to 500.
        prog = oe.parse(CODE)
        edited = ed.set_keyword(prog, "Extrude0", "extent_one", 500.0)
        call = dict(edited.calls())["Extrude0"]
        self.assertEqual(call.keyword("extent_one"), 500.0)
        # Re-emitted code is still parseable and other params untouched.
        reparsed = oe.round_trip(edited)
        self.assertEqual(
            dict(reparsed.calls())["Extrude0"].keyword("extent_two"), 0.0)

    def test_append_new_keyword(self):
        prog = oe.parse(CODE)
        edited = ed.set_keyword(prog, "Line0_0_0", "construction", True)
        self.assertIs(
            dict(edited.calls())["Line0_0_0"].keyword("construction"), True)

    def test_reparametrize_position(self):
        prog = oe.parse(CODE)
        edited = ed.reparametrize(prog, "Sketch0", position=[10.0, 20.0], size=2.0)
        call = dict(edited.calls())["Sketch0"]
        self.assertEqual(call.keyword("position"), [10.0, 20.0])
        self.assertEqual(call.keyword("size"), 2.0)

    def test_unknown_target_raises(self):
        prog = oe.parse(CODE)
        with self.assertRaises(ValueError):
            ed.set_keyword(prog, "Loops0", "x", 1)  # tuple init, not a command

    def test_does_not_mutate_original(self):
        prog = oe.parse(CODE)
        ed.set_keyword(prog, "Extrude0", "extent_one", 500.0)
        self.assertEqual(
            dict(prog.calls())["Extrude0"].keyword("extent_one"), 1000.0)


if __name__ == "__main__":
    unittest.main()
