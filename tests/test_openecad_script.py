"""Tests for the OpenECAD editable CAD-script format."""

import unittest

from harnesscad.domain.programs import openecad_script as oe


# A canonical OpenECAD program mirroring Algorithm 1 / Algorithm 3.
CODE = "\n".join([
    'SketchPlane0 = add_sketchplane(origin=[0.0, 0.0, 0.0], '
    'normal=[0.0, 0.0, 1.0], x_axis=[1.0, 0.0, 0.0])',
    'Loops0, Curves0_0 = [], []',
    'Line0_0_0 = add_line(start=[-1000.0, -750.0], end=[1000.0, -750.0])',
    'Line0_0_1 = add_line(start=[1000.0, -750.0], end=[1000.0, 750.0])',
    'Arc0_0_2 = add_arc(start=[1000.0, 750.0], end=[-1000.0, 750.0], '
    'mid=[0.0, 900.0])',
    'Line0_0_3 = add_line(start=[-1000.0, 750.0], end=[-1000.0, -750.0])',
    'Loop0_0 = add_loop(Curves0_0)',
    'Profile0 = add_profile(Loops0)',
    'Sketch0 = add_sketch(SketchPlane0, Profile0, position=[0.0, 0.0], '
    'size=1.0)',
    'Extrude0 = add_extrude(Sketch0, operation=1, type=0, '
    'extent_one=1000.0, extent_two=0.0)',
    'SketchPlane1 = add_sketchplane_ref(Extrude0, origin=[0.0, 0.0], '
    'type="sameplan", reverse=True)',
])


class TestParse(unittest.TestCase):
    def test_parses_all_statements(self):
        prog = oe.parse(CODE)
        self.assertEqual(len(prog.statements), 11)

    def test_tuple_init(self):
        prog = oe.parse(CODE)
        st = prog.statements[1]
        self.assertEqual(st.targets, ("Loops0", "Curves0_0"))
        self.assertEqual(st.values, ([], []))
        self.assertIsNone(st.call)

    def test_single_command_assignment(self):
        prog = oe.parse(CODE)
        st = prog.statements[0]
        self.assertEqual(st.targets, ("SketchPlane0",))
        self.assertIsInstance(st.call, oe.Call)
        self.assertEqual(st.call.func, oe.ADD_SKETCHPLANE)

    def test_keyword_and_positional_args(self):
        prog = oe.parse(CODE)
        sketch = prog.calls_of(oe.ADD_SKETCH)[0]
        self.assertEqual(len(sketch.positional()), 2)
        self.assertEqual(sketch.keyword("size"), 1.0)

    def test_negative_numbers(self):
        prog = oe.parse(CODE)
        line = prog.calls_of(oe.ADD_LINE)[0]
        self.assertEqual(line.keyword("start"), [-1000.0, -750.0])

    def test_variable_reference(self):
        prog = oe.parse(CODE)
        ref = prog.calls_of(oe.ADD_SKETCHPLANE_REF)[0]
        self.assertIsInstance(ref.positional()[0], oe.Ref)
        self.assertEqual(ref.positional()[0].name, "Extrude0")

    def test_string_and_bool_kwargs(self):
        prog = oe.parse(CODE)
        ref = prog.calls_of(oe.ADD_SKETCHPLANE_REF)[0]
        self.assertEqual(ref.keyword("type"), "sameplan")
        self.assertIs(ref.keyword("reverse"), True)


class TestEmit(unittest.TestCase):
    def test_integral_float_style(self):
        self.assertEqual(oe.emit_value(1000.0), "1000.0")
        self.assertEqual(oe.emit_value(-750.0), "-750.0")

    def test_int_vs_bool(self):
        self.assertEqual(oe.emit_value(1), "1")
        self.assertEqual(oe.emit_value(True), "True")

    def test_emit_call(self):
        call = oe.Call(oe.ADD_CIRCLE, (
            oe.Arg([0.0, 0.0], "center"), oe.Arg(5.0, "radius")))
        self.assertEqual(
            oe.emit_call(call), 'add_circle(center=[0.0, 0.0], radius=5.0)')


class TestRoundTrip(unittest.TestCase):
    def test_emit_parse_idempotent(self):
        prog = oe.parse(CODE)
        self.assertEqual(oe.emit(oe.round_trip(prog)), oe.emit(prog))

    def test_canonical_code_stable(self):
        # The canonical CODE is already in emit() canonical form.
        self.assertEqual(oe.emit(oe.parse(CODE)), CODE)


class TestValidation(unittest.TestCase):
    def test_unknown_command_rejected(self):
        with self.assertRaises(ValueError):
            oe.Call("add_spline")

    def test_bad_ref_rejected(self):
        with self.assertRaises(ValueError):
            oe.Ref("1bad")

    def test_non_assignment_rejected(self):
        with self.assertRaises(ValueError):
            oe.parse("add_line(start=[0,0], end=[1,1])")

    def test_target_value_mismatch(self):
        with self.assertRaises(ValueError):
            oe.Assign(("a", "b"), ([],))


if __name__ == "__main__":
    unittest.main()
