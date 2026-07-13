import unittest

from harnesscad.domain.programs.validate.cadquery_workplane import (
    calls_from_code,
    is_valid_code,
    validate_calls,
    validate_code,
)


def errors(diags):
    return [d for d in diags if d.severity == "error"]


class TestCallsFromCode(unittest.TestCase):
    def test_extract_chain(self):
        code = 'cq.Workplane("XY").lineTo(1, 0).lineTo(1, 1).close().extrude(0.5)'
        chain = calls_from_code(code)
        self.assertEqual(
            [m for m, _ in chain], ["lineTo", "lineTo", "close", "extrude"]
        )

    def test_argcount(self):
        chain = calls_from_code('cq.Workplane("XY").circle(5).extrude(2)')
        self.assertEqual(chain, [("circle", 1), ("extrude", 1)])

    def test_assignment_form(self):
        code = 'r = cq.Workplane("XY").box(1, 2, 3)'
        self.assertEqual(calls_from_code(code), [("box", 3)])


class TestValidPrograms(unittest.TestCase):
    def test_line_close_extrude(self):
        code = 'cq.Workplane("XY").lineTo(1, 0).lineTo(1, 1).close().extrude(0.5)'
        self.assertTrue(is_valid_code(code))
        self.assertEqual(errors(validate_code(code)), [])

    def test_circle_extrude(self):
        self.assertTrue(is_valid_code('cq.Workplane("XY").circle(5).extrude(2)'))

    def test_box_primitive(self):
        self.assertTrue(is_valid_code('cq.Workplane("XY").box(1, 1, 1)'))


class TestPendingViolations(unittest.TestCase):
    def test_extrude_no_pending_wire(self):
        # extrude straight after Workplane -- nothing to build
        diags = validate_calls([("extrude", 1)])
        self.assertTrue(any("no pending wire" in d.message for d in errors(diags)))

    def test_extrude_with_unfused_edges(self):
        code = 'cq.Workplane("XY").lineTo(1, 0).lineTo(1, 1).extrude(0.5)'
        diags = validate_code(code)
        self.assertTrue(
            any("never fused into a wire" in d.message for d in errors(diags))
        )

    def test_close_no_open_path(self):
        diags = validate_calls([("close", 0)])
        self.assertTrue(any("no open path" in d.message for d in errors(diags)))

    def test_loft_needs_two_profiles(self):
        # one circle then loft -> only one profile
        diags = validate_calls([("circle", 1), ("loft", 0)])
        self.assertTrue(
            any("at least 2 profiles" in d.message for d in errors(diags))
        )

    def test_loft_two_profiles_ok(self):
        diags = validate_calls(
            [("circle", 1), ("workplane", 1), ("circle", 1), ("loft", 0)]
        )
        # switching planes with no dangling edges + 2 wires: no error
        self.assertEqual(errors(diags), [])

    def test_boolean_without_base(self):
        diags = validate_calls([("cut", 1)])
        self.assertTrue(any("no base solid" in d.message for d in errors(diags)))


class TestPlaneWarnings(unittest.TestCase):
    def test_incompatible_planes_warn(self):
        # two profiles on two planes fed to a single extrude
        diags = validate_calls(
            [("circle", 1), ("workplane", 1), ("circle", 1), ("extrude", 1)]
        )
        self.assertTrue(
            any("incompatible planes" in d.message for d in diags)
        )

    def test_plane_switch_with_open_edges_warns(self):
        diags = validate_calls([("lineTo", 2), ("workplane", 1)])
        self.assertTrue(
            any("unfused pending edges" in d.message for d in diags)
        )

    def test_dangling_wire_warning(self):
        diags = validate_calls([("circle", 1)])
        self.assertTrue(any("never consumed" in d.message for d in diags))


if __name__ == "__main__":
    unittest.main()
