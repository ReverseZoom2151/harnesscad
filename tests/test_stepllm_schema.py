import unittest

from formats.stepllm_parser import Entity, Enum, Real, Ref, serialize_entity
from formats.stepllm_schema import (
    axis2_placement_3d, cartesian_point, check_attributes, circle,
    direction, entity_def, known, line, make, plane,
)


class TestDefs(unittest.TestCase):
    def test_known(self):
        self.assertTrue(known("CARTESIAN_POINT"))
        self.assertFalse(known("FOOBAR"))

    def test_arity(self):
        self.assertEqual(entity_def("AXIS2_PLACEMENT_3D").arity, 4)
        self.assertEqual(entity_def("PLANE").arity, 2)

    def test_unknown_lookup_raises(self):
        with self.assertRaises(KeyError):
            entity_def("NOPE")


class TestConstructors(unittest.TestCase):
    def test_cartesian_point_serializes(self):
        e = cartesian_point(1, 0, 0, 0)
        self.assertEqual(serialize_entity(e), "#1=CARTESIAN_POINT('',(0.,0.,0.));")

    def test_cartesian_point_fractional(self):
        e = cartesian_point(1, 1.5, 0, 0)
        self.assertEqual(e.params[1][0], Real("1.5"))

    def test_direction(self):
        e = direction(2, 0, 0, 1)
        self.assertEqual(serialize_entity(e), "#2=DIRECTION('',(0.,0.,1.));")

    def test_placement_refs(self):
        e = axis2_placement_3d(4, 1, 2, 3)
        self.assertEqual(e.params[1:], [Ref(1), Ref(2), Ref(3)])

    def test_circle_and_plane_and_line(self):
        self.assertEqual(circle(5, 4, 2.0).keyword, "CIRCLE")
        self.assertEqual(plane(6, 4).params, ["", Ref(4)])
        self.assertEqual(line(7, 1, 2).params, ["", Ref(1), Ref(2)])

    def test_make_arity_guard(self):
        with self.assertRaises(ValueError):
            make("PLANE", 1, "")  # missing position


class TestCheckAttributes(unittest.TestCase):
    def test_valid_instance_no_problems(self):
        self.assertEqual(check_attributes(cartesian_point(1, 0, 0, 0)), [])

    def test_unknown_entity_skipped(self):
        e = Entity(1, "SOMETHING_ELSE", ["", Ref(2)])
        self.assertEqual(check_attributes(e), [])

    def test_wrong_arity_reported(self):
        e = Entity(1, "PLANE", [""])  # missing ref
        problems = check_attributes(e)
        self.assertEqual(len(problems), 1)
        self.assertIn("expected 2 attributes", problems[0])

    def test_wrong_kind_reported(self):
        # PLANE.position must be a ref, not a string.
        e = Entity(1, "PLANE", ["", "not_a_ref"])
        problems = check_attributes(e)
        self.assertEqual(len(problems), 1)
        self.assertIn("position", problems[0])

    def test_bool_attribute_kind(self):
        e = Entity(1, "EDGE_CURVE",
                   ["", Ref(2), Ref(3), Ref(4), Enum("T")])
        self.assertEqual(check_attributes(e), [])

    def test_bad_bool_reported(self):
        e = Entity(1, "EDGE_CURVE",
                   ["", Ref(2), Ref(3), Ref(4), Enum("PLANE")])
        self.assertEqual(len(check_attributes(e)), 1)

    def test_reflist_and_reallist(self):
        good = Entity(1, "EDGE_LOOP", ["", [Ref(2), Ref(3)]])
        self.assertEqual(check_attributes(good), [])
        bad = Entity(2, "EDGE_LOOP", ["", [Ref(2), 3]])
        self.assertEqual(len(check_attributes(bad)), 1)

    def test_int_accepted_for_real(self):
        # A radius given as an int literal is a valid real per part-21.
        e = Entity(1, "CIRCLE", ["", Ref(2), 5])
        self.assertEqual(check_attributes(e), [])


if __name__ == "__main__":
    unittest.main()
