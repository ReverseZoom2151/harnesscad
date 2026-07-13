"""Tests for reconstruction.rlcad_command_spec (RLCAD extended DSL + actions)."""

import unittest

from harnesscad.domain.reconstruction.tokens import rlcad_command_spec as cs


class TestExtrudeCommand(unittest.TestCase):
    def test_distinct_faces_required(self):
        with self.assertRaises(ValueError):
            cs.ExtrudeCommand(3, 3, cs.UNION)

    def test_dsl(self):
        e = cs.ExtrudeCommand(1, 2, cs.UNION)
        self.assertEqual(e.to_dsl(), "add_extrude(1, 2, union)")
        self.assertEqual(e.action_type, cs.EXTRUDE)

    def test_bad_op(self):
        with self.assertRaises(ValueError):
            cs.ExtrudeCommand(1, 2, "chamfer")


class TestRevolveCommand(unittest.TestCase):
    def test_dsl(self):
        r = cs.RevolveCommand(5, cs.SUBTRACTION)
        self.assertEqual(r.to_dsl(), "add_revolve(5, subtraction)")
        self.assertEqual(r.action_type, cs.REVOLVE)

    def test_default_op_newbody(self):
        self.assertEqual(cs.RevolveCommand(7).op, cs.NEWBODY)


class TestActionEncoding(unittest.TestCase):
    def test_extrude_roundtrip(self):
        e = cs.ExtrudeCommand(4, 9, cs.INTERSECTION)
        a = cs.encode_action(e)
        self.assertEqual(a, (4, 9, cs.INTERSECTION, cs.EXTRUDE))
        self.assertEqual(cs.decode_action(a), e)

    def test_revolve_roundtrip_fs_eq_fe(self):
        r = cs.RevolveCommand(6, cs.UNION)
        a = cs.encode_action(r)
        self.assertEqual(a, (6, 6, cs.UNION, cs.REVOLVE))
        self.assertEqual(cs.decode_action(a), r)

    def test_extrude_invalid_when_faces_equal(self):
        self.assertFalse(cs.is_valid_action((2, 2, cs.UNION, cs.EXTRUDE)))

    def test_revolve_invalid_when_faces_differ(self):
        self.assertFalse(cs.is_valid_action((2, 3, cs.UNION, cs.REVOLVE)))

    def test_valid_actions(self):
        self.assertTrue(cs.is_valid_action((1, 2, cs.NEWBODY, cs.EXTRUDE)))
        self.assertTrue(cs.is_valid_action((8, 8, cs.NEWBODY, cs.REVOLVE)))

    def test_unknown_op_or_type_invalid(self):
        self.assertFalse(cs.is_valid_action((1, 2, "bad", cs.EXTRUDE)))
        self.assertFalse(cs.is_valid_action((1, 1, cs.UNION, "loft")))

    def test_wrong_length_invalid(self):
        self.assertFalse(cs.is_valid_action((1, 2, cs.UNION)))

    def test_decode_rejects_invalid(self):
        with self.assertRaises(ValueError):
            cs.decode_action((2, 2, cs.UNION, cs.EXTRUDE))

    def test_action_to_index(self):
        self.assertEqual(
            cs.action_to_index((4, 9, cs.INTERSECTION, cs.EXTRUDE)),
            (4, 9, cs.BOOLEAN_INDEX[cs.INTERSECTION], cs.ACTION_TYPE_INDEX[cs.EXTRUDE]))


class TestModelSequence(unittest.TestCase):
    def test_dsl_and_counts(self):
        seq = cs.ModelSequence(0, (
            cs.ExtrudeCommand(1, 2, cs.NEWBODY),
            cs.RevolveCommand(3, cs.UNION),
            cs.RevolveCommand(4, cs.SUBTRACTION),
        ))
        self.assertEqual(
            seq.to_dsl(),
            "G0; add_extrude(1, 2, newbody); add_revolve(3, union); "
            "add_revolve(4, subtraction)")
        self.assertEqual(seq.revolve_count(), 2)
        self.assertEqual(seq.action_types(),
                         (cs.EXTRUDE, cs.REVOLVE, cs.REVOLVE))

    def test_empty_sequence_dsl(self):
        self.assertEqual(cs.ModelSequence(2).to_dsl(), "G2")

    def test_validate_sequence(self):
        seq = cs.ModelSequence(0, (
            cs.ExtrudeCommand(1, 2), cs.RevolveCommand(3)))
        self.assertTrue(cs.validate_sequence(seq))

    def test_vocab_sizes(self):
        self.assertEqual(len(cs.BOOLEAN_OPS), 4)
        self.assertEqual(len(cs.ACTION_TYPES), 2)


if __name__ == "__main__":
    unittest.main()
