"""Tests for domain.programs.featurescript_ops."""

import unittest

from harnesscad.domain.programs.featurescript_ops import (
    FeatureScriptError,
    op_names,
    validate_program,
)


def _valid():
    return [
        {"op": "sketch", "params": {}},
        {"op": "extrude", "params": {"distance": 10},
         "queries": [{"op_index": 0, "role": "face"}]},
        # fillet references an edge created by the extrude (op 1) -- not the sketch.
        {"op": "fillet", "params": {"radius": 1.0},
         "queries": [{"op_index": 1, "role": "edge"}]},
        {"op": "pattern", "params": {"count": 4},
         "queries": [{"op_index": 1, "role": "solid"}]},
    ]


class VocabTest(unittest.TestCase):
    def test_has_extended_ops(self):
        names = op_names()
        for op in ("loft", "revolve", "chamfer", "shell", "sweep", "boolean"):
            self.assertIn(op, names)
        self.assertGreaterEqual(len(names), 15)


class ValidateTest(unittest.TestCase):
    def test_valid_program(self):
        validate_program(_valid())  # no raise

    def test_far_back_reference_allowed(self):
        prog = _valid()
        # chamfer referencing the sketch (op 0), several ops back.
        prog.append({"op": "chamfer", "params": {"distance": 0.5},
                     "queries": [{"op_index": 0, "role": "edge"}]})
        validate_program(prog)

    def test_unknown_op(self):
        with self.assertRaises(FeatureScriptError):
            validate_program([{"op": "warp", "params": {}}])

    def test_missing_param(self):
        with self.assertRaises(FeatureScriptError):
            validate_program([
                {"op": "sketch", "params": {}},
                {"op": "extrude", "params": {}, "queries": [{"op_index": 0, "role": "face"}]},
            ])

    def test_query_must_be_earlier(self):
        with self.assertRaises(FeatureScriptError):
            validate_program([
                {"op": "sketch", "params": {}},
                {"op": "fillet", "params": {"radius": 1.0},
                 "queries": [{"op_index": 1, "role": "edge"}]},  # self-reference
            ])

    def test_query_role_must_be_produced(self):
        with self.assertRaises(FeatureScriptError):
            validate_program([
                {"op": "sketch", "params": {}},
                {"op": "fillet", "params": {"radius": 1.0},
                 "queries": [{"op_index": 0, "role": "solid"}]},  # sketch makes no solid
            ])

    def test_needs_query_enforced(self):
        with self.assertRaises(FeatureScriptError):
            validate_program([
                {"op": "sketch", "params": {}},
                {"op": "extrude", "params": {"distance": 5}},  # no query
            ])

    def test_boolean_kind(self):
        with self.assertRaises(FeatureScriptError):
            validate_program([
                {"op": "sketch", "params": {}},
                {"op": "extrude", "params": {"distance": 5},
                 "queries": [{"op_index": 0, "role": "face"}]},
                {"op": "boolean", "params": {"kind": "xor"},
                 "queries": [{"op_index": 1, "role": "solid"}]},
            ])

    def test_pattern_count(self):
        with self.assertRaises(FeatureScriptError):
            validate_program([
                {"op": "sketch", "params": {}},
                {"op": "extrude", "params": {"distance": 5},
                 "queries": [{"op_index": 0, "role": "face"}]},
                {"op": "pattern", "params": {"count": 1},
                 "queries": [{"op_index": 1, "role": "solid"}]},
            ])


if __name__ == "__main__":
    unittest.main()
