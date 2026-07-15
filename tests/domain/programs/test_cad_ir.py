"""Tests for domain.programs.cad_ir."""

import unittest

from harnesscad.domain.programs.cad_ir import (
    CadIRError,
    Operation,
    Skill,
    instantiate_skill,
    topological_order,
    validate_program,
)


def _prog():
    return [
        {"name": "s1", "op": "sketch", "produces": ["face1"]},
        {"name": "e1", "op": "extrude", "consumes": ["face1"], "produces": ["solid1"]},
        {"name": "f1", "op": "fillet", "consumes": ["solid1"], "produces": ["solid2"]},
    ]


class ValidateTest(unittest.TestCase):
    def test_valid(self):
        validate_program(_prog())  # no raise

    def test_undefined_dependency(self):
        with self.assertRaises(CadIRError):
            validate_program([{"name": "e1", "op": "extrude", "consumes": ["ghost"]}])

    def test_double_produce(self):
        with self.assertRaises(CadIRError):
            validate_program(
                [
                    {"name": "a", "op": "sketch", "produces": ["f"]},
                    {"name": "b", "op": "sketch", "produces": ["f"]},
                ]
            )

    def test_duplicate_name(self):
        with self.assertRaises(CadIRError):
            validate_program(
                [{"name": "a", "op": "sketch"}, {"name": "a", "op": "extrude"}]
            )


class OrderTest(unittest.TestCase):
    def test_topo_order(self):
        self.assertEqual(topological_order(_prog()), ["s1", "e1", "f1"])

    def test_order_independent_of_input_permutation(self):
        p = _prog()
        shuffled = [p[2], p[0], p[1]]
        self.assertEqual(topological_order(shuffled), ["s1", "e1", "f1"])

    def test_cycle_detected(self):
        cyc = [
            {"name": "a", "op": "x", "consumes": ["eb"], "produces": ["ea"]},
            {"name": "b", "op": "y", "consumes": ["ea"], "produces": ["eb"]},
        ]
        with self.assertRaises(CadIRError):
            topological_order(cyc)


class SkillTest(unittest.TestCase):
    def _skill(self):
        return Skill(
            name="counterbore",
            parameters={"radius": None, "depth": 5.0},
            template=[
                {"name": "hole_sketch", "op": "sketch", "produces": ["hf"],
                 "params": {"r": "$radius"}},
                {"name": "hole_cut", "op": "extrude", "consumes": ["hf"],
                 "produces": ["cut"], "params": {"d": "$depth"}},
            ],
        )

    def test_instantiate(self):
        ops = instantiate_skill(self._skill(), {"radius": 3.0})
        self.assertEqual(len(ops), 2)
        self.assertEqual(ops[0].params["r"], 3.0)
        self.assertEqual(ops[1].params["d"], 5.0)  # default used

    def test_missing_required(self):
        with self.assertRaises(CadIRError):
            instantiate_skill(self._skill(), {})

    def test_unknown_argument(self):
        with self.assertRaises(CadIRError):
            instantiate_skill(self._skill(), {"radius": 3.0, "bogus": 1})

    def test_result_is_valid_program(self):
        ops = instantiate_skill(self._skill(), {"radius": 2.0})
        self.assertEqual(topological_order(ops), ["hole_sketch", "hole_cut"])


if __name__ == "__main__":
    unittest.main()
