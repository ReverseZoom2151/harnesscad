"""Tests for domain.reconstruction.scene.assembly_trace."""

import unittest

from harnesscad.domain.reconstruction.scene.assembly_trace import (
    Part,
    assembly_order,
    build_trace,
    component_numeracy,
    trace_faithfulness,
)


def _chair():
    return [
        Part("seat", "seat"),
        Part("leg1", "leg", parent="seat"),
        Part("leg2", "leg", parent="seat"),
        Part("back", "back", parent="seat"),
    ]


class OrderTest(unittest.TestCase):
    def test_parent_before_child(self):
        order = assembly_order(_chair())
        names = [p.name for p in order]
        self.assertEqual(names[0], "seat")
        self.assertLess(names.index("seat"), names.index("leg1"))

    def test_dangling_parent(self):
        with self.assertRaises(ValueError):
            assembly_order([Part("a", "x", parent="ghost")])

    def test_cycle(self):
        with self.assertRaises(ValueError):
            assembly_order([Part("a", "x", parent="b"), Part("b", "y", parent="a")])


class TraceTest(unittest.TestCase):
    def test_trace_grows(self):
        trace = build_trace(_chair())
        self.assertEqual(len(trace), 5)  # s0..s4
        self.assertEqual(trace[0], tuple())
        self.assertEqual(len(trace[-1]), 4)

    def test_faithfulness_perfect(self):
        self.assertEqual(trace_faithfulness(build_trace(_chair())), 1.0)

    def test_faithfulness_detects_removal(self):
        p = _chair()
        trace = build_trace(p)
        # corrupt: a step that removes a part
        bad = trace[:2] + [tuple()] + trace[2:]
        self.assertLess(trace_faithfulness(bad), 1.0)


class NumeracyTest(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(
            component_numeracy(_chair(), {"leg": 2, "seat": 1, "back": 1}), 1.0
        )

    def test_wrong_count(self):
        # target expects 4 legs, only 2 present
        self.assertAlmostEqual(
            component_numeracy(_chair(), {"leg": 4, "seat": 1}), 0.5
        )

    def test_empty_target(self):
        with self.assertRaises(ValueError):
            component_numeracy(_chair(), {})


if __name__ == "__main__":
    unittest.main()
