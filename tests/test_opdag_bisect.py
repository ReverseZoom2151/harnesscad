"""Tests for OpDAG.bisect — the content-hashed op-history binary search.

``bisect(predicate)`` returns the index of the first op after whose inclusion a
monotone predicate flips from good (True) to bad (False), or None when it never
flips. It is pure: it reads only the recorded op list, never replays.
"""

import unittest

from harnesscad.core.cisp.ops import NewSketch, AddCircle, AddRectangle, Extrude
from harnesscad.core.state.opdag import OpDAG


def _fill(dag, ops):
    for op in ops:
        dag.append(op)
    return dag


class TestBisect(unittest.TestCase):
    def test_empty_history_returns_none(self):
        self.assertIsNone(OpDAG().bisect(lambda ops: True))

    def test_finds_first_op_that_flips(self):
        # 6 ops; predicate "good" while len(prefix) <= 3 -> first bad is index 3.
        dag = _fill(OpDAG(), [NewSketch() for _ in range(6)])
        idx = dag.bisect(lambda ops: len(ops) <= 3)
        self.assertEqual(idx, 3)

    def test_first_op_already_bad(self):
        dag = _fill(OpDAG(), [NewSketch() for _ in range(4)])
        idx = dag.bisect(lambda ops: False)  # bad from the very first prefix
        self.assertEqual(idx, 0)

    def test_never_flips_returns_none(self):
        dag = _fill(OpDAG(), [NewSketch() for _ in range(4)])
        self.assertIsNone(dag.bisect(lambda ops: True))  # always good

    def test_predicate_on_op_content(self):
        # The "bad" op is the extrude with distance 0 (a real content predicate).
        ops = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", w=10.0, h=5.0),
            Extrude(sketch="sk1", distance=5.0),
            Extrude(sketch="sk1", distance=0.0),   # index 3 introduces the fault
            AddCircle(sketch="sk1", r=1.0),
        ]
        dag = _fill(OpDAG(), ops)

        def _good(prefix):
            # good == no zero-distance extrude present yet
            return not any(isinstance(o, Extrude) and o.distance == 0.0
                           for o in prefix)

        self.assertEqual(dag.bisect(_good), 3)

    def test_bisect_is_logarithmic_and_side_effect_free(self):
        calls = {"n": 0}
        dag = _fill(OpDAG(), [NewSketch() for _ in range(1000)])
        head_before = dag.head_hash

        def _pred(prefix):
            calls["n"] += 1
            return len(prefix) <= 500

        self.assertEqual(dag.bisect(_pred), 500)
        # O(log n): ~10 evaluations for 1000 ops, nowhere near linear.
        self.assertLess(calls["n"], 20)
        # No replay / mutation: history is untouched.
        self.assertEqual(len(dag), 1000)
        self.assertEqual(dag.head_hash, head_before)

    def test_single_bad_op(self):
        dag = _fill(OpDAG(), [Extrude(sketch="sk1", distance=0.0)])
        self.assertEqual(dag.bisect(lambda ops: False), 0)
        self.assertIsNone(_fill(OpDAG(), [NewSketch()]).bisect(lambda ops: True))


if __name__ == "__main__":
    unittest.main()
