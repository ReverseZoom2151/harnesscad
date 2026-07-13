"""Tests for datagen/contrastcad_permute.py — shape-preserving permutation."""

import collections
import unittest

from harnesscad.data.datagen.contrastcad_permute import (
    cyclic_shift_loop,
    permute_sequence,
    swap_circle_loops,
)
from harnesscad.data.datagen.contrastcad_rre import CIRCLE, EXTRUDE, LINE, SOL


def _loop_p1():
    return [
        {"type": SOL},
        {"type": LINE, "x": 1, "y": 1},
        {"type": LINE, "x": 2, "y": 2},
        {"type": LINE, "x": 3, "y": 3},
        {"type": LINE, "x": 4, "y": 4},
    ]


def _pair_p3():
    return [
        {"type": SOL},
        {"type": CIRCLE, "x": 1, "y": 1, "r": 5},
        {"type": SOL},
        {"type": CIRCLE, "x": 9, "y": 9, "r": 2},
        {"type": EXTRUDE, "w": 0, "delta1": 10, "delta2": 0},
    ]


def _multiset(seq):
    return collections.Counter(tuple(sorted(c.items())) for c in seq)


class TestCyclicShiftLoop(unittest.TestCase):
    def test_shift_rotates_curves(self):
        shifted = cyclic_shift_loop(_loop_p1(), 1)
        xs = [c["x"] for c in shifted if c["type"] == LINE]
        self.assertEqual(xs, [2, 3, 4, 1])

    def test_shift_preserves_sol_head(self):
        shifted = cyclic_shift_loop(_loop_p1(), 2)
        self.assertEqual(shifted[0]["type"], SOL)

    def test_full_rotation_is_identity(self):
        self.assertEqual(cyclic_shift_loop(_loop_p1(), 4), _loop_p1())

    def test_shape_multiset_preserved(self):
        self.assertEqual(_multiset(cyclic_shift_loop(_loop_p1(), 3)),
                         _multiset(_loop_p1()))


class TestSwapCircleLoops(unittest.TestCase):
    def test_swaps_two_circles(self):
        out = swap_circle_loops(_pair_p3())
        circles = [c for c in out if c["type"] == CIRCLE]
        self.assertEqual((circles[0]["x"], circles[1]["x"]), (9, 1))

    def test_extrusion_preserved(self):
        out = swap_circle_loops(_pair_p3())
        self.assertEqual(out[-1]["type"], EXTRUDE)

    def test_shape_multiset_preserved(self):
        self.assertEqual(_multiset(swap_circle_loops(_pair_p3())),
                         _multiset(_pair_p3()))


class TestPermuteSequence(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(permute_sequence(_loop_p1(), 5),
                         permute_sequence(_loop_p1(), 5))

    def test_preserves_shape_multiset(self):
        seq = _loop_p1() + [{"type": EXTRUDE, "w": 0, "delta1": 5, "delta2": 0}]
        out = permute_sequence(seq, 7)
        self.assertEqual(_multiset(out), _multiset(seq))

    def test_length_preserved(self):
        out = permute_sequence(_pair_p3(), 3)
        self.assertEqual(len(out), len(_pair_p3()))

    def test_p3_multiset_preserved(self):
        out = permute_sequence(_pair_p3(), 11)
        self.assertEqual(_multiset(out), _multiset(_pair_p3()))


if __name__ == "__main__":
    unittest.main()
