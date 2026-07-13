"""Tests for datagen/contrastcad_rre.py — RRE data augmentation."""

import random
import unittest

from harnesscad.data.datagen.contrastcad_rre import (
    ARC,
    EXTRUDE,
    EXTRUDE_TYPES,
    LINE,
    SOL,
    EOS,
    randomize_extrusions,
    replace_lines_with_arcs,
    rre_augment,
    split_pairs,
    swap_pairs,
)


def _seq_a():
    return [
        {"type": SOL},
        {"type": LINE, "x": 10, "y": 20},
        {"type": LINE, "x": 30, "y": 40},
        {"type": LINE, "x": 50, "y": 60},
        {"type": EXTRUDE, "w": 0, "delta1": 100, "delta2": 0, "s": 128, "b": 0},
        {"type": EOS},
    ]


def _seq_b():
    return [
        {"type": SOL},
        {"type": "C", "x": 5, "y": 5, "r": 3},
        {"type": EXTRUDE, "w": 1, "delta1": 200, "delta2": 5, "s": 64, "b": 1},
        {"type": EOS},
    ]


class TestReplaceLinesWithArcs(unittest.TestCase):
    def test_deterministic(self):
        a = replace_lines_with_arcs(_seq_a(), 7, 0.5)
        b = replace_lines_with_arcs(_seq_a(), 7, 0.5)
        self.assertEqual(a, b)

    def test_full_replacement_turns_all_lines_to_arcs(self):
        out = replace_lines_with_arcs(_seq_a(), 1, replace_prob=1.0)
        arcs = [c for c in out if c["type"] == ARC]
        lines = [c for c in out if c["type"] == LINE]
        self.assertEqual(len(arcs), 3)
        self.assertEqual(len(lines), 0)

    def test_endpoint_preserved_and_params_in_range(self):
        out = replace_lines_with_arcs(_seq_a(), 2, replace_prob=1.0)
        src_lines = [c for c in _seq_a() if c["type"] == LINE]
        arcs = [c for c in out if c["type"] == ARC]
        for src, arc in zip(src_lines, arcs):
            self.assertEqual((arc["x"], arc["y"]), (src["x"], src["y"]))
            self.assertTrue(1 <= arc["theta"] <= 255)
            self.assertIn(arc["c"], (0, 1))

    def test_zero_prob_is_noop(self):
        out = replace_lines_with_arcs(_seq_a(), 3, replace_prob=0.0)
        self.assertEqual(out, _seq_a())

    def test_invalid_prob(self):
        with self.assertRaises(ValueError):
            replace_lines_with_arcs(_seq_a(), 1, replace_prob=1.5)


class TestRandomizeExtrusions(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(randomize_extrusions(_seq_a(), 9),
                         randomize_extrusions(_seq_a(), 9))

    def test_resamples_type_and_distances(self):
        out = randomize_extrusions(_seq_a(), 5)
        ext = [c for c in out if c["type"] == EXTRUDE][0]
        self.assertIn(ext["w"], EXTRUDE_TYPES)
        self.assertTrue(0 <= ext["delta1"] <= 255)
        self.assertTrue(0 <= ext["delta2"] <= 255)

    def test_preserves_non_resampled_params(self):
        out = randomize_extrusions(_seq_a(), 5)
        ext = [c for c in out if c["type"] == EXTRUDE][0]
        self.assertEqual(ext["s"], 128)
        self.assertEqual(ext["b"], 0)

    def test_non_extrusions_unchanged(self):
        out = randomize_extrusions(_seq_a(), 5)
        self.assertEqual([c for c in out if c["type"] != EXTRUDE],
                         [c for c in _seq_a() if c["type"] != EXTRUDE])


class TestSplitPairs(unittest.TestCase):
    def test_split_counts(self):
        seq = _seq_a()[:-1] + _seq_b()  # two extrusion pairs then a tail
        pairs = split_pairs(seq)
        extr = [p for p in pairs if p[-1]["type"] == EXTRUDE]
        self.assertEqual(len(extr), 2)

    def test_pair_ends_with_extrusion(self):
        pairs = split_pairs(_seq_a())
        self.assertEqual(pairs[0][-1]["type"], EXTRUDE)


class TestSwapPairs(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(swap_pairs(_seq_a(), _seq_b(), 4, 1.0),
                         swap_pairs(_seq_a(), _seq_b(), 4, 1.0))

    def test_full_swap_pulls_in_donor_pair(self):
        out = swap_pairs(_seq_a(), _seq_b(), 1, swap_prob=1.0)
        # donor pair is the circle+extrusion pair from seq_b
        self.assertTrue(any(c["type"] == "C" for c in out))

    def test_zero_prob_keeps_own_pairs(self):
        out = swap_pairs(_seq_a(), _seq_b(), 1, swap_prob=0.0)
        self.assertEqual(out, [dict(c) for c in _seq_a()])


class TestRREAugment(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(rre_augment(_seq_a(), 11, _seq_b()),
                         rre_augment(_seq_a(), 11, _seq_b()))

    def test_seed_changes_output(self):
        self.assertNotEqual(rre_augment(_seq_a(), 1, _seq_b()),
                            rre_augment(_seq_a(), 2, _seq_b()))

    def test_runs_without_donor(self):
        out = rre_augment(_seq_a(), 3, replace_prob=1.0)
        self.assertTrue(any(c["type"] == ARC for c in out))

    def test_accepts_random_instance(self):
        rng = random.Random(0)
        out = rre_augment(_seq_a(), rng, _seq_b())
        self.assertIsInstance(out, list)


if __name__ == "__main__":
    unittest.main()
