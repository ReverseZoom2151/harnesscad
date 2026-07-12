import unittest

from datagen.pllm_program_augment import (
    BOOLEAN_OPS, Program, W_MAX, diversify, expand_append, expand_spawn,
    procedural_workspace, program_length, shorten_remove_boolean,
)
import random


def _prog():
    return Program([["sketch", "extrude"], ["sketch2", "extrude2"]], ["union"])


class TestLength(unittest.TestCase):
    def test_counts_ops_and_booleans(self):
        self.assertEqual(program_length(_prog()), 2 + 2 + 1)

    def test_tuple_form(self):
        self.assertEqual(program_length(([["a"]], [])), 1)


class TestExpandAppend(unittest.TestCase):
    def test_appends_to_last(self):
        out = expand_append(_prog(), ["fillet"])
        self.assertEqual(out.workspaces[-1][-1], "fillet")
        self.assertEqual(program_length(out), 6)

    def test_does_not_mutate_original(self):
        p = _prog()
        expand_append(p, ["x"])
        self.assertEqual(program_length(p), 5)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            expand_append(Program([], []), ["x"])


class TestExpandSpawn(unittest.TestCase):
    def test_adds_workspace_and_boolean(self):
        out = expand_spawn(_prog(), ["hole"], "cut")
        self.assertEqual(len(out.workspaces), 3)
        self.assertEqual(out.booleans[-1], "cut")

    def test_respects_cap(self):
        p = Program([["a"], ["b"], ["c"], ["d"], ["e"]], ["union"] * 4)
        self.assertEqual(len(p.workspaces), W_MAX)
        out = expand_spawn(p, ["f"], "union")
        self.assertEqual(len(out.workspaces), W_MAX)  # unchanged at cap

    def test_bad_boolean(self):
        with self.assertRaises(ValueError):
            expand_spawn(_prog(), ["x"], "blend")


class TestProceduralWorkspace(unittest.TestCase):
    def test_deterministic(self):
        a = procedural_workspace(random.Random(7))
        b = procedural_workspace(random.Random(7))
        self.assertEqual(a, b)

    def test_length_range(self):
        ws = procedural_workspace(random.Random(1), min_ops=2, max_ops=2)
        self.assertEqual(len(ws), 2)

    def test_bad_range(self):
        with self.assertRaises(ValueError):
            procedural_workspace(random.Random(0), min_ops=0)


class TestShorten(unittest.TestCase):
    def test_removes_boolean_and_workspace(self):
        out = shorten_remove_boolean(_prog())
        self.assertEqual(out.booleans, [])
        self.assertEqual(len(out.workspaces), 1)
        self.assertLess(program_length(out), program_length(_prog()))

    def test_keeps_base_workspace(self):
        out = shorten_remove_boolean(_prog())
        self.assertEqual(out.workspaces[0], ["sketch", "extrude"])

    def test_no_boolean_raises(self):
        with self.assertRaises(ValueError):
            shorten_remove_boolean(Program([["a"]], []))


class TestDiversify(unittest.TestCase):
    def test_produces_longer_and_shorter(self):
        out = diversify(_prog(), seed=3, n_expand=1, n_shorten=1)
        orig = out["lengths"]["original"]
        self.assertGreater(out["lengths"]["expanded"][0], orig)
        self.assertLess(out["lengths"]["shortened"][0], orig)

    def test_deterministic(self):
        a = diversify(_prog(), seed=5)
        b = diversify(_prog(), seed=5)
        self.assertEqual(a["lengths"], b["lengths"])
        self.assertEqual(a["expanded"][0].workspaces, b["expanded"][0].workspaces)

    def test_expansion_respects_cap(self):
        p = Program([["a"], ["b"], ["c"], ["d"], ["e"]], ["union"] * 4)
        out = diversify(p, seed=1, n_expand=3, n_shorten=0)
        self.assertEqual(out["expanded"], [])  # already at W_MAX

    def test_shorten_stops_when_no_booleans(self):
        out = diversify(_prog(), seed=1, n_expand=0, n_shorten=5)
        # only one boolean available -> at most one shortened variant
        self.assertEqual(len(out["shortened"]), 1)

    def test_boolean_ops_constant(self):
        self.assertEqual(set(BOOLEAN_OPS), {"union", "cut", "intersect"})


if __name__ == "__main__":
    unittest.main()
