"""Tests for HNC-CAD code-tree control (generation.hnc_code_control)."""

import unittest

from harnesscad.agents.generation.code_tree_control import (
    LOOP,
    PROFILE,
    SEP,
    SOLID,
    CodeNode,
    CodeTree,
    FeatureLayout,
    autocomplete_mask,
    edit_code,
    level_mask,
    serialize,
)


def sample_tree():
    # one solid, two profiles, loops (2, 2) -> paper's [S,SEP,P,L,L,SEP,P,L,L]
    return CodeTree(solid=7, profiles=((3, (1, 2)), (4, (5, 6))))


class TestSerialize(unittest.TestCase):
    def test_paper_order(self):
        els = serialize(sample_tree())
        kinds = [SEP if e == SEP else e.level for e in els]
        self.assertEqual(
            kinds,
            [SOLID, SEP, PROFILE, LOOP, LOOP, SEP, PROFILE, LOOP, LOOP],
        )

    def test_codes_preserved(self):
        els = serialize(sample_tree())
        self.assertEqual(els[0], CodeNode(SOLID, 7))
        self.assertEqual(els[2], CodeNode(PROFILE, 3))
        self.assertEqual(els[3], CodeNode(LOOP, 1))


class TestFeatureLayout(unittest.TestCase):
    def setUp(self):
        self.layout = FeatureLayout(loop_size=10, profile_size=5, solid_size=3)

    def test_feature_size(self):
        # 10 + 5 + 3 + 1 separator
        self.assertEqual(self.layout.feature_size, 19)
        self.assertEqual(self.layout.sep_slot, 18)

    def test_slots_disjoint(self):
        self.assertEqual(self.layout.slot(CodeNode(LOOP, 0)), 0)
        self.assertEqual(self.layout.slot(CodeNode(PROFILE, 0)), 10)
        self.assertEqual(self.layout.slot(CodeNode(SOLID, 0)), 15)
        self.assertEqual(self.layout.slot(SEP), 18)

    def test_onehot(self):
        vec = self.layout.onehot(CodeNode(SOLID, 2))
        self.assertEqual(sum(vec), 1)
        self.assertEqual(vec[17], 1)

    def test_code_out_of_range(self):
        with self.assertRaises(ValueError):
            self.layout.slot(CodeNode(SOLID, 3))


class TestLevelMask(unittest.TestCase):
    def test_fix_solid_only(self):
        mask = level_mask(sample_tree(), {SOLID})
        els = mask.elements
        for el, f in zip(els, mask.fixed):
            if el == SEP:
                self.assertTrue(f)
            elif el.level == SOLID:
                self.assertTrue(f)
            else:
                self.assertFalse(f)

    def test_fix_solid_and_profile(self):
        mask = level_mask(sample_tree(), {SOLID, PROFILE})
        loops_fixed = [f for el, f in zip(mask.elements, mask.fixed)
                       if el != SEP and el.level == LOOP]
        self.assertTrue(all(not f for f in loops_fixed))

    def test_unknown_level(self):
        with self.assertRaises(ValueError):
            level_mask(sample_tree(), {"bogus"})


class TestEditCode(unittest.TestCase):
    def test_edit_solid(self):
        t = edit_code(sample_tree(), SOLID, 0, 99)
        self.assertEqual(t.solid, 99)

    def test_edit_profile(self):
        t = edit_code(sample_tree(), PROFILE, 1, 88)
        self.assertEqual(t.profiles[1][0], 88)
        self.assertEqual(t.profiles[0][0], 3)  # untouched

    def test_edit_loop_flat_index(self):
        # flat loop order: 1,2,5,6 -> index 2 is the first loop of profile 1
        t = edit_code(sample_tree(), LOOP, 2, 77)
        self.assertEqual(t.profiles[1][1], (77, 6))

    def test_edit_loop_out_of_range(self):
        with self.assertRaises(IndexError):
            edit_code(sample_tree(), LOOP, 99, 0)

    def test_original_unchanged(self):
        orig = sample_tree()
        edit_code(orig, SOLID, 0, 1)
        self.assertEqual(orig.solid, 7)


class TestAutocompleteMask(unittest.TestCase):
    def test_known_nodes_fixed(self):
        # user supplied the solid and the first loop
        mask = autocomplete_mask(sample_tree(), {(SOLID, 0), (LOOP, 0)})
        fixed_addr = []
        loop_i = 0
        prof_i = -1
        for el, f in zip(mask.elements, mask.fixed):
            if el == SEP:
                continue
            if el.level == SOLID:
                addr = (SOLID, 0)
            elif el.level == PROFILE:
                prof_i += 1
                addr = (PROFILE, prof_i)
            else:
                addr = (LOOP, loop_i)
                loop_i += 1
            if f:
                fixed_addr.append(addr)
        self.assertIn((SOLID, 0), fixed_addr)
        self.assertIn((LOOP, 0), fixed_addr)
        self.assertNotIn((LOOP, 1), fixed_addr)

    def test_generated_positions(self):
        mask = autocomplete_mask(sample_tree(), set())
        # nothing known -> all 7 code nodes to be generated (2 SEP stay fixed)
        gen = mask.generated_positions()
        self.assertEqual(len(gen), 7)  # S, P, L, L, P, L, L


if __name__ == "__main__":
    unittest.main()
