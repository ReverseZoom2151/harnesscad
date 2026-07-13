"""Tests for dataengine.cadeditor_edit_typing (CAD-Editor variant pairing)."""
import unittest

from harnesscad.data.dataengine.edits.edit_typing import (
    classify_edit_type,
    group_by_prefix,
    pair_group,
    pair_variants,
    EditPairing,
)


class TestClassify(unittest.TestCase):
    def test_add_when_original_is_seed(self):
        self.assertEqual(classify_edit_type("seed_origInput", "seed_v1"), "add")

    def test_delete_when_edited_is_seed(self):
        self.assertEqual(classify_edit_type("seed_v1", "seed_origInput"), "delete")

    def test_modify_when_neither_is_seed(self):
        self.assertEqual(classify_edit_type("seed_v1", "seed_v2"), "modify")

    def test_custom_sentinel(self):
        self.assertEqual(classify_edit_type("A_BASE", "A_x", sentinel="BASE"), "add")


class TestGroupByPrefix(unittest.TestCase):
    def test_buckets_by_prefix(self):
        items = [
            {"name": "AAAAAAAA_1"},
            {"name": "AAAAAAAA_2"},
            {"name": "BBBBBBBB_1"},
        ]
        groups = group_by_prefix(items, key_len=8)
        self.assertEqual(list(groups.keys()), ["AAAAAAAA", "BBBBBBBB"])
        self.assertEqual(len(groups["AAAAAAAA"]), 2)

    def test_order_preserved(self):
        items = [{"name": "z_1"}, {"name": "a_1"}]
        self.assertEqual(list(group_by_prefix(items, key_len=1).keys()), ["z", "a"])


class TestPairGroup(unittest.TestCase):
    def _items(self):
        return [
            {"name": "s_origInput", "original_sequence": "S0"},
            {"name": "s_v1", "original_sequence": "S1"},
            {"name": "s_v2", "original_sequence": "S2"},
        ]

    def test_forward_and_reverse_emitted(self):
        pairs = pair_group(self._items())
        # 3 combinations * 2 directions = 6
        self.assertEqual(len(pairs), 6)

    def test_types_are_directional(self):
        pairs = pair_group(self._items())
        # origInput -> v1 is add; v1 -> origInput is delete
        fwd = next(p for p in pairs
                   if p.original_name == "s_origInput" and p.edited_name == "s_v1")
        rev = next(p for p in pairs
                   if p.original_name == "s_v1" and p.edited_name == "s_origInput")
        self.assertEqual(fwd.edit_type, "add")
        self.assertEqual(rev.edit_type, "delete")
        # v1 <-> v2 both modify
        mod = next(p for p in pairs
                   if p.original_name == "s_v1" and p.edited_name == "s_v2")
        self.assertEqual(mod.edit_type, "modify")

    def test_sequences_swapped_on_reverse(self):
        pairs = pair_group(self._items())
        fwd = next(p for p in pairs
                   if p.original_name == "s_origInput" and p.edited_name == "s_v1")
        self.assertEqual(fwd.original_sequence, "S0")
        self.assertEqual(fwd.edited_sequence, "S1")

    def test_cap_enforced(self):
        items = [{"name": f"s_v{i}", "original_sequence": str(i)} for i in range(10)]
        pairs = pair_group(items, cap=5)
        self.assertEqual(len(pairs), 5)

    def test_singleton_group_empty(self):
        self.assertEqual(pair_group([{"name": "x", "original_sequence": "q"}]), [])

    def test_to_dict_shape(self):
        p = EditPairing("a", "b", "SA", "SB", "modify")
        self.assertEqual(p.to_dict(), {
            "original_pic_name": "a", "edited_pic_name": "b",
            "original_sequence": "SA", "edited_sequence": "SB", "type": "modify",
        })


class TestPairVariants(unittest.TestCase):
    def test_only_same_seed_paired(self):
        items = [
            {"name": "AAAAAAAA_origInput", "original_sequence": "a0"},
            {"name": "AAAAAAAA_v1", "original_sequence": "a1"},
            {"name": "BBBBBBBB_v1", "original_sequence": "b1"},  # lone -> no pairs
        ]
        pairs = pair_variants(items)
        self.assertEqual(len(pairs), 2)  # only the A bucket, both directions
        names = {(p.original_name, p.edited_name) for p in pairs}
        self.assertIn(("AAAAAAAA_origInput", "AAAAAAAA_v1"), names)
        self.assertNotIn(("BBBBBBBB_v1", "AAAAAAAA_v1"),
                         {(p.original_name, p.edited_name) for p in pairs})

    def test_deterministic(self):
        items = [{"name": f"S_v{i}", "original_sequence": str(i)} for i in range(4)]
        self.assertEqual([p.to_dict() for p in pair_variants(items)],
                         [p.to_dict() for p in pair_variants(items)])


if __name__ == "__main__":
    unittest.main()
