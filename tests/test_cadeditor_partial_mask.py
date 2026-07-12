"""Tests for editing.cadeditor_partial_mask (CAD-Editor fine-grained masking)."""
import unittest

from editing.cadeditor_partial_mask import (
    MASK,
    compare_tokens,
    merge_consecutive_masks,
    generate_mask,
    mask_span_count,
    parse_components,
)


class TestCompareTokens(unittest.TestCase):
    def test_identical_preserved(self):
        self.assertEqual(compare_tokens("line,14,14", "line,14,14"), "line,14,14")

    def test_partial_component_mask(self):
        self.assertEqual(compare_tokens("line,14,14", "line,13,13"),
                         "line," + MASK + "," + MASK)

    def test_one_component_differs(self):
        self.assertEqual(compare_tokens("line,14,14", "line,14,20"),
                         "line,14," + MASK)

    def test_different_command_whole_mask(self):
        self.assertEqual(compare_tokens("line,14,14", "arc,14,14"), MASK)

    def test_length_mismatch_masks_surplus(self):
        # circle has more params; command matches, surplus position masked.
        self.assertEqual(compare_tokens("circle,1,2", "circle,1,2,3"),
                         "circle,1,2," + MASK)

    def test_parse_components(self):
        self.assertEqual(parse_components("circle,1,2,3"), ["circle", "1", "2", "3"])


class TestMergeConsecutiveMasks(unittest.TestCase):
    def test_collapse_runs(self):
        self.assertEqual(
            merge_consecutive_masks([MASK, MASK, "a", MASK, MASK, MASK, "b"]),
            [MASK, "a", MASK, "b"])

    def test_partial_masks_not_merged(self):
        toks = ["line," + MASK, "line," + MASK]
        self.assertEqual(merge_consecutive_masks(toks), toks)

    def test_no_masks_unchanged(self):
        self.assertEqual(merge_consecutive_masks(["a", "b"]), ["a", "b"])


class TestGenerateMask(unittest.TestCase):
    def test_identical_sequences_no_mask(self):
        seq = ["line,1,1", "line,2,2", "<curve_end>"]
        self.assertEqual(generate_mask(seq, seq), seq)

    def test_deletion_becomes_single_mask(self):
        original = ["a", "b", "c", "d"]
        edited = ["a", "d"]
        out = generate_mask(original, edited)
        self.assertEqual(out, ["a", MASK, "d"])

    def test_insertion_becomes_mask(self):
        original = ["a", "d"]
        edited = ["a", "b", "c", "d"]
        out = generate_mask(original, edited)
        self.assertEqual(out, ["a", MASK, "d"])

    def test_partial_modify_preserves_command(self):
        original = ["line,14,14", "<curve_end>"]
        edited = ["line,13,13", "<curve_end>"]
        out = generate_mask(original, edited)
        self.assertEqual(out, ["line," + MASK + "," + MASK, "<curve_end>"])

    def test_deterministic(self):
        original = ["a", "line,1,2", "c", "x"]
        edited = ["a", "line,1,9", "c"]
        self.assertEqual(generate_mask(original, edited),
                         generate_mask(original, edited))

    def test_merge_flag_off_keeps_runs(self):
        original = ["a", "b", "c", "d"]
        edited = ["a"]
        merged = generate_mask(original, edited, merge=True)
        raw = generate_mask(original, edited, merge=False)
        self.assertEqual(merged, ["a", MASK])
        self.assertEqual(raw.count(MASK), 1)  # single replace/delete opcode span


class TestMaskSpanCount(unittest.TestCase):
    def test_counts_contiguous_regions(self):
        masked = ["a", MASK, "b", "line," + MASK, "c"]
        self.assertEqual(mask_span_count(masked), 2)

    def test_zero_when_unmasked(self):
        self.assertEqual(mask_span_count(["a", "b"]), 0)

    def test_adjacent_partial_and_bare_one_region(self):
        masked = [MASK, "line," + MASK, "z"]
        self.assertEqual(mask_span_count(masked), 1)


if __name__ == "__main__":
    unittest.main()
