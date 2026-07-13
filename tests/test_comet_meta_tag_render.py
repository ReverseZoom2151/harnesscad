"""Tests for terse meta-tag rendering + consecutive-origin bundling."""

import unittest

from harnesscad.agents.context.meta_tags import (
    ACT_PRIORITY,
    ORIGIN_PRIORITY,
    Row,
    pick_highest,
    render_rows,
    short_tag_block,
)


class TestPickHighest(unittest.TestCase):
    def test_picks_highest_priority(self):
        tags = ["ORIGIN:EXTERNAL", "ORIGIN:FILE_EDIT"]
        self.assertEqual(pick_highest(tags, ORIGIN_PRIORITY), "ORIGIN:FILE_EDIT")

    def test_unknown_scores_zero(self):
        self.assertEqual(pick_highest(["ORIGIN:MYSTERY"], ORIGIN_PRIORITY), "")

    def test_deterministic_regardless_of_order(self):
        a = pick_highest(["FLAG:ACT_EDIT", "FLAG:ACT_FAIL"], ACT_PRIORITY)
        b = pick_highest(["FLAG:ACT_FAIL", "FLAG:ACT_EDIT"], ACT_PRIORITY)
        self.assertEqual(a, b)
        self.assertEqual(a, "FLAG:ACT_FAIL")


class TestShortTagBlock(unittest.TestCase):
    def test_full_block(self):
        tags = ["ORIGIN:FILE_EDIT", "FLAG:ACT_EDIT", "FLAG:USER_FEEDBACK", "IMPORTANCE:HIGH"]
        self.assertEqual(short_tag_block(tags), "(O:FILE_EDIT A:EDIT F:USER_FEEDBACK I:H)")

    def test_act_suppressed_on_user(self):
        tags = ["ORIGIN:USER", "FLAG:ACT_EDIT"]
        self.assertEqual(short_tag_block(tags), "(O:USER)")

    def test_importance_med_suppressed(self):
        tags = ["ORIGIN:EXTERNAL", "IMPORTANCE:MED"]
        self.assertEqual(short_tag_block(tags), "(O:EXTERNAL)")

    def test_importance_low_shown(self):
        tags = ["ORIGIN:EXTERNAL", "IMPORTANCE:LOW"]
        self.assertEqual(short_tag_block(tags), "(O:EXTERNAL I:L)")

    def test_empty_when_no_axis(self):
        self.assertEqual(short_tag_block(["random", "another"]), "")

    def test_only_highest_per_axis(self):
        tags = ["ORIGIN:EXTERNAL", "ORIGIN:USER"]
        # USER outranks EXTERNAL
        self.assertEqual(short_tag_block(tags), "(O:USER)")


class TestRenderRows(unittest.TestCase):
    def test_plain_rows_unmerged(self):
        rows = [
            Row("mem_1", "did A", "when A", ["ORIGIN:USER"]),
            Row("mem_2", "did B", "when B", ["ORIGIN:FILE_EDIT"]),
        ]
        out = render_rows(rows)
        self.assertEqual(len(out), 2)
        self.assertIn("[mem_1] (O:USER) did A | when A", out[0])

    def test_user_rows_never_merged(self):
        rows = [Row(f"mem_{i}", f"s{i}", "t", ["ORIGIN:USER"]) for i in range(3)]
        out = render_rows(rows)
        self.assertEqual(len(out), 3)

    def test_consecutive_origin_bundled(self):
        rows = [
            Row("mem_a", "s1", "t1", ["ORIGIN:EXTERNAL"]),
            Row("mem_b", "s2", "t2", ["ORIGIN:EXTERNAL"]),
            Row("mem_c", "s3", "t3", ["ORIGIN:EXTERNAL"]),
        ]
        out = render_rows(rows, max_merge=3)
        self.assertEqual(len(out), 1)
        self.assertIn("s1; s2; s3", out[0])
        self.assertIn("[mem_a+b+c]", out[0])

    def test_bundle_chunked_by_max_merge(self):
        rows = [Row(f"mem_{c}", f"s{c}", "t", ["ORIGIN:EXTERNAL"]) for c in "abcd"]
        out = render_rows(rows, max_merge=2)
        # 4 rows / max_merge 2 -> two bundle lines
        self.assertEqual(len(out), 2)
        self.assertIn("sa; sb", out[0])
        self.assertIn("sc; sd", out[1])

    def test_single_run_not_merged(self):
        rows = [
            Row("mem_a", "s1", "t1", ["ORIGIN:EXTERNAL"]),
            Row("mem_b", "s2", "t2", ["ORIGIN:USER"]),
        ]
        out = render_rows(rows)
        self.assertEqual(len(out), 2)
        self.assertIn("[mem_a]", out[0])

    def test_passive_prefix(self):
        rows = [Row("mem_x", "keep", "t", ["ORIGIN:USER"], recall_mode="passive")]
        out = render_rows(rows)
        self.assertIn("(passive)", out[0])

    def test_bad_max_merge(self):
        with self.assertRaises(ValueError):
            render_rows([], max_merge=0)


if __name__ == "__main__":
    unittest.main()
