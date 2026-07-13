"""Tests for programs.t2cq2_output_cleaning."""

from __future__ import annotations

import unittest

from harnesscad.domain.programs.t2cq2_output_cleaning import (
    EOS_TOKEN,
    REASON_EMPTY,
    REASON_NO_EOS,
    REASON_NO_EXPORT,
    canonicalize_export,
    clean_corpus,
    clean_output,
    find_export_shapes,
    strip_export_statements,
    strip_markdown_fence,
    strip_response_prefix,
    truncate_at_eos,
)

SCRIPT = (
    "import cadquery as cq\n"
    'part_1 = cq.Workplane("XY").box(1.0, 2.0, 3.0)\n'
    'cq.exporters.export(part_1, "/tmp/old.stl")\n'
)


class TruncationTest(unittest.TestCase):
    def test_cuts_at_eos_and_rstrips(self):
        self.assertEqual(truncate_at_eos("abc \n" + EOS_TOKEN + "junk"), "abc")

    def test_missing_eos_returns_none(self):
        self.assertIsNone(truncate_at_eos("abc"))

    def test_only_first_eos_matters(self):
        self.assertEqual(truncate_at_eos("a" + EOS_TOKEN + "b" + EOS_TOKEN), "a")


class PrefixAndFenceTest(unittest.TestCase):
    def test_response_prefix_removed(self):
        self.assertEqual(
            strip_response_prefix("### Instruction:\nx\n\n### Response:\ncode"),
            "code",
        )

    def test_response_prefix_absent_is_stripped_text(self):
        self.assertEqual(strip_response_prefix("  code  "), "code")

    def test_python_fence_removed(self):
        self.assertEqual(strip_markdown_fence("```python\ncode\n```"), "code")

    def test_bare_fence_removed(self):
        self.assertEqual(strip_markdown_fence("```\ncode\n```"), "code")

    def test_unfenced_text_unchanged(self):
        self.assertEqual(strip_markdown_fence("code"), "code")


class ExportHandlingTest(unittest.TestCase):
    def test_finds_shape_argument(self):
        self.assertEqual(find_export_shapes(SCRIPT), ["part_1"])

    def test_finds_multiple_exports_in_order(self):
        code = SCRIPT + 'cq.exporters.export(part_2, "b.stl")\n'
        self.assertEqual(find_export_shapes(code), ["part_1", "part_2"])

    def test_no_export_returns_empty(self):
        self.assertEqual(find_export_shapes("import cadquery as cq\n"), [])

    def test_strip_removes_truncated_export_line(self):
        code = SCRIPT + "cq.exporters.expo"
        stripped = strip_export_statements(code)
        self.assertNotIn("cq.exporters", stripped)
        self.assertIn("part_1 = cq.Workplane", stripped)

    def test_canonicalize_keeps_first_shape_and_new_path(self):
        out = canonicalize_export(SCRIPT, "./cq/0001.stl")
        self.assertIsNotNone(out)
        self.assertTrue(out.endswith('cq.exporters.export(part_1, "./cq/0001.stl")'))
        self.assertEqual(out.count("cq.exporters.export"), 1)
        self.assertNotIn("old.stl", out)

    def test_canonicalize_without_export_returns_none(self):
        self.assertIsNone(canonicalize_export("import cadquery as cq\n", "a.stl"))


class CleanOutputTest(unittest.TestCase):
    def test_full_pipeline(self):
        raw = (
            "### Instruction:\nmake a box\n\n### Response:\n"
            "```python\n" + SCRIPT + "```\n" + EOS_TOKEN + " trailing garbage"
        )
        res = clean_output(raw, "./stl/7.stl")
        self.assertTrue(res.ok)
        self.assertEqual(res.shape, "part_1")
        self.assertEqual(res.exports_found, 1)
        self.assertTrue(res.truncated_at_eos)
        self.assertIn("import cadquery as cq", res.code)
        self.assertTrue(res.code.endswith('cq.exporters.export(part_1, "./stl/7.stl")'))
        self.assertNotIn("garbage", res.code)
        self.assertNotIn("```", res.code)

    def test_missing_eos_rejected(self):
        res = clean_output(SCRIPT, "a.stl")
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REASON_NO_EOS)
        self.assertIsNone(res.code)

    def test_missing_eos_allowed_when_not_required(self):
        res = clean_output(SCRIPT, "a.stl", require_eos=False)
        self.assertTrue(res.ok)
        self.assertFalse(res.truncated_at_eos)

    def test_no_export_rejected(self):
        res = clean_output("import cadquery as cq\n" + EOS_TOKEN, "a.stl")
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REASON_NO_EXPORT)

    def test_empty_body_rejected(self):
        res = clean_output("   " + EOS_TOKEN, "a.stl")
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REASON_EMPTY)

    def test_body_that_is_only_export_rejected(self):
        raw = 'cq.exporters.export(part_1, "x.stl")\n' + EOS_TOKEN
        res = clean_output(raw, "a.stl")
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REASON_EMPTY)

    def test_deterministic(self):
        raw = SCRIPT + EOS_TOKEN
        self.assertEqual(clean_output(raw, "a.stl"), clean_output(raw, "a.stl"))


class CorpusTest(unittest.TestCase):
    def test_stats(self):
        items = [
            (SCRIPT + EOS_TOKEN, "0.stl"),
            (SCRIPT, "1.stl"),
            ("import cadquery as cq\n" + EOS_TOKEN, "2.stl"),
            (EOS_TOKEN, "3.stl"),
        ]
        results, stats = clean_corpus(items)
        self.assertEqual(len(results), 4)
        self.assertEqual(stats.total, 4)
        self.assertEqual(stats.kept, 1)
        self.assertEqual(stats.dropped_no_eos, 1)
        self.assertEqual(stats.dropped_no_export, 1)
        self.assertEqual(stats.dropped_empty, 1)
        self.assertAlmostEqual(stats.drop_rate, 0.75)

    def test_empty_corpus(self):
        results, stats = clean_corpus([])
        self.assertEqual(results, [])
        self.assertEqual(stats.total, 0)
        self.assertAlmostEqual(stats.drop_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
