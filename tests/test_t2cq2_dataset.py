"""Tests for dataengine.t2cq2_dataset."""

from __future__ import annotations

import json
import unittest

from dataengine.t2cq2_dataset import (
    CadQueryRecord,
    annotation_outcomes,
    build_annotation_request,
    build_retry_prompt,
    export_targets,
    format_prompt,
    format_training_example,
    ground_truth_path,
    parse_jsonl,
    split_response,
    to_jsonl,
    uid_for_stem,
)

RECORD = CadQueryRecord("a box", 'import cadquery as cq\np = cq.Workplane("XY")')


class RecordTest(unittest.TestCase):
    def test_parse_jsonl(self):
        line = json.dumps({"input": "a box", "output": "code"})
        records = parse_jsonl([line, "", line])
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].input, "a box")
        self.assertEqual(records[0].output, "code")

    def test_missing_key_raises(self):
        with self.assertRaises(ValueError):
            parse_jsonl([json.dumps({"input": "x"})])

    def test_round_trip(self):
        records = [RECORD, CadQueryRecord("a cylinder", "code2")]
        self.assertEqual(parse_jsonl(to_jsonl(records).split("\n")), records)

    def test_to_jsonl_newlines_are_escaped(self):
        text = to_jsonl([RECORD])
        self.assertEqual(len(text.split("\n")), 1)
        self.assertIn("\\n", text)


class PromptTest(unittest.TestCase):
    def test_prompt_format(self):
        self.assertEqual(
            format_prompt("a box"),
            "### Instruction:\na box\n\n### Response:\n",
        )

    def test_training_example_appends_output(self):
        text = format_training_example(RECORD)
        self.assertTrue(text.startswith("### Instruction:\na box"))
        self.assertTrue(text.endswith(RECORD.output))

    def test_split_response_round_trips_prompt(self):
        self.assertEqual(
            split_response(format_training_example(RECORD)), RECORD.output
        )

    def test_split_response_without_marker(self):
        self.assertEqual(split_response("  raw code  "), "raw code")

    def test_split_response_uses_first_marker(self):
        self.assertEqual(split_response("### Response:\na\n### Response:\nb"),
                         "a\n### Response:\nb")


class AnnotationTest(unittest.TestCase):
    def test_request_contains_json_and_export_target(self):
        req = build_annotation_request('{"cad": 1}', "00010001", "./stl/")
        self.assertIn('{"cad": 1}', req)
        self.assertIn("./stl/00010001.stl", req)
        self.assertIn("don't need to use show()", req)

    def test_retry_prompt_uses_last_five_stderr_lines(self):
        stderr = "\n".join(f"line{i}" for i in range(10))
        prompt = build_retry_prompt("code_here", stderr)
        self.assertIn("code_here", prompt)
        self.assertIn("line9", prompt)
        self.assertIn("line5", prompt)
        self.assertNotIn("line4", prompt)

    def test_retry_prompt_short_stderr(self):
        self.assertIn("boom", build_retry_prompt("c", "boom"))


class AnnotationOutcomesTest(unittest.TestCase):
    def test_success_on_first_or_second_attempt(self):
        log = {
            "0001": [0],
            "0002": [1, 0],
            "0003": [1, 1],
            "0004": [1, 1, 0],  # third attempt is beyond the loop's budget
        }
        stats = annotation_outcomes(log, "./cq")
        self.assertEqual(stats.success_count, 2)
        self.assertEqual(stats.fail_count, 2)
        self.assertEqual(
            stats.failed_scripts, ("./cq/0003.py", "./cq/0004.py")
        )
        self.assertAlmostEqual(stats.success_rate, 0.5)

    def test_empty_log(self):
        stats = annotation_outcomes({}, "./cq")
        self.assertEqual(stats.success_count, 0)
        self.assertAlmostEqual(stats.success_rate, 0.0)

    def test_deterministic_ordering(self):
        log = {"b": [1, 1], "a": [1, 1]}
        self.assertEqual(
            annotation_outcomes(log, "./cq").failed_scripts,
            ("./cq/a.py", "./cq/b.py"),
        )


class PathTest(unittest.TestCase):
    def test_uid_bucketing(self):
        self.assertEqual(uid_for_stem("00010001"), "0001/00010001")

    def test_short_stem_raises(self):
        with self.assertRaises(ValueError):
            uid_for_stem("0001")

    def test_ground_truth_path(self):
        self.assertEqual(
            ground_truth_path("00010001.stl", "../stlcq"),
            "../stlcq/0001/00010001.stl",
        )

    def test_export_targets(self):
        targets = export_targets(["00010001.stl", "00020002.stl"], "./cq")
        self.assertEqual(targets[0], "./cq/00010001.stl")
        self.assertEqual(targets[1], "./cq/00020002.stl")
        self.assertEqual(len(targets), 2)


if __name__ == "__main__":
    unittest.main()
