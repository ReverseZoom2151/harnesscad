"""The CADBench exec-log loader's manifest-only + taxonomy invariants.

Kernel-free and instant: these assert the raw logs are manifest-only (nothing
vendored, degrade-to-empty) and that the extracted status-code / error-category
taxonomy is non-empty, machine-checkable, and always present as facts.
"""

from __future__ import annotations

import unittest

from harnesscad.eval.corpus.fixtures import cadbench_exec_logs as cel
from harnesscad.eval.corpus.fixtures import loader


class TestManifest(unittest.TestCase):

    def test_manifest_only_mit_nothing_vendored(self):
        m = cel.manifest()
        self.assertEqual(m.license, "MIT")
        self.assertEqual(len(m.entries), cel.EXPECTED_LOG_FILES)
        self.assertEqual(m.verify_vendored(), [])
        for e in m.entries:
            self.assertIsNone(e.vendored, e.name)
            self.assertEqual(e.role, "exec_log", e.name)
            self.assertTrue(e.resource.endswith("_logs.json"), e.name)
            self.assertEqual(len(e.sha256), 64, e.name)

    def test_reachable_through_hub(self):
        self.assertIs(loader("cadbench_exec_logs"), cel)


class TestStatusCodes(unittest.TestCase):

    def test_three_codes_summing_to_total(self):
        self.assertEqual({sc.code for sc in cel.STATUS_CODES}, {0, 1, 2})
        self.assertEqual(
            sum(sc.observed_rows for sc in cel.STATUS_CODES), cel.TOTAL_ROWS)
        for sc in cel.STATUS_CODES:
            self.assertEqual(cel.STATUS_ROW_COUNTS[sc.code], sc.observed_rows)

    def test_predicate_reads_only_status(self):
        for sc in cel.STATUS_CODES:
            self.assertTrue(sc.matches(sc.code))
            self.assertFalse(sc.matches(sc.code + 100))
        self.assertEqual(cel.status_code(1).name, "success")
        self.assertIsNone(cel.status_code(99))


class TestErrorTaxonomy(unittest.TestCase):

    def test_non_empty_and_unique_ids(self):
        self.assertTrue(cel.ERROR_CATEGORIES)
        ids = [c.cid for c in cel.ERROR_CATEGORIES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_each_category_matches_its_own_example_scoped_to_status_zero(self):
        for c in cel.ERROR_CATEGORIES:
            self.assertTrue(c.signatures, c.cid)
            self.assertTrue(c.example, c.cid)
            self.assertGreater(c.observed_rows, 0, c.cid)
            # machine-checkable: fires on its own recorded example under status 0
            self.assertTrue(c.matches(0, c.example), c.cid)
            # never fires on a success / timeout row
            self.assertFalse(c.matches(1, c.example), c.cid)
            self.assertFalse(c.matches(2, c.example), c.cid)
            self.assertIn(c.cid, cel.classify_error(0, c.example), c.cid)

    def test_success_and_timeout_yield_no_error_category(self):
        self.assertEqual(cel.classify_error(1, "Success"), [])
        self.assertEqual(cel.classify_error(2, "Code Execution Timeout"), [])


class TestDegradeAndSelfcheck(unittest.TestCase):

    def test_log_files_is_a_list(self):
        # Facts (taxonomy) are always present; raw files may be empty.
        files = cel.log_files()
        self.assertIsInstance(files, list)
        self.assertLessEqual(len(files), cel.EXPECTED_LOG_FILES)

    def test_selfcheck_exits_zero(self):
        self.assertEqual(cel.main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
