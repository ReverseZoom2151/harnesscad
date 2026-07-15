"""Tests for the assembly-checks dispatch surface."""

import argparse
import unittest

from harnesscad.domain.assembly import registry as asm


class TestCheck(unittest.TestCase):
    def test_overlapping_boxes_flagged(self):
        boxes = {
            "a": (0.0, 0.0, 0.0, 3.0, 3.0, 3.0),
            "b": (1.0, 1.0, 1.0, 4.0, 4.0, 4.0),  # 2x2x2 overlap volume 8
        }
        res = asm.check(boxes)
        # the two AABBs overlap -> the check must not pass.
        self.assertFalse(res.passed)
        self.assertGreaterEqual(res.checked_pairs, 1)
        self.assertTrue(res.clips)

    def test_disjoint_boxes_clear(self):
        boxes = {
            "a": (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            "b": (5.0, 5.0, 5.0, 6.0, 6.0, 6.0),
        }
        res = asm.check(boxes)
        self.assertTrue(res.passed)
        self.assertEqual(res.clips, [])


class TestRouting(unittest.TestCase):
    def test_discover_lists_interference_route(self):
        rows = asm.discover()
        routes = {r["route"] for r in rows}
        self.assertIn("check", routes)
        groups = {r["group"] for r in rows}
        self.assertIn("interference", groups)

    def test_discover_rows_are_well_formed(self):
        for r in asm.discover():
            self.assertEqual(
                set(r), {"group", "route", "module", "doc", "present"}
            )
            self.assertIsInstance(r["present"], bool)
            self.assertTrue(r["module"].startswith("harnesscad.domain.assembly."))

    def test_routed_modules_subset_of_declared(self):
        declared = {r["module"] for r in asm.discover()}
        for m in asm.routed_modules():
            self.assertIn(m, declared)

    def test_determinism(self):
        self.assertEqual(asm.discover(), asm.discover())
        self.assertEqual(asm.routed_modules(), asm.routed_modules())


class TestCli(unittest.TestCase):
    def test_add_arguments_and_run_json(self):
        parser = argparse.ArgumentParser()
        asm.add_arguments(parser)
        args = parser.parse_args(["--json"])
        self.assertEqual(asm.run_cli(args), 0)

    def test_run_cli_text(self):
        parser = argparse.ArgumentParser()
        asm.add_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(asm.run_cli(args), 0)

    def test_main_returns_zero(self):
        self.assertEqual(asm.main(["--json"]), 0)


if __name__ == "__main__":
    unittest.main()
