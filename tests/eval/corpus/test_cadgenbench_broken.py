"""The cadgenbench broken-jig loader's pinned-score invariants.

Kernel-free and instant: these assert the committed reference scores are
internally consistent (the regression substrate's whole point) and that the
manifest-only loader degrades cleanly. No geometry engine is touched.
"""

from __future__ import annotations

import unittest

from harnesscad.eval.corpus.fixtures import cadgenbench_broken as cgb
from harnesscad.eval.corpus.fixtures import loader


class TestManifest(unittest.TestCase):

    def test_manifest_apache_and_manifest_only(self):
        m = cgb.manifest()
        self.assertEqual(m.license, "Apache-2.0")
        self.assertEqual(len(m.entries), 32)
        # Nothing vendored: every entry resolves only through resources/.
        for e in m.entries:
            self.assertIsNone(e.vendored, e.name)
            self.assertTrue(e.resource, e.name)
            self.assertEqual(len(e.sha256), 64, e.name)
        self.assertEqual(m.verify_vendored(), [])

    def test_role_census(self):
        roles = {}
        for e in cgb.manifest().entries:
            roles[e.role] = roles.get(e.role, 0) + 1
        self.assertEqual(
            roles,
            {"gt": 4, "candidate_correct": 4, "broken": 11, "sub_volume": 13},
        )

    def test_reachable_through_hub(self):
        self.assertIs(loader("cadgenbench_broken"), cgb)


class TestPinnedScores(unittest.TestCase):

    def test_counts(self):
        self.assertEqual(len(cgb.PINNED_SCORES), 19)
        self.assertEqual(len(cgb.pass_cases()), 8)
        self.assertEqual(len(cgb.broken_cases()), 11)

    def test_pass_cases_score_full(self):
        for fx in cgb.pass_cases():
            self.assertEqual(fx.expected_score, cgb.PASS_SCORE, fx.entry_name)
            self.assertTrue(fx.passes, fx.entry_name)

    def test_every_broken_below_pass_threshold(self):
        worst_pass = min(f.expected_score for f in cgb.pass_cases())
        for fx in cgb.broken_cases():
            self.assertLess(fx.expected_score, worst_pass, fx.entry_name)
            self.assertLess(fx.expected_score, cgb.DEFAULT_IOU_THRESHOLD,
                            fx.entry_name)
            self.assertFalse(fx.passes, fx.entry_name)

    def test_scores_in_unit_interval(self):
        for fx in cgb.scored_fixtures():
            self.assertGreaterEqual(fx.expected_score, 0.0, fx.entry_name)
            self.assertLessEqual(fx.expected_score, 1.0, fx.entry_name)

    def test_broken_cases_name_defect_and_builder(self):
        seen = set()
        for fx in cgb.broken_cases():
            self.assertTrue(fx.defect, fx.entry_name)
            self.assertIn(fx.builder, cgb._BUILDERS, fx.entry_name)
            seen.add(fx.builder)
        self.assertEqual(seen, set(cgb._BUILDERS))

    def test_pinned_keys_resolve_to_manifest_entries(self):
        names = {e.name for e in cgb.manifest().entries}
        for fx in cgb.scored_fixtures():
            self.assertIn(fx.entry_name, names)


class TestRamp(unittest.TestCase):
    """The one kernel-free piece of the grader, ported verbatim."""

    def test_endpoints_and_midpoint(self):
        self.assertEqual(cgb.iou_to_interface_score(1.00), 1.0)
        self.assertEqual(cgb.iou_to_interface_score(0.95), 1.0)
        self.assertEqual(cgb.iou_to_interface_score(0.80), 0.0)
        self.assertEqual(cgb.iou_to_interface_score(0.50), 0.0)
        self.assertAlmostEqual(cgb.iou_to_interface_score(0.875), 0.5)

    def test_monotonic_non_decreasing(self):
        prev = -1.0
        for i in range(0, 101):
            v = cgb.iou_to_interface_score(i / 100.0)
            self.assertGreaterEqual(v, prev - 1e-12)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
            prev = v


class TestVerifyAgainstSeam(unittest.TestCase):
    """The kernel-injection seam mirrors the pose loader's measure injection."""

    def test_perfect_scorer_matches_every_pin(self):
        # A stub "grader" that just returns the pinned score reproduces the
        # reference exactly -- proving the seam threads expected vs got.
        def perfect(candidate_path, fixture_dir):
            for fx in cgb.scored_fixtures():
                if fx.path == candidate_path:
                    return fx.expected_score
            raise AssertionError(candidate_path)

        results = cgb.verify_against(perfect)
        ran = [r for r in results if "skipped" not in r]
        if not ran:
            self.skipTest("no resources checkout; fixtures unresolved")
        for r in ran:
            self.assertTrue(r["passed"], r)

    def test_selfcheck_exits_zero(self):
        self.assertEqual(cgb.main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
