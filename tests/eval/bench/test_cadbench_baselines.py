"""Tests for eval.bench.imports.cadbench_baselines.

CADBench's committed baseline leaderboard values, vendored as typed
``(model, modality, bench, difficulty) -> metric`` rows, plus the adapter onto
the hard-corpus board's ``Standing`` row type. Values are checked against the
files vendored under ``imports/cadbench/`` (SHA-pinned in MANIFEST.json), so the
expected numbers below are the CADBench authors' own committed outputs.
"""

import unittest

from harnesscad.eval.bench.imports import cadbench_baselines as cb
from harnesscad.eval.bench import imports as hub


class TestManifest(unittest.TestCase):
    def test_license_is_mit_and_vendored_shas_verify(self):
        m = cb.manifest()
        self.assertEqual(m.license, "MIT")
        self.assertEqual(m.source_repo, "CADBench")
        self.assertEqual(m.verify_vendored(), [])

    def test_only_json_metric_outputs_are_vendored(self):
        # No task/dataset content: every entry is a metrics/per_label JSON.
        m = cb.manifest()
        for e in m.entries:
            self.assertTrue((e.vendored or "").endswith(".json"), e.name)
            self.assertIn(e.role, ("metrics", "per_label"))
            self.assertIn("tested_models", e.resource or "")

    def test_expected_file_counts(self):
        m = cb.manifest()
        self.assertEqual(len(m.by_role("metrics")), cb.EXPECTED_METRICS_FILES)
        self.assertEqual(len(m.by_role("per_label")), cb.EXPECTED_PER_LABEL_FILES)


class TestRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = cb.rows()

    def test_row_count_and_unique_keys(self):
        self.assertEqual(len(self.rows), cb.EXPECTED_ROWS)  # 186 + 124*3 = 558
        keys = [r.key for r in self.rows]
        self.assertEqual(len(set(keys)), len(keys))

    def test_combos_and_benches(self):
        combos = {(r.model, r.modality) for r in self.rows}
        self.assertEqual(len(combos), cb.EXPECTED_COMBOS)
        # every bench has an overall row.
        for bench in cb.BENCHES:
            self.assertTrue(any(r.bench == bench and r.difficulty == "overall"
                                for r in self.rows), bench)

    def test_only_banded_benches_carry_difficulties(self):
        banded = {r.bench for r in self.rows if r.difficulty in ("easy", "medium", "hard")}
        self.assertEqual(banded, set(cb.BANDED_BENCHES))
        # benchM / benchO carry an overall row but no per-difficulty rows.
        for bench in ("benchM", "benchO"):
            diffs = {r.difficulty for r in self.rows if r.bench == bench}
            self.assertEqual(diffs, {"overall"}, bench)

    def test_deterministic(self):
        self.assertEqual([r.key for r in cb.rows()], [r.key for r in self.rows])

    def test_metric_accessors_both_schemas(self):
        for r in self.rows:
            iou = r.aligned_iou_mean()
            self.assertIsNotNone(iou, r.label)
            self.assertGreaterEqual(iou, 0.0)
            self.assertLessEqual(iou, 1.0)
            vsr = r.vsr()
            self.assertIsNotNone(vsr, r.label)
            self.assertTrue(0.0 <= vsr <= 100.0, (r.label, vsr))
            self.assertGreater(r.sample_count(), 0)
            self.assertGreaterEqual(r.aligned_chamfer_mean(), 0.0)

    def test_known_committed_value_cadcoder_multiview_benchA(self):
        # Verified from the vendored benchA_metrics.json (cadcoder/multiview/r1):
        # Aligned IoU Mean 0.24284..., VSR 86.1333...
        row = next(r for r in self.rows
                   if r.key == ("cadcoder", "multiview", "benchA", "overall"))
        self.assertAlmostEqual(row.aligned_iou_mean(), 0.24284609969485707, places=9)
        self.assertAlmostEqual(row.vsr(), 86.13333333333333, places=6)
        self.assertEqual(row.sample_count(), cb.BENCH_TASK_COUNT)

    def test_known_per_label_value_cadcoder_multiview_benchA_easy(self):
        # From benchA_per_label_metrics.json easy.success_only: mean 0.264268...
        # VSR = success_only.count / adjusted.count = 958 / 1000 = 95.8%.
        row = next(r for r in self.rows
                   if r.key == ("cadcoder", "multiview", "benchA", "easy"))
        self.assertAlmostEqual(row.aligned_iou_mean(), 0.26426855081075185, places=9)
        self.assertAlmostEqual(row.vsr(), 95.8, places=6)
        self.assertEqual(row.sample_count(), 1000)

    def test_ourimages_models_have_empty_modality(self):
        # cadevolve_ourimages / cadrille_ourimages name no modality dir.
        for model in ("cadevolve_ourimages", "cadrille_ourimages"):
            mods = {r.modality for r in self.rows if r.model == model}
            self.assertEqual(mods, {""}, model)


class TestStandingAdapter(unittest.TestCase):
    def setUp(self):
        from harnesscad.eval.leaderboard.hardcorpus_board import Standing
        self.Standing = Standing
        self.rows = cb.rows()

    def test_to_standing_maps_vsr_and_iou(self):
        row = next(r for r in self.rows
                   if r.key == ("cadcoder", "multiview", "benchA", "overall"))
        s = cb.to_standing(row)
        self.assertIsInstance(s, self.Standing)
        self.assertEqual(s.n, cb.BENCH_TASK_COUNT)
        # weak_rate == VSR (validity), oracle not run for an external baseline.
        self.assertAlmostEqual(s.weak_rate * 100.0, row.vsr(), places=3)
        self.assertEqual(s.oracle_rate, 0.0)
        self.assertEqual(s.oracle_solved, 0)
        self.assertEqual(s.mean_iou, row.aligned_iou_mean())
        self.assertLessEqual(s.weak_passed, s.n)

    def test_to_standings_default_overall_only(self):
        standings = cb.to_standings()
        self.assertEqual(len(standings),
                         sum(1 for r in self.rows if r.difficulty == "overall"))
        self.assertTrue(all(isinstance(s, self.Standing) for s in standings))

    def test_to_standings_all_difficulties(self):
        self.assertEqual(len(cb.to_standings(difficulty="")), len(self.rows))

    def test_oracle_run_outranks_field_only_baselines(self):
        from harnesscad.eval.leaderboard.hardcorpus_board import ranking
        run = self.Standing(name="harness-run", n=100, oracle_solved=40,
                            weak_passed=60)
        board = [run] + cb.to_standings()[:20]
        self.assertEqual(ranking(board)[0].name, "harness-run")


class TestScorecardAdapter(unittest.TestCase):
    def test_maps_to_cd_ir_iou_and_ranks(self):
        from harnesscad.eval.bench.protocols.tiered_leaderboard import rank_leaderboard
        rows = [r for r in cb.rows() if r.difficulty == "overall"][:10]
        sc = [cb.to_scorecard_row(r) for r in rows]
        for d, r in zip(sc, rows):
            self.assertEqual(d["iou"], r.aligned_iou_mean())
            self.assertAlmostEqual(d["ir"] + r.vsr(), 100.0, places=6)
        ranked = rank_leaderboard(sc, metric="iou")
        self.assertEqual(len(ranked), len(sc))
        self.assertEqual(ranked[0]["rank"], 1)


class TestDegradeAndHub(unittest.TestCase):
    def test_rows_degrade_to_empty_on_absent_manifest(self):
        orig = cb.manifest
        try:
            cb.manifest = lambda: (_ for _ in ()).throw(OSError("absent"))
            self.assertEqual(cb.rows(), [])
            self.assertEqual(cb.to_standings(), [])
        finally:
            cb.manifest = orig

    def test_reachable_through_hub_but_not_a_brief_source(self):
        self.assertIn("cadbench_baselines", hub.LOADERS)
        self.assertIs(hub.loader("cadbench_baselines"), cb)
        self.assertNotIn("cadbench_baselines", hub.BRIEF_LOADERS)
        self.assertEqual(hub.briefs_from("cadbench_baselines"), [])

    def test_availability_census(self):
        av = hub.availability()["cadbench_baselines"]
        self.assertEqual(av["total"], cb.EXPECTED_METRICS_FILES
                         + cb.EXPECTED_PER_LABEL_FILES)
        self.assertEqual(av["vendored"], av["total"])


class TestSelfcheck(unittest.TestCase):
    def test_selfcheck_exits_zero(self):
        self.assertEqual(cb.main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
