import unittest

from harnesscad.data.dataengine.selftrain.gift_geometric_feedback import Candidate
from harnesscad.data.dataengine.selftrain.gift_bootstrap_loop import (
    amortization_gap, bootstrap_round, bootstrap_selftrain,
    inverse_temperature_schedule, pass_at_1, pass_at_k,
)


class TestInverseTemperature(unittest.TestCase):
    def test_small_budget_hotter_than_large(self):
        sched = inverse_temperature_schedule([8, 16, 32, 64, 128])
        self.assertGreater(sched[8], sched[128])
        # monotone non-increasing in budget
        temps = [sched[b] for b in sorted(sched)]
        self.assertEqual(temps, sorted(temps, reverse=True))

    def test_endpoints(self):
        sched = inverse_temperature_schedule([8, 128], temp_high=1.2, temp_low=0.2)
        self.assertAlmostEqual(sched[8], 1.2)
        self.assertAlmostEqual(sched[128], 0.2)

    def test_singleton(self):
        sched = inverse_temperature_schedule([16], temp_low=0.2)
        self.assertAlmostEqual(sched[16], 0.2)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            inverse_temperature_schedule([])


class TestPassMetrics(unittest.TestCase):
    def setUp(self):
        # first score = greedy/single-shot, max = oracle
        self.scores = [[0.6, 0.9, 0.8], [0.7, 0.7, 0.95]]

    def test_pass_at_1(self):
        self.assertAlmostEqual(pass_at_1(self.scores), (0.6 + 0.7) / 2)

    def test_pass_at_k(self):
        self.assertAlmostEqual(pass_at_k(self.scores), (0.9 + 0.95) / 2)

    def test_amortization_gap(self):
        g = amortization_gap(self.scores)
        self.assertAlmostEqual(g["pass_at_1"], 0.65)
        self.assertAlmostEqual(g["pass_at_k"], 0.925)
        self.assertAlmostEqual(g["gap"], 0.275)
        self.assertGreater(g["relative_gap"], 0)

    def test_gap_zero_when_greedy_is_best(self):
        g = amortization_gap([[0.9, 0.5], [0.8, 0.2]])
        self.assertAlmostEqual(g["gap"], 0.0)


class TestBootstrapRound(unittest.TestCase):
    def test_collects_srs_and_fda(self):
        base = [("i1", "gt1")]

        def sampler(image_id, gt_code):
            return [Candidate("alt", 0.93), Candidate("near", 0.7),
                    Candidate("junk", 0.1)]

        out = bootstrap_round(base, sampler, render_fn=lambda p: "r:" + p)
        self.assertEqual(out["srs"], 1)
        self.assertEqual(out["fda"], 1)

    def test_accepts_tuples(self):
        base = [("i1", "gt1")]
        out = bootstrap_round(base, lambda i, g: [("alt", 0.93)])
        self.assertEqual(out["srs"], 1)


class TestBootstrapSelftrain(unittest.TestCase):
    def test_accumulates_and_dedups_across_rounds(self):
        base = [("i1", "gt1"), ("i2", "gt2")]

        def sampler(image_id, gt_code):
            if image_id == "i1":
                return [Candidate("alt1", 0.95), Candidate("miss1", 0.7)]
            return [Candidate("alt2", 0.92)]

        out = bootstrap_selftrain(base, sampler, rounds=3,
                                  render_fn=lambda p: "r:" + p)
        # deterministic sampler => second round adds nothing => early stop
        self.assertEqual(len(out["history"]), 2)
        self.assertEqual(out["history"][1]["srs_added"], 0)
        self.assertEqual(out["history"][1]["fda_added"], 0)
        # i1 -> alt1 (SRS), i2 -> alt2 (SRS); i1 -> miss1 (FDA)
        self.assertEqual(out["counts"]["srs"], 2)
        self.assertEqual(out["counts"]["fda"], 1)
        self.assertEqual(out["counts"]["base"], 2)
        self.assertEqual(out["counts"]["total"], 5)
        # base pairs preserved at the front
        self.assertEqual(out["augmented"][:2], base)

    def test_growth_history_records_totals(self):
        base = [("i1", "gt1")]
        out = bootstrap_selftrain(base, lambda i, g: [Candidate("a", 0.95)],
                                  rounds=1)
        self.assertEqual(out["history"][0]["srs_total"], 1)


if __name__ == "__main__":
    unittest.main()
