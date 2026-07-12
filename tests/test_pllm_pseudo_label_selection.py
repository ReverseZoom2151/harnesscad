import unittest

from dataengine.pllm_pseudo_label_selection import (
    Candidate, NEAR_TIE, accept_pseudo_label, executable_candidates,
    select_dataset, select_representative,
)


class TestExecutableFilter(unittest.TestCase):
    def test_drops_non_executable(self):
        cands = [Candidate("a", 0.1, 5, True), Candidate("b", 0.05, 4, False)]
        execs = executable_candidates(cands)
        self.assertEqual([c.program for c in execs], ["a"])

    def test_accepts_tuple_form(self):
        execs = executable_candidates([("a", 0.1, 5, True)])
        self.assertEqual(execs[0].program, "a")


class TestSelectRepresentative(unittest.TestCase):
    def test_lowest_chamfer_wins(self):
        cands = [Candidate("a", 0.3, 5, True), Candidate("b", 0.1, 9, True),
                 Candidate("c", 0.2, 3, True)]
        self.assertEqual(select_representative(cands).program, "b")

    def test_near_tie_prefers_shorter(self):
        # b is marginally worse than a but within NEAR_TIE -> shorter b wins.
        cands = [Candidate("a", 0.10000, 9, True),
                 Candidate("b", 0.10005, 4, True)]
        self.assertEqual(select_representative(cands).program, "b")

    def test_beyond_tie_keeps_best_chamfer(self):
        cands = [Candidate("a", 0.10, 9, True), Candidate("b", 0.20, 4, True)]
        # gap 0.10 >> NEAR_TIE, so lowest chamfer a wins despite being longer.
        self.assertEqual(select_representative(cands).program, "a")

    def test_all_non_executable_returns_none(self):
        cands = [Candidate("a", 0.1, 5, False)]
        self.assertIsNone(select_representative(cands))

    def test_deterministic_full_tie(self):
        cands = [Candidate("z", 0.1, 4, True), Candidate("a", 0.1, 4, True)]
        # equal chamfer and length -> program identity breaks tie ("a" < "z").
        self.assertEqual(select_representative(cands).program, "a")

    def test_negative_tie_raises(self):
        with self.assertRaises(ValueError):
            select_representative([Candidate("a", 0.1, 5, True)], near_tie=-1.0)


class TestAcceptPseudoLabel(unittest.TestCase):
    def test_accept_within_threshold(self):
        out = accept_pseudo_label([Candidate("a", 0.05, 5, True)], cd_threshold=0.1)
        self.assertTrue(out["accepted"])
        self.assertEqual(out["reason"], "accepted")
        self.assertEqual(out["program"], "a")

    def test_reject_below_confidence(self):
        out = accept_pseudo_label([Candidate("a", 0.5, 5, True)], cd_threshold=0.1)
        self.assertFalse(out["accepted"])
        self.assertEqual(out["reason"], "below_confidence")
        # program still reported for inspection
        self.assertEqual(out["program"], "a")

    def test_reject_no_executable(self):
        out = accept_pseudo_label([Candidate("a", 0.01, 5, False)], cd_threshold=0.1)
        self.assertFalse(out["accepted"])
        self.assertEqual(out["reason"], "no_executable")

    def test_boundary_accepts(self):
        out = accept_pseudo_label([Candidate("a", 0.1, 5, True)], cd_threshold=0.1)
        self.assertTrue(out["accepted"])

    def test_negative_threshold_raises(self):
        with self.assertRaises(ValueError):
            accept_pseudo_label([Candidate("a", 0.1, 5, True)], cd_threshold=-1)


class TestSelectDataset(unittest.TestCase):
    def test_yield_and_counts(self):
        data = {
            "s1": [Candidate("p1", 0.02, 5, True)],
            "s2": [Candidate("p2", 0.9, 5, True)],          # below confidence
            "s3": [Candidate("p3", 0.01, 5, False)],        # no executable
            "s4": [Candidate("p4a", 0.03, 8, True),
                   Candidate("p4b", 0.030005, 4, True)],    # near tie -> shorter
        }
        out = select_dataset(data, cd_threshold=0.1)
        self.assertEqual(out["counts"]["total"], 4)
        self.assertEqual(out["counts"]["accepted"], 2)
        self.assertAlmostEqual(out["yield"], 0.5)
        progs = {r["shape_id"]: r["program"] for r in out["accepted"]}
        self.assertEqual(progs["s1"], "p1")
        self.assertEqual(progs["s4"], "p4b")

    def test_empty(self):
        out = select_dataset({}, cd_threshold=0.1)
        self.assertEqual(out["yield"], 0.0)

    def test_near_tie_constant(self):
        self.assertGreater(NEAR_TIE, 0)


if __name__ == "__main__":
    unittest.main()
