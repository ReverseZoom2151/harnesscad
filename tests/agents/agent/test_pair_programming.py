import unittest

from harnesscad.agents.agent.pair_programming import (
    ACCEPT_TOKEN,
    Round,
    SwitchPolicy,
    run_pair_loop,
    select_best,
    should_switch,
)


class SwitchPolicyTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(SwitchPolicy.parse("err2"), SwitchPolicy("err", 2))
        self.assertEqual(SwitchPolicy.parse("fixed3"), SwitchPolicy("fixed", 3))
        self.assertEqual(SwitchPolicy.parse("none"), SwitchPolicy("none", 0))
        self.assertEqual(SwitchPolicy.parse(""), SwitchPolicy("err", 1))

    def test_parse_invalid(self):
        with self.assertRaises(ValueError):
            SwitchPolicy.parse("wobble")

    def test_should_switch_err(self):
        p = SwitchPolicy("err", 2)
        self.assertFalse(should_switch(p, 1, 0))
        self.assertTrue(should_switch(p, 2, 0))

    def test_should_switch_fixed(self):
        p = SwitchPolicy("fixed", 2)
        self.assertFalse(should_switch(p, 1, 0))
        self.assertTrue(should_switch(p, 1, 1))

    def test_should_switch_none(self):
        self.assertFalse(should_switch(SwitchPolicy("none", 0), 99, 99))


class SelectBestTests(unittest.TestCase):
    def test_prefers_passing_then_score_then_recency(self):
        rounds = [
            Round("a", False, 0.1, "", False, False),
            Round("b", False, 0.9, "", False, False),
            Round("c", True, 0.2, "", False, False),
        ]
        self.assertEqual(select_best(rounds), 2)  # passing wins

    def test_recency_tiebreak(self):
        rounds = [
            Round("a", True, 0.5, "", False, False),
            Round("b", True, 0.5, "", False, False),
        ]
        self.assertEqual(select_best(rounds), 1)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            select_best([])


class RunPairLoopTests(unittest.TestCase):
    def test_accept_on_second_round(self):
        outputs = iter(["bad", "good"])

        def generate(prompt):
            return next(outputs)

        def review(prompt):
            return ACCEPT_TOKEN if "good" in prompt else "please fix line 1"

        res = run_pair_loop("make a widget", generate, review, max_iters=4)
        self.assertTrue(res.accepted)
        self.assertEqual(res.artifact, "good")
        self.assertEqual(res.iters, 1)
        self.assertEqual(len(res.rounds), 2)

    def test_check_evidence_recorded(self):
        outputs = iter(["bad", "good"])

        def generate(prompt):
            return next(outputs)

        def review(prompt):
            return ACCEPT_TOKEN if "good" in prompt else "fix"

        def check(artifact):
            ok = artifact == "good"
            return ok, "" if ok else "does not compile", 1.0 if ok else 0.0

        res = run_pair_loop("t", generate, review, check=check, max_iters=4)
        self.assertTrue(res.accepted)
        self.assertFalse(res.rounds[0].check_ok)
        self.assertTrue(res.rounds[1].check_ok)

    def test_no_acceptance_returns_argmax_quality(self):
        outputs = iter(["a", "b", "c"])
        scores = {"a": 0.1, "b": 0.9, "c": 0.5}

        def generate(prompt):
            return next(outputs)

        def review(prompt):
            return "keep fixing"

        def check(artifact):
            return False, "err", scores[artifact]

        res = run_pair_loop("t", generate, review, check=check, max_iters=2)
        self.assertFalse(res.accepted)
        self.assertEqual(res.artifact, "b")  # highest score across all candidates
        self.assertEqual(res.iters, 2)


if __name__ == "__main__":
    unittest.main()
