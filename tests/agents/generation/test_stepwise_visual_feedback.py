import unittest

from harnesscad.agents.generation.stepwise_visual_feedback import (
    NEGATIVE,
    POSITIVE,
    build_svf_plan,
    intermediate_steps,
    refine_with_svf,
    ultimate_step,
)


class TestBuildSvfPlan(unittest.TestCase):
    def test_lengths(self):
        plan = build_svf_plan(3)
        # 3 intermediate + 1 ultimate
        self.assertEqual(len(plan), 4)
        self.assertEqual(len(intermediate_steps(plan)), 3)

    def test_visibility_growth(self):
        plan = intermediate_steps(build_svf_plan(3))
        self.assertEqual(plan[0].visible, (0,))
        self.assertEqual(plan[1].visible, (0, 1))
        self.assertEqual(plan[2].visible, (0, 1, 2))

    def test_highlight_and_hide(self):
        plan = intermediate_steps(build_svf_plan(3))
        # step k highlights k and hides all priors j<k
        self.assertEqual(plan[2].highlighted, 2)
        self.assertEqual(plan[2].hidden, (0, 1))
        self.assertEqual(plan[0].hidden, ())

    def test_ultimate(self):
        u = ultimate_step(build_svf_plan(3))
        self.assertEqual(u.kind, "ultimate")
        self.assertEqual(u.visible, (0, 1, 2))
        self.assertIsNone(u.highlighted)
        self.assertEqual(u.hidden, ())

    def test_cot_pairing(self):
        plan = intermediate_steps(build_svf_plan(3, cot_len=2))
        self.assertEqual(plan[0].cot_step, 0)
        self.assertEqual(plan[1].cot_step, 1)
        self.assertIsNone(plan[2].cot_step)  # no t_2 available

    def test_zero_triples(self):
        with self.assertRaises(ValueError):
            build_svf_plan(0)


class TestRefineWithSvf(unittest.TestCase):
    def test_positive_stops_immediately(self):
        res = refine_with_svf(
            "code0",
            n_triples=2,
            judge_fn=lambda c, p: POSITIVE,
            refine_fn=lambda c, f: c + "!",
            max_rounds=3,
        )
        self.assertTrue(res.converged)
        self.assertEqual(res.rounds, 1)
        self.assertEqual(res.code, "code0")
        self.assertEqual(res.verdicts, [POSITIVE])

    def test_negative_refines_up_to_cap(self):
        calls = []

        def judge(c, p):
            calls.append(c)
            return NEGATIVE

        res = refine_with_svf(
            "c",
            n_triples=1,
            judge_fn=judge,
            refine_fn=lambda c, f: c + "x",
            max_rounds=2,
        )
        self.assertFalse(res.converged)
        self.assertEqual(res.rounds, 2)
        self.assertEqual(res.code, "cxx")
        self.assertEqual(res.verdicts, [NEGATIVE, NEGATIVE])

    def test_negative_then_positive(self):
        seq = iter([NEGATIVE, POSITIVE])
        res = refine_with_svf(
            "c",
            n_triples=1,
            judge_fn=lambda c, p: next(seq),
            refine_fn=lambda c, f: c + "1",
            max_rounds=5,
        )
        self.assertTrue(res.converged)
        self.assertEqual(res.rounds, 2)
        self.assertEqual(res.code, "c1")

    def test_judge_uses_plan(self):
        captured = {}

        def judge(c, plan):
            captured["n"] = len(plan)
            return POSITIVE

        refine_with_svf(
            "c",
            n_triples=4,
            judge_fn=judge,
            refine_fn=lambda c, f: c,
            max_rounds=1,
        )
        self.assertEqual(captured["n"], 5)  # 4 intermediate + 1 ultimate

    def test_bad_verdict(self):
        with self.assertRaises(ValueError):
            refine_with_svf(
                "c",
                n_triples=1,
                judge_fn=lambda c, p: "maybe",
                refine_fn=lambda c, f: c,
            )

    def test_bad_max_rounds(self):
        with self.assertRaises(ValueError):
            refine_with_svf(
                "c",
                n_triples=1,
                judge_fn=lambda c, p: POSITIVE,
                refine_fn=lambda c, f: c,
                max_rounds=0,
            )


if __name__ == "__main__":
    unittest.main()
