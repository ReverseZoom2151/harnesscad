"""Tests for the Fara-style verified-CAD-trajectory schema + CUAVerifierBench."""

import unittest

from harnesscad.agents.cua.verified_trajectory import (
    CUAVerifierBench, OracleVerdict, REJECTED, TrajectoryStep, UNLABELED,
    VERIFIED, VerifiedTrajectory, VerifierExample, label_trajectory,
)


def _traj(n=2, brief="build a block"):
    steps = [TrajectoryStep(index=i, observation={"marks": i},
                            action={"verb": "click", "target": "Pad"})
             for i in range(n)]
    return VerifiedTrajectory(brief=brief, steps=steps, trajectory_id="t1")


class TestOracleVerdict(unittest.TestCase):
    def test_constructors_and_labels(self):
        self.assertEqual(OracleVerdict.verified("ok").label, VERIFIED)
        self.assertEqual(OracleVerdict.rejected("bad").label, REJECTED)
        self.assertEqual(OracleVerdict.unlabeled().label, UNLABELED)
        self.assertTrue(OracleVerdict.verified().ok)
        self.assertFalse(OracleVerdict.rejected().ok)

    def test_is_labeled(self):
        self.assertTrue(OracleVerdict.verified().is_labeled)
        self.assertFalse(OracleVerdict.unlabeled().is_labeled)

    def test_dict_roundtrip(self):
        v = OracleVerdict.verified("detail", source="cad_oracle")
        self.assertEqual(OracleVerdict.from_dict(v.to_dict()), v)


class TestTrajectory(unittest.TestCase):
    def test_unlabeled_by_default(self):
        t = _traj()
        self.assertEqual(t.labeled_fraction(), 0.0)
        self.assertFalse(t.is_fully_verified())

    def test_label_trajectory_fills_verdicts(self):
        t = _traj(2)
        labelled = label_trajectory(
            t, step_labeller=lambda s: OracleVerdict.verified("step ok"),
            final_verdict=OracleVerdict.verified("part grades out"))
        self.assertEqual(labelled.labeled_fraction(), 1.0)
        self.assertTrue(labelled.is_fully_verified())
        # original is untouched (returns a new trajectory)
        self.assertEqual(t.labeled_fraction(), 0.0)

    def test_partial_label_not_fully_verified(self):
        t = _traj(2)
        labelled = label_trajectory(
            t,
            step_labeller=lambda s: (OracleVerdict.verified() if s.index == 0
                                     else OracleVerdict.rejected("miss")),
            final_verdict=OracleVerdict.rejected("part wrong"))
        self.assertEqual(labelled.labeled_fraction(), 1.0)  # both labeled...
        self.assertFalse(labelled.is_fully_verified())      # ...but not all VERIFIED

    def test_dict_roundtrip(self):
        t = label_trajectory(_traj(2),
                             step_labeller=lambda s: OracleVerdict.verified(),
                             final_verdict=OracleVerdict.verified())
        back = VerifiedTrajectory.from_dict(t.to_dict())
        self.assertEqual(back.to_dict(), t.to_dict())


class TestVerifierExample(unittest.TestCase):
    def test_false_positive_is_judge_pass_oracle_fail(self):
        ex = VerifierExample("t", oracle_ok=False, judge_ok=True)
        self.assertTrue(ex.false_positive)
        self.assertFalse(ex.false_negative)
        self.assertFalse(ex.agrees)

    def test_false_negative_is_judge_fail_oracle_pass(self):
        ex = VerifierExample("t", oracle_ok=True, judge_ok=False)
        self.assertTrue(ex.false_negative)
        self.assertFalse(ex.false_positive)


class TestCUAVerifierBench(unittest.TestCase):
    def test_empty_metrics(self):
        m = CUAVerifierBench().metrics()
        self.assertEqual(m["n"], 0)
        self.assertEqual(m["accuracy"], 0.0)

    def test_measures_judge_error(self):
        b = CUAVerifierBench()
        b.add("t1", oracle_ok=True, judge_ok=True)    # agree
        b.add("t2", oracle_ok=True, judge_ok=True)    # agree
        b.add("t3", oracle_ok=False, judge_ok=True)   # FALSE POSITIVE
        b.add("t4", oracle_ok=True, judge_ok=False)   # false negative
        m = b.metrics()
        self.assertEqual(m["n"], 4)
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["false_positives"], 1)
        self.assertEqual(m["false_negatives"], 1)
        self.assertEqual(m["oracle_negatives"], 1)
        self.assertEqual(m["oracle_positives"], 3)
        # FP rate is over oracle-failed trajectories: 1/1.
        self.assertEqual(m["false_positive_rate"], 1.0)
        # FN rate is over oracle-passed trajectories: 1/3.
        self.assertAlmostEqual(m["false_negative_rate"], 1 / 3)

    def test_perfect_judge_is_accurate(self):
        b = CUAVerifierBench()
        for i in range(3):
            b.add("t%d" % i, oracle_ok=True, judge_ok=True)
        b.add("bad", oracle_ok=False, judge_ok=False)
        m = b.metrics()
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["false_positive_rate"], 0.0)

    def test_add_from_verdicts(self):
        b = CUAVerifierBench()
        b.add_from_verdicts("t", OracleVerdict.verified(), OracleVerdict.rejected())
        self.assertEqual(len(b), 1)
        self.assertTrue(b.examples[0].false_negative)


if __name__ == "__main__":
    unittest.main()
