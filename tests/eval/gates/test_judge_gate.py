"""An uncalibrated judge may not instruct. Same class of bug as the washer, one
layer up: an unaudited evaluator inside a loop."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from harnesscad.eval.gates import judge_gate as jg


def _records():
    # A judge whose distance is LOW for parts the oracle accepted: it agrees.
    good = [{"distance": 0.05 + 0.01 * i, "accepted": True} for i in range(10)]
    bad = [{"distance": 0.80 + 0.01 * i, "accepted": False} for i in range(10)]
    return good + bad


def _bad_records():
    # A judge that is uncorrelated with the oracle: it is noise wearing a number.
    return [{"distance": 0.5, "accepted": i % 2 == 0} for i in range(20)]


class TestCalibration(unittest.TestCase):
    def test_a_faithful_judge_calibrates_and_clears_the_bar(self):
        cal = jg.calibrate("vlm-judge", _records())
        self.assertEqual(cal.precision, 1.0)
        self.assertEqual(cal.agreement, 1.0)
        self.assertGreater(cal.kendall, jg.MIN_KENDALL)
        self.assertTrue(cal.reliable)

    def test_a_noisy_judge_calibrates_and_FAILS_the_bar(self):
        cal = jg.calibrate("vlm-judge", _bad_records())
        self.assertFalse(cal.reliable)

    def test_calibrating_on_nothing_is_refused(self):
        with self.assertRaises(ValueError):
            jg.calibrate("vlm-judge", [])


class TestPolicy(unittest.TestCase):
    def test_the_shipped_vlm_judge_is_uncalibrated_today(self):
        # This is a fact about the repository, and it is why VLMJudgeCheck is
        # capped at INFO. When somebody calibrates it, this test changes.
        self.assertFalse(jg.is_calibrated("vlm-judge"))

    def test_an_uncalibrated_judge_is_refused_the_channel(self):
        with self.assertRaises(jg.UncalibratedJudge):
            jg.require_calibrated("vlm-judge")

    def test_a_calibrated_judge_is_licensed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cal.json")
            cal = jg.calibrate("vlm-judge", _records())
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"judges": {"vlm-judge": cal.to_dict()}}, fh)
            self.assertTrue(jg.is_calibrated("vlm-judge", path))
            self.assertEqual(jg.require_calibrated("vlm-judge", path=path).precision,
                             1.0)

    def test_gate_passes_when_every_judge_is_advisory(self):
        self.assertTrue(jg.check().ok)

    def test_gate_FAILS_a_model_facing_judge_with_no_calibration(self):
        rep = jg.check(judges={"vlm-judge": {"model_facing": True}})
        self.assertFalse(rep.ok)
        self.assertIn("MODEL-FACING", rep.violations[0])


if __name__ == "__main__":
    unittest.main()
