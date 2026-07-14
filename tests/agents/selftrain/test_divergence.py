"""The first-divergence detector, validated against the KNOWN pressure regression.

The brief said the detector must identify op 3 on the 14b's ``trap_hole_oversize``
regression, and that if it does not, it is wrong. It does, and this asserts it on
the op streams lifted verbatim from ``assets/pressure/results.json``.
"""

from __future__ import annotations

import unittest

from harnesscad.agents.selftrain import divergence
from harnesscad.eval.pressure import briefs as briefs_mod

# Verbatim from assets/pressure/results.json:
#   model=qwen2.5-coder:14b brief=trap_hole_oversize
#   loop=blind    attempt 1 -> solved   (diameter 12, what the brief demanded)
#   loop=harness  attempt 1 -> solved, then the fleet said
#       "infeasible-plan: hole diameter 12 mm >= plate/stock wall 10 mm"
#       (a FALSE statement: those are orthogonal dimensions)
#   loop=harness  attempt 2 -> the 14b changed exactly one field, 12 -> 8, and lost.
_PREFIX = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 40, "h": 40},
    {"op": "extrude", "sketch": "sk1", "distance": 10},
]
CORRECT = _PREFIX + [
    {"op": "hole", "face_or_sketch": "solid", "x": 20, "y": 20,
     "diameter": 12, "through": True, "kind": "simple", "depth": None},
]
REGRESSED = _PREFIX + [
    {"op": "hole", "face_or_sketch": "solid", "x": 20, "y": 20,
     "diameter": 8, "through": True, "kind": "simple", "depth": None},
]


class TestTheKnownRegression(unittest.TestCase):

    def setUp(self):
        self.brief = briefs_mod.brief_by_id("trap_hole_oversize")

    def test_the_detector_finds_op_3(self):
        report = divergence.analyse(self.brief, REGRESSED)
        self.assertTrue(report.ok)
        self.assertEqual(report.first_divergence, 3,
                         "the detector must blame the HOLE op (0-based index 3), "
                         "the one field the 14b changed. It said %r."
                         % (report.first_divergence,))
        self.assertEqual(report.steps[3].op, "hole")
        self.assertEqual(report.blame, "model")

    def test_the_correct_attempt_has_no_divergence(self):
        # The same plan with the diameter the brief actually asked for. A detector
        # that fires on this is a detector with the fleet's disease.
        report = divergence.analyse(self.brief, CORRECT)
        self.assertTrue(report.ok)
        self.assertIsNone(report.first_divergence, report.detail)

    def test_the_prefix_is_not_condemned(self):
        # This is the whole point of sec. 12.8.3: ops 0-2 are byte-identical in the
        # correct and the regressed stream, so a per-step reward MUST rate them the
        # same. Outcome-only supervision gives all four ops a 0 and teaches the
        # model that its correct extrude was a mistake.
        report = divergence.analyse(self.brief, REGRESSED)
        rewards = divergence.step_rewards(report, REGRESSED)
        self.assertEqual([r.reward for r in rewards], [1.0, 1.0, 1.0, -1.0])
        self.assertEqual([r.divergent for r in rewards],
                         [False, False, False, True])

    def test_the_plate_prefix_is_recoverable(self):
        # After the extrude, the solid plate has 8% EXCESS material (the un-bored
        # hole) -- and a subtractive op is still coming, so it is recoverable and
        # is not a divergence. Getting this wrong would blame op 2.
        report = divergence.analyse(self.brief, REGRESSED)
        extrude = report.steps[2]
        self.assertEqual(extrude.op, "extrude")
        self.assertFalse(extrude.divergent)
        self.assertTrue(extrude.can_remove_later)
        self.assertGreater(extrude.excess, divergence.TOL)


class TestPolarity(unittest.TestCase):

    def test_op_polarity(self):
        self.assertEqual(divergence._polarity({"op": "extrude"}), "add")
        self.assertEqual(divergence._polarity({"op": "hole"}), "remove")
        self.assertEqual(divergence._polarity({"op": "shell"}), "remove")
        self.assertEqual(divergence._polarity({"op": "new_sketch"}), "none")
        self.assertEqual(
            divergence._polarity({"op": "boolean", "kind": "union"}), "add")
        self.assertEqual(
            divergence._polarity({"op": "boolean", "kind": "cut"}), "remove")

    def test_an_unknown_op_is_assumed_to_do_anything(self):
        # Soundness: if we do not know what an op can do, we must not claim the
        # plan was unrecoverable. An unknown op keeps every escape route open.
        self.assertEqual(divergence._polarity({"op": "wormhole"}), "both")
        self.assertTrue(divergence._can_add([{"op": "wormhole"}]))
        self.assertTrue(divergence._can_remove([{"op": "wormhole"}]))


class TestDegenerate(unittest.TestCase):

    def test_empty_stream(self):
        brief = briefs_mod.brief_by_id("l_bracket")
        report = divergence.analyse(brief, [])
        self.assertFalse(report.ok)
        self.assertIsNone(report.first_divergence)

    def test_a_refused_op_is_a_hard_divergence(self):
        brief = briefs_mod.brief_by_id("trap_hole_oversize")
        ops = _PREFIX + [{"op": "fillet", "radius": 500.0}]
        report = divergence.analyse(brief, ops)
        self.assertEqual(report.first_divergence, 3)

    def test_determinism(self):
        brief = briefs_mod.brief_by_id("trap_hole_oversize")
        a = divergence.analyse(brief, REGRESSED)
        b = divergence.analyse(brief, REGRESSED)
        self.assertEqual(a.to_dict(), b.to_dict())


if __name__ == "__main__":                                # pragma: no cover
    unittest.main()
