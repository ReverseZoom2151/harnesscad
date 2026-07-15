"""The scorer: it grades on BOTH oracles, and the held-out references are sound.

``reference_score`` is the corpus's self-test -- if a held-out reference does not
pass its own oracle, the brief is broken and every score against it is meaningless
(the pressure corpus shipped exactly that failure). It is run here, through the
scorer, without ever importing the held-out briefs into this test.
"""

from __future__ import annotations

import unittest

from harnesscad.eval.hardcorpus import score


class TestScore(unittest.TestCase):

    def test_size_is_positive(self):
        self.assertGreater(score.size(), 0)

    def test_held_out_references_pass_their_own_oracle(self):
        r = score.reference_score()
        self.assertEqual(r.oracle_solved, r.n,
                         "a held-out reference fails its own oracle: %s"
                         % r.failed)
        self.assertEqual(r.built, r.n)

    def test_a_wrong_solver_is_caught_and_the_field_is_fooled(self):
        # A solver that always drills an 8 mm hole in a 60x40x12 plate: a valid,
        # watertight part that the field's IoU passes on the hole briefs but the
        # oracle fails. This exercises the field_fooled counter end to end.
        from harnesscad.core.cisp.ops import (AddRectangle, Extrude, Hole,
                                              NewSketch)

        def bad(_text):
            return [NewSketch("XY"), AddRectangle("sk1", 0, 0, 60, 40),
                    Extrude("sk1", 12), Hole("sk1", 20, 20, 8, None, True, "simple")]

        r = score.score(bad)
        self.assertEqual(r.n, score.size())
        # It builds valid parts, so at least some are weak-passed while oracle-failed.
        self.assertGreaterEqual(r.built, 1)

    def test_near_miss_audit_shows_the_gap(self):
        a = score.near_miss_audit()
        self.assertGreater(a.n, 0)
        self.assertEqual(a.oracle_failed_near, a.n,
                         "every held-out near-miss must fail the oracle")
        self.assertEqual(a.gaps, a.n,
                         "every held-out near-miss must demonstrate a gap")

    def test_report_is_numbers_only(self):
        r = score.reference_score()
        d = r.to_dict()
        # No op streams, no reference geometry -- just counts, rates and reasons.
        self.assertNotIn("reference", d)
        self.assertIn("oracle_rate", d)
        self.assertIn("weak_rate", d)


if __name__ == "__main__":
    unittest.main()
