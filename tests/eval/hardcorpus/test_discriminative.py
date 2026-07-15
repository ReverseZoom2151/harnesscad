"""The headline, asserted: every near-miss fools the field's grader and not ours.

For each case: the CORRECT answer must pass BOTH graders (the control), and the
NEAR-MISS must be scored CORRECT by the field's oracle (or at least by its geometric
family) while the measured oracle scores it WRONG. If a control breaks, the brief is
broken and the case proves nothing; if a near-miss stops fooling the field, the case
is no longer discriminative and should be removed, not silently kept.
"""

from __future__ import annotations

import unittest

from harnesscad.eval.hardcorpus import discriminative as disc


class TestDiscriminative(unittest.TestCase):

    def setUp(self):
        self.verdicts = [disc.grade_case(nm) for nm in disc.CASES]

    def test_controls_hold_for_every_case(self):
        for v in self.verdicts:
            self.assertTrue(v.controls_hold,
                            "%s: the CORRECT answer does not pass both graders "
                            "(weak=%s oracle=%s) -- the case is broken"
                            % (v.id, v.weak_correct.get("passes"),
                               v.oracle_correct.get("solved")))

    def test_every_near_miss_fails_the_measured_oracle(self):
        for v in self.verdicts:
            self.assertFalse(v.oracle_near.get("solved"),
                             "%s: the near-miss PASSED the measured oracle -- it is "
                             "not a near-miss" % v.id)
            self.assertFalse(v.oracle_near.get("probes_ok"),
                             "%s: the near-miss passed the probe family; the probe "
                             "is not discriminating" % v.id)

    def test_every_near_miss_at_least_fools_the_geometric_family(self):
        # MUSE gates on watertight+manifold+valid; every near-miss must pass that.
        for v in self.verdicts:
            self.assertTrue(v.defeats_geometric_family,
                            "%s: the near-miss did not even pass MUSE's geometric "
                            "stage; it would be caught upstream" % v.id)

    def test_the_headline_cases_fool_iou_and_chamfer_too(self):
        # dia/pos/cbore/fillet must beat the FULL Text2CAD-Bench grader.
        full = {v.id: v for v in self.verdicts if v.scored == "full"}
        for name in ("dia_hole", "pos_hole", "cbore_plain", "fillet_edges"):
            self.assertIn(name, full,
                          "%s no longer beats IoU+Chamfer; the headline weakened"
                          % name)
            self.assertGreaterEqual(full[name].weak_near["iou"], disc.weak.IOU_MATCH)

    def test_shell_face_is_the_muse_blind_spot(self):
        v = next(x for x in self.verdicts if x.id == "shell_face")
        # volume, genus and watertight are IDENTICAL to the correct part.
        wc, wn = v.weak_correct, v.weak_near
        self.assertEqual(wc.get("watertight"), wn.get("watertight"))
        self.assertEqual(wc.get("manifold"), wn.get("manifold"))
        oc, on = v.oracle_correct["measured"], v.oracle_near["measured"]
        self.assertAlmostEqual(oc["volume"], on["volume"], places=3,
                               msg="the wrong-face shell must have identical volume")
        self.assertTrue(wn.get("valid"),
                        "MUSE's geometric stage must PASS this wrong part")

    def test_the_table_renders(self):
        text = disc.table(self.verdicts)
        self.assertIn("DISCRIMINATIVE TABLE", text)
        self.assertIn("GAP", text)


if __name__ == "__main__":
    unittest.main()
