"""The precision floor gate: a rule with precision 0.4 must not be mergeable."""

from __future__ import annotations

import json
import unittest

from harnesscad.eval.gates import precision_floor as pf


def _score(name, *, precision=None, fp=0, tp=0, fp_codes=None, false_positives=()):
    return {
        "name": name, "tier": "lint", "tp": tp, "fp": fp, "fn": 0, "tn": 0,
        "out_of_scope": 0, "abstained": False,
        "precision": precision, "recall": None, "f1": None,
        "false_positives": list(false_positives),
        "false_negatives": [], "codes": {}, "fp_codes": dict(fp_codes or {}),
        "errored": 0,
    }


def _report(scores):
    return {
        "oracle": "fleet", "ok": True, "backend": "frep", "known_good": 8,
        "known_bad": 8,
        "fleet": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "false_positive_rate": 0.0},
        "verifiers": scores,
    }


class TestSoundnessProjection(unittest.TestCase):
    def test_measured_verifier_is_model_facing(self):
        self.assertTrue(pf.is_model_facing("brep-validity"))
        self.assertEqual(pf.model_facing_codes("brep-validity"), ["*"])

    def test_heuristic_verifier_is_not_model_facing(self):
        self.assertFalse(pf.is_model_facing("completeness"))
        self.assertFalse(pf.is_model_facing("precheck"))

    def test_kernel_preflight_is_model_facing_only_for_its_theorems(self):
        codes = pf.model_facing_codes("kernel-preflight")
        self.assertIn("preflight-THICKNESS_TOO_LARGE", codes)
        self.assertIn("preflight-ZERO_VOLUME", codes)
        # The unsound one (report.md, fleet hole 4: a 50x30x6 plate filleted at
        # r=3.1 is valid and the rule rejects it) may NEVER be promoted.
        self.assertNotIn("preflight-RADIUS_TOO_LARGE", codes)


class TestGate(unittest.TestCase):
    def test_committed_baseline_passes_the_live_shape(self):
        base = pf.baseline()
        self.assertIn("verifiers", base)
        # Every model-facing verifier is committed at 1.0 and nowhere else.
        for name, entry in base["verifiers"].items():
            if entry["model_facing"]:
                self.assertEqual(entry["precision_floor"], pf.MODEL_FACING_FLOOR,
                                 "%s is model-facing; its floor must be 1.0" % name)

    def test_a_new_unregistered_verifier_fails_the_build(self):
        rep = _report([_score("brand-new-rule", precision=1.0, tp=3)])
        gate = pf.check(rep, floors={"verifiers": {}})
        self.assertFalse(gate.ok)
        self.assertEqual([v.kind for v in gate.violations], ["unregistered"])

    def test_a_model_facing_rule_with_precision_0_4_fails(self):
        # brep-validity is MEASURED -> model-facing. Pretend it started lying.
        rep = _report([_score("brep-validity", precision=0.4, tp=2, fp=3,
                              fp_codes={"invalid-brep": 3},
                              false_positives=["washer_80x8_bore30"])])
        floors = {"verifiers": {"brep-validity": {
            "model_facing": True, "precision_floor": 1.0}}}
        gate = pf.check(rep, floors=floors)
        self.assertFalse(gate.ok)
        kinds = sorted(v.kind for v in gate.violations)
        self.assertEqual(kinds, ["below-floor", "model-facing-fp"])
        self.assertGreater(gate.model_facing_false_positive_rate, 0.0)

    def test_a_model_facing_floor_below_one_is_itself_a_violation(self):
        # Nobody gets to lower the floor on the channel that instructs the model.
        rep = _report([_score("brep-validity", precision=0.9, tp=9, fp=1,
                              fp_codes={"invalid-brep": 1},
                              false_positives=["p"])])
        floors = {"verifiers": {"brep-validity": {
            "model_facing": True, "precision_floor": 0.9}}}
        gate = pf.check(rep, floors=floors)
        self.assertFalse(gate.ok)
        self.assertIn("below-floor", [v.kind for v in gate.violations])

    def test_heuristic_false_positives_do_not_fail_the_model_facing_channel(self):
        # completeness fires on every part. It is HEURISTIC: the planner's
        # soundness gate drops it, so no model ever sees it. It regresses only
        # if its precision drops below the committed value.
        rep = _report([_score("completeness", precision=0.5, tp=8, fp=8,
                              fp_codes={"missing-metadata": 36},
                              false_positives=["a", "b"])])
        floors = {"verifiers": {"completeness": {
            "model_facing": False, "precision_floor": 0.5}}}
        gate = pf.check(rep, floors=floors)
        self.assertTrue(gate.ok, [v.message for v in gate.violations])
        self.assertEqual(gate.model_facing_false_positive_rate, 0.0)

    def test_a_heuristic_rule_that_regresses_fails(self):
        rep = _report([_score("completeness", precision=0.3, tp=3, fp=7,
                              fp_codes={"missing-metadata": 7})])
        floors = {"verifiers": {"completeness": {
            "model_facing": False, "precision_floor": 0.5}}}
        gate = pf.check(rep, floors=floors)
        self.assertFalse(gate.ok)

    def test_report_serialises(self):
        rep = _report([_score("precheck", precision=1.0, tp=6)])
        floors = {"verifiers": {"precheck": {
            "model_facing": False, "precision_floor": 1.0}}}
        gate = pf.check(rep, floors=floors)
        json.dumps(gate.to_dict())
        self.assertTrue(gate.ok)
        self.assertIn("PASS", pf.format_text(gate))


if __name__ == "__main__":
    unittest.main()
