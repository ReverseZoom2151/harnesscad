"""Preference pairs: the label is a measurement, and the fleet never touches it."""

from __future__ import annotations

import unittest

from harnesscad.agents.selftrain import preference
from harnesscad.agents.selftrain.trajectory import Trajectory

OPS_GOOD = [{"op": "new_sketch", "plane": "XY"}, {"op": "extrude", "distance": 10}]
OPS_BAD = [{"op": "new_sketch", "plane": "XY"}, {"op": "extrude", "distance": 3}]
OPS_UGLY = [{"op": "new_sketch", "plane": "XY"}]


def _t(tid, brief, ops, *, apply_ok=True, gate_ok=True, envelope_ok=True,
       shape_ok=True, diagnostics=None) -> Trajectory:
    return Trajectory(
        trajectory_id=tid, brief_id=brief, brief_text="a plate", prompt="a plate",
        ops=list(ops), parse_ok=True,
        diagnostics=list(diagnostics or []),
        verdict={"apply_ok": apply_ok, "gate_ok": gate_ok,
                 "envelope_ok": envelope_ok, "shape_ok": shape_ok,
                 "accepted": all([apply_ok, gate_ok, envelope_ok, shape_ok]),
                 "shape_iou": 0.99},
    )


class TestPairReward(unittest.TestCase):

    def test_it_is_ordinal_and_lexicographic(self):
        self.assertEqual(preference.pair_reward({"apply_ok": False}), 0.0)
        self.assertEqual(preference.pair_reward({"apply_ok": True}), 1.0)
        self.assertEqual(preference.pair_reward(
            {"apply_ok": True, "gate_ok": True}), 2.0)
        self.assertEqual(preference.pair_reward(
            {"apply_ok": True, "gate_ok": True, "envelope_ok": True}), 3.0)
        self.assertEqual(preference.pair_reward(
            {"apply_ok": True, "gate_ok": True, "envelope_ok": True,
             "shape_ok": True}), 4.0)


class TestDPO(unittest.TestCase):

    def test_a_correct_and_an_incorrect_part_on_one_brief_is_a_pair(self):
        ts = [_t("good", "b", OPS_GOOD),
              _t("bad", "b", OPS_BAD, envelope_ok=False, shape_ok=False)]
        pairs = preference.build_dpo(ts)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].chosen_id, "good")
        self.assertEqual(pairs[0].rejected_id, "bad")
        self.assertGreater(pairs[0].chosen_reward, pairs[0].rejected_reward)

    def test_pairs_never_cross_a_brief(self):
        ts = [_t("good", "b1", OPS_GOOD),
              _t("bad", "b2", OPS_BAD, envelope_ok=False, shape_ok=False)]
        self.assertEqual(preference.build_dpo(ts), [])

    def test_ties_carry_no_signal_and_are_dropped(self):
        ts = [_t("a", "b", OPS_GOOD), _t("b", "b", OPS_BAD)]   # both fully correct
        self.assertEqual(preference.build_dpo(ts), [])

    def test_strict_refuses_a_pair_of_two_wrong_answers(self):
        # "less wrong" is a preference over FAILURE MODES. Training on it teaches
        # the model to prefer a well-formed wrong part, which is the reward-hacking
        # direction the ledger warns about.
        ts = [_t("meh", "b", OPS_BAD, envelope_ok=False, shape_ok=False),
              _t("worse", "b", OPS_UGLY, gate_ok=False, envelope_ok=False,
                 shape_ok=False)]
        self.assertEqual(len(preference.build_dpo(ts, strict=True)), 0)
        self.assertEqual(len(preference.build_dpo(ts, strict=False)), 1)

    def test_the_label_source_is_stated_on_every_record(self):
        ts = [_t("good", "b", OPS_GOOD),
              _t("bad", "b", OPS_BAD, envelope_ok=False, shape_ok=False)]
        row = preference.build_dpo(ts)[0].to_dict()
        self.assertIn("NEVER the verifier fleet", row["label_source"])

    def test_a_false_fleet_diagnostic_does_not_flip_the_label(self):
        # THE RULE. The fleet fired 'infeasible-plan' on the CORRECT part 40 times
        # in the pressure run. If the fleet were the labeller, this pair would be
        # inverted and DPO would memorise it.
        ts = [_t("good", "b", OPS_GOOD,
                 diagnostics=[{"code": "infeasible-plan", "severity": "error"}]),
              _t("bad", "b", OPS_BAD, envelope_ok=False, shape_ok=False,
                 diagnostics=[])]
        pairs = preference.build_dpo(ts)
        self.assertEqual(pairs[0].chosen_id, "good")


class TestKTO(unittest.TestCase):

    def test_every_parsed_stream_is_one_unpaired_record(self):
        ts = [_t("good", "b1", OPS_GOOD),
              _t("bad", "b2", OPS_BAD, envelope_ok=False, shape_ok=False)]
        rows = preference.build_kto(ts)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r.desirable for r in rows], [True, False])

    def test_kto_needs_no_sibling_on_the_same_brief(self):
        # This is why KTO is the better first bet: DPO gets nothing from a lone
        # candidate, KTO gets a label.
        ts = [_t("lonely", "b", OPS_GOOD)]
        self.assertEqual(preference.build_dpo(ts), [])
        self.assertEqual(len(preference.build_kto(ts)), 1)

    def test_stats_report_the_imbalance(self):
        ts = [_t("g", "b", OPS_GOOD),
              _t("x", "b", OPS_BAD, envelope_ok=False, shape_ok=False)]
        s = preference.preference_stats(preference.build_dpo(ts),
                                        preference.build_kto(ts))
        self.assertEqual(s["kto_desirable"], 1)
        self.assertEqual(s["kto_undesirable"], 1)
        self.assertAlmostEqual(s["kto_imbalance"], 0.5)


if __name__ == "__main__":                                # pragma: no cover
    unittest.main()
