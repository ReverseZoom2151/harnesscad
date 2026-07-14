"""RFT: what the filter lets through, and what it must never let through."""

from __future__ import annotations

import unittest

from harnesscad.agents.selftrain import rft
from harnesscad.agents.selftrain.trajectory import Trajectory


def _t(tid: str, brief: str, ops, *, apply_ok=True, gate_ok=True,
       envelope_ok=True, shape_ok=True, parse_ok=True) -> Trajectory:
    return Trajectory(
        trajectory_id=tid, brief_id=brief, brief_text="a plate",
        prompt="a plate", ops=list(ops), parse_ok=parse_ok,
        verdict={"apply_ok": apply_ok, "gate_ok": gate_ok,
                 "envelope_ok": envelope_ok, "shape_ok": shape_ok,
                 "accepted": all([apply_ok, gate_ok, envelope_ok, shape_ok]),
                 "shape_iou": 0.99, "blind_spots": ["something"]},
    )


OPS_A = [{"op": "new_sketch", "plane": "XY"}, {"op": "extrude", "distance": 10}]
OPS_B = [{"op": "new_sketch", "plane": "XY"}, {"op": "extrude", "distance": 12}]


class TestPolicies(unittest.TestCase):

    def test_full_requires_the_conjunction(self):
        good = _t("g", "b", OPS_A)
        self.assertEqual(len(rft.build([good], policy="full")), 1)
        for kw in ("apply_ok", "gate_ok", "envelope_ok", "shape_ok"):
            bad = _t("x", "b", OPS_A, **{kw: False})
            self.assertEqual(len(rft.build([bad], policy="full")), 0,
                             "full policy must reject a candidate failing %s" % kw)

    def test_envelope_only_is_the_many_to_one_hole(self):
        # A part with the right bbox/volume/probes and the WRONG geometry. The v1
        # grader would have made this a training pair.
        wrong = _t("w", "b", OPS_A, shape_ok=False)
        self.assertEqual(len(rft.build([wrong], policy="envelope_only")), 1)
        self.assertEqual(len(rft.build([wrong], policy="full")), 0)

    def test_gate_only_is_the_production_ceiling(self):
        # The gate has never read the brief. It accepts a well-formed part that
        # answers the wrong question -- which is exactly what self-training on a
        # user's brief would have to live with, and it is why the yield gap
        # between gate_only and full is worth reporting.
        wrong = _t("w", "b", OPS_A, envelope_ok=False, shape_ok=False)
        self.assertEqual(len(rft.build([wrong], policy="gate_only")), 1)
        self.assertEqual(len(rft.build([wrong], policy="full")), 0)

    def test_unparseable_is_never_a_training_pair(self):
        t = _t("u", "b", [], parse_ok=False)
        self.assertEqual(len(rft.build([t], policy="gate_only")), 0)


class TestDedup(unittest.TestCase):

    def test_identical_streams_on_one_brief_collapse(self):
        ts = [_t("a", "b", OPS_A), _t("b", "b", OPS_A), _t("c", "b", OPS_B)]
        self.assertEqual(len(rft.build(ts, dedup=True)), 2)
        self.assertEqual(len(rft.build(ts, dedup=False)), 3)

    def test_the_same_stream_on_two_briefs_is_two_facts(self):
        ts = [_t("a", "b1", OPS_A), _t("b", "b2", OPS_A)]
        self.assertEqual(len(rft.build(ts, dedup=True)), 2)


class TestStats(unittest.TestCase):

    def test_many_to_one_gap_is_reported(self):
        ts = [_t("a", "b", OPS_A),                       # full
              _t("c", "b", OPS_B, shape_ok=False)]       # envelope but not shape
        s = rft.acceptance_stats(ts)
        self.assertEqual(s["by_policy"]["full"]["accepted"], 1)
        self.assertEqual(s["by_policy"]["envelope_only"]["accepted"], 2)
        self.assertEqual(s["many_to_one_gap"], 1)

    def test_the_distribution_warning_travels_with_the_data(self):
        self.assertIn("OUR habits", rft.DISTRIBUTION_WARNING)


class TestRationalization(unittest.TestCase):

    def test_a_reference_record_is_labelled_as_a_humans_answer(self):
        from harnesscad.eval.pressure import briefs as briefs_mod
        brief = briefs_mod.brief_by_id("l_bracket")
        rows = rft.rationalized_records([brief])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source, "rationalized")
        self.assertIn("HUMAN", rows[0].blind_spots[0])


if __name__ == "__main__":                                # pragma: no cover
    unittest.main()
