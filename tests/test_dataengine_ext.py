"""Tests for the extended data-engine capabilities (Scale-AI playbook):
distribution auditing, active-learning selection, consensus/QC labeling, and
design-intent capture. Dependency-free: synthetic op-dicts + hand-built records.
"""

import unittest

from dataengine.distribution_audit import audit_distribution, op_tags, family_of
from dataengine.active_learning import (
    select_informative, uncertainty_of, signature, uncertainty_scorer,
)
from dataengine.consensus import consensus_label
from dataengine.intent import IntentAnnotation, attach_intent, intents_of
from dataengine.trajectory import Action, Step, Trajectory


def _op(tag, **kw):
    d = {"op": tag}
    d.update(kw)
    return d


# =====================================================================
# Distribution auditing
# =====================================================================

class TestDistributionAudit(unittest.TestCase):
    def _corpus(self):
        # extrude deliberately dominates the op mix; two part families present.
        return [
            {"ops": [_op("new_sketch")] + [_op("extrude")] * 8, "generator": "plate"},
            {"ops": [_op("boolean")] + [_op("extrude")] * 8, "generator": "bracket"},
        ]

    def test_flags_over_represented_op(self):
        report = audit_distribution(self._corpus())
        over_tags = [f["tag"] for f in report.over_represented]
        self.assertIn("extrude", over_tags)
        # extrude is 16/18 of all ops.
        self.assertEqual(report.op_tag_freq["extrude"], 16)
        self.assertEqual(report.n_ops, 18)

    def test_under_represented_flagged(self):
        report = audit_distribution(self._corpus())
        under_tags = [f["tag"] for f in report.under_represented]
        # new_sketch / boolean each appear once -> under-represented vs uniform.
        self.assertIn("new_sketch", under_tags)

    def test_family_coverage(self):
        report = audit_distribution(self._corpus())
        self.assertEqual(report.coverage, 2)
        self.assertEqual(set(report.families), {"plate", "bracket"})

    def test_feature_and_ngram_histograms(self):
        report = audit_distribution(self._corpus())
        self.assertEqual(report.feature_types.get("extrude"), 16)
        # a new_sketch>extrude transition exists.
        self.assertIn("new_sketch>extrude", report.op_ngram_freq)

    def test_target_distribution_divergence(self):
        # A target that expects extrude to be rare -> KL/chi-square > 0, extrude over.
        target = {"new_sketch": 1.0, "boolean": 1.0, "extrude": 1.0}
        report = audit_distribution(self._corpus(), target=target)
        self.assertGreater(report.divergence["kl"], 0.0)
        self.assertGreater(report.divergence["chi_square"], 0.0)
        self.assertIn("extrude", [f["tag"] for f in report.over_represented])

    def test_to_dict_and_render(self):
        report = audit_distribution(self._corpus())
        d = report.to_dict()
        self.assertEqual(d["n_ops"], 18)
        self.assertIn("extrude", report.render())

    def test_op_tags_from_trajectory(self):
        traj = Trajectory(steps=[
            Step(0, {}, Action(tool_call=_op("new_sketch")), 1.0, {}, "applied"),
            Step(1, {}, Action(tool_call=_op("extrude")), -1.0, {}, "rolled-back"),
        ])
        # The rolled-back step is skipped: only the op that stuck is counted.
        self.assertEqual(op_tags(traj), ["new_sketch"])


# =====================================================================
# Active-learning selection
# =====================================================================

class TestActiveLearning(unittest.TestCase):
    def _candidates(self):
        novel = {"id": "novel", "ops": [_op("new_sketch"), _op("extrude")],
                 "uncertainty": 0.9, "generator": "plate"}
        dup1 = {"id": "dup1", "ops": [_op("new_sketch"), _op("add_circle"),
                                      _op("boolean")],
                "uncertainty": 0.1, "generator": "bracket"}
        dup2 = {"id": "dup2", "ops": [_op("new_sketch"), _op("add_circle"),
                                      _op("boolean")],
                "uncertainty": 0.1, "generator": "bracket"}
        return novel, dup1, dup2

    def test_high_uncertainty_novel_ranks_first(self):
        novel, dup1, dup2 = self._candidates()
        picked = select_informative([dup1, dup2, novel], 2)
        self.assertEqual(picked[0]["id"], "novel")

    def test_diverse_selection_drops_duplicate(self):
        novel, dup1, dup2 = self._candidates()
        picked = select_informative([dup1, dup2, novel], 2)
        ids = [c["id"] for c in picked]
        self.assertIn("novel", ids)
        # Only one of the two identical-signature duplicates is chosen.
        self.assertEqual(len({"dup1", "dup2"} & set(ids)), 1)

    def test_k_clamped_and_empty(self):
        novel, dup1, dup2 = self._candidates()
        self.assertEqual(select_informative([], 3), [])
        self.assertEqual(len(select_informative([novel, dup1], 10)), 2)
        self.assertEqual(select_informative([novel], 0), [])

    def test_pluggable_scorer(self):
        a = {"id": "a", "score": 0.2, "ops": [_op("extrude")]}
        b = {"id": "b", "score": 0.9, "ops": [_op("extrude")]}
        picked = select_informative([a, b], 1, scorer=lambda c: c["score"])
        self.assertEqual(picked[0]["id"], "b")
        # The default uncertainty scorer reads the uncertainty proxy.
        self.assertEqual(uncertainty_of({"uncertainty": 0.7}), 0.7)
        self.assertEqual(uncertainty_scorer({"score": 0.4}), 0.4)


# =====================================================================
# Consensus / QC labeling
# =====================================================================

class TestConsensus(unittest.TestCase):
    def test_accepts_above_threshold(self):
        res = consensus_label(["bracket", "bracket", "bracket", "gusset"])
        self.assertTrue(res.accepted)
        self.assertEqual(res.label, "bracket")
        self.assertAlmostEqual(res.agreement, 0.75)
        self.assertTrue(res.disagreement)  # not unanimous

    def test_rejects_split_vote(self):
        res = consensus_label(["a", "a", "b", "b"])
        self.assertFalse(res.accepted)
        self.assertIsNone(res.label)
        self.assertAlmostEqual(res.agreement, 0.5)
        self.assertTrue(res.disagreement)

    def test_unanimous(self):
        res = consensus_label(["x", "x", "x"])
        self.assertTrue(res.accepted)
        self.assertEqual(res.agreement, 1.0)
        self.assertEqual(res.pairwise_agreement, 1.0)
        self.assertFalse(res.disagreement)

    def test_gold_spot_check(self):
        res = consensus_label(["bracket", "bracket", "gusset"], gold="bracket")
        self.assertTrue(res.gold_matches)
        res2 = consensus_label(["gusset", "gusset", "bracket"], gold="bracket")
        self.assertFalse(res2.gold_matches)

    def test_vote_shapes_and_pairwise(self):
        # dict/tuple votes are unwrapped; pairwise agreement computed.
        res = consensus_label([{"label": "p"}, ("ann2", "p"), "q"])
        self.assertEqual(res.majority, "p")
        # 3 votes: pairs = 3, same-vote pairs = 1 (the two 'p's) -> 1/3.
        self.assertAlmostEqual(res.pairwise_agreement, 1.0 / 3.0)

    def test_empty(self):
        res = consensus_label([])
        self.assertFalse(res.accepted)
        self.assertIsNone(res.label)
        self.assertEqual(res.n_votes, 0)


# =====================================================================
# Design-intent capture
# =====================================================================

class TestIntent(unittest.TestCase):
    def test_annotation_round_trip(self):
        ann = IntentAnnotation("rib resists bending load",
                               {"process": "cnc", "min_wall_mm": 2.0},
                               op="extrude", index=1)
        self.assertEqual(IntentAnnotation.from_dict(ann.to_dict()), ann)

    def test_attach_to_step_infers_op_and_index(self):
        step = Step(index=2, state_before={},
                    action=Action(tool_call=_op("extrude")),
                    reward=1.0, state_after={}, outcome="applied")
        ann = attach_intent(step, "extrude to full plate thickness",
                            {"min_wall_mm": 2.0})
        self.assertEqual(ann.op, "extrude")
        self.assertEqual(ann.index, 2)
        got = intents_of(step)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0], ann)

    def test_attach_to_trajectory_metadata(self):
        traj = Trajectory()
        ann = attach_intent(traj, "a load-bearing bracket",
                            {"load_case": "cantilever"})
        self.assertEqual(traj.metadata["intents"][0]["rationale"],
                         "a load-bearing bracket")
        got = intents_of(traj)
        self.assertEqual(got[0], ann)

    def test_intent_json_serialisable(self):
        import json
        ann = IntentAnnotation("why", {"k": 1}, op="hole", index=0)
        s = json.dumps(ann.to_dict(), sort_keys=True)
        self.assertEqual(IntentAnnotation.from_dict(json.loads(s)), ann)


if __name__ == "__main__":
    unittest.main()
