"""Scoring tests for the third adapter wave of the bench registry.

The second wave is pinned by ``test_registry.py``. This module does the same for
the third wave: every newly-bound metric is materialised, kept out of
``unadapted()``, and -- for a representative sample across the buckets -- actually
*scored* on a tiny hand-built pred/gold pair, asserting the metric's own
semantics (identical -> best, disjoint -> worst where that is meaningful). A
metric bound but never scored is not really wired, so these are scoring tests,
not just discovery tests.
"""

from __future__ import annotations

import unittest

from harnesscad.eval.bench import registry as bench


#: Every metric the third wave added, by name.
WAVE_THREE = (
    "geometry.scaled_chamfer_reward",
    "geometry.compiler_chamfer",
    "geometry.solver_feedback",
    "geometry.design_distance_curve",
    "geometry.edit_relation_preservation",
    "geometry.identity_preservation",
    "geometry.occupancy_jsd",
    "geometry.constraint_satisfaction",
    "geometry.typed_requirements",
    "geometry.cae_feasibility",
    "geometry.assemblability",
    "sequence.edit_operation_f1",
    "sequence.reward_hacking",
    "sequence.cad_sequence_f1",
    "vision.quantity_alignment",
    "vision.defect_confusion",
    "vision.description_match",
    "vision.dfm_feature_recognition",
    "vision.mfr_quantity",
    "vision.volume_fraction_error",
    "vision.orthographic_reasoning",
    "vision.dimension_extraction",
    "retrieval.pairwise_edge",
    "retrieval.grounding",
    "retrieval.text_to_cad",
    "retrieval.ranking_agreement",
    "retrieval.choice_optimality",
    "retrieval.tool_retrieval",
    "retrieval.designqa",
    "retrieval.qa_grade",
    "retrieval.qa_evidence",
    "generative.cad_qa_accuracy",
    "generative.judge_human_agreement",
    "generative.feasibility_correlation",
    "generative.perceived_actual_gap",
)


def _side(**kw):
    """One side (pred or gold) of a sample: only the keys a metric reads."""
    return dict(kw)


def _score(name, pred, gold):
    return bench.metric(name).score(pred, gold)


class DiscoveryTest(unittest.TestCase):
    def test_every_new_metric_is_discovered_and_bound(self):
        known = {m.name: m for m in bench.metrics()}
        indexed = {m.dotted for m in bench.metrics()}
        for name in WAVE_THREE:
            self.assertIn(name, known, f"{name} was not discovered")
            self.assertNotIn(known[name].dotted, bench.unadapted(),
                             f"{name}'s module is still listed unadapted")
            self.assertIn(known[name].kind, bench.KINDS)
            for key in known[name].inputs:
                self.assertIn(key, bench.INPUT_KINDS, f"{name} needs unknown {key}")
        # discovery surface stays a surface: unadapted is non-empty and every
        # stated reason names a still-unadapted module.
        self.assertTrue(bench.unadapted())
        unadapted = set(bench.unadapted())
        for dotted, reason in bench.UNADAPTED_REASONS:
            self.assertIn(dotted, unadapted, f"{dotted} has a reason but is adapted")
            self.assertTrue(reason.strip())

    def test_no_module_is_adapted_twice_except_the_voxel_pair(self):
        counts = {}
        for m in bench.metrics():
            counts.setdefault(m.dotted, []).append(m.name)
        for dotted, names in counts.items():
            if len(names) > 1:
                self.assertEqual(
                    sorted(names),
                    sorted(["geometry.voxel_iou_grid", "geometry.voxel_iou_points"]),
                    f"{dotted} adapted by {names}")


class ScoringTest(unittest.TestCase):
    """Representative metrics score sanely: identical -> best, disjoint -> worst."""

    def test_compiler_chamfer_zero_on_identical_positive_on_disjoint(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        same = _score("geometry.compiler_chamfer",
                      _side(points=pts), _side(points=pts))
        far = _score("geometry.compiler_chamfer",
                     _side(points=pts),
                     _side(points=[(10.0, 10.0, 10.0), (11.0, 10.0, 10.0),
                                   (10.0, 11.0, 10.0)]))
        self.assertEqual(same, 0.0)
        self.assertGreater(far, same)

    def test_scaled_chamfer_reward_perfect_on_identity(self):
        pts = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)]
        v = _score("geometry.scaled_chamfer_reward",
                   _side(points=pts), _side(points=pts))
        self.assertEqual(v["reward"], 1.0)
        self.assertEqual(v["scd"], 0.0)

    def test_quantity_alignment_best_on_equal_counts(self):
        best = _score("vision.quantity_alignment",
                      _side(counts=[3.0, 5.0]), _side(counts=[3.0, 5.0]))
        worse = _score("vision.quantity_alignment",
                       _side(counts=[0.0, 0.0]), _side(counts=[10.0, 10.0]))
        self.assertEqual(best["mae"], 0.0)
        self.assertEqual(best["score"], 1.0)
        self.assertGreater(worse["mae"], best["mae"])
        self.assertLess(worse["score"], best["score"])

    def test_dfm_feature_recognition_f1_one_identical_zero_disjoint(self):
        same = _score("vision.dfm_feature_recognition",
                      _side(feature_names=["chamfer"]),
                      _side(feature_names=["chamfer"]))
        disjoint = _score("vision.dfm_feature_recognition",
                          _side(feature_names=["chamfer"]),
                          _side(feature_names=["rectangular pocket"]))
        self.assertEqual(same["f1"], 1.0)
        self.assertEqual(disjoint["f1"], 0.0)

    def test_designqa_token_f1_one_identical_zero_disjoint(self):
        same = _score("retrieval.designqa",
                      _side(answer_text="the quick brown fox"),
                      _side(answer_text="the quick brown fox"))
        disjoint = _score("retrieval.designqa",
                          _side(answer_text="red green blue"),
                          _side(answer_text="one two three"))
        self.assertEqual(same["token_f1"], 1.0)
        self.assertEqual(disjoint["token_f1"], 0.0)

    def test_text_to_cad_recall_top_hit_vs_miss(self):
        hit = _score("retrieval.text_to_cad",
                     _side(id_ranking={"ranked": [1, 2, 3]}),
                     _side(id_ranking={"gt": 1}))
        miss = _score("retrieval.text_to_cad",
                      _side(id_ranking={"ranked": [2, 3, 4]}),
                      _side(id_ranking={"gt": 1}))
        self.assertEqual(hit["recall_at_1"], 100.0)
        self.assertEqual(miss["recall_at_1"], 0.0)

    def test_pairwise_edge_perfect_vs_scrambled_partition(self):
        best = _score("retrieval.pairwise_edge",
                      _side(cluster_labels=[0, 0, 1, 1]),
                      _side(cluster_labels=[0, 0, 1, 1]))
        worse = _score("retrieval.pairwise_edge",
                       _side(cluster_labels=[0, 1, 0, 1]),
                       _side(cluster_labels=[0, 0, 1, 1]))
        self.assertEqual(best["edge_accuracy"], 1.0)
        self.assertLess(worse["edge_accuracy"], best["edge_accuracy"])

    def test_judge_human_agreement_signs_track_correlation(self):
        pos = _score("generative.judge_human_agreement",
                     _side(scores=[1.0, 2.0, 3.0, 4.0]),
                     _side(scores=[1.0, 2.0, 3.0, 4.0]))
        neg = _score("generative.judge_human_agreement",
                     _side(scores=[1.0, 2.0, 3.0, 4.0]),
                     _side(scores=[4.0, 3.0, 2.0, 1.0]))
        self.assertEqual(pos["pearson"], 1.0)
        self.assertEqual(neg["pearson"], -1.0)

    def test_typed_requirements_all_pass_vs_all_fail(self):
        contract = {"requirements": [["max_stress", "stress", "<=", 250.0]]}
        passing = _score("geometry.typed_requirements",
                         _side(requirement_contract={"measurements": {"max_stress": 100.0}}),
                         _side(requirement_contract=contract))
        failing = _score("geometry.typed_requirements",
                         _side(requirement_contract={"measurements": {"max_stress": 400.0}}),
                         _side(requirement_contract=contract))
        self.assertEqual(passing["mean_requirement_pass"], 1.0)
        self.assertEqual(failing["mean_requirement_pass"], 0.0)

    def test_orthographic_reasoning_all_right_vs_all_wrong(self):
        truth = {"parameters": {"a": 1.0, "b": 2.0}}
        right = _score("vision.orthographic_reasoning",
                       _side(parameters={"a": 1.0, "b": 2.0}), _side(**truth))
        wrong = _score("vision.orthographic_reasoning",
                       _side(parameters={"a": 9.0, "b": 9.0}), _side(**truth))
        self.assertEqual(right["accuracy"], 1.0)
        self.assertEqual(wrong["accuracy"], 0.0)

    def test_constraint_satisfaction_all_met_vs_all_violated(self):
        met = _score("geometry.constraint_satisfaction",
                     _side(constraint_values={"g1": -1.0, "g2": 0.0}), _side())
        viol = _score("geometry.constraint_satisfaction",
                      _side(constraint_values={"g1": 1.0, "g2": 2.0}), _side())
        self.assertEqual(met["constraint_satisfaction"], 1.0)
        self.assertEqual(met["feasible"], 1.0)
        self.assertEqual(viol["constraint_satisfaction"], 0.0)
        self.assertEqual(viol["feasible"], 0.0)

    def test_choice_optimality_optimal_vs_worst(self):
        costs = {"choice": 0, "costs": [1.0, 2.0, 3.0]}
        opt = _score("retrieval.choice_optimality",
                     _side(option_choice={"choice": 0}),
                     _side(option_choice=costs))
        bad = _score("retrieval.choice_optimality",
                     _side(option_choice={"choice": 2}),
                     _side(option_choice=costs))
        self.assertEqual(opt["optimal"], 1.0)
        self.assertEqual(bad["optimal"], 0.0)

    def test_edit_operation_f1_identical_vs_disjoint(self):
        same = _score("sequence.edit_operation_f1",
                      _side(edit_ops=["extrude", "fillet"]),
                      _side(edit_ops=["extrude", "fillet"]))
        disjoint = _score("sequence.edit_operation_f1",
                          _side(edit_ops=["extrude"]),
                          _side(edit_ops=["mirror"]))
        self.assertEqual(same["f1"], 1.0)
        self.assertEqual(disjoint["f1"], 0.0)

    def test_perceived_actual_gap_zero_when_calibrated(self):
        calibrated = _score("generative.perceived_actual_gap",
                            _side(feasibility=0.7), _side(feasibility=0.7))
        overclaim = _score("generative.perceived_actual_gap",
                           _side(feasibility=1.0), _side(feasibility=0.0))
        self.assertEqual(calibrated["gap"], 0.0)
        self.assertEqual(overclaim["gap"], 1.0)

    def test_identity_preservation_perfect_vs_leaky_edit(self):
        clean = _score("geometry.identity_preservation",
                       _side(edit_entities={"after": [1, 2, 3, 5], "modified": []}),
                       _side(edit_entities={"before": [1, 2, 3, 4],
                                            "intended_region": [4, 5]}))
        leaky = _score("geometry.identity_preservation",
                       _side(edit_entities={"after": [1, 5, 6], "modified": []}),
                       _side(edit_entities={"before": [1, 2, 3, 4],
                                            "intended_region": [4]}))
        self.assertEqual(clean["preservation"], 1.0)
        self.assertLess(leaky["preservation"], clean["preservation"])

    def test_determinism_of_a_sample_of_new_metrics(self):
        cases = {
            "vision.quantity_alignment": (_side(counts=[1.0, 2.0]),
                                          _side(counts=[1.0, 3.0])),
            "retrieval.designqa": (_side(answer_text="a b c"),
                                   _side(answer_text="a b d")),
            "generative.judge_human_agreement": (_side(scores=[1.0, 2.0, 3.0]),
                                                 _side(scores=[1.0, 3.0, 2.0])),
        }
        for name, (pred, gold) in cases.items():
            self.assertEqual(_score(name, pred, gold), _score(name, pred, gold), name)


class RivalTest(unittest.TestCase):
    """New rivals join the right families and cannot be pooled in a suite."""

    def test_new_chamfer_rivals_registered(self):
        family = dict(bench.RIVAL_FAMILIES)["chamfer_distance_3d"]
        self.assertIn("geometry.scaled_chamfer_reward", family)
        self.assertIn("geometry.compiler_chamfer", family)
        self.assertIn("geometry.design_distance_curve",
                      dict(bench.RIVAL_FAMILIES)["chamfer_distance_2d"])

    def test_clustering_agreement_family_is_enforced(self):
        family = dict(bench.RIVAL_FAMILIES)["clustering_agreement"]
        self.assertEqual(sorted(family),
                         ["retrieval.clustering_external", "retrieval.pairwise_edge"])
        conflicts = bench._rival_conflicts(list(family))
        self.assertTrue(conflicts)

    def test_running_two_chamfer_rivals_together_is_refused(self):
        s = {"id": "x", "pred": _side(points=[(0.0, 0.0, 0.0)] * 4),
             "gold": _side(points=[(0.0, 0.0, 0.0)] * 4)}
        scaled = bench.metric("geometry.scaled_chamfer_reward")
        compiler = bench.metric("geometry.compiler_chamfer")
        with self.assertRaises(bench.RivalBlendError):
            bench.run_suite("geometry_smoke", [s],
                            extra_metrics=[scaled, compiler])

    def test_no_suite_blends_a_new_rival_pair(self):
        families = bench.rivals()
        for suite_name in bench.suites():
            chosen = set(bench.suite(suite_name).metric_names)
            for family, members in families.items():
                self.assertLessEqual(len(chosen.intersection(members)), 1,
                                     f"suite {suite_name!r} blends {family!r}")


if __name__ == "__main__":
    unittest.main()
