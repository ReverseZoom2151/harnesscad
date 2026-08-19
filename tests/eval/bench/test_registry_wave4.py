"""Scoring tests for the fourth adapter wave of the bench registry.

Same contract as ``test_registry_wave3.py``: every metric this wave bound is
materialised, leaves ``unadapted()``, and is actually *scored* on a hand-built
pred/gold pair whose expected number is worked out by hand from the underlying
module's published definition. A metric bound but never scored is not wired, so
each case below asserts a real computed value, not merely "a float came back".

The wave also pins the two refusals that matter: the FEA adapter must refuse a
row upstream flags ``is_oracle=False`` (it answers a regression question, not a
correctness one), and the PPA adapter must refuse an empty sketch (where the
paper's accuracies degenerate to a vacuous 1.0 and its Chamfer term is +inf).
"""

from __future__ import annotations

import unittest

from harnesscad.eval.bench import registry as bench


#: Every metric the fourth wave added, by name.
WAVE_FOUR = (
    "geometry.manufacturability",
    "geometry.functionality",
    "geometry.edit_preservation",
    "geometry.fea_oracle",
    "geometry.rubric_deductions",
    "sketch.autoconstraint_f1",
    "sketch.primitive_prediction",
    "sequence.structure_consistency",
    "vision.scene_reconstruction",
)


def _side(**kw):
    """One side (pred or gold) of a sample: only the keys a metric reads."""
    return dict(kw)


def _score(name, pred, gold):
    return bench.metric(name).score(pred, gold)


class DiscoveryTest(unittest.TestCase):
    def test_every_new_metric_is_discovered_and_bound(self):
        known = {m.name: m for m in bench.metrics()}
        for name in WAVE_FOUR:
            self.assertIn(name, known, f"{name} was not discovered")
            self.assertNotIn(known[name].dotted, bench.unadapted(),
                             f"{name}'s module is still listed unadapted")
            self.assertIn(known[name].kind, bench.KINDS)
            for key in known[name].inputs:
                self.assertIn(key, bench.INPUT_KINDS, f"{name} needs unknown {key}")

    def test_the_ledger_is_complete_in_both_directions(self):
        """Unadapted is a ledger, not a silence: every entry carries a reason."""
        unadapted = set(bench.unadapted())
        stated = set(bench.reasons())
        self.assertTrue(unadapted)
        self.assertEqual(sorted(unadapted - stated), [],
                         "unadapted modules with no recorded reason")
        self.assertEqual(sorted(stated - unadapted), [],
                         "reasons naming modules that are in fact adapted")
        for _dotted, reason in bench.UNADAPTED_REASONS:
            self.assertTrue(reason.strip())


class MusePillarScoringTest(unittest.TestCase):
    """The two remaining MUSE design-intent pillars (Dong et al., Table 8)."""

    GOOD_MANUFACTURING = {
        "material": "Aluminum", "process": "CNC Milling",
        "components": [{"name": "plate", "wall_thickness": 3.0,
                        "bbox": [100.0, 50.0, 5.0]}],
        "clearances": [["slot", 0.2]],
    }
    #: PLA cannot be CNC-milled (Table 5); the wall is under the 1.0 mm process
    #: minimum and the 0.001 mm seam is below the 0.05 mm tolerance floor.
    BAD_MANUFACTURING = {
        "material": "PLA", "process": "CNC Milling",
        "components": [{"name": "plate", "wall_thickness": 0.2}],
        "clearances": [["slot", 0.001]],
    }

    GOOD_FUNCTIONAL = {
        "structures": ["leg", "seat", "armrest"],
        "must_have": ["leg", "seat"], "nice_to_have": ["armrest"],
        "parameters": {"height": 40.0}, "parameter_ranges": {"height": [30.0, 50.0]},
        "ground_contacts": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        "center_of_mass": [1.0, 1.0],
        "load_bearing_members": [{"name": "leg", "thickness": 2.0,
                                  "connected": True}],
    }
    #: Missing a must-have structure, a parameter outside Omega, two collinear
    #: contacts, a centre of mass off the support polygon and a thin broken leg.
    BAD_FUNCTIONAL = {
        "structures": ["seat"],
        "must_have": ["leg", "seat"], "nice_to_have": ["armrest"],
        "parameters": {"height": 99.0}, "parameter_ranges": {"height": [30.0, 50.0]},
        "ground_contacts": [[0.0, 0.0], [1.0, 0.0]],
        "center_of_mass": [5.0, 5.0],
        "load_bearing_members": [{"name": "leg", "thickness": 0.1,
                                  "connected": False}],
    }

    def test_manufacturability_both_sub_criteria_pass_and_fail(self):
        good = _score("geometry.manufacturability",
                      _side(manufacturing_design=self.GOOD_MANUFACTURING),
                      _side(manufacturing_design=self.GOOD_MANUFACTURING))
        self.assertEqual(good, {"manufacturable": 1.0, "well_toleranced": 1.0,
                                "average": 1.0})
        bad = _score("geometry.manufacturability",
                     _side(manufacturing_design=self.BAD_MANUFACTURING),
                     _side(manufacturing_design=self.BAD_MANUFACTURING))
        self.assertEqual(bad, {"manufacturable": 0.0, "well_toleranced": 0.0,
                               "average": 0.0})

    def test_manufacturability_splits_when_only_the_tolerance_fails(self):
        design = dict(self.GOOD_MANUFACTURING)
        # 0.10 mm x the 10x gap factor = a 1.0 mm admissible ceiling for CNC.
        design["clearances"] = [["slot", 5.0]]
        mixed = _score("geometry.manufacturability",
                       _side(manufacturing_design=design),
                       _side(manufacturing_design=design))
        self.assertEqual(mixed["manufacturable"], 1.0)
        self.assertEqual(mixed["well_toleranced"], 0.0)
        self.assertEqual(mixed["average"], 0.5)

    def test_functionality_both_sub_criteria_pass_and_fail(self):
        good = _score("geometry.functionality",
                      _side(functional_design=self.GOOD_FUNCTIONAL),
                      _side(functional_design=self.GOOD_FUNCTIONAL))
        self.assertEqual(good, {"functional": 1.0, "robust": 1.0, "average": 1.0})
        bad = _score("geometry.functionality",
                     _side(functional_design=self.BAD_FUNCTIONAL),
                     _side(functional_design=self.BAD_FUNCTIONAL))
        self.assertEqual(bad, {"functional": 0.0, "robust": 0.0, "average": 0.0})

    def test_functionality_splits_when_only_robustness_fails(self):
        design = dict(self.GOOD_FUNCTIONAL)
        design["center_of_mass"] = [9.0, 9.0]   # outside the support polygon
        mixed = _score("geometry.functionality",
                       _side(functional_design=design),
                       _side(functional_design=design))
        self.assertEqual(mixed["functional"], 1.0)
        self.assertEqual(mixed["robust"], 0.0)
        self.assertEqual(mixed["average"], 0.5)


class EditPreservationTest(unittest.TestCase):
    """VoxHammer unedited-region preservation, latents keyed by voxel."""

    COORDS = [[0, 0, 0], [1, 0, 0]]
    SOURCE = [[1.0, 0.0], [0.0, 1.0]]

    def _gold(self, keep):
        return _side(latent_voxels={"coords": self.COORDS, "values": self.SOURCE,
                                    "keep": keep})

    def test_identical_latents_score_zero_error(self):
        out = _score("geometry.edit_preservation",
                     _side(latent_voxels={"coords": self.COORDS,
                                          "values": self.SOURCE}),
                     self._gold(self.COORDS))
        self.assertEqual(out, {"preservation_mse": 0.0,
                               "preservation_max_error": 0.0})

    def test_one_drifted_channel_gives_the_hand_computed_mse(self):
        # One of four scalar slots moved by 3.0: MSE = 9/4, worst voxel L2 = 3.
        out = _score("geometry.edit_preservation",
                     _side(latent_voxels={"coords": self.COORDS,
                                          "values": [[1.0, 3.0], [0.0, 1.0]]}),
                     self._gold(self.COORDS))
        self.assertEqual(out["preservation_mse"], 2.25)
        self.assertEqual(out["preservation_max_error"], 3.0)

    def test_the_keep_mask_excludes_the_edited_voxel(self):
        # The drift sits on voxel (0,0,0); masking to (1,0,0) must ignore it.
        out = _score("geometry.edit_preservation",
                     _side(latent_voxels={"coords": self.COORDS,
                                          "values": [[1.0, 3.0], [0.0, 1.0]]}),
                     self._gold([[1, 0, 0]]))
        self.assertEqual(out["preservation_mse"], 0.0)

    def test_an_empty_keep_intersection_is_refused_not_scored(self):
        with self.assertRaises(ValueError):
            _score("geometry.edit_preservation",
                   _side(latent_voxels={"coords": self.COORDS,
                                        "values": self.SOURCE}),
                   self._gold([[9, 9, 9]]))


class FeaOracleTest(unittest.TestCase):
    """Grading a solver scalar against the closed-form answer key."""

    def _oracle(self):
        from harnesscad.eval.bench import analytic_fea
        return analytic_fea.case("tension_rod", "max_displacement")

    def test_the_exact_closed_form_answer_is_inside_the_band(self):
        oracle = self._oracle()
        out = _score("geometry.fea_oracle",
                     _side(fea_answer={"computed": oracle.value}),
                     _side(fea_answer={"case_id": "tension_rod",
                                       "metric": "max_displacement"}))
        self.assertEqual(out["within_tolerance"], 1.0)
        self.assertEqual(out["relative_deviation"], 0.0)
        self.assertEqual(out["gating"], 1.0)

    def test_a_fifty_percent_error_is_outside_a_ten_percent_band(self):
        oracle = self._oracle()
        self.assertEqual(oracle.tolerance_percent, 10.0)
        out = _score("geometry.fea_oracle",
                     _side(fea_answer={"computed": oracle.value * 1.5}),
                     _side(fea_answer={"case_id": "tension_rod",
                                       "metric": "max_displacement"}))
        self.assertEqual(out["within_tolerance"], 0.0)
        self.assertAlmostEqual(out["relative_deviation"], 0.5, places=12)

    def test_a_five_percent_error_stays_inside_a_ten_percent_band(self):
        oracle = self._oracle()
        out = _score("geometry.fea_oracle",
                     _side(fea_answer={"computed": oracle.value * 1.05}),
                     _side(fea_answer={"case_id": "tension_rod",
                                       "metric": "max_displacement"}))
        self.assertEqual(out["within_tolerance"], 1.0)

    def test_a_non_oracle_row_is_refused(self):
        """Upstream flags these two rows as regression goldens, not truth."""
        with self.assertRaises(ValueError):
            _score("geometry.fea_oracle",
                   _side(fea_answer={"computed": 1.0}),
                   _side(fea_answer={"case_id": "fixed_fixed_udl",
                                     "metric": "max_displacement"}))


class RubricDeductionsTest(unittest.TestCase):
    """Deterministic rubric scoring: measured context vs a weighted rubric."""

    ITEMS = [
        {"item_id": "i1", "primary_category": "geometry", "max_points": 1.0,
         "normalized_weight": 0.5,
         "deduction_rules": [{"rule_code": "global_geometry_invalid",
                              "deduction_ratio": 1.0}]},
        {"item_id": "i2", "primary_category": "assembly", "max_points": 1.0,
         "normalized_weight": 0.5,
         "deduction_rules": [{"rule_code": "component_count_mismatch",
                              "deduction_ratio": 1.0}]},
    ]
    CLEAN = {"sandbox_ok": True, "code_valid": True, "geometry_valid": True,
             "watertight": True, "manifold": True, "self_intersection_free": True,
             "normal_consistency": True, "volume_valid": True, "bbox_valid": True,
             "occt_valid": True, "bbox": (0.0, 0.0, 0.0, 10.0, 10.0, 10.0),
             "solid_count": 3}

    def _gold(self):
        return _side(rubric_case={"items": self.ITEMS, "expected_components": 3})

    def test_a_clean_context_triggers_no_deduction(self):
        out = _score("geometry.rubric_deductions",
                     _side(rubric_case={"context": self.CLEAN}), self._gold())
        self.assertEqual(out["weighted_score"], 1.0)
        self.assertEqual(out["mean_item_score"], 1.0)
        self.assertEqual(out["n_deductions"], 0.0)

    def test_both_rules_firing_zeroes_the_weighted_score(self):
        broken = dict(self.CLEAN, geometry_valid=False, solid_count=1)
        out = _score("geometry.rubric_deductions",
                     _side(rubric_case={"context": broken}), self._gold())
        self.assertEqual(out["weighted_score"], 0.0)
        self.assertEqual(out["n_deductions"], 2.0)

    def test_one_rule_firing_leaves_exactly_its_weight_behind(self):
        # Only the component count is wrong: the 0.5-weighted geometry item stands.
        miscounted = dict(self.CLEAN, solid_count=5)
        out = _score("geometry.rubric_deductions",
                     _side(rubric_case={"context": miscounted}), self._gold())
        self.assertEqual(out["weighted_score"], 0.5)
        self.assertEqual(out["mean_item_score"], 0.5)
        self.assertEqual(out["n_deductions"], 1.0)


class AutoconstraintF1Test(unittest.TestCase):
    """Vitruvion/SketchGraphs auto-constraint edge F1 over EdgeOp rows."""

    GOLD = [["coincident", [1, 2]], ["parallel", [3, 4]]]

    def test_identical_constraint_sets_score_one(self):
        out = _score("sketch.autoconstraint_f1",
                     _side(constraint_ops=self.GOLD),
                     _side(constraint_ops=self.GOLD))
        self.assertEqual(out, {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                               "num_correct": 2.0})

    def test_one_of_two_edges_right_gives_a_half_f1(self):
        out = _score("sketch.autoconstraint_f1",
                     _side(constraint_ops=[["coincident", [1, 2]],
                                           ["tangent", [5, 6]]]),
                     _side(constraint_ops=self.GOLD))
        self.assertEqual(out["precision"], 0.5)
        self.assertEqual(out["recall"], 0.5)
        self.assertEqual(out["f1"], 0.5)
        self.assertEqual(out["num_correct"], 1.0)

    def test_a_wrong_label_on_the_right_references_is_a_miss(self):
        out = _score("sketch.autoconstraint_f1",
                     _side(constraint_ops=[["perpendicular", [1, 2]],
                                           ["perpendicular", [3, 4]]]),
                     _side(constraint_ops=self.GOLD))
        self.assertEqual(out["f1"], 0.0)
        self.assertEqual(out["num_correct"], 0.0)


class PrimitivePredictionTest(unittest.TestCase):
    """PPA primitive-prediction accuracies (Wang et al., Eq. 21-23)."""

    #: Coordinates are already in [0, 1] -- the module's stated precondition.
    GOLD = [["line", 1, [0.1, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0]],
            ["circle", 1, [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.2]]]

    def test_identical_sketches_score_one_on_every_axis(self):
        out = _score("sketch.primitive_prediction",
                     _side(sketch_primitives=self.GOLD),
                     _side(sketch_primitives=self.GOLD))
        self.assertEqual(out["acc_ptype"], 1.0)
        self.assertEqual(out["acc_flag"], 1.0)
        self.assertEqual(out["acc_ppar"], 1.0)
        self.assertEqual(out["chamfer"], 0.0)
        self.assertEqual(out["matched"], 2.0)

    def test_a_flipped_boolean_flag_halves_only_the_flag_accuracy(self):
        flipped = [self.GOLD[0],
                   ["circle", 0, [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.2]]]
        out = _score("sketch.primitive_prediction",
                     _side(sketch_primitives=flipped),
                     _side(sketch_primitives=self.GOLD))
        self.assertEqual(out["acc_flag"], 0.5)
        self.assertEqual(out["acc_ptype"], 1.0)
        self.assertEqual(out["acc_ppar"], 1.0)

    def test_a_displaced_primitive_costs_parameter_accuracy_and_chamfer(self):
        moved = [["line", 1, [0.1, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0]], self.GOLD[1]]
        out = _score("sketch.primitive_prediction",
                     _side(sketch_primitives=moved),
                     _side(sketch_primitives=self.GOLD))
        self.assertEqual(out["acc_ptype"], 1.0)
        self.assertEqual(out["acc_ppar"], 0.5)
        self.assertGreater(out["chamfer"], 0.0)

    def test_an_empty_sketch_is_refused_rather_than_scored_vacuously(self):
        with self.assertRaises(ValueError):
            _score("sketch.primitive_prediction",
                   _side(sketch_primitives=[]),
                   _side(sketch_primitives=self.GOLD))


class StructureConsistencyTest(unittest.TestCase):
    """GeoFusion structure F1 over the parsed sketch/extrusion tree."""

    CURVE = {"kind": "line", "params": [11, 11, 20, 11]}

    def _solid(self, n_faces):
        faces = [{"loops": [{"curves": [self.CURVE]}]} for _ in range(n_faces)]
        return {"pairs": [{"sketch": {"faces": faces},
                           "extrusion": {"params": [0] * 10}}]}

    def test_identical_trees_match_exactly(self):
        out = _score("sequence.structure_consistency",
                     _side(solid_tree=self._solid(1)),
                     _side(solid_tree=self._solid(1)))
        self.assertEqual(out, {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                               "structure_match": 1.0})

    def test_a_spurious_face_costs_precision_but_not_recall(self):
        # 4 gold paths, 7 predicted, 4 shared -> P = 4/7, R = 1.
        out = _score("sequence.structure_consistency",
                     _side(solid_tree=self._solid(2)),
                     _side(solid_tree=self._solid(1)))
        self.assertAlmostEqual(out["precision"], 4.0 / 7.0, places=12)
        self.assertEqual(out["recall"], 1.0)
        self.assertAlmostEqual(out["f1"], 8.0 / 11.0, places=12)
        self.assertEqual(out["structure_match"], 0.0)

    def test_a_missing_face_costs_recall_but_not_precision(self):
        out = _score("sequence.structure_consistency",
                     _side(solid_tree=self._solid(1)),
                     _side(solid_tree=self._solid(2)))
        self.assertEqual(out["precision"], 1.0)
        self.assertAlmostEqual(out["recall"], 4.0 / 7.0, places=12)


class SceneReconstructionTest(unittest.TestCase):
    """Sketch2CAD scene-graph reconstruction: pose, class and pose errors."""

    GOLD_OBJECTS = [{"shape": "cube", "position": [0.0, 0.0, 0.0],
                     "rotation": [0.0, 0.0], "size": [1.0, 1.0, 1.0]}]

    def _gold(self, pose=2):
        return _side(scene={"pose": pose, "objects": self.GOLD_OBJECTS})

    def test_a_perfect_scene_scores_one_with_zero_error(self):
        out = _score("vision.scene_reconstruction",
                     _side(scene={"pose": 2, "objects": self.GOLD_OBJECTS}),
                     self._gold())
        self.assertEqual(out["pose_accuracy"], 1.0)
        self.assertEqual(out["classification_f1"], 1.0)
        self.assertEqual(out["position_error_x"], 0.0)
        self.assertEqual(out["rotation_error_yaw"], 0.0)
        self.assertEqual(out["matched"], 1.0)

    def test_a_wrong_class_and_pose_gives_the_hand_computed_errors(self):
        wrong = [{"shape": "cylinder", "position": [2.0, 0.0, 0.0],
                  "rotation": [90.0, 0.0], "size": [1.0, 3.0, 1.0]}]
        out = _score("vision.scene_reconstruction",
                     _side(scene={"pose": 1, "objects": wrong}), self._gold())
        self.assertEqual(out["pose_accuracy"], 0.0)
        self.assertEqual(out["classification_f1"], 0.0)
        self.assertEqual(out["position_error_x"], 2.0)
        self.assertEqual(out["position_error_y"], 0.0)
        self.assertEqual(out["size_error_y"], 2.0)
        self.assertEqual(out["rotation_error_yaw"], 90.0)
        self.assertEqual(out["rotation_error_pitch"], 0.0)

    def test_rotation_error_wraps_the_short_way_round(self):
        spun = [{"shape": "cube", "position": [0.0, 0.0, 0.0],
                 "rotation": [350.0, 0.0], "size": [1.0, 1.0, 1.0]}]
        out = _score("vision.scene_reconstruction",
                     _side(scene={"pose": 2, "objects": spun}), self._gold())
        self.assertEqual(out["rotation_error_yaw"], 10.0)
        self.assertEqual(out["classification_f1"], 1.0)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
