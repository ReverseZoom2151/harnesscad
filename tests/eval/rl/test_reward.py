"""Tests for the composite, target-comparing RL reward.

Real computed values only: a known cube against itself, a known offset case, a
known half-scale case, and an unbuildable case gated to exactly 0.
"""

from __future__ import annotations

import math
import unittest

from harnesscad.core.cisp.protocol import ApplyOpsResult
from harnesscad.core.diagnostics import Diagnostic, Severity
from harnesscad.eval.rl.reward import (
    RLCAD_WEIGHTS,
    RewardBreakdown,
    ShapeSample,
    apply_frame,
    composite_reward,
    composite_reward_value,
    exec_gate,
    iou_term,
    mmd_term,
    nc_term,
    process_score,
    shared_unit_cube_frame,
)


def box_sample(edge=10.0, offset=(0.0, 0.0, 0.0), per_face=4):
    """Surface samples + outward normals for an axis-aligned box."""
    step = [(-0.5 + (i + 0.5) / per_face) for i in range(per_face)]
    pts, nrm = [], []
    for axis in range(3):
        for side in (-1.0, 1.0):
            for u in step:
                for v in step:
                    p = [0.0, 0.0, 0.0]
                    p[axis] = side * 0.5
                    p[(axis + 1) % 3] = u
                    p[(axis + 2) % 3] = v
                    n = [0.0, 0.0, 0.0]
                    n[axis] = side
                    pts.append(tuple(p[d] * edge + offset[d] for d in range(3)))
                    nrm.append(tuple(n))
    return ShapeSample.of(pts, nrm)


OK = ApplyOpsResult(True, 1, "digest")
OK_WARNED = ApplyOpsResult(
    True, 1, "digest", [Diagnostic(Severity.WARNING, "under-constrained", "m")])
REJECTED = ApplyOpsResult(False, 0, "digest", [], rejected={"op": "extrude"})
ERRORED = ApplyOpsResult(
    True, 1, "digest", [Diagnostic(Severity.ERROR, "empty-solid", "m")])

SMALL_EMD = {"emd_max_points": 24}


class TestShapeSample(unittest.TestCase):
    def test_of_freezes_points_and_normals(self):
        s = ShapeSample.of([[1, 2, 3]], [[0, 0, 1]])
        self.assertEqual(s.points, ((1.0, 2.0, 3.0),))
        self.assertEqual(s.normals, ((0.0, 0.0, 1.0),))
        self.assertEqual(s.oriented(), [((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))])

    def test_normals_must_be_parallel_to_points(self):
        with self.assertRaises(ValueError):
            ShapeSample.of([(0, 0, 0), (1, 1, 1)], [(0, 0, 1)])

    def test_an_empty_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            ShapeSample.of([])

    def test_without_normals_oriented_is_none(self):
        self.assertIsNone(ShapeSample.of([(0, 0, 0)]).oriented())


class TestExecGate(unittest.TestCase):
    def test_a_clean_result_opens_the_gate(self):
        self.assertEqual(exec_gate(OK), 1.0)

    def test_a_warning_does_not_close_the_gate(self):
        self.assertEqual(exec_gate(OK_WARNED), 1.0)

    def test_a_rejected_batch_closes_the_gate(self):
        self.assertEqual(exec_gate(REJECTED), 0.0)

    def test_an_error_diagnostic_closes_the_gate_even_when_ok(self):
        self.assertEqual(exec_gate(ERRORED), 0.0)

    def test_the_gate_is_zero_not_negative(self):
        # reward_from_apply returns -1.0 on failure; a MULTIPLICATIVE gate must
        # be 0, or it would flip the sign of the geometry terms.
        self.assertEqual(exec_gate(REJECTED), 0.0)
        self.assertNotEqual(exec_gate(REJECTED), -1.0)

    def test_serialised_diagnostics_gate_identically(self):
        class Bag(object):
            ok = True
            diagnostics = [{"severity": "error", "code": "x", "message": "y"}]

        self.assertEqual(exec_gate(Bag()), 0.0)


class TestProcessScore(unittest.TestCase):
    def test_clean_is_one(self):
        self.assertEqual(process_score(OK), 1.0)

    def test_each_warning_shaves_a_tenth(self):
        self.assertAlmostEqual(process_score(OK_WARNED), 0.9)

    def test_a_failed_result_is_zero_not_minus_one(self):
        self.assertEqual(process_score(REJECTED), 0.0)


class TestFrame(unittest.TestCase):
    def test_the_target_frame_maps_the_target_into_the_unit_cube(self):
        target = box_sample(edge=10.0)
        frame = shared_unit_cube_frame(target.points)
        mapped = apply_frame(target.points, frame)
        self.assertAlmostEqual(max(p[0] for p in mapped), 0.5)
        self.assertAlmostEqual(min(p[0] for p in mapped), -0.5)

    def test_the_shared_frame_preserves_relative_offset(self):
        # This is the whole reason per-cloud normalisation is refused: a
        # candidate offset by half an edge must still LOOK offset after mapping.
        target = box_sample(edge=10.0)
        moved = box_sample(edge=10.0, offset=(5.0, 0.0, 0.0))
        frame = shared_unit_cube_frame(target.points)
        mapped = apply_frame(moved.points, frame)
        self.assertAlmostEqual(max(p[0] for p in mapped), 1.0)


class TestTerms(unittest.TestCase):
    def test_a_cube_against_itself_is_iou_one(self):
        cube = box_sample()
        self.assertEqual(iou_term(cube, cube), 1.0)

    def test_disjoint_cubes_are_iou_zero(self):
        a = box_sample(edge=10.0)
        b = box_sample(edge=10.0, offset=(100.0, 0.0, 0.0))
        self.assertEqual(iou_term(a, b), 0.0)

    def test_iou_accepts_precomputed_voxels(self):
        a = ShapeSample.of([(0.0, 0.0, 0.0)], voxels=[(0, 0, 0), (1, 0, 0)])
        b = ShapeSample.of([(0.0, 0.0, 0.0)], voxels=[(0, 0, 0)])
        self.assertAlmostEqual(iou_term(a, b), 0.5)

    def test_a_cube_against_itself_is_mmd_one(self):
        cube = box_sample()
        r_mmd, cd, emd, _notes = mmd_term(cube, cube, **SMALL_EMD)
        self.assertAlmostEqual(cd, 0.0)
        self.assertAlmostEqual(emd, 0.0)
        self.assertAlmostEqual(r_mmd, 1.0)

    def test_r_mmd_is_the_documented_affine_map_of_the_distances(self):
        target = box_sample(edge=10.0)
        moved = box_sample(edge=10.0, offset=(2.0, 0.0, 0.0))
        r_mmd, cd, emd, _notes = mmd_term(target, moved, **SMALL_EMD)
        self.assertAlmostEqual(
            r_mmd, 1.0 - ((cd + emd) / 2.0) / math.sqrt(3.0))

    def test_a_translated_cube_loses_mmd_reward(self):
        target = box_sample(edge=10.0)
        moved = box_sample(edge=10.0, offset=(2.0, 0.0, 0.0))
        r_mmd, _cd, _emd, _n = mmd_term(target, moved, **SMALL_EMD)
        self.assertLess(r_mmd, 1.0)
        self.assertGreater(r_mmd, 0.0)

    def test_subsampling_is_reported_never_silent(self):
        cube = box_sample()
        _r, _cd, _emd, notes = mmd_term(cube, cube, emd_max_points=8)
        self.assertTrue(any("stride-subsampled" in n for n in notes))

    def test_nc_of_a_cube_against_itself_is_one(self):
        cube = box_sample()
        self.assertAlmostEqual(nc_term(cube, cube), 1.0)

    def test_nc_is_none_without_normals(self):
        cube = box_sample()
        bare = ShapeSample.of(cube.points)
        self.assertIsNone(nc_term(bare, cube))


class TestCompositeReward(unittest.TestCase):
    def test_weights_are_the_published_rlcad_values(self):
        self.assertEqual(RLCAD_WEIGHTS, {"iou": 0.2, "mmd": 0.5, "nc": 0.3})

    def test_a_known_cube_against_itself_scores_exactly_one(self):
        cube = box_sample()
        r = composite_reward(OK, cube, cube, **SMALL_EMD)
        self.assertIsInstance(r, RewardBreakdown)
        self.assertAlmostEqual(r.iou, 1.0)
        self.assertAlmostEqual(r.mmd, 1.0)
        self.assertAlmostEqual(r.nc, 1.0)
        self.assertAlmostEqual(r.geometric, 1.0)
        self.assertAlmostEqual(r.total, 1.0)

    def test_the_total_is_the_weighted_sum_of_the_three_terms(self):
        target = box_sample(edge=10.0)
        cand = box_sample(edge=10.0, offset=(2.0, 0.0, 0.0))
        r = composite_reward(OK, target, cand, **SMALL_EMD)
        expected = 0.2 * r.iou + 0.5 * r.mmd + 0.3 * r.nc
        self.assertAlmostEqual(r.geometric, expected)
        self.assertAlmostEqual(r.total, expected)

    def test_a_known_offset_case_scores_strictly_between_zero_and_one(self):
        target = box_sample(edge=10.0)
        cand = box_sample(edge=10.0, offset=(3.0, 0.0, 0.0))
        r = composite_reward(OK, target, cand, **SMALL_EMD)
        self.assertGreater(r.total, 0.0)
        self.assertLess(r.total, 1.0)
        # The offset shows up in every channel, which is the point of the
        # composite: IoU alone would already be well under 1.
        self.assertLess(r.iou, 1.0)
        self.assertLess(r.mmd, 1.0)

    def test_iou_alone_is_insufficient_and_the_composite_says_so(self):
        # RLCAD's ablation in miniature: a candidate that fills the target
        # volume but whose surface orientation is wrong keeps IoU high while the
        # composite drops.  Same points, all normals rotated 90 degrees.
        target = box_sample(edge=10.0)
        tilted = ShapeSample.of(
            target.points, [(n[2], n[0], n[1]) for n in target.normals])
        r = composite_reward(OK, target, tilted, **SMALL_EMD)
        self.assertAlmostEqual(r.iou, 1.0)
        self.assertAlmostEqual(r.mmd, 1.0)
        self.assertLess(r.nc, 0.5)
        self.assertLess(r.total, 1.0)
        self.assertAlmostEqual(r.total, 0.2 + 0.5 + 0.3 * r.nc)

    def test_a_half_scale_cube_is_caught_by_the_world_space_iou(self):
        target = box_sample(edge=10.0)
        half = box_sample(edge=5.0)
        r = composite_reward(OK, target, half, **SMALL_EMD)
        self.assertLess(r.iou, 0.5)
        self.assertLess(r.total, 1.0)

    def test_an_unbuildable_candidate_is_gated_to_exactly_zero(self):
        cube = box_sample()
        r = composite_reward(REJECTED, cube, cube, **SMALL_EMD)
        self.assertEqual(r.total, 0.0)
        self.assertEqual(r.exec_gate, 0.0)
        # No partial credit may leak from a malformed sample: the geometry is
        # not even computed.
        self.assertIsNone(r.iou)
        self.assertIsNone(r.mmd)
        self.assertIsNone(r.nc)
        self.assertTrue(any("exec gate 0" in n for n in r.notes))

    def test_an_error_diagnostic_also_gates_to_zero(self):
        cube = box_sample()
        self.assertEqual(composite_reward(ERRORED, cube, cube).total, 0.0)

    def test_a_gated_result_needs_no_geometry_at_all(self):
        self.assertEqual(composite_reward(REJECTED, None, None).total, 0.0)

    def test_a_built_result_without_samples_is_an_error(self):
        with self.assertRaises(ValueError):
            composite_reward(OK, None, None)

    def test_the_process_term_is_off_by_default(self):
        cube = box_sample()
        clean = composite_reward(OK, cube, cube, **SMALL_EMD)
        warned = composite_reward(OK_WARNED, cube, cube, **SMALL_EMD)
        self.assertAlmostEqual(clean.total, warned.total)
        self.assertAlmostEqual(warned.process, 0.9)

    def test_the_process_term_applies_when_weighted_in(self):
        cube = box_sample()
        r = composite_reward(
            OK_WARNED, cube, cube, w_geom=0.8, w_process=0.2, **SMALL_EMD)
        self.assertAlmostEqual(r.total, 0.8 * 1.0 + 0.2 * 0.9)

    def test_missing_normals_redistribute_the_nc_weight_and_say_so(self):
        target = box_sample(edge=10.0)
        cand = ShapeSample.of(box_sample(edge=10.0, offset=(2.0, 0.0, 0.0)).points)
        bare_target = ShapeSample.of(target.points)
        r = composite_reward(OK, bare_target, cand, **SMALL_EMD)
        self.assertIsNone(r.nc)
        # 0.2 and 0.5 renormalised to sum to 1: weights become 2/7 and 5/7.
        self.assertAlmostEqual(
            r.geometric, (0.2 * r.iou + 0.5 * r.mmd) / 0.7)
        self.assertTrue(any(n.startswith("nc:") for n in r.notes))

    def test_composite_reward_value_returns_the_scalar(self):
        cube = box_sample()
        self.assertAlmostEqual(
            composite_reward_value(OK, cube, cube, **SMALL_EMD), 1.0)


class TestSaturationGuard(unittest.TestCase):
    def test_a_perfect_match_is_flagged_saturated(self):
        cube = box_sample()
        r = composite_reward(OK, cube, cube, **SMALL_EMD)
        self.assertTrue(r.saturated)
        self.assertAlmostEqual(r.headroom, 0.0)
        self.assertTrue(any("SATURATED" in n for n in r.notes))

    def test_a_middling_match_is_not_flagged(self):
        target = box_sample(edge=10.0)
        cand = box_sample(edge=10.0, offset=(3.0, 0.0, 0.0))
        r = composite_reward(OK, target, cand, **SMALL_EMD)
        self.assertFalse(r.saturated)
        self.assertGreater(r.headroom, 0.0)

    def test_dominance_is_reported_per_term(self):
        cube = box_sample()
        r = composite_reward(OK, cube, cube, **SMALL_EMD)
        self.assertEqual(r.dominant, "mmd")           # weight 0.5 is the largest
        self.assertAlmostEqual(r.dominance, 0.5)
        self.assertAlmostEqual(sum(r.contributions.values()), r.geometric)

    def test_a_single_live_term_is_flagged_as_dominating(self):
        cube = box_sample()
        r = composite_reward(
            OK, cube, cube, weights={"iou": 1.0, "mmd": 0.0, "nc": 0.0},
            **SMALL_EMD)
        self.assertAlmostEqual(r.dominance, 1.0)
        self.assertTrue(any("DOMINATED" in n for n in r.notes))

    def test_detail_sensitivity_amplifies_variation_near_the_top(self):
        target = box_sample(edge=10.0)
        near = box_sample(edge=10.0, offset=(0.05, 0.0, 0.0))
        flat = composite_reward(OK, target, near, **SMALL_EMD)
        sharp = composite_reward(
            OK, target, near, detail_sensitivity=4.0, **SMALL_EMD)
        # Same raw R_geom, larger distance from a perfect 1.0 after shaping.
        self.assertAlmostEqual(sharp.geometric, flat.geometric)
        self.assertLess(sharp.total, flat.total)
        self.assertAlmostEqual(sharp.total, flat.geometric ** 4.0)

    def test_detail_sensitivity_preserves_ordering(self):
        target = box_sample(edge=10.0)
        good = box_sample(edge=10.0, offset=(0.5, 0.0, 0.0))
        bad = box_sample(edge=10.0, offset=(3.0, 0.0, 0.0))
        g = composite_reward(OK, target, good, detail_sensitivity=4.0, **SMALL_EMD)
        b = composite_reward(OK, target, bad, detail_sensitivity=4.0, **SMALL_EMD)
        self.assertGreater(g.total, b.total)

    def test_saturation_is_measured_before_shaping(self):
        # Turning the sensitivity knob up must not hide the condition it exists
        # to counteract.
        cube = box_sample()
        r = composite_reward(OK, cube, cube, detail_sensitivity=8.0, **SMALL_EMD)
        self.assertTrue(r.saturated)

    def test_a_non_positive_sensitivity_is_rejected(self):
        cube = box_sample()
        with self.assertRaises(ValueError):
            composite_reward(OK, cube, cube, detail_sensitivity=0.0)

    def test_to_dict_exposes_every_component(self):
        cube = box_sample()
        d = composite_reward(OK, cube, cube, **SMALL_EMD).to_dict()
        self.assertEqual(
            sorted(d),
            ["contributions", "distances", "exec_gate", "geometric", "notes",
             "process", "saturation", "shaped", "terms", "total", "weights"])
        self.assertEqual(sorted(d["terms"]), ["iou", "mmd", "nc"])
        self.assertTrue(d["saturation"]["saturated"])


if __name__ == "__main__":
    unittest.main()
