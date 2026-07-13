"""Tests for CAD Score composition and editing no-op renormalization."""
import unittest

from harnesscad.eval.bench.protocols.cad_score import (
    EDIT_HEADROOM_FLOOR,
    EDITING_AXIS_WEIGHTS,
    GENERATION_AXIS_WEIGHTS,
    AxisScores,
    EditHeadroomError,
    StaleBaselineError,
    build_edit_baseline,
    cad_score,
    check_baseline_fresh,
    check_edit_headroom,
    editing_score,
    generation_score,
    noop_ceiling,
    renormalize_shape,
    shape_similarity,
    weighted_axis_mean,
)


class TestShapeSimilarity(unittest.TestCase):
    def test_mean_of_sub_metrics(self):
        self.assertAlmostEqual(shape_similarity(0.9, 0.7), 0.8)

    def test_missing_sub_metric_drops_out(self):
        self.assertAlmostEqual(shape_similarity(0.9, None), 0.9)

    def test_no_sub_metric_is_none_not_zero(self):
        self.assertIsNone(shape_similarity(None, None))


class TestComposition(unittest.TestCase):
    def test_generation_weights(self):
        axes = AxisScores(shape=0.89, interface=0.0, topology=1.0)
        # Worked example from the metric doc: 0.4*0.89 + 0.4*0 + 0.2*1 = 0.556
        self.assertAlmostEqual(generation_score(axes), 0.556, places=6)

    def test_validity_gate_zeroes(self):
        axes = AxisScores(shape=1.0, interface=1.0, topology=1.0)
        self.assertEqual(cad_score(axes, is_valid=False), 0.0)

    def test_missing_axis_renormalizes_weights(self):
        # No authored sub-volumes: interface drops out, its 0.4 redistributes
        # over shape (0.4) and topology (0.2) rather than diluting the mean.
        axes = AxisScores(shape=0.5, interface=None, topology=1.0)
        expected = (0.4 * 0.5 + 0.2 * 1.0) / 0.6
        self.assertAlmostEqual(generation_score(axes), expected, places=9)

    def test_no_axes_scores_zero(self):
        self.assertEqual(cad_score(AxisScores(), is_valid=True), 0.0)

    def test_equal_weighting_when_weights_none(self):
        axes = AxisScores(shape=0.0, interface=1.0, topology=0.5)
        self.assertAlmostEqual(weighted_axis_mean(axes, None), 0.5, places=9)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(GENERATION_AXIS_WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(sum(EDITING_AXIS_WEIGHTS.values()), 1.0)


class TestEditRenormalization(unittest.TestCase):
    def test_noop_maps_to_zero(self):
        self.assertEqual(renormalize_shape(0.9, 0.9), 0.0)

    def test_perfect_maps_to_one(self):
        self.assertAlmostEqual(renormalize_shape(1.0, 0.9), 1.0)

    def test_worse_than_noop_floors_at_zero(self):
        self.assertEqual(renormalize_shape(0.5, 0.9), 0.0)

    def test_halfway_through_headroom(self):
        self.assertAlmostEqual(renormalize_shape(0.95, 0.9), 0.5, places=9)

    def test_zero_headroom_is_defensive(self):
        self.assertEqual(renormalize_shape(1.0, 1.0), 0.0)

    def test_noop_caps_at_ceiling(self):
        # No-op: shape == baseline, topology and interface untouched (perfect).
        axes = AxisScores(shape=0.92, interface=1.0, topology=1.0)
        score = editing_score(axes, baseline_shape=0.92)
        self.assertAlmostEqual(score, noop_ceiling(), places=9)
        self.assertAlmostEqual(score, 0.4, places=9)

    def test_real_edit_clears_the_noop(self):
        axes = AxisScores(shape=0.98, interface=1.0, topology=1.0)
        self.assertGreater(editing_score(axes, baseline_shape=0.92), noop_ceiling())

    def test_broken_topology_still_penalized_raw(self):
        axes = AxisScores(shape=1.0, interface=1.0, topology=0.0)
        # topology stays raw: 0.6*1 + 0.3*1 + 0.1*0 = 0.9
        self.assertAlmostEqual(editing_score(axes, baseline_shape=0.9), 0.9, places=9)

    def test_invalid_editing_candidate_scores_zero(self):
        axes = AxisScores(shape=1.0, interface=1.0, topology=1.0)
        self.assertEqual(editing_score(axes, baseline_shape=0.9, is_valid=False), 0.0)


class TestBaselineGuards(unittest.TestCase):
    def test_headroom_ok(self):
        self.assertAlmostEqual(check_edit_headroom(0.9), 0.1, places=9)

    def test_headroom_floor_rejects_unscorable_fixture(self):
        with self.assertRaises(EditHeadroomError):
            check_edit_headroom(1.0 - EDIT_HEADROOM_FLOOR / 2)

    def test_build_baseline_record(self):
        record = build_edit_baseline(
            baseline_shape=0.71,
            version="1.2.0",
            surface_distance_f1=0.68,
            volume_iou=0.74,
            alignment_rmse=0.012345,
        )
        self.assertAlmostEqual(record["shape_similarity_score"], 0.71)
        self.assertAlmostEqual(record["headroom"], 0.29, places=6)
        self.assertEqual(record["alignment_rmse"], 0.0123)
        self.assertEqual(record["metric_version"], "1.2.0")

    def test_fresh_baseline_passes(self):
        record = build_edit_baseline(baseline_shape=0.5, version="1.2.0")
        check_baseline_fresh(record, "1.2.0", fixture="201")

    def test_stale_baseline_raises(self):
        record = build_edit_baseline(baseline_shape=0.5, version="1.1.0")
        with self.assertRaises(StaleBaselineError):
            check_baseline_fresh(record, "1.2.0", fixture="201")


if __name__ == "__main__":
    unittest.main()
