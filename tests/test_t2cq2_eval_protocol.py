"""Tests for bench.t2cq2_eval_protocol."""

from __future__ import annotations

import unittest

from bench.t2cq2_eval_protocol import (
    CD_SCALE,
    aggregate,
    chamfer_distance,
    evaluate_corpus,
    evaluate_sample,
    f1_score,
    normalize_points,
    normalize_unit_cube,
    parse_match_results,
    voxelize,
    volumetric_iou,
)

CUBE = [
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (0.0, 2.0, 0.0),
    (0.0, 0.0, 2.0),
    (2.0, 2.0, 0.0),
    (2.0, 0.0, 2.0),
    (0.0, 2.0, 2.0),
    (2.0, 2.0, 2.0),
]


class NormalizeTest(unittest.TestCase):
    def test_centroid_centered_and_bbox_scaled(self):
        out = normalize_points(CUBE)
        # centroid at origin
        for axis in range(3):
            self.assertAlmostEqual(sum(p[axis] for p in out) / len(out), 0.0)
        # max extent becomes 1
        for axis in range(3):
            span = max(p[axis] for p in out) - min(p[axis] for p in out)
            self.assertAlmostEqual(span, 1.0)

    def test_scale_invariance(self):
        big = [(x * 7.0, y * 7.0, z * 7.0) for x, y, z in CUBE]
        for a, b in zip(normalize_points(CUBE), normalize_points(big)):
            for i in range(3):
                self.assertAlmostEqual(a[i], b[i])

    def test_degenerate_point_set(self):
        self.assertEqual(normalize_points([(1.0, 1.0, 1.0)]), [(0.0, 0.0, 0.0)])

    def test_empty(self):
        self.assertEqual(normalize_points([]), [])
        self.assertEqual(normalize_unit_cube([]), [])

    def test_unit_cube_lower_corner_at_origin(self):
        out = normalize_unit_cube([(1.0, 2.0, 3.0), (3.0, 4.0, 5.0)])
        self.assertEqual(out[0], (0.0, 0.0, 0.0))
        self.assertEqual(out[1], (1.0, 1.0, 1.0))


class ChamferTest(unittest.TestCase):
    def test_identical_sets_are_zero(self):
        self.assertAlmostEqual(chamfer_distance(CUBE, CUBE), 0.0)

    def test_known_value_sum_of_mean_squares(self):
        a = [(0.0, 0.0, 0.0)]
        b = [(0.0, 0.0, 3.0)]
        # 3^2 in each direction, summed
        self.assertAlmostEqual(chamfer_distance(a, b), 18.0)

    def test_symmetric(self):
        b = [(x + 0.5, y, z) for x, y, z in CUBE]
        self.assertAlmostEqual(
            chamfer_distance(CUBE, b), chamfer_distance(b, CUBE)
        )

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            chamfer_distance([], CUBE)


class F1Test(unittest.TestCase):
    def test_identical_sets_are_one(self):
        self.assertAlmostEqual(f1_score(CUBE, CUBE, threshold=0.01), 1.0)

    def test_disjoint_sets_are_zero(self):
        far = [(x + 100.0, y, z) for x, y, z in CUBE]
        self.assertAlmostEqual(f1_score(CUBE, far, threshold=0.02), 0.0)

    def test_threshold_is_strict_less_than(self):
        a = [(0.0, 0.0, 0.0)]
        b = [(0.02, 0.0, 0.0)]
        self.assertAlmostEqual(f1_score(a, b, threshold=0.02), 0.0)
        self.assertAlmostEqual(f1_score(a, b, threshold=0.021), 1.0)

    def test_partial_match_harmonic_mean(self):
        pred = [(0.0, 0.0, 0.0), (50.0, 0.0, 0.0)]
        gt = [(0.0, 0.0, 0.0)]
        # precision 0.5, recall 1.0 -> 2*0.5/1.5
        self.assertAlmostEqual(f1_score(pred, gt, threshold=0.02), 2.0 / 3.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            f1_score(CUBE, [])


class VoxelIouTest(unittest.TestCase):
    def test_identical_shapes_iou_one(self):
        self.assertAlmostEqual(volumetric_iou(CUBE, CUBE, 0.1), 1.0)

    def test_iou_invariant_to_translation_and_scale(self):
        moved = [(x * 3.0 + 9.0, y * 3.0, z * 3.0 - 4.0) for x, y, z in CUBE]
        self.assertAlmostEqual(volumetric_iou(CUBE, moved, 0.1), 1.0)

    def test_partial_overlap(self):
        a = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        b = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
        iou = volumetric_iou(a, b, 0.25)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_empty_union_is_one(self):
        self.assertAlmostEqual(volumetric_iou([], [], 0.02), 1.0)

    def test_voxelize_indices(self):
        self.assertEqual(
            voxelize([(0.0, 0.0, 0.0), (0.05, 0.0, 0.0)], 0.02),
            frozenset({(0, 0, 0), (2, 0, 0)}),
        )

    def test_voxel_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            voxelize(CUBE, 0.0)


class JudgeGateTest(unittest.TestCase):
    def test_parses_yes_lines_only(self):
        lines = [
            "00010001: Match: Yes",
            "00010002: Match: No",
            "  00010003: Match: Yes  ",
            "",
        ]
        self.assertEqual(
            parse_match_results(lines), ("00010001", "00010003")
        )


class CorpusTest(unittest.TestCase):
    def test_evaluate_sample_scale_invariant(self):
        pred = [(x * 5.0, y * 5.0, z * 5.0) for x, y, z in CUBE]
        m = evaluate_sample("u1", pred, CUBE)
        self.assertAlmostEqual(m.chamfer, 0.0)
        self.assertAlmostEqual(m.f1, 1.0)
        self.assertAlmostEqual(m.iou, 1.0)
        self.assertEqual(m.uid, "u1")

    def test_candidate_gate_filters(self):
        samples = [("a", CUBE, CUBE), ("b", CUBE, CUBE)]
        out = evaluate_corpus(samples, candidates=("a",))
        self.assertEqual([m.uid for m in out], ["a"])
        self.assertEqual(len(evaluate_corpus(samples)), 2)

    def test_aggregate_scales_chamfer_by_1000(self):
        pred = [(x + 0.4, y, z) for x, y, z in CUBE]
        metrics = evaluate_corpus([("a", pred, CUBE), ("b", CUBE, CUBE)])
        agg = aggregate(metrics)
        self.assertEqual(agg["n"], 2)
        raw = [m.chamfer for m in metrics]
        self.assertAlmostEqual(agg["cd_mean"], sum(raw) / 2 * CD_SCALE)
        self.assertAlmostEqual(agg["cd_median"], sum(raw) / 2 * CD_SCALE)
        self.assertGreaterEqual(agg["f1_mean"], 0.0)
        self.assertLessEqual(agg["iou_mean"], 1.0)

    def test_aggregate_empty(self):
        agg = aggregate([])
        self.assertEqual(agg["n"], 0)
        self.assertIsNone(agg["cd_mean"])

    def test_deterministic(self):
        pred = [(x + 0.1, y, z) for x, y, z in CUBE]
        self.assertEqual(
            evaluate_sample("u", pred, CUBE), evaluate_sample("u", pred, CUBE)
        )


if __name__ == "__main__":
    unittest.main()
