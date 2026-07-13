"""Tests for the E3D-Bench deterministic evaluation modules (paper 73)."""

import math
import unittest

from harnesscad.domain.geometry.transforms.e3dbench_umeyama import (
    Sim3,
    alignment_rmse,
    det3,
    jacobi_eigen,
    matmul,
    svd3,
    transpose,
    umeyama_alignment,
)
from harnesscad.eval.bench.vision.e3dbench_depth_metrics import (
    TAU_SPARSE,
    TAU_VIDEO,
    abs_rel,
    apply_scale,
    delta_inlier_ratio,
    evaluate_depth,
    median_scale_factor,
)
from harnesscad.eval.bench.vision.e3dbench_pose_metrics import (
    absolute_translation_error,
    evaluate_trajectory,
    relative_pose_error,
    rotation_angle_error,
    translation_error,
)
from harnesscad.eval.bench.geometry.e3dbench_pointmap_metrics import (
    accuracy,
    completeness,
    evaluate_reconstruction,
    normal_consistency,
    precision_recall_fscore,
)
from harnesscad.eval.bench.harness.e3dbench_harness import (
    BenchmarkHarness,
    MetricSpec,
    normalize_higher_is_better,
)


def _rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _rot_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


class LinAlgTests(unittest.TestCase):
    def test_jacobi_eigen_diagonal(self):
        a = [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]
        vals, vecs = jacobi_eigen(a)
        self.assertAlmostEqual(vals[0], 3.0, places=10)
        self.assertAlmostEqual(vals[1], 2.0, places=10)
        self.assertAlmostEqual(vals[2], 1.0, places=10)

    def test_jacobi_eigen_reconstructs(self):
        a = [[2.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 1.0]]
        vals, v = jacobi_eigen(a)
        # A V = V diag(vals)
        av = matmul(a, v)
        for j in range(3):
            for i in range(3):
                self.assertAlmostEqual(av[i][j], vals[j] * v[i][j], places=8)

    def test_svd3_reconstructs(self):
        a = [[1.0, 2.0, 0.0], [0.0, 1.0, -1.0], [2.0, 0.0, 1.0]]
        u, s, v = svd3(a)
        # U diag(s) V^T
        sd = [[s[c] if r == c else 0.0 for c in range(3)] for r in range(3)]
        recon = matmul(matmul(u, sd), transpose(v))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(recon[i][j], a[i][j], places=8)
        # descending singular values, non-negative
        self.assertGreaterEqual(s[0], s[1])
        self.assertGreaterEqual(s[1], s[2])
        self.assertGreaterEqual(s[2], -1e-12)

    def test_det3(self):
        self.assertAlmostEqual(det3([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 1.0)
        self.assertAlmostEqual(det3(_rot_z(0.7)), 1.0, places=10)


class UmeyamaTests(unittest.TestCase):
    def setUp(self):
        self.src = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, -1.0, 0.5]]

    def _apply(self, scale, R, t, pts):
        out = []
        for p in pts:
            rp = [scale * (R[r][0] * p[0] + R[r][1] * p[1] + R[r][2] * p[2]) + t[r]
                  for r in range(3)]
            out.append(rp)
        return out

    def test_recovers_known_sim3(self):
        R = _rot_z(0.5)
        scale = 2.5
        t = [3.0, -1.0, 4.0]
        dst = self._apply(scale, R, t, self.src)
        xform = umeyama_alignment(self.src, dst, with_scale=True)
        self.assertAlmostEqual(xform.scale, scale, places=6)
        self.assertAlmostEqual(alignment_rmse(self.src, dst, xform), 0.0, places=6)
        for r in range(3):
            for c in range(3):
                self.assertAlmostEqual(xform.R[r][c], R[r][c], places=6)

    def test_rigid_no_scale(self):
        R = _rot_x(0.9)
        t = [1.0, 2.0, -3.0]
        dst = self._apply(1.0, R, t, self.src)
        xform = umeyama_alignment(self.src, dst, with_scale=False)
        self.assertAlmostEqual(xform.scale, 1.0, places=9)
        self.assertAlmostEqual(alignment_rmse(self.src, dst, xform), 0.0, places=6)

    def test_proper_rotation(self):
        R = _rot_z(1.2)
        dst = self._apply(1.7, R, [0.0, 0.0, 0.0], self.src)
        xform = umeyama_alignment(self.src, dst)
        self.assertAlmostEqual(det3(xform.R), 1.0, places=6)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            umeyama_alignment([], [])

    def test_mismatched_raises(self):
        with self.assertRaises(ValueError):
            umeyama_alignment([[0, 0, 0]], [[0, 0, 0], [1, 1, 1]])


class DepthMetricTests(unittest.TestCase):
    def test_perfect_prediction(self):
        gt = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(abs_rel(gt, gt), 0.0)
        self.assertAlmostEqual(delta_inlier_ratio(gt, gt, TAU_SPARSE), 1.0)

    def test_abs_rel_value(self):
        pred = [1.1, 2.0, 2.7]
        gt = [1.0, 2.0, 3.0]
        # errors: 0.1, 0, 0.1 -> mean 0.2/3
        self.assertAlmostEqual(abs_rel(pred, gt), (0.1 + 0.0 + 0.3 / 3.0) / 3.0)

    def test_delta_threshold(self):
        pred = [1.0, 1.5, 2.0]
        gt = [1.0, 1.0, 2.0]
        # ratios: 1.0, 1.5, 1.0 -> under 1.25: 2/3
        self.assertAlmostEqual(delta_inlier_ratio(pred, gt, TAU_VIDEO), 2.0 / 3.0)

    def test_mask_and_invalid(self):
        pred = [1.0, 9.0, 2.0]
        gt = [1.0, 0.0, 2.0]  # middle invalid (gt<=0)
        self.assertAlmostEqual(abs_rel(pred, gt), 0.0)
        mask = [True, True, False]
        self.assertAlmostEqual(abs_rel(pred, gt, mask), 0.0)

    def test_median_scaling(self):
        gt = [2.0, 4.0, 6.0, 8.0]
        pred = [1.0, 2.0, 3.0, 4.0]  # exactly half
        factor = median_scale_factor(pred, gt)
        self.assertAlmostEqual(factor, 2.0)
        scaled = apply_scale(pred, factor)
        self.assertAlmostEqual(abs_rel(scaled, gt), 0.0)

    def test_evaluate_depth_report(self):
        gt = [2.0, 4.0, 6.0, 8.0]
        pred = [1.0, 2.0, 3.0, 4.0]
        rep = evaluate_depth(pred, gt, align_median=True, tau=TAU_VIDEO)
        self.assertAlmostEqual(rep["scale"], 2.0)
        self.assertAlmostEqual(rep["abs_rel"], 0.0)
        self.assertAlmostEqual(rep["delta"], 100.0)
        self.assertEqual(rep["count"], 4)

    def test_negative_pred_not_inlier(self):
        pred = [-1.0, 2.0]
        gt = [1.0, 2.0]
        self.assertAlmostEqual(delta_inlier_ratio(pred, gt, TAU_VIDEO), 0.5)


class PoseMetricTests(unittest.TestCase):
    def test_rotation_angle_error(self):
        self.assertAlmostEqual(rotation_angle_error(_rot_z(0.0), _rot_z(0.0)), 0.0)
        self.assertAlmostEqual(
            rotation_angle_error(_rot_z(0.0), _rot_z(math.radians(30.0))),
            30.0, places=6)

    def test_translation_error(self):
        self.assertAlmostEqual(translation_error([0, 0, 0], [3, 4, 0]), 5.0)

    def _traj(self, n):
        poses = []
        for i in range(n):
            poses.append((_rot_z(0.1 * i), [float(i), 0.5 * i, 0.0]))
        return poses

    def test_ate_zero_for_aligned(self):
        gt = self._traj(6)
        # predicted = gt transformed by a global sim3 (scale+rot+trans of centres)
        R = _rot_z(0.4)
        pred = []
        for Rp, t in gt:
            nt = [3.0 * (R[r][0] * t[0] + R[r][1] * t[1] + R[r][2] * t[2]) + (r + 1)
                  for r in range(3)]
            pred.append((Rp, nt))
        ate = absolute_translation_error(pred, gt, with_scale=True)
        self.assertAlmostEqual(ate, 0.0, places=6)

    def test_rpe_zero_identical(self):
        gt = self._traj(5)
        rpe_t, rpe_r = relative_pose_error(gt, gt)
        self.assertAlmostEqual(rpe_t, 0.0, places=9)
        self.assertAlmostEqual(rpe_r, 0.0, places=9)

    def test_evaluate_trajectory(self):
        gt = self._traj(5)
        rep = evaluate_trajectory(gt, gt)
        self.assertAlmostEqual(rep["ate"], 0.0, places=6)
        self.assertAlmostEqual(rep["rpe_trans"], 0.0, places=9)
        self.assertAlmostEqual(rep["rpe_rot"], 0.0, places=9)
        self.assertEqual(rep["frames"], 5)

    def test_rpe_detects_rotation_drift(self):
        gt = self._traj(5)
        pred = [(_rot_z(0.2 * i), t) for i, (_, t) in enumerate(gt)]
        _, rpe_r = relative_pose_error(pred, gt)
        self.assertGreater(rpe_r, 0.0)

    def test_too_few_frames(self):
        with self.assertRaises(ValueError):
            relative_pose_error([(_rot_z(0), [0, 0, 0])], [(_rot_z(0), [0, 0, 0])])


class PointmapMetricTests(unittest.TestCase):
    def setUp(self):
        self.gt = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                   [1.0, 1.0, 0.0], [0.5, 0.5, 0.0]]

    def test_accuracy_completeness_zero_identical(self):
        self.assertAlmostEqual(accuracy(self.gt, self.gt), 0.0)
        self.assertAlmostEqual(completeness(self.gt, self.gt), 0.0)

    def test_accuracy_value(self):
        pred = [[0.0, 0.0, 0.1]]  # 0.1 from nearest gt point
        self.assertAlmostEqual(accuracy(pred, self.gt), 0.1, places=9)

    def test_fscore_perfect(self):
        p, r, f = precision_recall_fscore(self.gt, self.gt, tau=0.01)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f, 1.0)

    def test_fscore_partial(self):
        pred = list(self.gt) + [[5.0, 5.0, 5.0]]  # one outlier predicted
        p, r, f = precision_recall_fscore(pred, self.gt, tau=0.01)
        self.assertAlmostEqual(p, 5.0 / 6.0)  # 5 of 6 predicted matched
        self.assertAlmostEqual(r, 1.0)  # all gt covered
        self.assertAlmostEqual(f, 2 * (5 / 6) * 1.0 / (5 / 6 + 1.0))

    def test_normal_consistency(self):
        normals = [[0.0, 0.0, 1.0]] * len(self.gt)
        nc = normal_consistency(self.gt, self.gt, normals, normals)
        self.assertAlmostEqual(nc, 1.0)
        flipped = [[0.0, 0.0, -1.0]] * len(self.gt)
        nc2 = normal_consistency(self.gt, self.gt, normals, flipped)
        self.assertAlmostEqual(nc2, 1.0)  # abs cosine

    def test_evaluate_reconstruction_alignment(self):
        # predicted cloud = gt scaled/rotated/translated; alignment should
        # drive accuracy/completeness to ~0
        R = _rot_z(0.6)
        pred = []
        for p in self.gt:
            rp = [2.0 * (R[r][0] * p[0] + R[r][1] * p[1] + R[r][2] * p[2]) + (r + 5)
                  for r in range(3)]
            pred.append(rp)
        rep = evaluate_reconstruction(pred, self.gt, tau=0.01, align=True)
        self.assertAlmostEqual(rep["accuracy"], 0.0, places=6)
        self.assertAlmostEqual(rep["completeness"], 0.0, places=6)
        self.assertAlmostEqual(rep["fscore"], 1.0, places=6)

    def test_evaluate_reconstruction_normals(self):
        normals = [[0.0, 0.0, 1.0]] * len(self.gt)
        rep = evaluate_reconstruction(self.gt, self.gt, align=False,
                                      pred_normals=normals, gt_normals=normals)
        self.assertAlmostEqual(rep["normal_consistency"], 1.0)


class HarnessTests(unittest.TestCase):
    def test_normalize_lower_better(self):
        vals = [10.0, 20.0, 30.0]
        norm = normalize_higher_is_better(vals, lower_is_better=True)
        self.assertAlmostEqual(norm[0], 1.0)  # smallest is best
        self.assertAlmostEqual(norm[2], 0.0)

    def test_normalize_higher_better(self):
        vals = [10.0, 20.0, 30.0]
        norm = normalize_higher_is_better(vals, lower_is_better=False)
        self.assertAlmostEqual(norm[0], 0.0)
        self.assertAlmostEqual(norm[2], 1.0)

    def test_normalize_all_equal(self):
        norm = normalize_higher_is_better([5.0, 5.0], lower_is_better=True)
        self.assertEqual(norm, [1.0, 1.0])

    def test_leaderboard(self):
        h = BenchmarkHarness([
            MetricSpec("abs_rel", lower_is_better=True),
            MetricSpec("delta", lower_is_better=False),
        ])
        # model A best on both metrics in both scenes
        h.add_result("s1", "A", {"abs_rel": 0.05, "delta": 95.0})
        h.add_result("s1", "B", {"abs_rel": 0.20, "delta": 60.0})
        h.add_result("s2", "A", {"abs_rel": 0.06, "delta": 90.0})
        h.add_result("s2", "B", {"abs_rel": 0.15, "delta": 70.0})
        board = h.leaderboard()
        self.assertEqual(board[0][0], "A")
        self.assertEqual(h.best_model(), "A")
        self.assertAlmostEqual(board[0][1], 1.0)  # A best on every metric/scene
        self.assertAlmostEqual(board[1][1], 0.0)

    def test_scene_scores_partial_metrics(self):
        h = BenchmarkHarness([MetricSpec("ate", True), MetricSpec("nc", False)])
        h.add_result("s1", "A", {"ate": 0.1})  # only one metric present
        h.add_result("s1", "B", {"ate": 0.5})
        scores = h.scene_scores("s1")
        self.assertAlmostEqual(scores["A"], 1.0)
        self.assertAlmostEqual(scores["B"], 0.0)

    def test_unknown_metric_raises(self):
        h = BenchmarkHarness([MetricSpec("ate", True)])
        with self.assertRaises(ValueError):
            h.add_result("s1", "A", {"bogus": 1.0})

    def test_models_and_scenes(self):
        h = BenchmarkHarness([MetricSpec("ate", True)])
        h.add_result("s2", "B", {"ate": 0.2})
        h.add_result("s1", "A", {"ate": 0.1})
        self.assertEqual(h.scenes(), ["s1", "s2"])
        self.assertEqual(h.models(), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
