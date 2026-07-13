"""Tests for geometry.cadcoder_solidalign (CAD-Coder SolidAlign pipeline)."""

import math
import random
import unittest

from harnesscad.domain.geometry.transforms.cadcoder_solidalign import (
    align_point_clouds,
    candidate_rotations,
    centroid,
    covariance,
    inertia_tensor,
    normalization_scale,
    normalize_points,
    principal_frame,
    reflection_signs,
    voxel_iou_score,
)
from harnesscad.domain.geometry.transforms.e3dbench_umeyama import det3, jacobi_eigen, matmul, transpose


def _box_points(nx, ny, nz, sx, sy, sz):
    """Regular grid of points filling an axis-aligned box with given extents."""
    pts = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pts.append([
                    (i / (nx - 1) - 0.5) * sx,
                    (j / (ny - 1) - 0.5) * sy,
                    (k / (nz - 1) - 0.5) * sz,
                ])
    return pts


def _rotate_z(pts, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [[c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]] for p in pts]


class TestDescriptors(unittest.TestCase):
    def test_centroid(self):
        pts = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]]
        self.assertEqual(centroid(pts), (0.5, 1.0, 1.5))

    def test_centroid_empty_raises(self):
        with self.assertRaises(ValueError):
            centroid([])

    def test_covariance_symmetric(self):
        pts = _box_points(4, 4, 4, 2.0, 4.0, 6.0)
        c = covariance(pts)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(c[i][j], c[j][i])
        # variance ordering follows extent: z > y > x
        self.assertGreater(c[2][2], c[1][1])
        self.assertGreater(c[1][1], c[0][0])

    def test_inertia_shares_axes_with_covariance(self):
        pts = _rotate_z(_box_points(4, 4, 4, 1.0, 3.0, 5.0), 0.6)
        _, _, v_cov = principal_frame(pts)
        _, v_in = jacobi_eigen(inertia_tensor(pts))
        # columns should match up to sign; compare |dot| ~ 1 for paired axes
        for c in range(3):
            dots = [abs(sum(v_cov[r][c] * v_in[r][cc] for r in range(3)))
                    for cc in range(3)]
            self.assertAlmostEqual(max(dots), 1.0, places=6)


class TestNormalization(unittest.TestCase):
    def test_normalization_scale_positive(self):
        self.assertAlmostEqual(normalization_scale([4.0, 0.0, 0.0]), 2.0)

    def test_normalization_scale_degenerate(self):
        with self.assertRaises(ValueError):
            normalization_scale([0.0, 0.0, 0.0])

    def test_normalize_points_recenters_and_scales(self):
        pts = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        out = normalize_points(pts, (1.0, 1.0, 0.0), 2.0)
        self.assertEqual(out[0], [0.5, -0.5, 0.0])
        self.assertEqual(out[1], [-0.5, 0.5, 0.0])

    def test_normalize_points_bad_scale(self):
        with self.assertRaises(ValueError):
            normalize_points([[0.0, 0.0, 0.0]], (0.0, 0.0, 0.0), 0.0)


class TestRotationFamily(unittest.TestCase):
    def test_four_sign_combinations(self):
        signs = reflection_signs()
        self.assertEqual(len(signs), 4)
        # each has an even number of -1 -> right-handed
        for s in signs:
            self.assertEqual(s.count(-1) % 2, 0)

    def test_candidates_are_proper_rotations(self):
        pts = _rotate_z(_box_points(4, 4, 4, 1.0, 3.0, 5.0), 0.4)
        _, _, vs = principal_frame(pts)
        _, _, vt = principal_frame(_box_points(4, 4, 4, 1.0, 3.0, 5.0))
        for r in candidate_rotations(vs, vt):
            self.assertAlmostEqual(det3(r), 1.0, places=6)
            # orthogonality: R^T R = I
            prod = matmul(transpose(r), r)
            for i in range(3):
                for j in range(3):
                    self.assertAlmostEqual(prod[i][j], 1.0 if i == j else 0.0,
                                           places=6)


class TestVoxelScore(unittest.TestCase):
    def test_identical_clouds_iou_one(self):
        pts = _box_points(3, 3, 3, 2.0, 2.0, 2.0)
        self.assertEqual(voxel_iou_score(pts, pts, resolution=8), 1.0)

    def test_both_empty(self):
        self.assertEqual(voxel_iou_score([], [], resolution=8), 1.0)

    def test_disjoint_clouds_low(self):
        a = _box_points(3, 3, 3, 1.0, 1.0, 1.0)
        b = [[p[0] + 100.0, p[1], p[2]] for p in a]
        self.assertLess(voxel_iou_score(a, b, resolution=8), 0.05)

    def test_bad_resolution(self):
        with self.assertRaises(ValueError):
            voxel_iou_score([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], resolution=0)


class TestAlignment(unittest.TestCase):
    def test_recovers_rotation(self):
        src = _box_points(6, 6, 6, 1.0, 2.0, 3.0)
        # target is the same box rotated 90deg about z and translated + scaled
        rot = _rotate_z(src, math.pi / 2.0)
        tgt = [[2.0 * p[0] + 10.0, 2.0 * p[1] - 5.0, 2.0 * p[2] + 3.0]
               for p in rot]
        result = align_point_clouds(src, tgt, resolution=16)
        self.assertGreater(result["iou"], 0.9)
        self.assertEqual(len(result["candidate_ious"]), 4)

    def test_aligned_lands_in_target_frame(self):
        src = _box_points(6, 6, 6, 1.0, 2.0, 3.0)
        rot = _rotate_z(src, math.pi / 2.0)
        tgt = [[2.0 * p[0] + 10.0, 2.0 * p[1] - 5.0, 2.0 * p[2] + 3.0]
               for p in rot]
        result = align_point_clouds(src, tgt, resolution=16)
        # aligned source centroid should match target centroid
        ca = centroid(result["aligned"])
        ct = centroid(tgt)
        for i in range(3):
            self.assertAlmostEqual(ca[i], ct[i], places=6)
        # every aligned point should coincide with a target point (set equality)
        aligned = result["aligned"]
        max_residual = max(min(math.dist(a, t) for t in tgt) for a in aligned)
        self.assertLess(max_residual, 1e-6)

    def test_deterministic(self):
        rng = random.Random(7)
        src = [[rng.uniform(-1, 1), rng.uniform(-2, 2), rng.uniform(-3, 3)]
               for _ in range(200)]
        tgt = _rotate_z(src, 0.7)
        r1 = align_point_clouds(src, tgt, resolution=12)
        r2 = align_point_clouds(src, tgt, resolution=12)
        self.assertEqual(r1["candidate_ious"], r2["candidate_ious"])
        self.assertEqual(r1["rotation"], r2["rotation"])


if __name__ == "__main__":
    unittest.main()
