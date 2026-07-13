import math
import unittest

from harnesscad.agents.generation.shape_metrics import (
    bbox_center, bbox_extent, center_to_origin, kabsch, icp,
    chamfer, f1_score, voxel_iou, voxel_resolution, absolute_space_metrics,
)


def _cube(size=10.0, offset=(0.0, 0.0, 0.0)):
    pts = []
    for x in (0.0, size):
        for y in (0.0, size):
            for z in (0.0, size):
                pts.append((x + offset[0], y + offset[1], z + offset[2]))
    return pts


def _rotate_z(pts, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [(p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]) for p in pts]


class TestCoregistration(unittest.TestCase):
    def test_bbox_center(self):
        self.assertEqual(bbox_center(_cube(10.0)), (5.0, 5.0, 5.0))

    def test_bbox_extent(self):
        self.assertEqual(bbox_extent(_cube(10.0)), (10.0, 10.0, 10.0))

    def test_center_to_origin(self):
        c = bbox_center(center_to_origin(_cube(10.0)))
        for v in c:
            self.assertAlmostEqual(v, 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            bbox_center([])


class TestKabsch(unittest.TestCase):
    def test_identity_for_aligned(self):
        pts = center_to_origin(_cube(10.0))
        R = kabsch(pts, pts)
        for r in range(3):
            for c in range(3):
                self.assertAlmostEqual(R[r][c], 1.0 if r == c else 0.0, places=4)

    def test_recovers_rotation(self):
        src = center_to_origin(_cube(10.0))
        dst = _rotate_z(src, 30.0)
        R = kabsch(src, dst)
        for p, q in zip(src, dst):
            rp = (R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2],
                  R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2],
                  R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2])
            for a, b in zip(rp, q):
                self.assertAlmostEqual(a, b, places=3)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            kabsch([(0, 0, 0)], [(0, 0, 0), (1, 1, 1)])


class TestICP(unittest.TestCase):
    def test_aligns_rotated_cube(self):
        target = _cube(10.0)
        source = _rotate_z(center_to_origin(target), 25.0)
        aligned, hist = icp(source, target, max_iter=30)
        self.assertLess(hist[-1], 1e-3)

    def test_history_nonincreasing_tail(self):
        target = _cube(10.0)
        source = _rotate_z(center_to_origin(target), 15.0)
        _, hist = icp(source, target)
        self.assertLessEqual(hist[-1], hist[0] + 1e-9)


class TestPointMetrics(unittest.TestCase):
    def test_chamfer_identical_zero(self):
        pts = _cube(10.0)
        self.assertAlmostEqual(chamfer(pts, pts), 0.0)

    def test_chamfer_positive(self):
        self.assertGreater(chamfer(_cube(10.0), _cube(10.0, (5.0, 0.0, 0.0))), 0.0)

    def test_f1_identical_one(self):
        pts = _cube(10.0)
        self.assertAlmostEqual(f1_score(pts, pts, tau=0.5), 1.0)

    def test_f1_disjoint_zero(self):
        a = _cube(1.0)
        b = _cube(1.0, (100.0, 0.0, 0.0))
        self.assertEqual(f1_score(a, b, tau=1.0), 0.0)

    def test_f1_bad_tau(self):
        with self.assertRaises(ValueError):
            f1_score(_cube(), _cube(), tau=0.0)


class TestVoxelIoU(unittest.TestCase):
    def test_identical_one(self):
        pts = _cube(10.0)
        self.assertAlmostEqual(voxel_iou(pts, pts), 1.0)

    def test_disjoint_zero(self):
        self.assertEqual(voxel_iou(_cube(1.0), _cube(1.0, (100.0, 0.0, 0.0))), 0.0)

    def test_adaptive_coarsening(self):
        small = voxel_resolution(_cube(10.0), _cube(10.0))
        self.assertEqual(small, 1.0)
        big = voxel_resolution(_cube(500.0), _cube(500.0))
        self.assertGreater(big, 1.0)


class TestFullPipeline(unittest.TestCase):
    def test_rotated_cube_scores_high(self):
        target = _cube(10.0)
        pred = _rotate_z(center_to_origin(target), 20.0)
        m = absolute_space_metrics(pred, target, tau=1.0, icp_iters=30)
        self.assertLess(m["chamfer"], 1.0)
        self.assertGreater(m["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
