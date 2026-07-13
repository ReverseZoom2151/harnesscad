import unittest
from harnesscad.domain.reconstruction.fewshot_partseg_features import (
    knn_indices, symmetric_eigenvalues, covariance_descriptors,
    edge_feature, point_features, _covariance,
)


class Tests(unittest.TestCase):
    def test_knn_excludes_self_and_orders(self):
        pts = [(0, 0, 0), (1, 0, 0), (5, 0, 0), (0.5, 0, 0)]
        nn = knn_indices(pts, 2)
        self.assertEqual(len(nn), 4)
        self.assertNotIn(0, nn[0])
        # nearest to origin is index 3 (0.5), then index 1 (1.0)
        self.assertEqual(nn[0], (3, 1))

    def test_knn_clamped_and_deterministic(self):
        pts = [(0, 0, 0), (1, 0, 0)]
        self.assertEqual(knn_indices(pts, 10)[0], (1,))
        self.assertEqual(knn_indices(pts, 5), knn_indices(pts, 5))

    def test_knn_negative_raises(self):
        with self.assertRaises(ValueError):
            knn_indices([(0, 0, 0)], -1)

    def test_eigenvalues_diagonal(self):
        cov = [[3.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]]
        self.assertEqual(symmetric_eigenvalues(cov), (3.0, 2.0, 1.0))

    def test_eigenvalues_general_symmetric(self):
        # Known symmetric matrix; eigenvalues sum to trace, descending.
        cov = [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 5.0]]
        eig = symmetric_eigenvalues(cov)
        self.assertEqual(len(eig), 3)
        self.assertTrue(eig[0] >= eig[1] >= eig[2])
        self.assertAlmostEqual(sum(eig), 9.0, places=6)  # trace
        # eigenvalues are {3,1,5}
        self.assertAlmostEqual(eig[0], 5.0, places=6)
        self.assertAlmostEqual(eig[1], 3.0, places=6)
        self.assertAlmostEqual(eig[2], 1.0, places=6)

    def test_descriptors_planar(self):
        # Points on a plane -> smallest eigenvalue ~0 -> high planarity.
        d = covariance_descriptors((4.0, 4.0, 0.0))
        self.assertAlmostEqual(d["curvature"], 0.0, places=6)
        self.assertAlmostEqual(d["planarity"], 1.0, places=6)
        self.assertAlmostEqual(d["linearity"], 0.0, places=6)

    def test_descriptors_linear(self):
        d = covariance_descriptors((9.0, 0.0, 0.0))
        self.assertAlmostEqual(d["linearity"], 1.0, places=6)
        self.assertAlmostEqual(d["scattering"], 0.0, places=6)

    def test_planar_cloud_low_curvature(self):
        # A grid on z=0 plane: every point's curvature should be ~0.
        pts = [(x, y, 0.0) for x in range(4) for y in range(4)]
        feats = point_features(pts, k=6)
        for row in feats:
            self.assertLess(row[4], 1e-6)  # curvature ~ 0 everywhere
        # Planarity is high on average (edge/corner points are less planar).
        avg_planarity = sum(row[1] for row in feats) / len(feats)
        self.assertGreater(avg_planarity, 0.5)

    def test_linear_cloud_high_linearity(self):
        pts = [(float(i), 0.0, 0.0) for i in range(8)]
        feats = point_features(pts, k=3)
        # interior points should be strongly linear
        self.assertGreater(feats[4][0], 0.9)

    def test_edge_feature_empty(self):
        self.assertEqual(edge_feature([(0, 0, 0)], 0, ()), (0.0, 0.0, 0.0, 0.0))

    def test_edge_feature_maxpool(self):
        pts = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, -3.0, 0.0)]
        ef = edge_feature(pts, 0, (1, 2))
        self.assertEqual(ef[0], 2.0)   # max dx
        self.assertEqual(ef[1], 0.0)   # max dy (both <=0 -> 0)
        self.assertAlmostEqual(ef[3], 3.0)  # farthest neighbour dist

    def test_point_features_shape(self):
        pts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
        feats = point_features(pts, k=3)
        self.assertEqual(len(feats), 5)
        self.assertTrue(all(len(r) == 9 for r in feats))

    def test_covariance_symmetry(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (2.0, 1.0, 0.0)]
        cov = _covariance(pts, [1, 2], pts[0])
        for a in range(3):
            for b in range(3):
                self.assertAlmostEqual(cov[a][b], cov[b][a])


if __name__ == "__main__":
    unittest.main()
