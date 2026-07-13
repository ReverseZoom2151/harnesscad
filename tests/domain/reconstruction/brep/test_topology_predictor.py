import unittest

from harnesscad.domain.reconstruction.brep.topology_predictor import (
    point_box_distance, edge_surface_score, topology_scores,
    predict_adjacency, predict, surface_edges,
)


class TestPointBoxDistance(unittest.TestCase):
    def test_inside_zero(self):
        self.assertEqual(point_box_distance((0.5, 0.5, 0.5), (0, 0, 0, 1, 1, 1)), 0.0)

    def test_outside_axis(self):
        self.assertAlmostEqual(point_box_distance((2.0, 0.5, 0.5), (0, 0, 0, 1, 1, 1)), 1.0)

    def test_corner(self):
        d = point_box_distance((2.0, 2.0, 0.5), (0, 0, 0, 1, 1, 1))
        self.assertAlmostEqual(d, 2 ** 0.5)


class TestEdgeSurfaceScore(unittest.TestCase):
    def test_on_surface_high(self):
        # both endpoints inside/on box -> score 1
        s = edge_surface_score((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                               (0.0, 0.0, 0.0, 1.0, 1.0, 0.0), tolerance=0.1)
        self.assertEqual(s, 1.0)

    def test_far_low(self):
        s = edge_surface_score((5.0, 5.0, 5.0), (6.0, 6.0, 6.0),
                               (0.0, 0.0, 0.0, 1.0, 1.0, 0.0), tolerance=0.1)
        self.assertEqual(s, 0.0)

    def test_threshold_boundary(self):
        # mean distance exactly tolerance/2 -> score 0.5
        s = edge_surface_score((0.0, 0.0, 0.05), (0.0, 0.0, 0.05),
                               (0.0, 0.0, 0.0, 1.0, 1.0, 0.0), tolerance=0.1)
        self.assertAlmostEqual(s, 0.5)

    def test_bad_tolerance(self):
        with self.assertRaises(ValueError):
            edge_surface_score((0, 0, 0), (1, 0, 0), (0, 0, 0, 1, 1, 0), 0.0)


class TestPredict(unittest.TestCase):
    def setUp(self):
        # unit square in z=0 plane, 4 boundary edges, one surface
        self.edges = (
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
            ((1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        )
        self.surfaces = ((0.0, 0.0, 0.0, 1.0, 1.0, 0.0),)

    def test_all_edges_bound_surface(self):
        adj = predict(self.edges, self.surfaces, tolerance=0.1)
        self.assertEqual(adj, ((True,), (True,), (True,), (True,)))

    def test_distant_edge_excluded(self):
        edges = self.edges + (((5.0, 5.0, 5.0), (6.0, 6.0, 6.0)),)
        adj = predict(edges, self.surfaces, tolerance=0.1)
        self.assertFalse(adj[4][0])

    def test_scores_matrix_shape(self):
        scores = topology_scores(self.edges, self.surfaces, 0.1)
        self.assertEqual(len(scores), 4)
        self.assertEqual(len(scores[0]), 1)

    def test_predict_adjacency_threshold(self):
        scores = ((0.6, 0.4), (0.5, 0.9))
        adj = predict_adjacency(scores, tau=0.5)
        self.assertEqual(adj, ((True, False), (False, True)))

    def test_surface_edges(self):
        adj = predict(self.edges, self.surfaces, tolerance=0.1)
        self.assertEqual(surface_edges(adj), ((0, 1, 2, 3),))

    def test_surface_edges_empty(self):
        self.assertEqual(surface_edges(()), ())


if __name__ == "__main__":
    unittest.main()
