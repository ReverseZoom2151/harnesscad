import unittest
from harnesscad.domain.reconstruction.fewshot_partseg_labelprop import (
    gaussian_knn_weights, symmetric_normalise, one_hot,
    propagate_iterative, propagate_closed_form, softmax_rows, predict,
)


class Tests(unittest.TestCase):
    def test_weights_symmetric_zero_diag(self):
        feats = [(0.0,), (1.0,), (5.0,)]
        w = gaussian_knn_weights(feats, k=1, sigma=1.0)
        for i in range(3):
            self.assertEqual(w[i][i], 0.0)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(w[i][j], w[j][i])

    def test_weights_decay_with_distance(self):
        feats = [(0.0,), (1.0,), (3.0,)]
        w = gaussian_knn_weights(feats, k=2, sigma=1.0)
        self.assertGreater(w[0][1], w[0][2])  # closer -> larger weight

    def test_weights_bad_sigma(self):
        with self.assertRaises(ValueError):
            gaussian_knn_weights([(0.0,)], 1, sigma=0.0)

    def test_symmetric_normalise_bounds(self):
        w = [[0.0, 1.0], [1.0, 0.0]]
        s = symmetric_normalise(w)
        self.assertAlmostEqual(s[0][1], 1.0)
        self.assertAlmostEqual(s[0][0], 0.0)

    def test_one_hot(self):
        y = one_hot([0, None, 1], 2)
        self.assertEqual(y, [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])

    def test_one_hot_out_of_range(self):
        with self.assertRaises(ValueError):
            one_hot([5], 2)

    def test_softmax_rows(self):
        out = softmax_rows([[0.0, 0.0]])
        self.assertAlmostEqual(out[0][0], 0.5)
        self.assertAlmostEqual(sum(out[0]), 1.0)

    def test_closed_form_matches_iterative(self):
        feats = [(0.0,), (0.2,), (5.0,), (5.2,)]
        w = gaussian_knn_weights(feats, k=2, sigma=1.0)
        s = symmetric_normalise(w)
        y = one_hot([0, None, 1, None], 2)
        alpha = 0.9
        zc = propagate_closed_form(s, y, alpha=alpha)
        zi = propagate_iterative(s, y, alpha=alpha, epochs=400)
        # Iterative fixed point is (1 - alpha) * closed-form (same argmax).
        for i in range(4):
            for c in range(2):
                self.assertAlmostEqual(zc[i][c] * (1 - alpha), zi[i][c],
                                       places=3)

    def test_alpha_range(self):
        s = [[0.0]]
        with self.assertRaises(ValueError):
            propagate_closed_form(s, [[1.0]], alpha=1.0)
        with self.assertRaises(ValueError):
            propagate_iterative(s, [[1.0]], alpha=-0.1)

    def test_predict_propagates_labels(self):
        # Two clusters, one labelled point each; unlabelled follow their cluster.
        feats = [(0.0,), (0.1,), (0.2,), (9.0,), (9.1,), (9.2,)]
        labels = [0, None, None, 1, None, None]
        preds, probs = predict(feats, labels, 2, k=3, sigma=1.0, alpha=0.9)
        self.assertEqual(preds[1], 0)
        self.assertEqual(preds[2], 0)
        self.assertEqual(preds[4], 1)
        self.assertEqual(preds[5], 1)
        self.assertEqual(len(probs), 6)

    def test_predict_iterative_agrees(self):
        feats = [(0.0,), (0.1,), (9.0,), (9.1,)]
        labels = [0, None, 1, None]
        pc, _ = predict(feats, labels, 2, k=2, method="closed_form")
        pi, _ = predict(feats, labels, 2, k=2, method="iterative", epochs=300)
        self.assertEqual(pc, pi)

    def test_predict_bad_method(self):
        with self.assertRaises(ValueError):
            predict([(0.0,), (1.0,)], [0, None], 2, method="nope")


if __name__ == "__main__":
    unittest.main()
