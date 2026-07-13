import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harnesscad.eval.bench.sketch.sketch_chamfer_2d import (
    chamfer_2d,
    histogram,
    nna_gap_to_half,
    one_nearest_neighbor_accuracy,
    sketch_cd,
    sketch_points,
    sqrt_bin_count,
)


class TestSketchMetrics(unittest.TestCase):
    def test_sketch_points(self):
        raster = [[1.0, 1.0], [0.0, 1.0]]  # one stroke pixel at (col=0,row=1)
        pts = sketch_points(raster, white=1.0)
        self.assertEqual(pts, [(0.0, 1.0)])

    def test_chamfer_identical_zero(self):
        a = [(0.0, 0.0), (1.0, 1.0)]
        self.assertAlmostEqual(chamfer_2d(a, a), 0.0)

    def test_chamfer_symmetric(self):
        a = [(0.0, 0.0)]
        b = [(3.0, 4.0)]
        # each direction contributes distance 5 -> total 10
        self.assertAlmostEqual(chamfer_2d(a, b), 10.0)
        self.assertAlmostEqual(chamfer_2d(a, b), chamfer_2d(b, a))

    def test_chamfer_squared(self):
        a = [(0.0, 0.0)]
        b = [(3.0, 4.0)]
        self.assertAlmostEqual(chamfer_2d(a, b, squared=True), 50.0)

    def test_chamfer_empty(self):
        with self.assertRaises(ValueError):
            chamfer_2d([], [(0.0, 0.0)])

    def test_sketch_cd_identical(self):
        raster = [[1.0, 0.0], [0.0, 1.0]]
        self.assertAlmostEqual(sketch_cd(raster, raster), 0.0)

    def test_sketch_cd_shifted(self):
        i = [[0.0, 1.0]]  # stroke at (0,0)
        g = [[1.0, 0.0]]  # stroke at (1,0)
        self.assertAlmostEqual(sketch_cd(i, g), 2.0)  # dist 1 both directions

    def test_1nna_perfectly_separable(self):
        # two well-separated clusters -> every NN shares label -> accuracy 1.0
        dist = [
            [0.0, 0.1, 5.0, 5.1],
            [0.1, 0.0, 5.1, 5.0],
            [5.0, 5.1, 0.0, 0.1],
            [5.1, 5.0, 0.1, 0.0],
        ]
        labels = [0, 0, 1, 1]
        self.assertEqual(one_nearest_neighbor_accuracy(dist, labels), 1.0)
        self.assertEqual(nna_gap_to_half(1.0), 0.5)

    def test_1nna_indistinguishable(self):
        # interleaved so each NN is the opposite label -> accuracy 0.0 (gap 0.5)
        dist = [
            [0.0, 0.1, 1.0, 1.0],
            [0.1, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.1],
            [1.0, 1.0, 0.1, 0.0],
        ]
        labels = [0, 1, 0, 1]
        acc = one_nearest_neighbor_accuracy(dist, labels)
        self.assertEqual(acc, 0.0)

    def test_1nna_errors(self):
        with self.assertRaises(ValueError):
            one_nearest_neighbor_accuracy([[0.0]], [0])
        with self.assertRaises(ValueError):
            one_nearest_neighbor_accuracy([[0.0, 1.0]], [0, 1])  # not square

    def test_sqrt_bin_count(self):
        self.assertEqual(sqrt_bin_count(100), 10)
        self.assertEqual(sqrt_bin_count(101), 11)  # ceil(10.05)
        self.assertEqual(sqrt_bin_count(1), 1)
        with self.assertRaises(ValueError):
            sqrt_bin_count(0)

    def test_histogram(self):
        vals = [0.0, 1.0, 2.0, 3.0]
        counts = histogram(vals, bins=2)
        self.assertEqual(sum(counts), 4)
        self.assertEqual(counts, [2, 2])  # max value 3.0 in last bucket

    def test_histogram_degenerate(self):
        self.assertEqual(histogram([5.0, 5.0, 5.0], bins=3), [3, 0, 0])

    def test_histogram_errors(self):
        with self.assertRaises(ValueError):
            histogram([], bins=2)
        with self.assertRaises(ValueError):
            histogram([1.0], bins=0)


if __name__ == "__main__":
    unittest.main()
