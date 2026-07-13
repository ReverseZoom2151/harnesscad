import unittest
from harnesscad.domain.reconstruction.recognize.fewshot_partseg_prototypes import (
    farthest_point_sampling, class_prototypes, build_prototypes,
    assign, segment,
)


class Tests(unittest.TestCase):
    def test_fps_picks_extremes(self):
        feats = [(0.0,), (1.0,), (10.0,), (5.0,)]
        idx = farthest_point_sampling(feats, 2, seed=0)
        # start=0, farthest is index 2 (10.0)
        self.assertEqual(idx, (0, 2))

    def test_fps_count_clamped(self):
        feats = [(0.0,), (1.0,)]
        self.assertEqual(len(farthest_point_sampling(feats, 9)), 2)

    def test_fps_empty(self):
        self.assertEqual(farthest_point_sampling([], 3), ())
        self.assertEqual(farthest_point_sampling([(0.0,)], 0), ())

    def test_fps_deterministic(self):
        feats = [(float(i), float(i * i)) for i in range(10)]
        self.assertEqual(farthest_point_sampling(feats, 4, seed=3),
                         farthest_point_sampling(feats, 4, seed=3))

    def test_single_prototype_is_mean(self):
        feats = [(0.0, 0.0), (2.0, 4.0)]
        protos = class_prototypes(feats, [0, 1], 1)
        self.assertEqual(protos, ((1.0, 2.0),))

    def test_multi_prototype_two_clusters(self):
        # Two tight clusters; two prototypes should separate them.
        feats = [(0.0,), (0.1,), (10.0,), (10.1,)]
        protos = class_prototypes(feats, [0, 1, 2, 3], 2, seed=0)
        self.assertEqual(len(protos), 2)
        centers = sorted(p[0] for p in protos)
        self.assertAlmostEqual(centers[0], 0.05, places=6)
        self.assertAlmostEqual(centers[1], 10.05, places=6)

    def test_build_prototypes_labels_sorted(self):
        feats = [(0.0,), (1.0,), (5.0,)]
        labs = [1, 1, 0]
        protos = build_prototypes(feats, labs, 1)
        self.assertEqual(protos[0][0], 0)  # label 0 first
        self.assertEqual(protos[1][0], 1)
        self.assertAlmostEqual(protos[1][1][0], 0.5)  # mean of 0,1

    def test_build_prototypes_length_mismatch(self):
        with self.assertRaises(ValueError):
            build_prototypes([(0.0,)], [0, 1], 1)

    def test_assign_nearest(self):
        protos = ((0, (0.0,)), (1, (10.0,)))
        labels = assign([(1.0,), (9.0,), (5.0,)], protos)
        # 5.0 ties? equidistant to 0 and 10 -> 5 distance each; lower label wins
        self.assertEqual(labels, (0, 1, 0))

    def test_assign_tie_breaks_low_label(self):
        protos = ((0, (0.0,)), (1, (2.0,)))
        self.assertEqual(assign([(1.0,)], protos), (0,))

    def test_assign_no_prototypes(self):
        with self.assertRaises(ValueError):
            assign([(0.0,)], ())

    def test_segment_end_to_end(self):
        support = [(0.0, 0.0), (0.1, 0.0), (9.0, 9.0), (9.1, 9.0)]
        labels = [0, 0, 1, 1]
        query = [(0.2, 0.1), (8.9, 9.2)]
        self.assertEqual(segment(support, labels, query), (0, 1))

    def test_segment_multi_prototype_background(self):
        # Background class 0 has two separated blobs; multi-proto captures both.
        support = [(0.0,), (0.1,), (20.0,), (20.1,), (10.0,)]
        labels = [0, 0, 0, 0, 1]
        query = [(0.05,), (19.9,), (10.1,)]
        out = segment(support, labels, query, n_prototypes=2, seed=0)
        self.assertEqual(out, (0, 0, 1))


if __name__ == "__main__":
    unittest.main()
