import math
import unittest

from drawings.sympoint_point_features import (
    BACKGROUND_SEMANTIC_ID,
    COORD_SCALE,
    FEATURE_DIM,
    STUFF_INSTANCE_ID,
    build_batch,
    build_point_cloud,
    center_coords,
    command_onehot,
    normalize_args,
    normalized_length,
    pad_to_min_points,
    point_feature,
    polar_angle,
    recompute_polar_feature,
    scale_coords,
)


class TestChannels(unittest.TestCase):
    def test_polar_angle_diagonal(self):
        self.assertAlmostEqual(polar_angle((1.0, 1.0)), 0.25)
        self.assertAlmostEqual(polar_angle((1.0, -1.0)), -0.25)

    def test_polar_angle_identifies_antipodes(self):
        self.assertAlmostEqual(polar_angle((2.0, 3.0)), polar_angle((-2.0, -3.0)), places=6)

    def test_polar_angle_bounded(self):
        self.assertTrue(-0.5 < polar_angle((0.0, 5.0)) < 0.5)

    def test_normalized_length_clips(self):
        self.assertAlmostEqual(normalized_length(70.0), 0.5)
        self.assertAlmostEqual(normalized_length(1e6), 1.0)
        self.assertAlmostEqual(normalized_length(-3.0), 0.0)

    def test_normalized_length_bad_scale(self):
        with self.assertRaises(ValueError):
            normalized_length(1.0, 0.0)

    def test_command_onehot(self):
        self.assertEqual(command_onehot(2), (0.0, 0.0, 1.0, 0.0))
        with self.assertRaises(ValueError):
            command_onehot(4)

    def test_point_feature_shape(self):
        feat = point_feature((1.0, 1.0), 140.0, 0)
        self.assertEqual(len(feat), FEATURE_DIM)
        self.assertAlmostEqual(feat[0], 0.25)
        self.assertAlmostEqual(feat[1], 1.0)
        self.assertEqual(feat[2:], (1.0, 0.0, 0.0, 0.0))

    def test_normalize_args(self):
        out = normalize_args((140.0, 70.0))
        self.assertAlmostEqual(out[0], 1.0)
        self.assertAlmostEqual(out[1], 0.5)
        with self.assertRaises(ValueError):
            normalize_args((1.0,), scale=0.0)


class TestCentering(unittest.TestCase):
    def test_mean(self):
        out = center_coords([(0.0, 0.0), (2.0, 4.0)], "mean")
        self.assertEqual(out, [(-1.0, -2.0), (1.0, 2.0)])

    def test_min(self):
        out = center_coords([(1.0, 5.0), (3.0, 9.0)], "min")
        self.assertEqual(out, [(0.0, 0.0), (2.0, 4.0)])

    def test_none_and_empty(self):
        self.assertEqual(center_coords([(1.0, 2.0)], "none"), [(1.0, 2.0)])
        self.assertEqual(center_coords([], "mean"), [])

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            center_coords([(0.0, 0.0)], "median")


class TestRecompute(unittest.TestCase):
    def test_polar_refreshed_after_shift(self):
        coords = [(1.0, 1.0)]
        feats = [list(point_feature(coords[0], 10.0, 1))]
        moved = [(1.0, -1.0)]
        out = recompute_polar_feature(feats, moved)
        self.assertAlmostEqual(out[0][0], -0.25)
        self.assertAlmostEqual(out[0][1], feats[0][1])
        self.assertEqual(out[0][2:], feats[0][2:])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            recompute_polar_feature([[0.0] * FEATURE_DIM], [(0.0, 0.0), (1.0, 1.0)])

    def test_bad_feature_dim(self):
        with self.assertRaises(ValueError):
            recompute_polar_feature([[0.0, 0.0]], [(1.0, 1.0)])

    def test_scale_coords_scales_length_channel(self):
        coords = [(1.0, 2.0)]
        feats = [list(point_feature(coords[0], 70.0, 0))]
        new_coords, new_feats = scale_coords(coords, feats, 2.0)
        self.assertEqual(new_coords, [(2.0, 4.0)])
        self.assertAlmostEqual(new_feats[0][1], 1.0)
        self.assertAlmostEqual(new_feats[0][0], polar_angle((2.0, 4.0)))


class TestBuildPointCloud(unittest.TestCase):
    def setUp(self):
        self.sample = build_point_cloud(
            points=[(0.0, 0.0), (140.0, 140.0)],
            lengths=[10.0, 200.0],
            commands=[0, 2],
            semantic_ids=[3, 35],
            instance_ids=[1, -1],
        )

    def test_coords_normalised_and_centered(self):
        cx = sum(p[0] for p in self.sample["coords"]) / 2
        self.assertAlmostEqual(cx, 0.0)
        self.assertAlmostEqual(self.sample["coords"][1][0], 0.5)

    def test_features(self):
        feats = self.sample["features"]
        self.assertEqual(len(feats[0]), FEATURE_DIM)
        self.assertAlmostEqual(feats[1][1], 1.0)  # length clipped at scale
        self.assertEqual(feats[1][2:], [0.0, 0.0, 1.0, 0.0])

    def test_raw_lengths_preserved(self):
        self.assertEqual(self.sample["lengths"], [10.0, 200.0])

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            build_point_cloud([(0.0, 0.0)], [1.0, 2.0], [0], [0], [0])

    def test_deterministic(self):
        again = build_point_cloud(
            points=[(0.0, 0.0), (140.0, 140.0)], lengths=[10.0, 200.0],
            commands=[0, 2], semantic_ids=[3, 35], instance_ids=[1, -1])
        self.assertEqual(again, self.sample)


class TestPadAndBatch(unittest.TestCase):
    def setUp(self):
        self.a = build_point_cloud([(0.0, 0.0)], [1.0], [0], [3], [0], norm="none")
        self.b = build_point_cloud([(1.0, 1.0), (2.0, 2.0)], [1.0, 2.0], [1, 3],
                                   [4, 35], [0, -1], norm="none")

    def test_pad(self):
        padded = pad_to_min_points(self.a, 4)
        self.assertEqual(len(padded["coords"]), 4)
        self.assertEqual(padded["semantic_ids"], [3, BACKGROUND_SEMANTIC_ID,
                                                  BACKGROUND_SEMANTIC_ID, BACKGROUND_SEMANTIC_ID])
        self.assertEqual(padded["instance_ids"][-1], STUFF_INSTANCE_ID)
        self.assertEqual(padded["features"][-1], [0.0] * FEATURE_DIM)
        self.assertEqual(padded["lengths"][-1], 0.0)

    def test_pad_noop(self):
        same = pad_to_min_points(self.b, 2)
        self.assertEqual(len(same["coords"]), 2)

    def test_batch_offsets(self):
        batch = build_batch([self.a, self.b], instance_stride=100)
        self.assertEqual(batch["offsets"], [1, 3])
        self.assertEqual(len(batch["coords"]), 3)

    def test_batch_instance_ids_globalised(self):
        batch = build_batch([self.a, self.b], instance_stride=100)
        self.assertEqual(batch["instance_ids"], [0, 100, -1])

    def test_batch_empty(self):
        batch = build_batch([])
        self.assertEqual(batch["offsets"], [])


class TestConstants(unittest.TestCase):
    def test_scale(self):
        self.assertEqual(COORD_SCALE, 140.0)
        self.assertEqual(FEATURE_DIM, 6)
        self.assertAlmostEqual(polar_angle((0.0, 0.0)), 0.0, places=6)
        self.assertLess(abs(polar_angle((1e-9, 1.0))), 0.5 + 1e-9)
        self.assertGreater(math.pi, 3.0)


if __name__ == "__main__":
    unittest.main()
