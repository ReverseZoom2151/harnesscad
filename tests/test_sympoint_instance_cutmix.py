import unittest

from drawings.sympoint_instance_cutmix import (
    InstanceQueue,
    cutmix,
    extract_instances,
    hflip,
    rotate,
    scale,
    shift,
    shuffle,
    vflip,
)
from drawings.sympoint_point_features import build_point_cloud, polar_angle


def make_sample():
    # two thing points (class 1, instance 0), one thing point (class 2, instance 1),
    # one stuff point (class 32 wall), one unlabelled thing point
    return build_point_cloud(
        points=[(10.0, 10.0), (20.0, 20.0), (30.0, 10.0), (40.0, 40.0), (50.0, 50.0)],
        lengths=[1.0, 2.0, 3.0, 100.0, 5.0],
        commands=[0, 0, 2, 0, 1],
        semantic_ids=[1, 1, 2, 32, 1],
        instance_ids=[0, 0, 1, -1, -1],
        norm="none",
    )


class TestGeometricAug(unittest.TestCase):
    def setUp(self):
        self.sample = build_point_cloud([(0.25, 0.5)], [1.0], [0], [1], [0], scale=1.0,
                                        norm="none")

    def test_hflip(self):
        out = hflip(self.sample, width=1.0)
        self.assertAlmostEqual(out["coords"][0][0], 0.75)
        self.assertAlmostEqual(out["coords"][0][1], 0.5)

    def test_vflip(self):
        out = vflip(self.sample, height=1.0)
        self.assertAlmostEqual(out["coords"][0][1], 0.5)
        out2 = vflip(build_point_cloud([(0.25, 0.25)], [1.0], [0], [1], [0], scale=1.0,
                                       norm="none"))
        self.assertAlmostEqual(out2["coords"][0][1], 0.75)

    def test_rotate_180_about_center(self):
        out = rotate(self.sample, 180.0)
        self.assertAlmostEqual(out["coords"][0][0], 0.75, places=6)
        self.assertAlmostEqual(out["coords"][0][1], 0.5, places=6)

    def test_rotate_360_identity(self):
        out = rotate(self.sample, 360.0)
        self.assertAlmostEqual(out["coords"][0][0], 0.25, places=6)

    def test_polar_feature_refreshed_by_aug(self):
        out = hflip(self.sample, width=1.0)
        self.assertAlmostEqual(out["features"][0][0], polar_angle(out["coords"][0]))
        self.assertNotAlmostEqual(out["features"][0][0], self.sample["features"][0][0])

    def test_shift(self):
        out = shift(self.sample, 1.0, -1.0)
        self.assertAlmostEqual(out["coords"][0][0], 1.25)
        self.assertAlmostEqual(out["coords"][0][1], -0.5)

    def test_scale_scales_length_channel(self):
        base = build_point_cloud([(10.0, 10.0)], [70.0], [0], [1], [0], norm="none")
        out = scale(base, 2.0)
        self.assertAlmostEqual(out["features"][0][1], 1.0)
        self.assertAlmostEqual(out["coords"][0][0], base["coords"][0][0] * 2)

    def test_labels_preserved(self):
        out = rotate(self.sample, 33.0)
        self.assertEqual(out["semantic_ids"], self.sample["semantic_ids"])
        self.assertEqual(out["lengths"], self.sample["lengths"])


class TestExtractInstances(unittest.TestCase):
    def test_only_thing_instances(self):
        insts = extract_instances(make_sample())
        self.assertEqual(len(insts), 2)
        self.assertEqual(insts[0]["semantic_ids"], [1, 1])
        self.assertEqual(insts[1]["semantic_ids"], [2])

    def test_stuff_and_unlabelled_skipped(self):
        insts = extract_instances(make_sample())
        harvested = [s for inst in insts for s in inst["semantic_ids"]]
        self.assertNotIn(32, harvested)
        self.assertEqual(len(harvested), 3)

    def test_sorted_deterministic(self):
        a = extract_instances(make_sample())
        b = extract_instances(make_sample())
        self.assertEqual(a, b)

    def test_custom_stuff_start(self):
        insts = extract_instances(make_sample(), stuff_start=2)
        self.assertEqual(len(insts), 1)
        self.assertEqual(insts[0]["semantic_ids"], [1, 1])


class TestInstanceQueue(unittest.TestCase):
    def test_bounded_fifo_newest_first(self):
        q = InstanceQueue(capacity=2)
        for i in range(3):
            q.push({"coords": [(float(i), 0.0)], "features": [[0.0] * 6],
                    "semantic_ids": [1], "instance_ids": [i], "lengths": [1.0]})
        self.assertEqual(len(q), 2)
        ids = [inst["instance_ids"][0] for inst in q]
        self.assertEqual(ids, [2, 1])

    def test_push_sample(self):
        q = InstanceQueue(capacity=10)
        q.push_sample(make_sample())
        self.assertEqual(len(q), 2)

    def test_zero_capacity(self):
        q = InstanceQueue(capacity=0)
        q.push_sample(make_sample())
        self.assertEqual(len(q), 0)

    def test_bad_capacity(self):
        with self.assertRaises(ValueError):
            InstanceQueue(capacity=-1)


class TestCutmix(unittest.TestCase):
    def test_paste_appends_points(self):
        sample = make_sample()
        q = InstanceQueue(capacity=10).push_sample(sample)
        out = cutmix(sample, q, 100.0, 0.0)
        self.assertEqual(len(out["coords"]), len(sample["coords"]) + 3)

    def test_pasted_points_are_shifted(self):
        sample = make_sample()
        q = InstanceQueue(capacity=10).push_sample(sample)
        out = cutmix(sample, q, 100.0, 5.0)
        pasted = out["coords"][len(sample["coords"]):]
        # queue is newest-first: instance (2,1) first, then (1,0); coords are
        # normalised by COORD_SCALE inside build_point_cloud
        self.assertAlmostEqual(pasted[0][0], 30.0 / 140.0 + 100.0)
        self.assertAlmostEqual(pasted[0][1], 10.0 / 140.0 + 5.0)

    def test_pasted_labels_carry_over(self):
        sample = make_sample()
        q = InstanceQueue(capacity=10).push_sample(sample)
        out = cutmix(sample, q, 100.0, 0.0)
        self.assertEqual(out["semantic_ids"][len(sample["coords"]):], [2, 1, 1])
        self.assertEqual(out["lengths"][len(sample["coords"]):], [3.0, 1.0, 2.0])

    def test_polar_feature_recomputed_everywhere(self):
        sample = make_sample()
        q = InstanceQueue(capacity=10).push_sample(sample)
        out = cutmix(sample, q, 100.0, 0.0)
        for feat, pt in zip(out["features"], out["coords"]):
            self.assertAlmostEqual(feat[0], polar_angle(pt))

    def test_empty_queue_is_identity(self):
        sample = make_sample()
        out = cutmix(sample, InstanceQueue(capacity=4), 10.0, 10.0)
        self.assertEqual(out["coords"], sample["coords"])
        self.assertEqual(out["semantic_ids"], sample["semantic_ids"])

    def test_deterministic(self):
        sample = make_sample()
        q1 = InstanceQueue(capacity=10).push_sample(make_sample())
        q2 = InstanceQueue(capacity=10).push_sample(make_sample())
        self.assertEqual(cutmix(sample, q1, 3.0, 4.0), cutmix(sample, q2, 3.0, 4.0))


class TestShuffle(unittest.TestCase):
    def test_permutation_keeps_channels_aligned(self):
        sample = make_sample()
        out = shuffle(sample, seed=7)
        self.assertEqual(sorted(out["semantic_ids"]), sorted(sample["semantic_ids"]))
        for pt, sem in zip(out["coords"], out["semantic_ids"]):
            idx = sample["coords"].index(pt)
            self.assertEqual(sample["semantic_ids"][idx], sem)

    def test_seeded_reproducible(self):
        sample = make_sample()
        self.assertEqual(shuffle(sample, 3), shuffle(sample, 3))

    def test_different_seeds_differ(self):
        sample = make_sample()
        orders = {tuple(shuffle(sample, s)["lengths"]) for s in range(8)}
        self.assertGreater(len(orders), 1)


if __name__ == "__main__":
    unittest.main()
