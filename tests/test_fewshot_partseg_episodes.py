import unittest
from harnesscad.domain.reconstruction.recognize.fewshot_partseg_episodes import (
    Sample, Episode, BACKGROUND, split_classes, remap_labels,
    build_episode, flatten_support,
)


def _sample(cls, n=3):
    feats = tuple((hash(cls) % 100 + 0.0, float(i)) for i in range(n))
    labs = tuple(cls for _ in range(n))
    return Sample(cls, feats, labs)


def _dataset(classes, per_class=4):
    return {c: [_sample(c) for _ in range(per_class)] for c in classes}


class Tests(unittest.TestCase):
    def test_split_disjoint_and_covers(self):
        train, test = split_classes(["a", "b", "c", "d", "e"], 3, seed=0)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 2)
        self.assertEqual(set(train) & set(test), set())
        self.assertEqual(set(train) | set(test), {"a", "b", "c", "d", "e"})

    def test_split_deterministic(self):
        self.assertEqual(split_classes(range(6), 3, seed=7),
                         split_classes(range(6), 3, seed=7))

    def test_split_out_of_range(self):
        with self.assertRaises(ValueError):
            split_classes(["a", "b"], 5)

    def test_remap_labels_background(self):
        lm = {"hole": 1, "chamfer": 2}
        out = remap_labels(("hole", "plane", "chamfer", "plane"), lm)
        self.assertEqual(out, (1, BACKGROUND, 2, BACKGROUND))

    def test_build_episode_structure(self):
        ds = _dataset(["hole", "pocket", "chamfer"], per_class=5)
        ep = build_episode(ds, ways=2, shots=1, queries=2, seed=1)
        self.assertIsInstance(ep, Episode)
        self.assertEqual(len(ep.ways), 2)
        self.assertEqual(len(ep.support), 2 * 1)
        self.assertEqual(len(ep.query), 2 * 2)
        # label map assigns 1..C
        self.assertEqual(sorted(ep.label_map.values()), [1, 2])

    def test_build_episode_remaps_points(self):
        ds = _dataset(["hole", "pocket", "chamfer"], per_class=5)
        ep = build_episode(ds, ways=2, shots=1, queries=1, seed=2)
        chosen = set(ep.ways)
        for s in ep.support + ep.query:
            for lab in s.point_labels:
                if s.cls in chosen:
                    self.assertEqual(lab, ep.label_map[s.cls])

    def test_build_episode_deterministic(self):
        ds = _dataset(["a", "b", "c", "d"], per_class=6)
        a = build_episode(ds, 2, 2, 2, seed=5)
        b = build_episode(ds, 2, 2, 2, seed=5)
        self.assertEqual(a.ways, b.ways)
        self.assertEqual([s.features for s in a.support],
                         [s.features for s in b.support])

    def test_build_episode_too_many_ways(self):
        ds = _dataset(["a", "b"], per_class=5)
        with self.assertRaises(ValueError):
            build_episode(ds, ways=3, shots=1, queries=1)

    def test_build_episode_too_few_samples(self):
        ds = _dataset(["a", "b"], per_class=2)
        with self.assertRaises(ValueError):
            build_episode(ds, ways=2, shots=2, queries=2)

    def test_flatten_support(self):
        ds = _dataset(["a", "b"], per_class=4)
        ep = build_episode(ds, ways=2, shots=2, queries=1, seed=0)
        feats, labs = flatten_support(ep)
        # 2 ways * 2 shots * 3 points each = 12 points
        self.assertEqual(len(feats), 12)
        self.assertEqual(len(labs), 12)


if __name__ == "__main__":
    unittest.main()
