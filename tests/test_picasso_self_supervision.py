"""Tests for drawings.picasso_self_supervision."""

from __future__ import annotations

import random
import unittest

from drawings.picasso_rasterizer import Circle, Line, rasterize
from drawings.picasso_self_supervision import (
    SelfSupervisionPair,
    generate_synthetic_sketch,
    make_self_supervision_dataset,
    make_self_supervision_pair,
    refine_by_rendering,
    render_consistency_loss,
    srn_training_pair,
)


class TestSyntheticGeneration(unittest.TestCase):
    def test_count_and_determinism(self):
        a = generate_synthetic_sketch(random.Random(7), n_primitives=5)
        b = generate_synthetic_sketch(random.Random(7), n_primitives=5)
        self.assertEqual(len(a), 5)
        self.assertEqual(a, b)

    def test_different_seed_differs(self):
        a = generate_synthetic_sketch(random.Random(1), n_primitives=6)
        b = generate_synthetic_sketch(random.Random(2), n_primitives=6)
        self.assertNotEqual(a, b)

    def test_in_bounds(self):
        prims = generate_synthetic_sketch(random.Random(3), n_primitives=8,
                                          margin=0.1)
        for p in prims:
            for attr in ("start", "end", "center", "mid", "pos"):
                pt = getattr(p, attr, None)
                if pt is not None:
                    self.assertGreaterEqual(pt[0], 0.0)
                    self.assertLessEqual(pt[0], 1.0)

    def test_empty(self):
        self.assertEqual(generate_synthetic_sketch(random.Random(0), 0), [])

    def test_only_lines(self):
        prims = generate_synthetic_sketch(random.Random(5), 4, types=("line",))
        self.assertTrue(all(isinstance(p, Line) for p in prims))


class TestSrnPair(unittest.TestCase):
    def test_pair_returns_image(self):
        prims = [Line((0.1, 0.1), (0.9, 0.9))]
        p, img = srn_training_pair(prims, width=16, height=16)
        self.assertEqual(p, prims)
        self.assertEqual(len(img), 16)


class TestSelfSupervisionPair(unittest.TestCase):
    def test_zero_loss_on_truth(self):
        prims = [Circle((0.5, 0.5), 0.3), Line((0.1, 0.1), (0.9, 0.5))]
        pair = make_self_supervision_pair(prims, width=32, height=32)
        self.assertAlmostEqual(pair.loss(prims), 0.0)

    def test_wrong_candidate_higher_loss(self):
        truth = [Circle((0.5, 0.5), 0.3)]
        wrong = [Circle((0.3, 0.3), 0.1)]
        pair = make_self_supervision_pair(truth, width=32, height=32)
        self.assertGreater(pair.loss(wrong), pair.loss(truth))

    def test_target_only_no_param_leak(self):
        # The contract: loss uses only the target image; verify a pair built from
        # a raw image (no hidden truth) still scores candidates.
        truth = [Line((0.0, 0.5), (1.0, 0.5))]
        img = rasterize(truth, 24, 24)
        pair = SelfSupervisionPair(target=img, width=24, height=24)
        self.assertAlmostEqual(pair.loss(truth), 0.0)
        self.assertEqual(pair._hidden_truth, [])


class TestDataset(unittest.TestCase):
    def test_deterministic_dataset(self):
        d1 = make_self_supervision_dataset(random.Random(11), 3, width=16, height=16)
        d2 = make_self_supervision_dataset(random.Random(11), 3, width=16, height=16)
        self.assertEqual(len(d1), 3)
        self.assertEqual([p.target for p in d1], [p.target for p in d2])

    def test_empty_dataset(self):
        self.assertEqual(
            make_self_supervision_dataset(random.Random(0), 0), []
        )


class TestRenderConsistencyAndRefine(unittest.TestCase):
    def test_consistency_loss_zero(self):
        prims = [Line((0.2, 0.2), (0.8, 0.8))]
        img = rasterize(prims, 32, 32)
        self.assertAlmostEqual(
            render_consistency_loss(prims, img, 32, 32), 0.0
        )

    def test_refine_reduces_loss(self):
        truth = [Circle((0.5, 0.5), 0.25)]
        target = rasterize(truth, 40, 40)
        # Start from a shifted circle.
        start = [Circle((0.6, 0.6), 0.25)]
        start_loss = render_consistency_loss(start, target, 40, 40)
        refined, refined_loss = refine_by_rendering(
            start, target, width=40, height=40, steps=30, step_size=0.05
        )
        self.assertLess(refined_loss, start_loss)

    def test_refine_no_labels_needed(self):
        # refine_by_rendering signature takes only candidate + target image.
        truth = [Line((0.3, 0.5), (0.7, 0.5))]
        target = rasterize(truth, 32, 32)
        start = [Line((0.3, 0.6), (0.7, 0.6))]
        refined, loss = refine_by_rendering(
            start, target, width=32, height=32, steps=25, step_size=0.05
        )
        self.assertLessEqual(
            loss, render_consistency_loss(start, target, 32, 32) + 1e-9
        )


if __name__ == "__main__":
    unittest.main()
