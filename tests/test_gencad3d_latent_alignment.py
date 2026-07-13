"""Tests for bench.gencad3d_latent_alignment (multimodal latent alignment)."""

import random
import unittest

from harnesscad.eval.bench.retrieval.latent_alignment import (
    AlignmentQuality,
    LinearAlignment,
    alignment_improvement,
    alignment_quality,
    cross_modal_topk_accuracy,
    fit_linear_alignment,
)


def _paired_latents(n=12, d=4, seed=0, noise=0.0, rotate=False):
    """Paired (geometry, cad) latents. If rotate, geometry is a linear transform
    of cad (misaligned but linearly recoverable); else geometry ~= cad + noise."""
    rng = random.Random(seed)
    cad = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
    geom = []
    for c in cad:
        if rotate:
            # simple shift/scale per dimension -> linear map recovers alignment
            g = [2.0 * c[i] + 0.5 * c[(i + 1) % d] for i in range(d)]
        else:
            g = [c[i] + rng.uniform(-noise, noise) for i in range(d)]
        geom.append(g)
    return geom, cad


class LinearAlignmentTests(unittest.TestCase):
    def test_recovers_identity_map(self):
        geom, cad = _paired_latents(n=20, d=3, seed=1, noise=0.0)  # geom == cad
        model = fit_linear_alignment(geom, cad)
        self.assertIsInstance(model, LinearAlignment)
        # projecting geom should reproduce cad closely
        proj = model.apply(geom[0])
        for a, b in zip(proj, cad[0]):
            self.assertAlmostEqual(a, b, places=3)

    def test_recovers_linear_transform(self):
        geom, cad = _paired_latents(n=30, d=4, seed=2, rotate=True)
        model = fit_linear_alignment(geom, cad, ridge=1e-9)
        proj = model.apply_all(geom)
        # aligned projections should match cad far better than raw geometry
        err_after = sum(abs(p[j] - cad[i][j])
                        for i, p in enumerate(proj) for j in range(4))
        err_before = sum(abs(geom[i][j] - cad[i][j])
                         for i in range(len(geom)) for j in range(4))
        self.assertLess(err_after, err_before)

    def test_deterministic(self):
        geom, cad = _paired_latents(seed=3)
        m1 = fit_linear_alignment(geom, cad)
        m2 = fit_linear_alignment(geom, cad)
        self.assertEqual(m1.weight, m2.weight)

    def test_dim_mismatch_apply(self):
        geom, cad = _paired_latents(d=4, seed=4)
        model = fit_linear_alignment(geom, cad)
        with self.assertRaises(ValueError):
            model.apply([1.0, 2.0])

    def test_unpaired_raises(self):
        with self.assertRaises(ValueError):
            fit_linear_alignment([[1.0, 2.0]], [[1.0], [2.0]])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            fit_linear_alignment([], [])


class AlignmentQualityTests(unittest.TestCase):
    def test_perfectly_aligned(self):
        geom, cad = _paired_latents(n=10, d=3, seed=5, noise=0.0)  # identical
        q = alignment_quality(geom, cad)
        self.assertIsInstance(q, AlignmentQuality)
        self.assertAlmostEqual(q.top1_accuracy, 1.0)
        self.assertAlmostEqual(q.mean_reciprocal_rank, 1.0)
        self.assertAlmostEqual(q.mean_paired_cosine, 1.0, places=6)
        self.assertGreater(q.margin, 0.0)

    def test_margin_positive_when_aligned(self):
        geom, cad = _paired_latents(n=15, d=4, seed=6, noise=0.05)
        q = alignment_quality(geom, cad)
        self.assertGreater(q.margin, 0.0)
        self.assertGreater(q.top1_accuracy, 0.5)

    def test_to_dict(self):
        geom, cad = _paired_latents(seed=7)
        d = alignment_quality(geom, cad).to_dict()
        self.assertIn("margin", d)
        self.assertIn("top1_accuracy", d)

    def test_unpaired_raises(self):
        with self.assertRaises(ValueError):
            alignment_quality([[1.0]], [[1.0], [2.0]])


class CrossModalTopkTests(unittest.TestCase):
    def test_top1_matches_quality(self):
        geom, cad = _paired_latents(n=12, d=4, seed=8, noise=0.05)
        acc1 = cross_modal_topk_accuracy(geom, cad, k=1)
        q = alignment_quality(geom, cad)
        self.assertAlmostEqual(acc1, q.top1_accuracy)

    def test_topk_monotonic(self):
        geom, cad = _paired_latents(n=12, d=4, seed=9, noise=0.3)
        a1 = cross_modal_topk_accuracy(geom, cad, k=1)
        a3 = cross_modal_topk_accuracy(geom, cad, k=3)
        self.assertLessEqual(a1, a3)
        self.assertLessEqual(a3, 1.0)

    def test_topn_is_one(self):
        geom, cad = _paired_latents(n=8, d=3, seed=10, noise=0.5)
        self.assertAlmostEqual(cross_modal_topk_accuracy(geom, cad, k=8), 1.0)

    def test_bad_k(self):
        geom, cad = _paired_latents(n=5, seed=11)
        with self.assertRaises(ValueError):
            cross_modal_topk_accuracy(geom, cad, k=99)


class AlignmentImprovementTests(unittest.TestCase):
    def test_linear_map_improves_alignment(self):
        # misaligned-but-linearly-recoverable modalities
        geom, cad = _paired_latents(n=40, d=4, seed=12, rotate=True)
        report = alignment_improvement(geom, cad, ridge=1e-9)
        self.assertGreaterEqual(report["margin_delta"], 0.0)
        # after alignment top-1 should be at least as good as before
        self.assertGreaterEqual(report["after"]["top1_accuracy"],
                                report["before"]["top1_accuracy"])

    def test_report_structure(self):
        geom, cad = _paired_latents(n=10, seed=13, noise=0.1)
        report = alignment_improvement(geom, cad)
        self.assertIn("before", report)
        self.assertIn("after", report)
        self.assertIn("margin_delta", report)


if __name__ == "__main__":
    unittest.main()
