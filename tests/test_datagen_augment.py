"""Tests for datagen/augment.py — deterministic parametric augmentation.

Each augmented variant must (a) be produced deterministically from (sample, seed),
(b) keep a structurally identical, still-buildable op stream, and (c) re-verify on
a fresh StubBackend session. Dependency-free (StubBackend only).
"""

import unittest

from harnesscad.io.backends.stub import StubBackend
from harnesscad.data.datagen import generate_dataset
from harnesscad.data.datagen.parametric_augment import augment
from harnesscad.data.datagen.pipeline import Sample
from harnesscad.core.loop import HarnessSession


def _op_tags(ops):
    return [o["op"] for o in ops]


class TestAugment(unittest.TestCase):
    def setUp(self):
        # One verified sample of each default family to augment.
        self.samples = generate_dataset(3, seed=5, backend_factory=StubBackend)
        self.assertTrue(self.samples)

    def test_produces_fixed_number_of_variants(self):
        for s in self.samples:
            variants = augment(s, seed=7)
            self.assertEqual(len(variants), 5)
            for v in variants:
                self.assertIsInstance(v, Sample)

    def test_deterministic(self):
        for s in self.samples:
            a = [v.to_dict() for v in augment(s, seed=13)]
            b = [v.to_dict() for v in augment(s, seed=13)]
            self.assertEqual(a, b)

    def test_seed_changes_perturbations(self):
        s = self.samples[0]
        a = [v.to_dict() for v in augment(s, seed=1)]
        b = [v.to_dict() for v in augment(s, seed=2)]
        # The mirror/rotate variants are seed-independent, but the two perturb
        # variants must differ between seeds.
        self.assertNotEqual(a, b)

    def test_structure_preserved(self):
        for s in self.samples:
            for v in augment(s, seed=9):
                self.assertEqual(_op_tags(v.ops), _op_tags(s.ops))

    def test_variants_rebuild_on_fresh_backend(self):
        for s in self.samples:
            for v in augment(s, seed=21):
                session = HarnessSession(StubBackend())
                result = session.apply_ops(v.reference_ops())
                self.assertTrue(result.ok,
                                f"{v.summary.get('augmentation')} failed to build")

    def test_variant_records_lineage(self):
        s = self.samples[0]
        variants = augment(s, seed=4)
        kinds = [v.summary["augmentation"] for v in variants]
        self.assertIn("mirror_x", kinds)
        self.assertIn("mirror_y", kinds)
        self.assertIn("rotate_90", kinds)
        for v in variants:
            self.assertEqual(v.summary["source_generator"], s.generator)

    def test_perturb_changes_dimensions(self):
        s = self.samples[0]
        variants = augment(s, seed=8)
        perturbs = [v for v in variants if v.summary["augmentation"].startswith("perturb")]
        self.assertTrue(perturbs)
        # At least one perturb variant changes a numeric op field vs the source.
        changed = any(v.ops != s.ops for v in perturbs)
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
