import math
import unittest

from harnesscad.io.ingest.orientation import (
    OrientationDistribution,
    OrientationMode,
    Rotation,
    angular_distance,
    coarse_to_fine_samples,
    product_of_experts,
    reference_cube_corners,
    symmetry_equivalents,
)


class RotationTests(unittest.TestCase):
    def test_normalizes_and_canonicalizes_antipodes(self):
        self.assertEqual(Rotation(2, 0, 0, 0), Rotation(-2, 0, 0, 0))

    def test_rejects_invalid_rotations(self):
        with self.assertRaises(ValueError):
            Rotation(0, 0, 0, 0)
        with self.assertRaises(ValueError):
            Rotation(float("nan"), 0, 0, 0)
        with self.assertRaises(ValueError):
            Rotation.from_axis_angle((0, 0, 0), 1)
        with self.assertRaises(ValueError):
            Rotation.from_axis_angle((1, 2), 1)

    def test_composition_and_angular_distance(self):
        quarter = Rotation.from_axis_angle((0, 0, 1), math.pi / 2)
        half = quarter.compose(quarter)
        self.assertAlmostEqual(angular_distance(Rotation.identity(), half), math.pi)
        self.assertAlmostEqual(angular_distance(quarter, quarter), 0)
        self.assertAlmostEqual(half.apply((1, 0, 0))[0], -1)

    def test_reference_cube_encoding(self):
        corners = reference_cube_corners(Rotation.identity())
        self.assertEqual(len(corners), 8)
        self.assertEqual(corners[0], (-0.5, -0.5, -0.5))
        self.assertEqual(len(set(corners)), 8)

    def test_explicit_symmetry_modes_are_deduplicated(self):
        half = Rotation.from_axis_angle((0, 0, 1), math.pi)
        modes = symmetry_equivalents(
            Rotation.identity(), [Rotation.identity(), half, half]
        )
        self.assertEqual(len(modes), 2)


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.identity = Rotation.identity()
        self.half = Rotation.from_axis_angle((0, 0, 1), math.pi)

    def test_normalization_and_diagnostics(self):
        distribution = OrientationDistribution(
            (
                OrientationMode(self.identity, 3, "upright"),
                OrientationMode(self.half, 1, "inverted"),
            )
        )
        self.assertAlmostEqual(sum(m.weight for m in distribution.modes), 1)
        self.assertAlmostEqual(distribution.confidence, 0.75)
        self.assertEqual(distribution.best.label, "upright")
        self.assertGreater(distribution.entropy, 0)
        self.assertGreater(distribution.ambiguity, 0)
        self.assertLess(distribution.ambiguity, 1)

    def test_distribution_validation(self):
        with self.assertRaises(ValueError):
            OrientationDistribution(())
        with self.assertRaises(ValueError):
            OrientationDistribution((OrientationMode(self.identity, 0),))
        with self.assertRaises(ValueError):
            OrientationMode(self.identity, -1)

    def test_product_of_named_experts(self):
        candidates = [
            OrientationMode(self.identity, 1, "upright"),
            OrientationMode(self.half, 1, "inverted"),
        ]
        result = product_of_experts(
            candidates,
            {
                "gravity": lambda rotation: 4
                if angular_distance(rotation, self.identity) < 0.1
                else 1,
                "image": lambda rotation: 2
                if angular_distance(rotation, self.identity) < 0.1
                else 3,
            },
        )
        self.assertEqual(result.best.label, "upright")
        self.assertAlmostEqual(result.modes[0].weight, 8 / 11)

    def test_expert_validation(self):
        mode = [OrientationMode(self.identity, 1)]
        with self.assertRaises(ValueError):
            product_of_experts(mode, {"bad": lambda _: -1})
        with self.assertRaises(ValueError):
            product_of_experts(mode, {"zero": lambda _: 0})
        with self.assertRaises(ValueError):
            product_of_experts(mode, {"ok": lambda _: 1}, expert_weights={"x": 2})


class SamplerTests(unittest.TestCase):
    def test_seed_replays_exactly_and_records_provenance(self):
        distribution = OrientationDistribution(
            (
                OrientationMode(Rotation.identity(), 2),
                OrientationMode(
                    Rotation.from_axis_angle((1, 0, 0), math.pi), 1
                ),
            )
        )
        first = coarse_to_fine_samples(
            distribution, seed=91, coarse_count=3, fine_per_mode=2
        )
        second = coarse_to_fine_samples(
            distribution, seed=91, coarse_count=3, fine_per_mode=2
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first.rotations), 3 + 2 * (1 + 2))
        self.assertEqual(first.provenance.seed, 91)
        self.assertEqual(len(first.provenance.distribution_digest), 64)
        # The exact mode is emitted before its local refinements.
        self.assertEqual(first.rotations[3], distribution.modes[0].rotation)
        self.assertLessEqual(
            angular_distance(first.rotations[3], first.rotations[4]),
            first.provenance.fine_radius_radians + 1e-12,
        )

    def test_sampler_validation(self):
        distribution = OrientationDistribution(
            (OrientationMode(Rotation.identity(), 1),)
        )
        with self.assertRaises(TypeError):
            coarse_to_fine_samples(distribution, seed=True)
        with self.assertRaises(ValueError):
            coarse_to_fine_samples(distribution, seed=1, coarse_count=-1)
        with self.assertRaises(ValueError):
            coarse_to_fine_samples(
                distribution, seed=1, fine_radius_radians=math.pi + 0.1
            )


if __name__ == "__main__":
    unittest.main()
