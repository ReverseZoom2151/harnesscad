"""Tests for datagen.vitruvion_primitive_noise."""

import math
import random
import unittest

from harnesscad.data.datagen.vitruvion_primitive_noise import (
    PrimitiveNoiseConfig,
    noisify_entity,
    noisify_sketch,
    truncated_normal,
)
from harnesscad.domain.geometry.vitruvion_sketch_norm import (
    VArc,
    VCircle,
    VLine,
    VPoint,
    entity_from_params,
    parameterize_entity,
)


class TestTruncatedNormal(unittest.TestCase):
    def test_support_is_scaled_bounds(self):
        rng = random.Random(7)
        for _ in range(500):
            value = truncated_normal(rng, -0.15, 0.15, 0.15)
            self.assertLessEqual(abs(value), 0.15 * 0.15 + 1e-12)

    def test_deterministic_in_seed(self):
        a = [truncated_normal(random.Random(3), -1.0, 1.0, 0.2) for _ in range(1)]
        b = [truncated_normal(random.Random(3), -1.0, 1.0, 0.2) for _ in range(1)]
        self.assertEqual(a, b)

    def test_bad_bounds(self):
        with self.assertRaises(ValueError):
            truncated_normal(random.Random(0), 1.0, 1.0, 0.1)


class TestNoisifyEntity(unittest.TestCase):
    def test_point_moves_within_the_scaled_bound(self):
        point = VPoint(x=0.1, y=-0.2)
        noisy = noisify_entity(point, random.Random(1), std=0.15, max_diff=0.15)
        bound = 0.15 * 0.15
        self.assertLessEqual(abs(noisy.x - 0.1), bound)
        self.assertLessEqual(abs(noisy.y + 0.2), bound)
        self.assertNotEqual((noisy.x, noisy.y), (0.1, -0.2))

    def test_literal_bounds_interpretation_is_wider(self):
        point = VPoint(x=0.0, y=0.0)
        std_units = noisify_entity(point, random.Random(2), std=0.15, max_diff=0.15)
        literal = noisify_entity(
            point, random.Random(2), std=0.15, max_diff=0.15, bounds_in_std_units=False
        )
        # Same rng stream, different truncation -> the literal reading is not clipped
        # to 0.0225 and generally lands further out.
        self.assertLessEqual(abs(std_units.x), 0.0225 + 1e-12)
        self.assertGreater(abs(literal.x), abs(std_units.x))

    def test_input_is_not_mutated(self):
        point = VPoint(x=0.1, y=0.1)
        noisify_entity(point, random.Random(1))
        self.assertEqual((point.x, point.y), (0.1, 0.1))

    def test_construction_flag_survives(self):
        line = entity_from_params([-0.2, -0.2, 0.2, 0.2])
        line.is_construction = True
        noisy = noisify_entity(line, random.Random(4))
        self.assertIsInstance(noisy, VLine)
        self.assertTrue(noisy.is_construction)

    def test_line_endpoints_move(self):
        line = entity_from_params([-0.3, 0.0, 0.3, 0.0])
        noisy = noisify_entity(line, random.Random(5), std=0.15, max_diff=0.15)
        before = parameterize_entity(line)
        after = parameterize_entity(noisy)
        self.assertNotEqual(before, after)
        for x, y in zip(before, after):
            self.assertLessEqual(abs(x - y), 0.15 * 0.15 + 1e-9)

    def test_circle_stays_a_circle(self):
        circle = VCircle(xCenter=0.0, yCenter=0.0, radius=0.3)
        noisy = noisify_entity(circle, random.Random(6))
        self.assertIsInstance(noisy, VCircle)
        self.assertNotAlmostEqual(noisy.radius, 0.3, places=6)

    def test_arc_center_stays_within_max_diff(self):
        arc = VArc(xCenter=0.0, yCenter=0.0, radius=0.3, startParam=0.2,
                   endParam=2.4)
        for seed in range(20):
            noisy = noisify_entity(arc, random.Random(seed), std=0.15, max_diff=0.15)
            self.assertIsInstance(noisy, VArc)
            self.assertLessEqual(abs(noisy.xCenter), 0.15)
            self.assertLessEqual(abs(noisy.yCenter), 0.15)
            self.assertTrue(math.isfinite(noisy.radius))

    def test_zero_std_raises(self):
        with self.assertRaises(ValueError):
            noisify_entity(VPoint(), random.Random(0), std=0.0)

    def test_max_trials_exhausted_returns_original(self):
        arc = VArc(xCenter=0.0, yCenter=0.0, radius=0.3, startParam=0.0, endParam=1.0)
        noisy = noisify_entity(arc, random.Random(0), std=0.15, max_diff=0.15,
                               max_trials=0)
        self.assertEqual(parameterize_entity(noisy), parameterize_entity(arc))


class TestNoisifySketch(unittest.TestCase):
    def setUp(self):
        self.entities = [
            VCircle(xCenter=0.0, yCenter=0.0, radius=0.25),
            entity_from_params([-0.4, -0.1, 0.4, 0.1]),
            VPoint(x=0.2, y=0.2),
        ]

    def test_deterministic_in_seed(self):
        a = noisify_sketch(self.entities, seed=11)
        b = noisify_sketch(self.entities, seed=11)
        self.assertEqual(
            [parameterize_entity(e) for e in a], [parameterize_entity(e) for e in b]
        )

    def test_different_seeds_differ(self):
        a = noisify_sketch(self.entities, seed=1)
        b = noisify_sketch(self.entities, seed=2)
        self.assertNotEqual(
            [parameterize_entity(e) for e in a], [parameterize_entity(e) for e in b]
        )

    def test_disabled_config_is_identity(self):
        out = noisify_sketch(self.entities, seed=1, config=PrimitiveNoiseConfig(enabled=False))
        self.assertEqual(
            [parameterize_entity(e) for e in out],
            [parameterize_entity(e) for e in self.entities],
        )
        self.assertIsNot(out[0], self.entities[0])

    def test_types_preserved(self):
        out = noisify_sketch(self.entities, seed=3)
        self.assertEqual([type(e) for e in out], [VCircle, VLine, VPoint])

    def test_originals_untouched(self):
        noisify_sketch(self.entities, seed=3)
        self.assertEqual(self.entities[0].radius, 0.25)


if __name__ == "__main__":
    unittest.main()
