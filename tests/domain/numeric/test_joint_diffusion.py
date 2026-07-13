import random
import unittest

from harnesscad.domain.numeric.joint_diffusion import (
    continuous_slice,
    corrupt_primitive,
    corrupt_sketch,
    discrete_slices,
    is_permutation_equivariant,
    joint_posterior_mean,
)
from harnesscad.domain.reconstruction.tokens.sketchdnn import (
    FEATURE_DIM,
    decode_primitive,
    encode_primitive,
)


def _is_simplex(v):
    return abs(sum(v) - 1.0) < 1e-9 and all(x >= 0.0 for x in v)


class TestLayoutHelpers(unittest.TestCase):
    def test_discrete_slices(self):
        self.assertEqual(discrete_slices(), [(0, 2), (2, 7)])

    def test_continuous_slice(self):
        self.assertEqual(continuous_slice(), (7, FEATURE_DIM))


class TestCorruptPrimitive(unittest.TestCase):
    def test_discrete_blocks_stay_simplex(self):
        vec = encode_primitive("LINE", [0.1, 0.2, 0.3, 0.4])
        out = corrupt_primitive(vec, 0.5, random.Random(0))
        self.assertTrue(_is_simplex(out[0:2]))
        self.assertTrue(_is_simplex(out[2:7]))

    def test_length_preserved(self):
        vec = encode_primitive("CIRCLE", [1.0, 2.0, 0.5])
        out = corrupt_primitive(vec, 0.5, random.Random(1))
        self.assertEqual(len(out), FEATURE_DIM)

    def test_deterministic(self):
        vec = encode_primitive("ARC", [0, 0, 1, 1, 0.3])
        a = corrupt_primitive(vec, 0.4, random.Random(9))
        b = corrupt_primitive(vec, 0.4, random.Random(9))
        self.assertEqual(a, b)

    def test_low_noise_preserves_class(self):
        vec = encode_primitive("POINT", [3.0, 4.0])
        rng = random.Random(5)
        keeps = 0
        for _ in range(100):
            out = corrupt_primitive(vec, 0.999, rng)
            _, cls, _ = decode_primitive(out)
            if cls == "POINT":
                keeps += 1
        self.assertGreater(keeps, 90)

    def test_continuous_low_noise_close_to_clean(self):
        vec = encode_primitive("LINE", [0.5, 0.5, 0.5, 0.5])
        out = corrupt_primitive(vec, 0.9999, random.Random(2))
        cs, ce = continuous_slice()
        for i in range(cs, ce):
            if vec[i] != 0.0:
                self.assertAlmostEqual(out[i], vec[i], delta=0.1)

    def test_bad_alpha_bar(self):
        vec = encode_primitive("POINT", [0.0, 0.0])
        with self.assertRaises(ValueError):
            corrupt_primitive(vec, 1.5, random.Random(0))


class TestCorruptSketch(unittest.TestCase):
    def test_corrupt_sketch_shapes(self):
        prims = [
            encode_primitive("LINE", [0, 0, 1, 1]),
            encode_primitive("CIRCLE", [0, 0, 1]),
        ]
        out = corrupt_sketch(prims, 0.5, [1, 2])
        self.assertEqual(len(out), 2)
        for row in out:
            self.assertEqual(len(row), FEATURE_DIM)

    def test_seed_length_mismatch(self):
        prims = [encode_primitive("POINT", [0, 0])]
        with self.assertRaises(ValueError):
            corrupt_sketch(prims, 0.5, [1, 2])


class TestPermutationEquivariance(unittest.TestCase):
    def setUp(self):
        self.prims = [
            encode_primitive("LINE", [0.1, 0.2, 0.3, 0.4]),
            encode_primitive("CIRCLE", [0.5, 0.6, 0.2]),
            encode_primitive("ARC", [0.0, 0.1, 0.2, 0.3, 0.4]),
        ]
        self.seeds = [11, 22, 33]

    def test_equivariant_reverse_perm(self):
        self.assertTrue(
            is_permutation_equivariant(self.prims, self.seeds, [2, 0, 1], 0.5)
        )

    def test_equivariant_identity_perm(self):
        self.assertTrue(
            is_permutation_equivariant(self.prims, self.seeds, [0, 1, 2], 0.7)
        )

    def test_equivariant_swap(self):
        self.assertTrue(
            is_permutation_equivariant(self.prims, self.seeds, [1, 0, 2], 0.3)
        )

    def test_bad_perm(self):
        with self.assertRaises(ValueError):
            is_permutation_equivariant(self.prims, self.seeds, [0, 0, 1], 0.5)


class TestJointPosteriorMean(unittest.TestCase):
    def test_discrete_blocks_simplex(self):
        v0 = encode_primitive("CIRCLE", [0.1, 0.2, 0.3])
        vt = corrupt_primitive(v0, 0.5, random.Random(4))
        mean = joint_posterior_mean(vt, v0, 0.8, 0.4, 0.5)
        self.assertTrue(_is_simplex(mean[0:2]))
        self.assertTrue(_is_simplex(mean[2:7]))

    def test_continuous_interpolates(self):
        v0 = encode_primitive("LINE", [1.0, 1.0, 1.0, 1.0])
        vt = list(v0)
        cs, ce = continuous_slice()
        for i in range(cs, ce):
            vt[i] = 0.0
        mean = joint_posterior_mean(vt, v0, 0.8, 0.4, 0.5)
        # posterior mean of continuous is a convex-ish blend between x0 and xt
        for i in range(cs, ce):
            if v0[i] == 1.0:
                self.assertGreater(mean[i], 0.0)
                self.assertLess(mean[i], 1.0)

    def test_length_preserved(self):
        v0 = encode_primitive("POINT", [0.5, 0.5])
        vt = corrupt_primitive(v0, 0.5, random.Random(0))
        mean = joint_posterior_mean(vt, v0, 0.8, 0.4, 0.5)
        self.assertEqual(len(mean), FEATURE_DIM)


if __name__ == "__main__":
    unittest.main()
