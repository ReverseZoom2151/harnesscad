import math
import unittest

from numeric.geofusion_state_space import (
    selective_scan, curvature_descriptor, conditioning_vector,
    hierarchical_pe, depthwise_conv1d, geometric_state_mixer,
    film_modulate, gmamba_flops, sigmoid,
)


class TestSelectiveScan(unittest.TestCase):
    def test_zero_input_zero_output(self):
        L, d = 4, 3
        z = tuple((0.0,) * d for _ in range(L))
        a = tuple((0.5,) * d for _ in range(L))
        b = tuple((1.0,) * d for _ in range(L))
        c = tuple((1.0,) * d for _ in range(L))
        g = tuple((1.0,) * d for _ in range(L))
        out, h = selective_scan(z, a, b, c, g)
        self.assertEqual(out, z)
        self.assertEqual(h, (0.0,) * d)

    def test_geometric_series(self):
        # Constant scalar kernels A=a, B=1, C=1, G=0, z_k=1, h0=0.
        # h_{k+1} = a h_k + 1 ; out_k = h_k. So out = 0, 1, 1+a, 1+a+a^2, ...
        L = 5
        a_val = 0.5
        z = tuple((1.0,) for _ in range(L))
        a = tuple((a_val,) for _ in range(L))
        b = tuple((1.0,) for _ in range(L))
        c = tuple((1.0,) for _ in range(L))
        g = tuple((0.0,) for _ in range(L))
        out, _ = selective_scan(z, a, b, c, g)
        expected = []
        h = 0.0
        for _ in range(L):
            expected.append(h)
            h = a_val * h + 1.0
        self.assertEqual([o[0] for o in out], expected)

    def test_g_passthrough(self):
        # C=0, G=1 -> output is exactly the input regardless of state.
        L, d = 3, 2
        z = tuple((float(k), float(k + 1)) for k in range(L))
        a = tuple((0.9,) * d for _ in range(L))
        b = tuple((0.3,) * d for _ in range(L))
        c = tuple((0.0,) * d for _ in range(L))
        g = tuple((1.0,) * d for _ in range(L))
        out, _ = selective_scan(z, a, b, c, g)
        self.assertEqual(out, z)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            selective_scan(((1.0,),), ((1.0,), (1.0,)), ((1.0,),), ((1.0,),), ((1.0,),))

    def test_empty(self):
        out, h = selective_scan((), (), (), (), ())
        self.assertEqual(out, ())


class TestGeometricConditioning(unittest.TestCase):
    def test_curvature_line(self):
        self.assertEqual(curvature_descriptor("line"), 0.0)

    def test_curvature_arc(self):
        self.assertAlmostEqual(curvature_descriptor("arc", radius=4.0), 0.25)
        self.assertAlmostEqual(curvature_descriptor("circle", radius=2.0), 0.5)

    def test_curvature_arc_bad_radius(self):
        with self.assertRaises(ValueError):
            curvature_descriptor("arc", radius=0.0)

    def test_curvature_general(self):
        self.assertEqual(curvature_descriptor("bspline", angular_deviation=0.7), 0.7)
        with self.assertRaises(ValueError):
            curvature_descriptor("bspline")

    def test_conditioning_vector(self):
        self.assertEqual(conditioning_vector(2.0, 3, 0.25), (2.0, 3.0, 0.25))

    def test_hierarchical_pe_deterministic(self):
        a = hierarchical_pe(1, 2, 3, 8)
        b = hierarchical_pe(1, 2, 3, 8)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 8)
        for v in a:
            self.assertLessEqual(abs(v), 1.0)

    def test_hierarchical_pe_distinct(self):
        self.assertNotEqual(hierarchical_pe(1, 2, 3, 8),
                            hierarchical_pe(2, 2, 3, 8))

    def test_hierarchical_pe_bad_dim(self):
        with self.assertRaises(ValueError):
            hierarchical_pe(0, 0, 0, 7)


class TestDepthwiseConv(unittest.TestCase):
    def test_identity_kernel(self):
        # kernel [0, 1] (causal, last tap = current) is identity.
        z = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        out = depthwise_conv1d(z, (0.0, 1.0))
        self.assertEqual(out, z)

    def test_causal_shift(self):
        # kernel [1, 0] -> output at k is input at k-1 (0 at boundary).
        z = ((1.0,), (2.0,), (3.0,))
        out = depthwise_conv1d(z, (1.0, 0.0))
        self.assertEqual(out, ((0.0,), (1.0,), (2.0,)))

    def test_bias(self):
        z = ((1.0,),)
        out = depthwise_conv1d(z, (1.0,), bias=(10.0,))
        self.assertEqual(out, ((11.0,),))

    def test_empty(self):
        self.assertEqual(depthwise_conv1d((), (1.0,)), ())


class TestGeometricStateMixer(unittest.TestCase):
    def test_shapes_and_determinism(self):
        # d=2, m=2 -> w1: 4x2, w2: 2x2
        a = (1.0, 1.0)
        b = (1.0, 1.0)
        z = (0.5, -0.5)
        w1 = ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0))
        w2 = ((1.0, 0.0), (0.0, 1.0))
        out = geometric_state_mixer(a, b, z, w1, w2)
        # h_in = a*b*z = (0.5, -0.5); w1 -> [h, z] = (0.5, -0.5, 0.5, -0.5)
        # h=(0.5,-0.5) z=(0.5,-0.5); gated = h*sigmoid(z)
        gated0 = 0.5 * sigmoid(0.5)
        gated1 = -0.5 * sigmoid(-0.5)
        self.assertAlmostEqual(out[0], gated0)
        self.assertAlmostEqual(out[1], gated1)

    def test_odd_w1_raises(self):
        with self.assertRaises(ValueError):
            geometric_state_mixer((1.0,), (1.0,), (1.0,),
                                  ((1.0,),), ((1.0,),))


class TestFilmAndComplexity(unittest.TestCase):
    def test_film(self):
        kernels = ((1.0, 2.0), (3.0, 4.0))
        psi = (0.5, 0.5)
        out = film_modulate(kernels, psi)
        self.assertEqual(out, ((0.5, 1.0), (1.5, 2.0)))

    def test_flops_linear(self):
        # doubling L doubles flops (linear), not quadruples (quadratic).
        f1 = gmamba_flops(100, 16)
        f2 = gmamba_flops(200, 16)
        self.assertEqual(f2, 2 * f1)

    def test_flops_positive(self):
        self.assertGreater(gmamba_flops(10, 8), 0)


if __name__ == "__main__":
    unittest.main()
