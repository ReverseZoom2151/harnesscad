"""Tests for geometry.curv_sdf_tpms.

Checks spatial periodicity of each TPMS, zero-level membership at known points,
sign changes across the surface, and that the Lipschitz-normalised variant is a
conservative field (``|grad| <= 1 + eps``).
"""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.sdf import curv_sdf_tpms as G


def grad_mag(f, p, h=1e-6):
    g = []
    for i in range(3):
        pp, pm = list(p), list(p)
        pp[i] += h
        pm[i] -= h
        g.append((f(tuple(pp)) - f(tuple(pm))) / (2.0 * h))
    return math.sqrt(sum(c * c for c in g))


class TestGyroid(unittest.TestCase):
    def test_zero_at_origin(self):
        self.assertAlmostEqual(G.gyroid((0.0, 0.0, 0.0)), 0.0, places=12)

    def test_periodic(self):
        # default period 2*pi -> w=1, period 2*pi in each axis
        p = (0.3, 0.7, 1.1)
        shifted = (p[0] + 2 * math.pi, p[1], p[2])
        self.assertAlmostEqual(G.gyroid(p), G.gyroid(shifted), places=9)

    def test_period_scaling(self):
        # a gyroid of period 4 repeats every 4 units
        p = (0.5, 1.2, 2.3)
        self.assertAlmostEqual(G.gyroid(p, period=4.0),
                               G.gyroid((p[0] + 4.0, p[1], p[2]), period=4.0),
                               places=9)

    def test_lipschitz_normalised(self):
        f = lambda q: G.gyroid(q, lipschitz=True)
        for p in [(0.1, 0.2, 0.3), (1.0, 2.0, 0.5), (0.7, 0.7, 0.7)]:
            self.assertLessEqual(grad_mag(f, p), 1.0 + 1e-4)


class TestSchwarzP(unittest.TestCase):
    def test_zero_point(self):
        q = (math.pi / 2, math.pi / 2, math.pi / 2)
        self.assertAlmostEqual(G.schwarz_p(q), 0.0, places=9)

    def test_sign_change(self):
        self.assertGreater(G.schwarz_p((0.0, 0.0, 0.0)), 0.0)   # 3
        self.assertLess(G.schwarz_p((math.pi, math.pi, math.pi)), 0.0)  # -3

    def test_periodic(self):
        p = (0.4, 1.3, 2.2)
        self.assertAlmostEqual(G.schwarz_p(p),
                               G.schwarz_p((p[0] + 2 * math.pi, p[1], p[2])),
                               places=9)

    def test_lipschitz(self):
        f = lambda q: G.schwarz_p(q, lipschitz=True)
        for p in [(0.2, 0.5, 1.0), (1.5, 0.3, 0.8)]:
            self.assertLessEqual(grad_mag(f, p), 1.0 + 1e-4)


class TestSchwarzD(unittest.TestCase):
    def test_zero_at_origin(self):
        self.assertAlmostEqual(G.schwarz_d((0.0, 0.0, 0.0)), 0.0, places=12)

    def test_periodic(self):
        p = (0.6, 0.9, 1.4)
        self.assertAlmostEqual(G.schwarz_d(p),
                               G.schwarz_d((p[0], p[1] + 2 * math.pi, p[2])),
                               places=9)

    def test_sign_change(self):
        # the field takes both signs over a period (partitions space)
        vals = [G.schwarz_d((a / 6.0, b / 6.0, c / 6.0))
                for a in range(-9, 10) for b in range(-9, 10) for c in range(-9, 10)]
        self.assertLess(min(vals), 0.0)
        self.assertGreater(max(vals), 0.0)

    def test_lipschitz(self):
        f = lambda q: G.schwarz_d(q, lipschitz=True)
        for p in [(0.3, 0.4, 0.5), (1.0, 0.2, 1.3)]:
            self.assertLessEqual(grad_mag(f, p), 1.0 + 1e-4)


class TestNeovius(unittest.TestCase):
    def test_periodic(self):
        p = (0.7, 1.1, 0.4)
        self.assertAlmostEqual(G.neovius(p),
                               G.neovius((p[0], p[1], p[2] + 2 * math.pi)),
                               places=9)

    def test_sign_change(self):
        self.assertGreater(G.neovius((0.0, 0.0, 0.0)), 0.0)  # 13
        self.assertLess(G.neovius((math.pi, 0.0, 0.0)), 0.0)  # 3(-1+1+1)+4(-1)= -1

    def test_lipschitz(self):
        f = lambda q: G.neovius(q, lipschitz=True)
        for p in [(0.2, 0.5, 0.9), (1.2, 0.3, 0.6)]:
            self.assertLessEqual(grad_mag(f, p), 1.0 + 1e-4)


if __name__ == "__main__":
    unittest.main()
