"""Tests for editing.voxhammer_trajectory."""
import unittest

from harnesscad.domain.editing.inversion_trajectory import (
    InversionCache,
    late_cfg,
    linear_schedule,
    reinject,
    taylor_flow_step,
)


class TestInversionCache(unittest.TestCase):
    def test_store_and_lookup(self):
        c = InversionCache()
        c.store(0.5, {(0, 0, 0): (1.0,)}, kv={"t": (2.0,)})
        self.assertTrue(c.has(0.5))
        self.assertEqual(c.latents_at(0.5), {(0, 0, 0): (1.0,)})
        self.assertEqual(c.kv_at(0.5), {"t": (2.0,)})

    def test_missing_raises(self):
        c = InversionCache()
        with self.assertRaises(KeyError):
            c.latents_at(0.1)
        with self.assertRaises(KeyError):
            c.kv_at(0.1)

    def test_timesteps_sorted(self):
        c = InversionCache()
        c.store(0.75, {})
        c.store(0.25, {})
        c.store(0.5, {})
        self.assertEqual(c.timesteps(), (0.25, 0.5, 0.75))

    def test_float_key_roundtrip(self):
        c = InversionCache()
        for t in linear_schedule(25):
            c.store(t, {(0, 0, 0): (t,)})
        for t in linear_schedule(25):
            self.assertTrue(c.has(t))


class TestReinject(unittest.TestCase):
    def test_reinject_preserves_keep(self):
        c = InversionCache()
        c.store(0.4, {(0, 0, 0): (9.0,), (1, 0, 0): (9.0,)})
        current = {(0, 0, 0): (1.0,), (1, 0, 0): (1.0,)}
        out = reinject(current, c, 0.4, {(0, 0, 0)})
        self.assertEqual(out[(0, 0, 0)], (9.0,))  # keep reinjected
        self.assertEqual(out[(1, 0, 0)], (1.0,))  # edit untouched


class TestLinearSchedule(unittest.TestCase):
    def test_endpoints(self):
        s = linear_schedule(4)
        self.assertEqual(s[0], 0.0)
        self.assertEqual(s[-1], 1.0)
        self.assertEqual(len(s), 5)

    def test_ascending(self):
        s = linear_schedule(10)
        self.assertEqual(list(s), sorted(s))

    def test_bad(self):
        with self.assertRaises(ValueError):
            linear_schedule(0)


class TestTaylorFlowStep(unittest.TestCase):
    def test_constant_field(self):
        # f constant -> pure Euler, second-order term vanishes
        out = taylor_flow_step((0.0, 0.0), 0.0, 0.5, lambda x, t: (2.0, -2.0))
        self.assertEqual(out, (1.0, -1.0))

    def test_linear_time_field_is_exact(self):
        # dx/dt = t  =>  x(t+dt) = x + dt*t + dt^2/2, matched exactly at 2nd order
        out = taylor_flow_step((0.0,), 1.0, 0.5, lambda x, t: (t,))
        expected = 0.0 + 0.5 * 1.0 + 0.5 * 0.5 * 0.5
        self.assertAlmostEqual(out[0], expected)

    def test_negative_dt_inversion(self):
        out = taylor_flow_step((5.0,), 0.0, -0.5, lambda x, t: (2.0,))
        self.assertEqual(out, (4.0,))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            taylor_flow_step((0.0, 0.0), 0.0, 0.1, lambda x, t: (1.0,))

    def test_deterministic(self):
        f = lambda x, t: tuple(v * 0.5 + t for v in x)
        a = taylor_flow_step((1.0, 2.0), 0.3, 0.2, f)
        b = taylor_flow_step((1.0, 2.0), 0.3, 0.2, f)
        self.assertEqual(a, b)


class TestLateCfg(unittest.TestCase):
    def test_active_in_interval(self):
        out = late_cfg((2.0,), (1.0,), omega=1.0, t=0.8)
        # (1+1)*2 - 1*1 = 3
        self.assertEqual(out, (3.0,))

    def test_inactive_outside_interval(self):
        out = late_cfg((2.0,), (1.0,), omega=1.0, t=0.2)
        self.assertEqual(out, (2.0,))

    def test_boundary_inclusive(self):
        self.assertEqual(late_cfg((2.0,), (0.0,), 1.0, 0.5), (4.0,))
        self.assertEqual(late_cfg((2.0,), (0.0,), 1.0, 1.0), (4.0,))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            late_cfg((1.0, 2.0), (1.0,), 1.0, 0.8)


if __name__ == "__main__":
    unittest.main()
