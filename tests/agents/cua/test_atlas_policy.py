"""Tests for the atlas computer-use PROMPT POLICY (coordinate half lives elsewhere)."""

import unittest

from harnesscad.agents.cua.atlas_policy import (
    ADVERTISED_FUNCTIONS, ComputerUsePolicy, DEFAULT_GRID, render_policy,
)
from harnesscad.io.cua.coordinate import (
    Denormalizer, ScreenInfo, normalize_function_call,
)


class TestPolicy(unittest.TestCase):
    def test_default_grid_matches_denormalizer_default(self):
        self.assertEqual(ComputerUsePolicy().grid, DEFAULT_GRID)
        self.assertEqual(Denormalizer().grid, DEFAULT_GRID)

    def test_policy_denormalizer_shares_the_grid(self):
        p = ComputerUsePolicy(grid=999)
        self.assertEqual(p.denormalizer().grid, p.grid)
        # grid endpoint denormalises to the last pixel via the policy's own map.
        self.assertEqual(p.denormalizer().denorm(999, 100), 99)

    def test_grid_statement_names_the_range(self):
        stmt = ComputerUsePolicy(grid=999).grid_statement()
        self.assertIn("999", stmt)

    def test_confirmation_gate(self):
        p = ComputerUsePolicy()
        self.assertTrue(p.needs_confirmation("save"))
        self.assertTrue(p.needs_confirmation("Delete"))
        self.assertFalse(p.needs_confirmation("click"))

    def test_dict_roundtrip_fields(self):
        d = ComputerUsePolicy().to_dict()
        self.assertEqual(d["grid"], DEFAULT_GRID)
        self.assertIn("click_at", d["functions"])


class TestAdvertisedFunctionsAreMapped(unittest.TestCase):
    def test_every_advertised_function_is_understood_by_coordinate(self):
        # The prompt must never promise a tool the coordinate mapper would drop:
        # each advertised function name maps to a NormalizedAction.
        for name in ADVERTISED_FUNCTIONS:
            args = {"x": 500, "y": 500, "text": "x", "keys": "ctrl+s",
                    "direction": "down"}
            action = normalize_function_call(name, args, width=100, height=100)
            self.assertIsNotNone(action, name)


class TestRender(unittest.TestCase):
    def setUp(self):
        self.screen = ScreenInfo(width=1920, height=1080, scale_factor=2.0)

    def test_render_injects_resolution_grid_and_objective(self):
        text = render_policy(ComputerUsePolicy(), self.screen,
                             "build a 30mm block")
        self.assertIn("1920x1080", text)
        self.assertIn("999", text)
        self.assertIn("build a 30mm block", text)
        # advertised action space is listed
        self.assertIn("click_at", text)

    def test_render_is_deterministic(self):
        a = render_policy(ComputerUsePolicy(), self.screen, "obj")
        b = render_policy(ComputerUsePolicy(), self.screen, "obj")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
