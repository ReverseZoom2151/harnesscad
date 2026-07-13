"""Tests for quality.spatialhero_composite_reward."""

import unittest

from harnesscad.eval.quality.reward.spatialhero_composite_reward import (
    CompositeReward,
    composite_reward,
)


def _full_components(**overrides):
    base = {
        "code_valid": 1.0,
        "execution_valid": 1.0,
        "dimension_accuracy": 1.0,
        "visual_quality": 1.0,
        "topology_valid": 1.0,
    }
    base.update(overrides)
    return base


class TestWeightValidation(unittest.TestCase):
    def test_default_weights_valid(self):
        cr = CompositeReward()
        self.assertAlmostEqual(sum(cr.weights.values()), 1.0)

    def test_bad_weights_rejected(self):
        with self.assertRaises(ValueError):
            CompositeReward(weights={"a": 0.5, "b": 0.2})

    def test_custom_weights_accepted(self):
        cr = CompositeReward(
            weights={"code_valid": 0.5, "topology_valid": 0.5},
            gate_keys=("code_valid",),
        )
        self.assertAlmostEqual(sum(cr.weights.values()), 1.0)


class TestGating(unittest.TestCase):
    def test_perfect_scores_full_reward(self):
        cr = CompositeReward()
        res = cr.compute(_full_components())
        self.assertAlmostEqual(res.total, 1.0)
        self.assertFalse(res.gated_out)

    def test_failed_code_gate_zeroes_reward(self):
        cr = CompositeReward()
        res = cr.compute(_full_components(code_valid=0.0))
        self.assertEqual(res.total, 0.0)
        self.assertTrue(res.gated_out)

    def test_failed_execution_gate_zeroes_reward(self):
        cr = CompositeReward()
        # even with strong other components, no execution -> 0
        res = cr.compute(_full_components(execution_valid=0.0))
        self.assertEqual(res.total, 0.0)
        self.assertTrue(res.gated_out)

    def test_missing_gate_component_fails(self):
        cr = CompositeReward()
        comps = {
            "code_valid": 1.0,
            "dimension_accuracy": 1.0,
            "visual_quality": 1.0,
            "topology_valid": 1.0,
        }  # execution_valid absent
        res = cr.compute(comps)
        self.assertTrue(res.gated_out)
        self.assertEqual(res.total, 0.0)

    def test_gates_pass_helper(self):
        cr = CompositeReward()
        self.assertTrue(cr.gates_pass(_full_components()))
        self.assertFalse(cr.gates_pass(_full_components(execution_valid=0.4)))


class TestWeightedSum(unittest.TestCase):
    def test_partial_scores(self):
        cr = CompositeReward()
        # code 1.0*0.2 + dim 0.5*0.3 + visual 0.0*0.3 + topo 1.0*0.2 = 0.55
        res = cr.compute(_full_components(
            dimension_accuracy=0.5, visual_quality=0.0))
        self.assertAlmostEqual(res.total, 0.55)
        self.assertFalse(res.gated_out)

    def test_contributions_reported(self):
        cr = CompositeReward()
        res = cr.compute(_full_components(dimension_accuracy=0.5))
        self.assertAlmostEqual(res.contributions["dimension_accuracy"], 0.15)
        self.assertAlmostEqual(res.contributions["code_valid"], 0.20)

    def test_values_clipped(self):
        cr = CompositeReward()
        # out-of-range component values are clipped into [0,1]
        res = cr.compute(_full_components(visual_quality=5.0))
        self.assertAlmostEqual(res.total, 1.0)

    def test_negative_values_clipped(self):
        cr = CompositeReward()
        res = cr.compute(_full_components(topology_valid=-3.0))
        # topo contributes 0 -> total 0.8
        self.assertAlmostEqual(res.total, 0.8)


class TestFunctionalShortcut(unittest.TestCase):
    def test_scalar_helper(self):
        val = composite_reward(_full_components())
        self.assertAlmostEqual(val, 1.0)

    def test_scalar_helper_gated(self):
        val = composite_reward(_full_components(code_valid=0.0))
        self.assertEqual(val, 0.0)

    def test_determinism(self):
        comps = _full_components(dimension_accuracy=0.37, visual_quality=0.61)
        a = composite_reward(comps)
        b = composite_reward(comps)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
