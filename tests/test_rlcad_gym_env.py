"""Tests for reconstruction.rlcad_gym_env (RLCAD deterministic gym mechanics)."""

import unittest

from harnesscad.domain.reconstruction.sequences import rlcad_gym_env as gym
from harnesscad.domain.reconstruction.tokens.rlcad_command_spec import (
    INTERSECTION, NEWBODY, SUBTRACTION, UNION,
)


def _box(x0, x1, y0, y1):
    return frozenset((x, y) for x in range(x0, x1) for y in range(y0, y1))


class TestBooleanApply(unittest.TestCase):
    def test_union(self):
        a = _box(0, 2, 0, 2)
        b = _box(1, 3, 0, 2)
        self.assertEqual(gym.boolean_apply(a, b, UNION), a | b)

    def test_intersection(self):
        a = _box(0, 2, 0, 2)
        b = _box(1, 3, 0, 2)
        self.assertEqual(gym.boolean_apply(a, b, INTERSECTION), a & b)

    def test_subtraction(self):
        a = _box(0, 3, 0, 2)
        b = _box(1, 3, 0, 2)
        self.assertEqual(gym.boolean_apply(a, b, SUBTRACTION), a - b)

    def test_newbody_from_empty(self):
        b = _box(0, 2, 0, 2)
        self.assertEqual(gym.boolean_apply(frozenset(), b, NEWBODY), b)

    def test_unknown_op(self):
        with self.assertRaises(ValueError):
            gym.boolean_apply(frozenset(), frozenset(), "loft")


class TestMetrics(unittest.TestCase):
    def test_iou_identity(self):
        a = _box(0, 3, 0, 3)
        self.assertEqual(gym.iou(a, a), 1.0)

    def test_iou_disjoint(self):
        a = _box(0, 2, 0, 2)
        b = _box(5, 7, 5, 7)
        self.assertEqual(gym.iou(a, b), 0.0)

    def test_iou_empty_empty(self):
        self.assertEqual(gym.iou(frozenset(), frozenset()), 1.0)

    def test_iou_half(self):
        a = _box(0, 2, 0, 2)     # 4 voxels
        b = _box(0, 2, 0, 4)     # 8 voxels, superset
        self.assertAlmostEqual(gym.iou(a, b), 4 / 8)

    def test_mmd_term_matches_iou_here(self):
        a = _box(0, 2, 0, 2)
        b = _box(0, 2, 0, 4)
        self.assertAlmostEqual(gym.mmd_term(a, b), gym.iou(a, b))

    def test_normal_consistency_identity(self):
        a = _box(0, 4, 0, 4)
        self.assertEqual(gym.normal_consistency(a, a), 1.0)

    def test_composite_perfect(self):
        a = _box(0, 3, 0, 3)
        self.assertAlmostEqual(gym.composite_reward(a, a), 1.0)

    def test_composite_bounds(self):
        a = _box(0, 2, 0, 2)
        b = _box(9, 11, 9, 11)
        r = gym.composite_reward(a, b)
        self.assertGreaterEqual(r, 0.0)
        self.assertLess(r, 1.0)


class TestGymEpisode(unittest.TestCase):
    def setUp(self):
        # Target = two side-by-side boxes; reachable by two NEWBODY unions.
        self.left = _box(0, 2, 0, 2)
        self.right = _box(2, 4, 0, 2)
        self.target = self.left | self.right
        self.actions = [
            gym.GymAction("L", NEWBODY, self.left),
            gym.GymAction("R", NEWBODY, self.right),
            gym.GymAction("far", NEWBODY, _box(9, 11, 9, 11)),
        ]
        self.env = gym.RevolveGymEnv(self.target, self.actions)

    def test_reset_empty(self):
        obs = self.env.reset()
        self.assertEqual(obs["n_voxels"], 0)
        self.assertEqual(self.env.state(), frozenset())

    def test_solve_sequence(self):
        self.env.reset()
        _, _, done1, info1 = self.env.step("L")
        self.assertTrue(info1["applied"])
        self.assertFalse(done1)
        _, reward2, done2, info2 = self.env.step("R")
        self.assertTrue(info2["solved"])
        self.assertTrue(done2)
        self.assertAlmostEqual(reward2, 1.0)
        self.assertAlmostEqual(info2["iou"], 1.0)

    def test_delta_iou_positive(self):
        self.env.reset()
        _, _, _, info = self.env.step("L")
        self.assertGreater(info["delta_iou"], 0.0)

    def test_invalid_action_penalty(self):
        self.env.reset()
        _, reward, _, info = self.env.step("nonexistent")
        self.assertEqual(reward, -1.0)
        self.assertFalse(info["applied"])
        self.assertEqual(self.env.state(), frozenset())

    def test_max_steps_termination(self):
        env = gym.RevolveGymEnv(self.target, self.actions, max_steps=1)
        env.reset()
        _, _, done, _ = env.step("far")
        self.assertTrue(done)

    def test_default_max_steps(self):
        self.assertEqual(self.env.max_steps, 2 * len(self.actions))


class TestValidityAndTrial(unittest.TestCase):
    def setUp(self):
        self.target = _box(0, 2, 0, 2)
        self.actions = [
            gym.GymAction("good", NEWBODY, _box(0, 2, 0, 2)),
            gym.GymAction("flagged", NEWBODY, _box(0, 1, 0, 1), valid=False),
            gym.GymAction("empty_sub", SUBTRACTION, frozenset()),
        ]
        self.env = gym.RevolveGymEnv(self.target, self.actions)
        self.env.reset()

    def test_valid_action_keys(self):
        self.assertEqual(self.env.valid_action_keys(), ["good"])

    def test_flagged_invalid(self):
        self.assertFalse(self.env.is_valid("flagged"))

    def test_empty_subtraction_invalid(self):
        self.assertFalse(self.env.is_valid("empty_sub"))

    def test_trial_does_not_commit(self):
        before = self.env.state()
        would = self.env.trial("good")
        self.assertEqual(would, self.target)
        self.assertEqual(self.env.state(), before)

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            gym.RevolveGymEnv(self.target, [
                gym.GymAction("x", NEWBODY, self.target),
                gym.GymAction("x", NEWBODY, self.target),
            ])


class TestMarkRevert(unittest.TestCase):
    def test_mark_and_revert(self):
        target = _box(0, 2, 0, 2)
        actions = [gym.GymAction("a", NEWBODY, _box(0, 1, 0, 1)),
                   gym.GymAction("b", NEWBODY, _box(1, 2, 0, 1))]
        env = gym.RevolveGymEnv(target, actions)
        env.reset()
        env.step("a")
        saved = env.state()
        env.mark()
        env.step("b")
        self.assertNotEqual(env.state(), saved)
        env.revert()
        self.assertEqual(env.state(), saved)

    def test_revert_without_mark(self):
        env = gym.RevolveGymEnv(_box(0, 1, 0, 1),
                                [gym.GymAction("a", NEWBODY, _box(0, 1, 0, 1))])
        env.reset()
        with self.assertRaises(RuntimeError):
            env.revert()


if __name__ == "__main__":
    unittest.main()
