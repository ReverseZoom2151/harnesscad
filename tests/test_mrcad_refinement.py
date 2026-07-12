"""Tests for editing.mrcad_refinement."""

import unittest

from editing.mrcad_schema import (
    DeletePoint,
    Design,
    MakeCurve,
    Message,
    MoveCurve,
    MovePoint,
    RemoveCurve,
    circle,
    line,
)
from editing.mrcad_refinement import (
    RefinementSession,
    Rollout,
    Round,
    apply_action,
    apply_actions,
    won,
)


class ApplyActionTest(unittest.TestCase):
    def test_make_and_remove(self):
        d = Design.empty()
        d = apply_action(d, MakeCurve(line((0, 0), (1, 0))))
        self.assertEqual(len(d), 1)
        d = apply_action(d, RemoveCurve(line((0, 0), (1, 0))))
        self.assertEqual(len(d), 0)

    def test_move_curve(self):
        d = Design((line((0, 0), (2, 0)),))
        d2 = apply_action(d, MoveCurve(line((0, 0), (2, 0)), (0, 5)))
        self.assertEqual(d2, Design((line((0, 5), (2, 5)),)))

    def test_move_curve_absent_is_noop(self):
        d = Design((line((0, 0), (2, 0)),))
        d2 = apply_action(d, MoveCurve(line((9, 9), (8, 8)), (0, 5)))
        self.assertEqual(d2, d)

    def test_move_point_shared_updates_all_curves(self):
        # Two curves share control point (0,0); moving it modifies both.
        shared = (0.0, 0.0)
        d = Design((line(shared, (2, 0)), circle(shared, (0, 4))))
        d2 = apply_action(d, MovePoint(shared, (1, 1)))
        self.assertEqual(d2, Design((line((1, 1), (2, 0)), circle((1, 1), (0, 4)))))

    def test_delete_point_deletes_all_sharing_curves(self):
        shared = (0.0, 0.0)
        d = Design((line(shared, (2, 0)), circle(shared, (0, 4)), line((5, 5), (6, 6))))
        d2 = apply_action(d, DeletePoint(shared))
        self.assertEqual(d2, Design((line((5, 5), (6, 6)),)))

    def test_unknown_action(self):
        with self.assertRaises(TypeError):
            apply_action(Design.empty(), object())


class ApplyActionsTest(unittest.TestCase):
    def test_left_to_right_composition(self):
        d = apply_actions(
            Design.empty(),
            [
                MakeCurve(line((0, 0), (1, 0))),
                MoveCurve(line((0, 0), (1, 0)), (0, 1)),
            ],
        )
        self.assertEqual(d, Design((line((0, 1), (1, 1)),)))

    def test_empty_sequence_is_identity(self):
        d = Design((circle((0, 0), (2, 0)),))
        self.assertEqual(apply_actions(d, []), d)


class SessionTest(unittest.TestCase):
    def test_rollout_starts_empty_and_chains(self):
        s = RefinementSession()
        self.assertEqual(s.current, Design.empty())
        s.play_round(Message(strokes=(((0, 0), (1, 0)),)), [MakeCurve(line((0, 0), (1, 0)))])
        s.play_round(Message(text="move it up"), [MoveCurve(line((0, 0), (1, 0)), (0, 2))])
        self.assertEqual(len(s.rounds), 2)
        self.assertEqual(s.current, Design((line((0, 2), (1, 2)),)))

    def test_round_indices_and_generation_flag(self):
        s = RefinementSession()
        s.play_round(Message(text="make a line"), [MakeCurve(line((0, 0), (1, 0)))])
        s.play_round(Message(text="move"), [])
        self.assertTrue(s.rounds[0].is_generation)
        self.assertFalse(s.rounds[1].is_generation)

    def test_rollout_invariants_hold(self):
        s = RefinementSession()
        s.play_round(Message(text="a"), [MakeCurve(line((0, 0), (1, 0)))])
        s.play_round(Message(text="b"), [MakeCurve(circle((0, 0), (2, 0)))])
        r = s.rollout()
        self.assertTrue(r.validate())
        self.assertEqual(r.designs()[-1], s.current)

    def test_manually_broken_rollout_fails_validation(self):
        bad = Rollout((
            Round(1, Design.empty(), Message(), (), Design((line((0, 0), (1, 0)),))),
        ))
        # result does not equal apply_actions(design, actions)
        self.assertFalse(bad.validate())

    def test_out_of_order_indices_fail_validation(self):
        r1 = Round(1, Design.empty(), Message(), (MakeCurve(line((0, 0), (1, 0))),),
                   Design((line((0, 0), (1, 0)),)))
        r_bad = Round(3, Design((line((0, 0), (1, 0)),)), Message(), (),
                      Design((line((0, 0), (1, 0)),)))
        self.assertFalse(Rollout((r1, r_bad)).validate())


class WonTest(unittest.TestCase):
    def test_won_threshold(self):
        dist = lambda a, b: 0.0 if a == b else 1.0
        d = Design((line((0, 0), (1, 0)),))
        self.assertTrue(won(d, d, 0.5, dist))
        self.assertFalse(won(d, Design.empty(), 0.5, dist))


if __name__ == "__main__":
    unittest.main()
