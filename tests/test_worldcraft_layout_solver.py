"""Tests for generation.worldcraft_layout_solver."""

import math
import unittest

from harnesscad.domain.reconstruction.worldcraft_layout_spec import (
    LayoutSpec,
    ObjectPlacement,
    Pose,
)
from harnesscad.agents.generation.worldcraft_layout_solver import (
    AlignAxis,
    FacePoint,
    MaxDistance,
    MinDistance,
    NonOverlap,
    OnTopOf,
    Proximity,
    SolveResult,
    WithinRoom,
    solve_layout,
    total_penalty,
)


def _obj(oid, x, y, z=0.5, hx=0.5, hy=0.5, hz=0.5, parent=None):
    return ObjectPlacement(oid, "box", (hx, hy, hz), Pose.at(x, y, z), parent_id=parent)


class TestConstraintPenalties(unittest.TestCase):
    def test_nonoverlap_zero_when_apart(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0))
        s.add(_obj("b", 5.0, 0.0))
        self.assertEqual(NonOverlap(a="a", b="b").penalty(s), 0.0)

    def test_nonoverlap_positive_when_intersecting(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0))
        s.add(_obj("b", 0.5, 0.0))
        self.assertGreater(NonOverlap(a="a", b="b").penalty(s), 0.0)

    def test_ontopof_satisfied(self):
        s = LayoutSpec()
        s.add(_obj("host", 0.0, 0.0, z=0.5, hx=1.0, hy=1.0, hz=0.5))  # top at z=1.0
        s.add(_obj("obj", 0.0, 0.0, z=1.25, hx=0.2, hy=0.2, hz=0.25))  # base at z=1.0
        c = OnTopOf(obj="obj", host="host")
        self.assertAlmostEqual(c.penalty(s), 0.0)
        self.assertTrue(c.is_satisfied(s))

    def test_ontopof_penalised_when_floating(self):
        s = LayoutSpec()
        s.add(_obj("host", 0.0, 0.0, z=0.5, hx=1.0, hy=1.0, hz=0.5))
        s.add(_obj("obj", 0.0, 0.0, z=3.0, hx=0.2, hy=0.2, hz=0.25))
        self.assertGreater(OnTopOf(obj="obj", host="host").penalty(s), 0.0)

    def test_align_axis(self):
        s = LayoutSpec()
        s.add(_obj("a", 1.0, 0.0))
        s.add(_obj("b", 1.0, 4.0))
        s.add(_obj("c", 1.0, 8.0))
        self.assertAlmostEqual(AlignAxis(objects=("a", "b", "c"), axis=0).penalty(s), 0.0)
        self.assertGreater(AlignAxis(objects=("a", "b", "c"), axis=1).penalty(s), 0.0)

    def test_min_max_distance(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0))
        s.add(_obj("b", 3.0, 0.0))
        self.assertEqual(MinDistance(a="a", b="b", distance=2.0).penalty(s), 0.0)
        self.assertAlmostEqual(MinDistance(a="a", b="b", distance=5.0).penalty(s), 2.0)
        self.assertEqual(MaxDistance(a="a", b="b", distance=5.0).penalty(s), 0.0)
        self.assertAlmostEqual(MaxDistance(a="a", b="b", distance=2.0).penalty(s), 1.0)

    def test_proximity(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0))
        s.add(_obj("b", 1.0, 0.0))  # footprints touch (gap 0)
        self.assertAlmostEqual(Proximity(a="a", b="b").penalty(s), 0.0)
        s2 = LayoutSpec()
        s2.add(_obj("a", 0.0, 0.0))
        s2.add(_obj("b", 5.0, 0.0))
        self.assertGreater(Proximity(a="a", b="b").penalty(s2), 0.0)

    def test_within_room(self):
        s = LayoutSpec(room_bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 3.0)))
        s.add(_obj("a", 5.0, 5.0))
        self.assertEqual(WithinRoom(obj="a").penalty(s), 0.0)
        s.add(_obj("b", 9.8, 5.0))  # extends to x=10.3 outside
        self.assertGreater(WithinRoom(obj="b").penalty(s), 0.0)

    def test_face_point(self):
        s = LayoutSpec()
        # object at origin, yaw 0 (facing +x); target on +x axis -> satisfied.
        s.add(ObjectPlacement("a", "box", (0.5, 0.5, 0.5), Pose.at(0.0, 0.0, 0.5)))
        self.assertAlmostEqual(FacePoint(obj="a", target=(1.0, 0.0)).penalty(s), 0.0)
        self.assertGreater(FacePoint(obj="a", target=(0.0, 1.0)).penalty(s), 0.0)


class TestSolver(unittest.TestCase):
    def test_solver_separates_overlapping_objects(self):
        s = LayoutSpec(room_bounds=((-10.0, -10.0, 0.0), (10.0, 10.0, 3.0)))
        s.add(_obj("a", 0.0, 0.0))
        s.add(_obj("b", 0.2, 0.1))
        cons = [
            NonOverlap(a="a", b="b", weight=10.0, hard=True),
            WithinRoom(obj="a", weight=1.0),
            WithinRoom(obj="b", weight=1.0),
        ]
        res = solve_layout(s, cons, seed=7, iterations=3000, move_scale=1.0, rotate=False)
        self.assertIsInstance(res, SolveResult)
        self.assertLess(res.final_cost, res.initial_cost)
        self.assertEqual(NonOverlap(a="a", b="b").penalty(res.layout), 0.0)

    def test_solver_deterministic(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0))
        s.add(_obj("b", 0.3, 0.0))
        cons = [NonOverlap(a="a", b="b", weight=5.0)]
        r1 = solve_layout(s, cons, seed=42, iterations=500)
        r2 = solve_layout(s, cons, seed=42, iterations=500)
        self.assertEqual(r1.layout.to_dict(), r2.layout.to_dict())
        self.assertEqual(r1.final_cost, r2.final_cost)

    def test_solver_does_not_mutate_input(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0))
        s.add(_obj("b", 0.3, 0.0))
        before = s.to_dict()
        solve_layout(s, [NonOverlap(a="a", b="b", weight=5.0)], seed=1, iterations=200)
        self.assertEqual(s.to_dict(), before)

    def test_satisfied_reports_hard_constraints(self):
        s = LayoutSpec(room_bounds=((-10.0, -10.0, 0.0), (10.0, 10.0, 3.0)))
        s.add(_obj("a", 0.0, 0.0))
        s.add(_obj("b", 0.1, 0.0))
        cons = [NonOverlap(a="a", b="b", weight=10.0, hard=True)]
        res = solve_layout(s, cons, seed=3, iterations=3000, rotate=False)
        self.assertTrue(res.satisfied(cons))

    def test_empty_movable_returns_input(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0))
        res = solve_layout(s, [], seed=0, iterations=100, movable=[])
        self.assertEqual(res.iterations, 0)
        self.assertEqual(res.accepted, 0)

    def test_total_penalty_sums(self):
        s = LayoutSpec()
        s.add(_obj("a", 0.0, 0.0))
        s.add(_obj("b", 3.0, 0.0))
        cons = [MinDistance(a="a", b="b", distance=5.0), MaxDistance(a="a", b="b", distance=1.0)]
        self.assertAlmostEqual(total_penalty(s, cons), 2.0 + 2.0)


if __name__ == "__main__":
    unittest.main()
