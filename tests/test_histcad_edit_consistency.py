import unittest

from reconstruction.histcad_sequence import Line, Circle, Arc, Constraint
from verifiers.histcad_edit_consistency import (
    constraint_residual, check_constraints, edit_consistency,
    propagate_equal_radius,
)


class TestResiduals(unittest.TestCase):
    def test_horizontal(self):
        self.assertAlmostEqual(constraint_residual("horizontal", [Line(0, 0, 5, 0)]), 0.0)
        self.assertAlmostEqual(constraint_residual("horizontal", [Line(0, 0, 5, 2)]), 2.0)

    def test_vertical(self):
        self.assertAlmostEqual(constraint_residual("vertical", [Line(0, 0, 0, 5)]), 0.0)

    def test_parallel(self):
        a, b = Line(0, 0, 1, 0), Line(0, 1, 1, 1)
        self.assertAlmostEqual(constraint_residual("parallel", [a, b]), 0.0)

    def test_perpendicular(self):
        a, b = Line(0, 0, 1, 0), Line(0, 0, 0, 1)
        self.assertAlmostEqual(constraint_residual("perpendicular", [a, b]), 0.0)

    def test_equal_radius(self):
        self.assertAlmostEqual(constraint_residual("equal", [Circle(0, 0, 2), Circle(5, 5, 2)]), 0.0)
        self.assertAlmostEqual(constraint_residual("equal", [Circle(0, 0, 2), Circle(5, 5, 3)]), 1.0)

    def test_concentric(self):
        self.assertAlmostEqual(constraint_residual("concentric", [Circle(0, 0, 2), Circle(0, 0, 5)]), 0.0)
        self.assertGreater(constraint_residual("concentric", [Circle(0, 0, 2), Circle(1, 0, 5)]), 0.0)

    def test_coincident(self):
        self.assertAlmostEqual(constraint_residual("coincident", [Line(0, 0, 1, 0), Line(1, 0, 2, 0)]), 0.0)

    def test_tangent_circle_line(self):
        # circle centered (0,0) r=1, line y=1 -> tangent
        circ, line = Circle(0, 0, 1), Line(-5, 1, 5, 1)
        self.assertAlmostEqual(constraint_residual("tangent", [circ, line]), 0.0, places=6)

    def test_tangent_two_circles(self):
        # externally tangent: distance 3 = 1 + 2
        self.assertAlmostEqual(constraint_residual("tangent", [Circle(0, 0, 1), Circle(3, 0, 2)]), 0.0, places=6)

    def test_fix(self):
        self.assertEqual(constraint_residual("fix", [Line(0, 0, 1, 1)]), 0.0)

    def test_unknown(self):
        with self.assertRaises(ValueError):
            constraint_residual("bogus", [Line(0, 0, 1, 0)])


class TestCheckConstraints(unittest.TestCase):
    def test_all_satisfied(self):
        prims = [Line(0, 0, 1, 0), Line(0, 0, 0, 1)]
        cons = [Constraint("horizontal", (0,)), Constraint("perpendicular", (0, 1))]
        rep = check_constraints(prims, cons)
        self.assertTrue(rep.all_satisfied)
        self.assertEqual(rep.violated, ())

    def test_violation(self):
        prims = [Line(0, 0, 1, 2)]
        rep = check_constraints(prims, [Constraint("horizontal", (0,))])
        self.assertFalse(rep.all_satisfied)
        self.assertEqual(len(rep.violated), 1)
        self.assertAlmostEqual(rep.max_residual, 2.0)


class TestEditConsistency(unittest.TestCase):
    def test_constraint_preserving_edit(self):
        prims = [Circle(0, 0, 2), Circle(5, 0, 2)]
        cons = [Constraint("equal", (0, 1))]
        # edit circle 0 radius to 3, then propagate to peer
        edited0 = [Circle(0, 0, 3), prims[1]]
        propagated = propagate_equal_radius(edited0, cons, 0)
        res = edit_consistency(prims, cons, propagated)
        self.assertTrue(res.consistent)
        self.assertTrue(res.after.all_satisfied)

    def test_unconstrained_edit_breaks(self):
        prims = [Circle(0, 0, 2), Circle(5, 0, 2)]
        cons = [Constraint("equal", (0, 1))]
        # edit circle 0 radius to 3 WITHOUT propagating -> breaks equality
        edited = [Circle(0, 0, 3), prims[1]]
        res = edit_consistency(prims, cons, edited)
        self.assertFalse(res.consistent)
        self.assertTrue(res.before.all_satisfied)

    def test_propagate_returns_updated_peer(self):
        prims = [Circle(0, 0, 2), Circle(5, 0, 2), Circle(9, 0, 2)]
        cons = [Constraint("equal", (0, 1)), Constraint("equal", (0, 2))]
        out = propagate_equal_radius([Circle(0, 0, 7), prims[1], prims[2]], cons, 0)
        self.assertAlmostEqual(out[1].r, 7.0)
        self.assertAlmostEqual(out[2].r, 7.0)


if __name__ == "__main__":
    unittest.main()
