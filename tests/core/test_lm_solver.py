"""Tests for the ezpz Levenberg-Marquardt 2D constraint solver.

Deterministic checks that the numerical solver (a THIRD, independent constraint
method alongside ``core.constraints.ConstraintGraph`` and ``SolveSpaceSketch``):

* actually drives a real 2D constraint system's residuals to zero (convergence),
* reports which variables remain under-constrained via the Jacobian null-space
  (the ezpz ``FreedomAnalysis`` port), and
* agrees with the abstract union-find DOF count on a well-constrained sketch.

No randomness, no wall clock -- every assertion is on a fixed system.
"""

import unittest
from math import hypot

from harnesscad.core.lm_solver import (
    FreedomReport,
    LMConfig,
    LMResult,
    SolveStatus,
    System2D,
    matrix_rank,
    solve_residuals,
)


class TestMatrixRank(unittest.TestCase):
    def test_full_rank_identity(self):
        rank, pivots = matrix_rank([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(rank, 2)
        self.assertEqual(pivots, [0, 1])

    def test_rank_deficient_reports_free_column(self):
        # second column is a multiple of the first -> rank 1, pivot only col 0.
        rank, pivots = matrix_rank([[1.0, 2.0], [2.0, 4.0]])
        self.assertEqual(rank, 1)
        self.assertEqual(pivots, [0])

    def test_zero_matrix_has_zero_rank(self):
        rank, pivots = matrix_rank([[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(rank, 0)
        self.assertEqual(pivots, [])

    def test_wide_matrix_pivot_columns(self):
        # 2 rows, 3 cols, independent rows -> rank 2, two pivot columns.
        rank, pivots = matrix_rank([[1.0, 0.0, 3.0], [0.0, 1.0, 4.0]])
        self.assertEqual(rank, 2)
        self.assertEqual(pivots, [0, 1])


class TestWellConstrainedSolve(unittest.TestCase):
    """A fully pinned point converges and reports zero free DOF."""

    def _system(self):
        s = System2D()
        p0 = s.add_point(0.0, 0.0)
        p1 = s.add_point(3.0, 4.0)  # deliberately off the solution
        s.fix(p0, 0.0, 0.0)
        s.distance(p0, p1, 5.0)
        s.pin_y(p1, 0.0)  # forces p1 onto the x-axis -> (5, 0)
        return s, p0, p1

    def test_converges(self):
        s, p0, p1 = self._system()
        res = s.solve()
        self.assertIsInstance(res, LMResult)
        self.assertIs(res.status, SolveStatus.CONVERGED)
        self.assertTrue(res.solved)
        self.assertLess(res.residual_norm, 1e-6)

    def test_solution_coordinates(self):
        s, p0, p1 = self._system()
        res = s.solve()
        x0, y0 = s.point(res, p0)
        x1, y1 = s.point(res, p1)
        self.assertAlmostEqual(x0, 0.0, places=5)
        self.assertAlmostEqual(y0, 0.0, places=5)
        self.assertAlmostEqual(x1, 5.0, places=4)
        self.assertAlmostEqual(y1, 0.0, places=4)
        # the constraint it was asked to satisfy actually holds.
        self.assertAlmostEqual(hypot(x1 - x0, y1 - y0), 5.0, places=4)

    def test_reports_no_free_dof(self):
        s, _, _ = self._system()
        res = s.solve()
        self.assertEqual(res.freedom.n_variables, 4)
        self.assertEqual(res.freedom.rank, 4)
        self.assertEqual(res.freedom.free_dof, 0)
        self.assertFalse(res.freedom.is_underconstrained)
        self.assertEqual(res.freedom.underconstrained, ())


class TestUnderConstrainedSolve(unittest.TestCase):
    """A point free to slide on a circle is flagged by the null-space analysis."""

    def test_one_free_dof_detected(self):
        s = System2D()
        p0 = s.add_point(0.0, 0.0)
        p1 = s.add_point(3.0, 1.0)
        s.fix(p0, 0.0, 0.0)
        s.distance(p0, p1, 5.0)  # p1 free to rotate around p0 -> 1 free DOF
        res = s.solve()

        # the distance constraint is still satisfied at whatever point it landed.
        x1, y1 = s.point(res, p1)
        self.assertAlmostEqual(hypot(x1, y1), 5.0, places=4)

        # 4 variables, rank 3 (2 from the fix + 1 from distance), 1 free DOF.
        self.assertEqual(res.freedom.n_variables, 4)
        self.assertEqual(res.freedom.rank, 3)
        self.assertEqual(res.freedom.free_dof, 1)
        self.assertTrue(res.freedom.is_underconstrained)
        self.assertEqual(len(res.freedom.underconstrained), 1)
        # the free variable is one of p1's two coordinates (index 2 or 3).
        self.assertIn(res.freedom.underconstrained[0], (2, 3))

    def test_totally_free_point(self):
        # a lone point with no constraints at all: both coordinates free.
        s = System2D()
        s.add_point(1.0, 2.0)
        res = s.solve()
        self.assertEqual(res.freedom.n_variables, 2)
        self.assertEqual(res.freedom.rank, 0)
        self.assertEqual(res.freedom.free_dof, 2)
        self.assertEqual(set(res.freedom.underconstrained), {0, 1})


class TestConstraintVocabulary(unittest.TestCase):
    def test_horizontal_and_vertical(self):
        s = System2D()
        p0 = s.add_point(0.0, 0.0)
        p1 = s.add_point(2.0, 3.0)
        p2 = s.add_point(5.0, 9.0)
        s.fix(p0, 0.0, 0.0)
        s.horizontal(p0, p1)  # equal y
        s.vertical(p0, p2)    # equal x
        s.pin_x(p1, 4.0)
        s.pin_y(p2, 7.0)
        res = s.solve()
        self.assertIs(res.status, SolveStatus.CONVERGED)
        x1, y1 = s.point(res, p1)
        x2, y2 = s.point(res, p2)
        self.assertAlmostEqual(y1, 0.0, places=5)  # horizontal with p0
        self.assertAlmostEqual(x1, 4.0, places=5)
        self.assertAlmostEqual(x2, 0.0, places=5)  # vertical with p0
        self.assertAlmostEqual(y2, 7.0, places=5)

    def test_coincident(self):
        s = System2D()
        p0 = s.add_point(0.0, 0.0)
        p1 = s.add_point(4.0, 4.0)
        s.fix(p0, 1.0, 2.0)
        s.coincident(p0, p1)
        res = s.solve()
        self.assertIs(res.status, SolveStatus.CONVERGED)
        self.assertEqual(s.point(res, p0), (res.params[0], res.params[1]))
        x0, y0 = s.point(res, p0)
        x1, y1 = s.point(res, p1)
        self.assertAlmostEqual(x0, x1, places=5)
        self.assertAlmostEqual(y0, y1, places=5)
        self.assertAlmostEqual(x0, 1.0, places=5)
        self.assertAlmostEqual(y0, 2.0, places=5)


class TestSolveResidualsDirect(unittest.TestCase):
    def test_scalar_root(self):
        # solve x^2 = 2 -> x = sqrt(2), starting from 1.0.
        res = solve_residuals([lambda v: [v[0] * v[0] - 2.0]], [1.0])
        self.assertIs(res.status, SolveStatus.CONVERGED)
        self.assertAlmostEqual(res.params[0], 2.0 ** 0.5, places=5)

    def test_determinism(self):
        def build():
            return solve_residuals(
                [lambda v: [v[0] - 3.0, v[1] + v[0] - 5.0]], [0.0, 0.0]
            )
        a = build()
        b = build()
        self.assertEqual(a.params, b.params)
        self.assertEqual(a.status, b.status)
        self.assertEqual(a.iterations, b.iterations)

    def test_custom_config_respected(self):
        cfg = LMConfig(max_iterations=1)
        # one iteration is not enough to converge a hard nonlinear system.
        res = solve_residuals(
            [lambda v: [v[0] * v[0] - 1e6]], [1.0], config=cfg
        )
        self.assertLessEqual(res.iterations, 1)


if __name__ == "__main__":
    unittest.main()
