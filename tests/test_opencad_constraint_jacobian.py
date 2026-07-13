import math
import unittest

from harnesscad.domain.numeric.opencad_constraint_jacobian import (
    Circle,
    Constraint,
    Line,
    Point,
    Sketch,
    SolveStatus,
    diagnose,
    jacobian,
    matrix_rank,
    residuals,
    solve,
)


def _two_point_sketch():
    sketch = Sketch()
    sketch.add(Point("p1", 0.0, 0.0))
    sketch.add(Point("p2", 3.0, 4.0))
    sketch.constrain(Constraint("c_fix", "fixed", "p1"))
    return sketch


class TestResiduals(unittest.TestCase):
    def test_horizontal_line_residual_zero_when_satisfied(self):
        sketch = Sketch()
        sketch.add(Line("l1", 0.0, 1.0, 5.0, 1.0))
        sketch.constrain(Constraint("c1", "horizontal", "l1"))
        self.assertEqual(residuals(sketch), [0.0])

    def test_distance_residual(self):
        sketch = _two_point_sketch()
        sketch.constrain(Constraint("c_d", "distance", "p1", "p2", 10.0))
        res = residuals(sketch)
        self.assertAlmostEqual(res[-1], 5.0 - 10.0)

    def test_perpendicular_and_parallel(self):
        sketch = Sketch()
        sketch.add(Line("a", 0, 0, 1, 0))
        sketch.add(Line("b", 0, 0, 0, 1))
        sketch.constrain(Constraint("perp", "perpendicular", "a", "b"))
        sketch.constrain(Constraint("par", "parallel", "a", "b"))
        res = residuals(sketch)
        self.assertAlmostEqual(res[0], 0.0)
        self.assertAlmostEqual(abs(res[1]), 1.0)

    def test_tangent_line_circle(self):
        sketch = Sketch()
        sketch.add(Line("l", 0, 2, 10, 2))
        sketch.add(Circle("c", 5.0, 0.0, 2.0))
        sketch.constrain(Constraint("t", "tangent", "l", "c"))
        self.assertAlmostEqual(residuals(sketch)[0], 0.0)

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            Constraint("x", "bogus", "a")


class TestJacobianAndRank(unittest.TestCase):
    def test_jacobian_shape_and_entries(self):
        sketch = Sketch()
        sketch.add(Line("l1", 0.0, 0.0, 5.0, 1.0))
        sketch.constrain(Constraint("c1", "horizontal", "l1"))
        jac = jacobian(sketch)
        self.assertEqual(len(jac), 1)
        self.assertEqual(len(jac[0]), 4)
        # d(y1 - y2)/dy1 = 1, /dy2 = -1, x columns are zero.
        self.assertAlmostEqual(jac[0][1], 1.0, places=5)
        self.assertAlmostEqual(jac[0][3], -1.0, places=5)
        self.assertAlmostEqual(jac[0][0], 0.0, places=6)

    def test_matrix_rank(self):
        self.assertEqual(matrix_rank([[1.0, 0.0], [0.0, 1.0]]), 2)
        self.assertEqual(matrix_rank([[1.0, 2.0], [2.0, 4.0]]), 1)
        self.assertEqual(matrix_rank([]), 0)


class TestDiagnose(unittest.TestCase):
    def test_underconstrained_dof(self):
        sketch = Sketch()
        sketch.add(Point("p", 1.0, 2.0))
        sketch.constrain(Constraint("c", "distance", "p", "p", 0.0))
        diag = diagnose(sketch)
        self.assertEqual(diag.status, SolveStatus.UNDERCONSTRAINED)
        self.assertGreater(diag.dof, 0)

    def test_fully_constrained_point(self):
        sketch = Sketch()
        sketch.add(Point("p", 1.0, 2.0))
        sketch.constrain(Constraint("fix", "fixed", "p"))
        diag = diagnose(sketch)
        self.assertEqual(diag.dof, 0)
        self.assertEqual(diag.rank, 2)
        self.assertEqual(diag.status, SolveStatus.SOLVED)

    def test_variable_index_map(self):
        sketch = Sketch()
        sketch.add(Circle("c", 0.0, 0.0, 3.0))
        sketch.constrain(Constraint("fix", "fixed", "c"))
        diag = diagnose(sketch)
        names = [v.parameter_name for v in diag.variables]
        self.assertEqual(names, ["cx", "cy", "radius"])
        self.assertTrue(all(v.entity_id == "c" for v in diag.variables))

    def test_constraint_row_spans(self):
        sketch = Sketch()
        sketch.add(Point("p1", 0.0, 0.0))
        sketch.add(Point("p2", 1.0, 1.0))
        sketch.constrain(Constraint("co", "coincident", "p1", "p2"))
        sketch.constrain(Constraint("fix", "fixed", "p1"))
        diag = diagnose(sketch)
        spans = [(c.constraint_id, c.row_start, c.row_count) for c in diag.constraints]
        self.assertEqual(spans, [("co", 0, 2), ("fix", 2, 2)])

    def test_overconstrained_conflict_detected(self):
        sketch = Sketch()
        sketch.add(Point("p1", 0.0, 0.0))
        sketch.add(Point("p2", 5.0, 0.0))
        sketch.constrain(Constraint("fa", "fixed", "p1"))
        sketch.constrain(Constraint("fb", "fixed", "p2"))
        sketch.constrain(Constraint("d", "distance", "p1", "p2", 9.0))
        diag = diagnose(sketch)
        self.assertEqual(diag.status, SolveStatus.OVERCONSTRAINED)
        self.assertIn("d", diag.over_constrained_ids)

    def test_under_constrained_variables_are_untouched_columns(self):
        sketch = Sketch()
        sketch.add(Line("l", 0.0, 0.0, 4.0, 1.0))
        sketch.constrain(Constraint("h", "horizontal", "l"))
        diag = diagnose(sketch)
        # x1 (index 0) and x2 (index 2) appear in no constraint row.
        self.assertEqual(diag.under_constrained_variables, [0, 2])

    def test_deterministic(self):
        a = diagnose(_two_point_sketch())
        b = diagnose(_two_point_sketch())
        self.assertEqual(a.rank, b.rank)
        self.assertEqual(a.nonzero_entries, b.nonzero_entries)


class TestSolve(unittest.TestCase):
    def test_solves_distance_constraint(self):
        sketch = _two_point_sketch()
        sketch.constrain(Constraint("d", "distance", "p1", "p2", 10.0))
        result = solve(sketch)
        p1 = result.sketch.entities["p1"]
        p2 = result.sketch.entities["p2"]
        self.assertAlmostEqual(p1.x, 0.0, places=5)
        self.assertAlmostEqual(p1.y, 0.0, places=5)
        self.assertAlmostEqual(math.hypot(p2.x - p1.x, p2.y - p1.y), 10.0, places=4)
        self.assertLessEqual(result.max_residual, 1e-4)

    def test_solves_horizontal_line(self):
        sketch = Sketch()
        sketch.add(Line("l", 0.0, 0.0, 5.0, 2.0))
        sketch.constrain(Constraint("h", "horizontal", "l"))
        result = solve(sketch)
        line = result.sketch.entities["l"]
        self.assertAlmostEqual(line.y1, line.y2, places=5)

    def test_no_constraints_is_underconstrained(self):
        sketch = Sketch()
        sketch.add(Point("p", 0.0, 0.0))
        result = solve(sketch)
        self.assertEqual(result.status, SolveStatus.UNDERCONSTRAINED)
        self.assertEqual(result.iterations, 0)

    def test_conflict_reported(self):
        sketch = Sketch()
        sketch.add(Point("p1", 0.0, 0.0))
        sketch.add(Point("p2", 5.0, 0.0))
        sketch.constrain(Constraint("fa", "fixed", "p1"))
        sketch.constrain(Constraint("fb", "fixed", "p2"))
        sketch.constrain(Constraint("d", "distance", "p1", "p2", 9.0))
        result = solve(sketch)
        self.assertEqual(result.status, SolveStatus.OVERCONSTRAINED)
        self.assertEqual(result.conflict_constraint_id, "d")

    def test_solve_is_deterministic(self):
        def build():
            s = _two_point_sketch()
            s.constrain(Constraint("d", "distance", "p1", "p2", 7.0))
            return s

        r1 = solve(build())
        r2 = solve(build())
        self.assertEqual(
            r1.sketch.entities["p2"].x, r2.sketch.entities["p2"].x
        )
        self.assertEqual(r1.iterations, r2.iterations)


if __name__ == "__main__":
    unittest.main()
