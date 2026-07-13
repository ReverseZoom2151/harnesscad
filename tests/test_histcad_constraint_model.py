import unittest

from harnesscad.domain.reconstruction.sequences.modeling_sequence import Line, Circle, Constraint
from harnesscad.core.state.constraint_model import (
    analyze, constraint_histogram, SketchStatus, CONSTRAINT_TYPES,
    PRIMITIVE_DOF, CONSTRAINT_DOF, MIN_REFS,
)


class TestVocabulary(unittest.TestCase):
    def test_ten_types(self):
        self.assertEqual(len(CONSTRAINT_TYPES), 10)
        self.assertEqual(set(CONSTRAINT_DOF), set(CONSTRAINT_TYPES))
        self.assertEqual(set(MIN_REFS), set(CONSTRAINT_TYPES))

    def test_primitive_dof(self):
        self.assertEqual(PRIMITIVE_DOF, {"line": 4, "circle": 3, "arc": 6})


class TestAnalyze(unittest.TestCase):
    def test_empty(self):
        a = analyze([], [])
        self.assertEqual(a.status, SketchStatus.EMPTY)

    def test_under_constrained(self):
        prims = [Line(0, 0, 1, 0)]
        a = analyze(prims, [Constraint("horizontal", (0,))])
        self.assertEqual(a.status, SketchStatus.UNDER)
        self.assertEqual(a.total_dof, 4)
        self.assertEqual(a.removed_dof, 1)
        self.assertEqual(a.net_dof, 3)

    def test_well_constrained_by_fix(self):
        prims = [Line(0, 0, 1, 0)]
        # fix removes all 4 dof of the line
        a = analyze(prims, [Constraint("fix", (0,))])
        self.assertEqual(a.removed_dof, 4)
        self.assertEqual(a.status, SketchStatus.WELL)

    def test_over_constrained_conflict(self):
        prims = [Line(0, 0, 1, 0)]
        a = analyze(prims, [Constraint("horizontal", (0,)),
                            Constraint("vertical", (0,))])
        self.assertEqual(a.status, SketchStatus.OVER)
        self.assertTrue(any(k == "horizontal-vertical" for k, _ in a.conflicts))
        self.assertFalse(a.consistent)

    def test_parallel_perpendicular_conflict(self):
        prims = [Line(0, 0, 1, 0), Line(0, 1, 1, 1)]
        a = analyze(prims, [Constraint("parallel", (0, 1)),
                            Constraint("perpendicular", (0, 1))])
        self.assertTrue(any(k == "parallel-perpendicular" for k, _ in a.conflicts))

    def test_redundant_duplicate(self):
        prims = [Line(0, 0, 1, 0), Line(0, 1, 1, 1)]
        a = analyze(prims, [Constraint("parallel", (0, 1)),
                            Constraint("parallel", (1, 0))])
        self.assertEqual(a.redundant, (1,))

    def test_arity_error(self):
        prims = [Line(0, 0, 1, 0), Line(0, 1, 1, 1)]
        a = analyze(prims, [Constraint("parallel", (0,))])
        self.assertEqual(a.arity_errors, (0,))
        self.assertFalse(a.consistent)

    def test_tuple_inputs(self):
        a = analyze([("line", 0, 0, 1, 0)], [("horizontal", (0,))])
        self.assertEqual(a.total_dof, 4)


class TestHistogram(unittest.TestCase):
    def test_histogram(self):
        h = constraint_histogram([Constraint("horizontal", (0,)),
                                  Constraint("horizontal", (1,)),
                                  Constraint("parallel", (0, 1))])
        self.assertEqual(h["horizontal"], 2)
        self.assertEqual(h["parallel"], 1)
        self.assertEqual(h["tangent"], 0)
        self.assertEqual(set(h), set(CONSTRAINT_TYPES))


if __name__ == "__main__":
    unittest.main()
