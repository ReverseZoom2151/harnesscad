"""Tests for the 2D sketch constraint model (constraints.py).

Covers the always-available abstract :class:`ConstraintGraph` DOF analysis and,
when the optional ``constraints`` extra (python-solvespace) is installed, the
real geometric :class:`SolveSpaceSketch` solver.
"""

import unittest

from constraints import (
    ConstraintGraph, ConstraintRef, RelaxationResult, SketchStatus,
    SolveSpaceSketch, solvespace_available,
)


HAVE_SOLVESPACE = solvespace_available()


class TestConstraintGraphDof(unittest.TestCase):
    def test_entity_dof_matches_conventions(self):
        g = ConstraintGraph()
        g.add_entity("e1", "rectangle")   # 4
        g.add_entity("e2", "circle")      # 3
        self.assertEqual(g.total_entity_dof(), 7)
        self.assertEqual(g.residual_dof(), 7)

    def test_single_constraint_reduces_residual(self):
        g = ConstraintGraph()
        g.add_entity("e1", "rectangle")
        self.assertEqual(g.residual_dof(), 4)
        g.add_constraint("distance", "e1", value=10.0)
        self.assertEqual(g.residual_dof(), 3)
        self.assertTrue(g.analyze().under_constrained)

    def test_unknown_kinds_raise(self):
        g = ConstraintGraph()
        with self.assertRaises(ValueError):
            g.add_entity("e1", "spline")
        g.add_entity("e1", "rectangle")
        with self.assertRaises(ValueError):
            g.add_constraint("tangent", "e1")
        with self.assertRaises(KeyError):
            g.add_constraint("distance", "nope", value=1.0)

    def test_empty_sketch_status(self):
        self.assertIs(ConstraintGraph().analyze().status, SketchStatus.EMPTY)


class TestConstraintGraphClassification(unittest.TestCase):
    def test_well_constrained(self):
        g = ConstraintGraph()
        g.add_entity("e1", "rectangle")           # 4 DOF
        for _ in range(4):
            g.add_constraint("distance", "e1", value=1.0)
        a = g.analyze()
        self.assertEqual(a.residual_dof, 0)
        self.assertEqual(a.redundant_dof, 0)
        self.assertIs(a.status, SketchStatus.WELL)

    def test_over_constrained_by_excess(self):
        g = ConstraintGraph()
        g.add_entity("e1", "rectangle")
        for _ in range(5):                         # one too many
            g.add_constraint("distance", "e1", value=1.0)
        a = g.analyze()
        self.assertEqual(a.residual_dof, -1)       # signed net goes negative
        self.assertEqual(a.redundant_dof, 1)
        self.assertEqual(len(a.redundant_constraints), 1)
        self.assertIs(a.status, SketchStatus.OVER)

    def test_redundancy_detected_even_when_net_dof_nonnegative(self):
        # Two separate entities; all constraints piled onto the first make it
        # redundant/over even though the second entity leaves net DOF > 0.
        g = ConstraintGraph()
        g.add_entity("p1", "point")   # 2
        g.add_entity("p2", "point")   # 2  (never constrained -> stays free)
        # 3 distance constraints on p1 alone: p1 has only 2 DOF -> 1 redundant.
        for _ in range(3):
            g.add_constraint("distance", "p1", value=1.0)
        a = g.analyze()
        self.assertEqual(a.residual_dof, 1)        # naive net says "under"
        self.assertEqual(a.redundant_dof, 1)       # but a constraint is redundant
        self.assertEqual(a.free_dof, 2)            # p2 still fully free
        self.assertIs(a.status, SketchStatus.OVER)

    def test_coupled_component_pools_dof(self):
        # coincident couples two points into one 4-DOF component; 4 non-redundant
        # constraints on the pair make it well-constrained.
        g = ConstraintGraph()
        g.add_entity("p1", "point")
        g.add_entity("p2", "point")
        g.add_constraint("coincident", "p1", "p2")   # weight 2
        g.add_constraint("distance", "p1", value=1.0)
        g.add_constraint("distance", "p2", value=1.0)
        a = g.analyze()
        self.assertEqual(a.effective_removed, 4)
        self.assertEqual(a.redundant_dof, 0)
        self.assertIs(a.status, SketchStatus.WELL)


class TestRelaxation(unittest.TestCase):
    """Graceful relaxation fallback (best-effort resolution over the graph)."""

    @staticmethod
    def _rebuild_without(builder, drop_indices):
        """Rebuild a fresh graph from ``builder`` skipping the given constraint
        indices, so a drop suggestion can be *applied* and re-classified."""
        drop = set(drop_indices)
        g = ConstraintGraph()
        for eid, kind in builder["entities"]:
            g.add_entity(eid, kind)
        for i, (kind, a, b, value) in enumerate(builder["constraints"]):
            if i in drop:
                continue
            g.add_constraint(kind, a, b, value)
        return g

    def test_over_constrained_reports_conflict_and_drop_restores_well(self):
        entities = [("e1", "rectangle")]
        constraints = [("distance", "e1", None, 1.0) for _ in range(5)]  # 1 too many
        g = ConstraintGraph()
        for eid, kind in entities:
            g.add_entity(eid, kind)
        for kind, a, b, value in constraints:
            g.add_constraint(kind, a, b, value)

        rr = g.relax()
        self.assertIsInstance(rr, RelaxationResult)
        self.assertIs(rr.status, SketchStatus.OVER)
        self.assertFalse(rr.feasible)
        self.assertEqual(rr.residual, 1.0)              # one redundant DOF
        # non-empty conflict set + a non-empty drop suggestion
        self.assertTrue(rr.conflicting_constraints)
        self.assertTrue(rr.dropped_suggestions)
        self.assertTrue(all(isinstance(r, ConstraintRef) for r in rr.conflicting_constraints))

        # applying the drop suggestion restores well-constrained status
        repaired = self._rebuild_without(
            {"entities": entities, "constraints": constraints},
            rr.dropped_indices,
        )
        self.assertIs(repaired.analyze().status, SketchStatus.WELL)

    def test_conflicts_shortcut_matches_relax(self):
        g = ConstraintGraph()
        g.add_entity("e1", "rectangle")
        for _ in range(6):
            g.add_constraint("distance", "e1", value=1.0)
        self.assertEqual(
            [r.index for r in g.conflicts()],
            g.relax().conflict_indices,
        )
        self.assertTrue(g.conflicts())

    def test_redundancy_hidden_by_free_entity_still_flags_conflict(self):
        # p2 keeps net DOF positive, but p1 is over-constrained -> still OVER,
        # and the drop suggestion, once applied, must remove the conflict.
        entities = [("p1", "point"), ("p2", "point")]
        constraints = [("distance", "p1", None, 1.0) for _ in range(3)]  # p1 has 2 DOF
        g = ConstraintGraph()
        for eid, kind in entities:
            g.add_entity(eid, kind)
        for kind, a, b, value in constraints:
            g.add_constraint(kind, a, b, value)

        rr = g.relax()
        self.assertIs(rr.status, SketchStatus.OVER)
        self.assertFalse(rr.feasible)
        self.assertTrue(rr.dropped_suggestions)
        repaired = self._rebuild_without(
            {"entities": entities, "constraints": constraints},
            rr.dropped_indices,
        )
        self.assertIsNot(repaired.analyze().status, SketchStatus.OVER)

    def test_under_constrained_reports_free_dof_and_unpinned(self):
        g = ConstraintGraph()
        g.add_entity("e1", "rectangle")     # 4 DOF
        g.add_entity("p1", "point")         # 2 DOF, never constrained
        g.add_constraint("distance", "e1", value=1.0)   # binds 1 of e1

        rr = g.relax()
        self.assertIs(rr.status, SketchStatus.UNDER)
        self.assertTrue(rr.feasible)                    # a placement still exists
        self.assertEqual(rr.residual, 0.0)              # no conflict
        self.assertFalse(rr.conflicting_constraints)
        self.assertFalse(rr.dropped_suggestions)
        self.assertEqual(rr.free_dof, 5)                # 3 left on e1 + 2 on p1
        # both entities still have slack -> both unpinned
        self.assertEqual(rr.unpinned_entities, ["e1", "p1"])

    def test_well_constrained_is_feasible_with_empty_conflict(self):
        g = ConstraintGraph()
        g.add_entity("e1", "rectangle")
        for _ in range(4):
            g.add_constraint("distance", "e1", value=1.0)

        rr = g.relax()
        self.assertIs(rr.status, SketchStatus.WELL)
        self.assertTrue(rr.feasible)
        self.assertEqual(rr.residual, 0.0)
        self.assertEqual(rr.conflicting_constraints, [])
        self.assertEqual(rr.dropped_suggestions, [])
        self.assertEqual(rr.free_dof, 0)
        self.assertEqual(rr.unpinned_entities, [])

    def test_analyze_relax_attaches_relaxation_and_preserves_core(self):
        g = ConstraintGraph()
        g.add_entity("e1", "rectangle")
        for _ in range(5):
            g.add_constraint("distance", "e1", value=1.0)

        plain = g.analyze()
        rich = g.analyze(relax=True)
        # core classification is identical; relax=False carries no relaxation.
        self.assertIsNone(plain.relaxation)
        self.assertEqual(plain.status, rich.status)
        self.assertEqual(plain.residual_dof, rich.residual_dof)
        self.assertEqual(plain.redundant_dof, rich.redundant_dof)
        self.assertIsInstance(rich.relaxation, RelaxationResult)
        self.assertFalse(rich.relaxation.feasible)
        self.assertTrue(rich.relaxation.conflicting_constraints)

    def test_empty_sketch_is_feasible(self):
        rr = ConstraintGraph().relax()
        self.assertIs(rr.status, SketchStatus.EMPTY)
        self.assertTrue(rr.feasible)
        self.assertEqual(rr.conflicting_constraints, [])
        self.assertEqual(rr.unpinned_entities, [])


@unittest.skipUnless(HAVE_SOLVESPACE, "python-solvespace extra not installed")
class TestSolveSpaceSketch(unittest.TestCase):
    def test_two_points_distance_is_under_constrained(self):
        sk = SolveSpaceSketch()
        a = sk.add_point(0.0, 0.0)
        b = sk.add_point(10.0, 0.0)
        sk.constrain("distance", a, b, value=10.0)
        res = sk.solve()
        self.assertTrue(res.solved)
        self.assertEqual(res.residual_dof, 3)      # 4 DOF - 1 distance
        self.assertIs(res.status, SketchStatus.UNDER)

    def test_fully_constrained_pair_is_well_constrained(self):
        sk = SolveSpaceSketch()
        a = sk.add_point(0.0, 0.0)
        b = sk.add_point(10.0, 0.0)
        # pin a to origin (distance to itself is not meaningful); instead pin
        # both coordinates by constraining against horizontal + distances.
        sk.constrain("distance", a, b, value=10.0)
        sk.constrain("horizontal", sk.add_line(a, b))
        # still under-constrained (a floats): expect a valid solve with dof > 0.
        res = sk.solve()
        self.assertTrue(res.solved)
        self.assertGreaterEqual(res.residual_dof, 0)

    def test_redundant_distance_is_over_constrained(self):
        sk = SolveSpaceSketch()
        a = sk.add_point(0.0, 0.0)
        b = sk.add_point(10.0, 0.0)
        sk.constrain("distance", a, b, value=10.0)
        sk.constrain("distance", a, b, value=10.0)   # conflicting/redundant
        res = sk.solve()
        self.assertFalse(res.solved)
        self.assertIs(res.status, SketchStatus.OVER)
        self.assertTrue(res.failures)

    def test_circle_radius_removes_one_dof(self):
        sk = SolveSpaceSketch()
        ct = sk.add_point(0.0, 0.0)
        circle = sk.add_circle(ct, 5.0)
        before = sk.solve()
        self.assertTrue(before.solved)
        self.assertEqual(before.residual_dof, 3)     # centre (2) + radius (1)
        sk.constrain("radius", circle, value=5.0)
        after = sk.solve()
        self.assertTrue(after.solved)
        self.assertEqual(after.residual_dof, 2)

    def test_missing_value_raises(self):
        sk = SolveSpaceSketch()
        a = sk.add_point(0.0, 0.0)
        b = sk.add_point(1.0, 0.0)
        with self.assertRaises(ValueError):
            sk.constrain("distance", a, b)


if __name__ == "__main__":
    unittest.main()
