"""Tests for reconstruction.sketchgraphs_graph."""

import unittest

from harnesscad.domain.reconstruction import sketchgraphs_graph as sg


class ConstructionTests(unittest.TestCase):
    def test_add_primitive_and_constraint(self):
        g = sg.SketchGraph()
        g.add_primitive("l0", "line")
        g.add_primitive("l1", "line")
        idx = g.add_constraint("perpendicular", ("l0", "l1"))
        self.assertEqual(idx, 0)
        self.assertEqual(g.num_nodes, 2)
        self.assertEqual(g.num_edges, 1)

    def test_duplicate_node_rejected(self):
        g = sg.SketchGraph()
        g.add_primitive("c0", "circle")
        with self.assertRaises(ValueError):
            g.add_primitive("c0", "line")

    def test_unknown_primitive_rejected(self):
        g = sg.SketchGraph()
        with self.assertRaises(ValueError):
            g.add_primitive("x", "torus")

    def test_spline_needs_dof(self):
        g = sg.SketchGraph()
        with self.assertRaises(ValueError):
            g.add_primitive("s0", "spline")
        g.add_primitive("s1", "spline", dof=8)
        self.assertEqual(g.dof_budget().primitive_dof, 8)

    def test_constraint_member_arity_validated(self):
        g = sg.SketchGraph()
        g.add_primitive("l0", "line")
        g.add_primitive("l1", "line")
        # mirror needs 3 members.
        with self.assertRaises(ValueError):
            g.add_constraint("mirror", ("l0", "l1"))

    def test_constraint_unknown_member(self):
        g = sg.SketchGraph()
        g.add_primitive("l0", "line")
        with self.assertRaises(KeyError):
            g.add_constraint("horizontal", ("nope",))


class EdgeKindTests(unittest.TestCase):
    def test_loop_edge_hyperedge(self):
        g = sg.SketchGraph()
        g.add_primitive("c0", "circle")
        g.add_primitive("l0", "line")
        g.add_primitive("l1", "line")
        g.add_constraint("radius", ("c0",), value=5.0)          # loop
        g.add_constraint("parallel", ("l0", "l1"))              # edge
        g.add_constraint("mirror", ("l0", "l1", "c0"))          # hyperedge
        self.assertEqual(len(g.loops()), 1)
        self.assertEqual(len(g.hyperedges()), 1)
        self.assertTrue(g.is_multigraph())

    def test_multi_edge_detection(self):
        g = sg.SketchGraph()
        g.add_primitive("l0", "line")
        g.add_primitive("l1", "line")
        g.add_constraint("parallel", ("l0", "l1"))
        self.assertFalse(g.has_multi_edges())
        g.add_constraint("equal", ("l0", "l1"))
        self.assertTrue(g.has_multi_edges())


class SubPrimitiveTests(unittest.TestCase):
    def test_subprimitive_adds_node_and_structural_edge(self):
        g = sg.SketchGraph()
        g.add_primitive("l0", "line")
        g.add_subprimitive("l0.start", "l0")
        self.assertEqual(g.num_nodes, 2)
        # one structural edge added automatically.
        self.assertEqual(g.num_edges, 1)
        self.assertEqual(len(g.constraint_edges()), 0)
        self.assertTrue(g.node("l0.start").is_subprimitive)
        self.assertEqual(g.node("l0.start").parent, "l0")

    def test_subprimitive_unknown_parent(self):
        g = sg.SketchGraph()
        with self.assertRaises(KeyError):
            g.add_subprimitive("p", "missing")

    def test_point_point_distance_via_subprimitives(self):
        g = sg.SketchGraph()
        g.add_primitive("l0", "line")
        g.add_primitive("l1", "line")
        g.add_subprimitive("l0.end", "l0")
        g.add_subprimitive("l1.start", "l1")
        g.add_constraint("distance", ("l0.end", "l1.start"), value=7.0)
        self.assertEqual(len(g.constraint_edges()), 1)


class RelationalAnalysisTests(unittest.TestCase):
    def test_degree_and_neighbors(self):
        g = sg.build_from_sketch(
            [("l0", "line"), ("l1", "line"), ("l2", "line")],
            [("parallel", ("l0", "l1")), ("perpendicular", ("l0", "l2"))],
        )
        self.assertEqual(g.degree("l0"), 2)
        self.assertEqual(g.degree("l1"), 1)
        self.assertEqual(g.neighbors("l0"), ("l1", "l2"))

    def test_loop_counts_degree_once(self):
        g = sg.SketchGraph()
        g.add_primitive("c0", "circle")
        g.add_constraint("radius", ("c0",), value=3.0)
        self.assertEqual(g.degree("c0"), 1)


class DofBudgetTests(unittest.TestCase):
    def test_two_coincident_lines(self):
        # 2 lines = 8 DOF; one coincident removes 2 -> 6 remaining.
        g = sg.build_from_sketch(
            [("l0", "line"), ("l1", "line")],
            [("coincident", ("l0", "l1"))],
        )
        b = g.dof_budget()
        self.assertEqual(b.primitive_dof, 8)
        self.assertEqual(b.removed_dof, 2)
        self.assertEqual(b.remaining_dof, 6)

    def test_mirror_is_variable(self):
        g = sg.build_from_sketch(
            [("l0", "line"), ("l1", "line"), ("a", "line")],
            [("mirror", ("l0", "l1", "a"))],
        )
        b = g.dof_budget()
        self.assertEqual(b.variable_constraints, 1)
        self.assertEqual(b.removed_dof, 0)  # mirror not counted

    def test_subprimitive_dof(self):
        g = sg.SketchGraph()
        g.add_primitive("l0", "line")         # 4 DOF
        g.add_subprimitive("l0.start", "l0")  # +2 DOF node, -2 structural edge
        b = g.dof_budget()
        self.assertEqual(b.primitive_dof, 6)
        self.assertEqual(b.removed_dof, 2)
        self.assertEqual(b.remaining_dof, 4)

    def test_can_be_over_determined(self):
        g = sg.SketchGraph()
        g.add_primitive("p0", "point")  # 2 DOF
        g.add_primitive("p1", "point")  # 2 DOF
        g.add_constraint("coincident", ("p0", "p1"))
        g.add_constraint("coincident", ("p0", "p1"))
        b = g.dof_budget()  # 4 - 4 = 0 here but demonstrates additive removal
        self.assertEqual(b.remaining_dof, 0)


if __name__ == "__main__":
    unittest.main()
