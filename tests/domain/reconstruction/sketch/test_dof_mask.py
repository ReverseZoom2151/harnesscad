"""Tests for reconstruction.sgraphs2_dof_mask."""

import unittest

from harnesscad.io.formats.onshape_json import EntityType, SubnodeType
from harnesscad.domain.reconstruction.sketch.dof_mask import (
    EDGE_DOF_REMOVED,
    NODE_DOF,
    EdgeOp,
    NodeOp,
    constraint_mask,
    constraint_types,
    cumulative_dof,
    dof_for_node,
    dof_removed_for_edge,
    mask_offset,
    mask_size,
    node_label_for_dof,
    sequence_dof,
    valid_constraints,
)


class TestTables(unittest.TestCase):
    def test_node_dof(self):
        self.assertEqual(NODE_DOF[EntityType.Point], 2)
        self.assertEqual(NODE_DOF[EntityType.Line], 4)
        self.assertEqual(NODE_DOF[EntityType.Circle], 3)
        self.assertEqual(NODE_DOF[EntityType.Arc], 5)

    def test_coincident_depends_on_the_type_pair(self):
        table = EDGE_DOF_REMOVED["coincident"]
        self.assertEqual(table[(EntityType.Point, EntityType.Point)], 2)
        self.assertEqual(table[(EntityType.Line, EntityType.Point)], 1)
        self.assertEqual(table[(EntityType.Circle, EntityType.Circle)], 3)

    def test_pair_keys_are_entity_types(self):
        for table in EDGE_DOF_REMOVED.values():
            for a, b in table:
                self.assertIsInstance(a, EntityType)
                self.assertIsInstance(b, EntityType)


class TestNodeLabels(unittest.TestCase):
    def test_subnode_collapses_to_point(self):
        self.assertIs(node_label_for_dof(SubnodeType.SN_Start), EntityType.Point)
        self.assertIs(node_label_for_dof(SubnodeType.SN_Center), EntityType.Point)
        self.assertIs(node_label_for_dof(EntityType.Arc), EntityType.Arc)

    def test_dof_for_node(self):
        self.assertEqual(dof_for_node(EntityType.Arc), 5)
        self.assertEqual(dof_for_node(SubnodeType.SN_End), 2)
        self.assertEqual(dof_for_node(EntityType.Spline), 0)


class TestEdgeDof(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            NodeOp(EntityType.External),
            NodeOp(EntityType.Line),
            NodeOp(EntityType.Point),
            NodeOp(EntityType.Circle),
        ]

    def test_pairwise_lookup(self):
        self.assertEqual(dof_removed_for_edge(EdgeOp("coincident", (1, 2)), self.nodes), 1)
        self.assertEqual(dof_removed_for_edge(EdgeOp("coincident", (2, 2)), self.nodes), 2)

    def test_symmetric_lookup(self):
        # (Line, Point) is tabulated; (Point, Line) must resolve to the same value.
        forward = dof_removed_for_edge(EdgeOp("midpoint", (1, 2)), self.nodes)
        reverse = dof_removed_for_edge(EdgeOp("midpoint", (2, 1)), self.nodes)
        self.assertEqual(forward, 2)
        self.assertEqual(reverse, 2)

    def test_self_loop_scored_against_own_pair(self):
        # horizontal on a single line -> the (Line, Line) entry.
        self.assertEqual(dof_removed_for_edge(EdgeOp("horizontal", (1,)), self.nodes), 1)

    def test_hyperedge_removes_nothing(self):
        self.assertEqual(dof_removed_for_edge(EdgeOp("coincident", (1, 2, 3)), self.nodes), 0)

    def test_external_node_removes_nothing(self):
        self.assertEqual(dof_removed_for_edge(EdgeOp("coincident", (0, 2)), self.nodes), 0)

    def test_unknown_constraint_removes_nothing(self):
        self.assertEqual(dof_removed_for_edge(EdgeOp("bogus", (1, 2)), self.nodes), 0)

    def test_inapplicable_pairing_removes_nothing(self):
        # radius is only defined for arc/circle; between a line and a point it is
        # free rather than an error.
        self.assertEqual(dof_removed_for_edge(EdgeOp("radius", (1, 2)), self.nodes), 0)

    def test_subnode_references_treated_as_points(self):
        nodes = [NodeOp(EntityType.External), NodeOp(EntityType.Line),
                 NodeOp(SubnodeType.SN_Start), NodeOp(SubnodeType.SN_End)]
        self.assertEqual(dof_removed_for_edge(EdgeOp("coincident", (2, 3)), nodes), 2)


class TestSequenceDof(unittest.TestCase):
    def test_deltas(self):
        ops = [
            NodeOp(EntityType.External),
            NodeOp(EntityType.Point),
            NodeOp(EntityType.Point),
            EdgeOp("coincident", (1, 2)),
        ]
        self.assertEqual(sequence_dof(ops), [0, 2, 2, -2])

    def test_cumulative_running_total(self):
        ops = [
            NodeOp(EntityType.External),
            NodeOp(EntityType.Line),      # +4
            EdgeOp("horizontal", (1,)),   # -1
            NodeOp(EntityType.Point),     # +2
            EdgeOp("coincident", (1, 2)), # -1
        ]
        self.assertEqual(cumulative_dof(ops), [0, 4, 3, 5, 4])

    def test_fully_constrained_point_reaches_zero(self):
        # A point fixed in place: +2 then -2.
        ops = [NodeOp(EntityType.External), NodeOp(EntityType.Point),
               EdgeOp("fix", (1,))]
        self.assertEqual(cumulative_dof(ops)[-1], 0)

    def test_empty_sequence(self):
        self.assertEqual(sequence_dof([]), [])
        self.assertEqual(cumulative_dof([]), [])


class TestValidConstraints(unittest.TestCase):
    def test_line_line_allows_parallel_not_radius(self):
        allowed = valid_constraints(EntityType.Line, EntityType.Line)
        self.assertIn("parallel", allowed)
        self.assertIn("perpendicular", allowed)
        self.assertNotIn("radius", allowed)
        self.assertNotIn("concentric", allowed)

    def test_circle_circle_allows_radius(self):
        allowed = valid_constraints(EntityType.Circle, EntityType.Circle)
        self.assertIn("radius", allowed)
        self.assertIn("concentric", allowed)
        self.assertNotIn("parallel", allowed)

    def test_order_insensitive(self):
        self.assertEqual(
            valid_constraints(EntityType.Arc, EntityType.Line),
            valid_constraints(EntityType.Line, EntityType.Arc),
        )

    def test_subnode_label_collapses(self):
        self.assertEqual(
            valid_constraints(EntityType.Line, SubnodeType.SN_Start),
            valid_constraints(EntityType.Line, EntityType.Point),
        )

    def test_line_point_allows_midpoint(self):
        self.assertIn("midpoint", valid_constraints(EntityType.Line, EntityType.Point))

    def test_unlisted_pairing_is_empty(self):
        self.assertEqual(valid_constraints(EntityType.Spline, EntityType.Spline), frozenset())

    def test_structural_subnode_excluded(self):
        self.assertNotIn("subnode", valid_constraints(EntityType.Line, EntityType.Point))
        self.assertNotIn("subnode", constraint_types())
        self.assertIn("subnode", constraint_types(include_structural=True))

    def test_constraint_types_sorted_and_stable(self):
        names = constraint_types()
        self.assertEqual(list(names), sorted(names))
        self.assertEqual(names, constraint_types())


class TestMaskLayout(unittest.TestCase):
    def test_offsets(self):
        self.assertEqual([mask_offset(i) for i in range(5)], [0, 1, 3, 6, 10])

    def test_size(self):
        self.assertEqual(mask_size(4), 10)
        self.assertEqual(mask_size(0), 0)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            mask_offset(-1)
        with self.assertRaises(ValueError):
            mask_size(-1)


class TestConstraintMask(unittest.TestCase):
    def setUp(self):
        self.labels = [
            EntityType.External,  # node 0: the origin
            EntityType.Line,      # node 1
            EntityType.Line,      # node 2
            EntityType.Circle,    # node 3
        ]
        self.types = constraint_types()
        self.mask = constraint_mask(self.labels, self.types)

    def _slot(self, i, j):
        return self.mask[mask_offset(i) + j]

    def _allowed(self, i, j):
        return {n for n, ok in zip(self.types, self._slot(i, j)) if ok}

    def test_shape(self):
        self.assertEqual(len(self.mask), mask_size(4))
        self.assertTrue(all(len(row) == len(self.types) for row in self.mask))

    def test_origin_rows_fully_permissive(self):
        self.assertTrue(all(self._slot(0, 0)))
        for i in range(1, 4):
            self.assertTrue(all(self._slot(i, 0)), f"node {i} vs origin")

    def test_line_line_slot(self):
        allowed = self._allowed(2, 1)
        self.assertIn("parallel", allowed)
        self.assertNotIn("radius", allowed)

    def test_line_circle_slot(self):
        allowed = self._allowed(3, 1)
        self.assertIn("tangent", allowed)
        self.assertIn("normal", allowed)
        self.assertNotIn("parallel", allowed)

    def test_self_slot_uses_own_pair(self):
        # Slot (i, i) is a self-loop: the circle/circle domain.
        self.assertIn("radius", self._allowed(3, 3))

    def test_matches_valid_constraints(self):
        for i in range(1, 4):
            for j in range(1, i + 1):
                self.assertEqual(
                    self._allowed(i, j),
                    set(valid_constraints(self.labels[i], self.labels[j])),
                )

    def test_empty_and_single_node(self):
        self.assertEqual(constraint_mask([]), [])
        single = constraint_mask([EntityType.External])
        self.assertEqual(len(single), 1)
        self.assertTrue(all(single[0]))

    def test_custom_type_order_respected(self):
        mask = constraint_mask(self.labels, ("radius", "parallel"))
        row = mask[mask_offset(2) + 1]
        self.assertEqual(row, [False, True])

    def test_deterministic(self):
        self.assertEqual(constraint_mask(self.labels), constraint_mask(self.labels))


if __name__ == "__main__":
    unittest.main()
