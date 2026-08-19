"""Tests for the JoinABLe B-rep entity-type + convexity fact tables."""

import unittest

from harnesscad.domain.geometry.topology import brep_entity_ids as bei


class BrepEntityIdsTest(unittest.TestCase):
    def test_entity_ids_contiguous_and_unique(self):
        ids = [int(m.value) for m in bei.EntityType]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sorted(ids), list(range(16)))

    def test_surface_curve_split(self):
        self.assertEqual(len(bei.SURFACE_TYPES), 8)
        self.assertEqual(len(bei.CURVE_TYPES), 8)
        self.assertTrue(bei.is_surface_type(bei.EntityType.NurbsSurfaceType))
        self.assertTrue(bei.is_curve_type(bei.EntityType.Line3DCurveType))
        self.assertTrue(bei.is_curve_type(bei.EntityType.Degenerate3DCurveType))

    def test_entity_name_id_round_trip(self):
        for member in bei.EntityType:
            self.assertEqual(bei.entity_name_to_id[member.name], int(member.value))
            self.assertEqual(bei.entity_id_to_name[int(member.value)], member.name)

    def test_entity_ids_match_source(self):
        self.assertEqual(bei.EntityType.PlaneSurfaceType, 0)
        self.assertEqual(bei.EntityType.NurbsSurfaceType, 7)
        self.assertEqual(bei.EntityType.Line3DCurveType, 8)
        self.assertEqual(bei.EntityType.Degenerate3DCurveType, 15)

    def test_convexity_six_states_contiguous(self):
        ids = [int(m.value) for m in bei.Convexity]
        self.assertEqual(sorted(ids), list(range(6)))

    def test_convexity_wire_names(self):
        self.assertEqual(bei.convexity_name_to_id["None"], 0)
        self.assertEqual(bei.convexity_name_to_id["Convex"], 1)
        self.assertEqual(bei.convexity_name_to_id["Concave"], 2)
        self.assertEqual(bei.convexity_name_to_id["Smooth"], 3)
        self.assertEqual(bei.convexity_name_to_id["Non-manifold"], 4)
        self.assertEqual(bei.convexity_name_to_id["Degenerate"], 5)

    def test_classify_covers_all_states(self):
        states = {bei.classify(w) for w in bei.convexity_name_to_id}
        self.assertEqual(len(states), 6)
        self.assertTrue(bei.is_convex(bei.classify("Convex")))
        self.assertTrue(bei.is_concave(bei.classify("Concave")))

    def test_classify_case_insensitive(self):
        self.assertIs(bei.classify("non-manifold"), bei.Convexity.Nonmanifold)

    def test_continuous_labels_are_subset(self):
        for label in ("convex", "concave", "smooth"):
            self.assertIn(label, bei.EDGE_CONVEXITY_TO_ID)

    def test_analytic_surface_bridge(self):
        for cls in ("Plane", "Cylinder", "Cone", "Sphere", "Torus"):
            self.assertIn(cls, bei.ANALYTIC_SURFACE_TO_ID)

    def test_axis_thresholds_and_predicate(self):
        self.assertEqual(bei.AXIS_ANGLE_TOL_DEG, 10.0)
        self.assertEqual(bei.AXIS_DISTANCE_TOL, 1e-2)
        self.assertTrue(bei.axis_lines_colinear(9.9, 9e-3))
        self.assertFalse(bei.axis_lines_colinear(10.0, 0.0))
        self.assertFalse(bei.axis_lines_colinear(0.0, 1e-2))

    def test_selfcheck_exits_zero(self):
        self.assertEqual(bei._selfcheck(), 0)


if __name__ == "__main__":
    unittest.main()
