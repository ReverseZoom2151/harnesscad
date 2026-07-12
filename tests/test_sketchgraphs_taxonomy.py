"""Tests for reconstruction.sketchgraphs_taxonomy."""

import unittest

from reconstruction import sketchgraphs_taxonomy as tax


class PrimitiveTaxonomyTests(unittest.TestCase):
    def test_appendix_a_dof_table(self):
        self.assertEqual(tax.primitive_dof("point"), 2)
        self.assertEqual(tax.primitive_dof("line"), 4)
        self.assertEqual(tax.primitive_dof("circle"), 3)
        self.assertEqual(tax.primitive_dof("arc"), 5)
        self.assertEqual(tax.primitive_dof("ellipse"), 5)

    def test_superset_of_cisp_ops(self):
        # SketchGraphs adds arc / ellipse / spline that cisp.ops lacks.
        self.assertIn("arc", tax.PRIMITIVE_SPECS)
        self.assertIn("ellipse", tax.PRIMITIVE_SPECS)
        self.assertIn("spline", tax.PRIMITIVE_SPECS)

    def test_spline_has_variable_dof(self):
        self.assertIsNone(tax.PRIMITIVE_SPECS["spline"].dof)
        with self.assertRaises(ValueError):
            tax.primitive_dof("spline")

    def test_unknown_primitive_raises(self):
        with self.assertRaises(KeyError):
            tax.primitive_dof("nurbs_surface")

    def test_primitive_frequency_ordering(self):
        ranked = tax.primitives_by_frequency()
        self.assertEqual(ranked[0].name, "line")  # 68.47 %, dominant
        freqs = [s.frequency_pct for s in ranked]
        self.assertEqual(freqs, sorted(freqs, reverse=True))


class ConstraintTaxonomyTests(unittest.TestCase):
    def test_coincident_is_pairwise_removes_two(self):
        spec = tax.CONSTRAINT_SPECS["coincident"]
        self.assertEqual(spec.member_arities, (2,))
        self.assertEqual(tax.constraint_dof("coincident"), 2)

    def test_horizontal_has_two_schemata(self):
        spec = tax.CONSTRAINT_SPECS["horizontal"]
        # Appendix B: horizontal appears as (local0) and (local0, local1).
        self.assertEqual(spec.member_arities, (1, 2))
        self.assertTrue(spec.allows_loop())
        self.assertFalse(spec.allows_hyperedge())

    def test_mirror_is_hyperedge_variable_dof(self):
        spec = tax.CONSTRAINT_SPECS["mirror"]
        self.assertEqual(spec.member_arities, (3,))
        self.assertTrue(spec.allows_hyperedge())
        self.assertIsNone(spec.dof_removed)
        with self.assertRaises(ValueError):
            tax.constraint_dof("mirror")

    def test_projected_is_external(self):
        spec = tax.CONSTRAINT_SPECS["projected"]
        self.assertTrue(spec.is_external)
        with self.assertRaises(ValueError):
            tax.constraint_dof("projected")

    def test_dimensional_constraints(self):
        self.assertTrue(tax.CONSTRAINT_SPECS["distance"].is_dimensional)
        self.assertTrue(tax.CONSTRAINT_SPECS["radius"].is_dimensional)
        self.assertTrue(tax.CONSTRAINT_SPECS["angle"].is_dimensional)
        self.assertFalse(tax.CONSTRAINT_SPECS["parallel"].is_dimensional)
        self.assertFalse(tax.CONSTRAINT_SPECS["coincident"].is_dimensional)

    def test_distance_schema_numeric_params(self):
        spec = tax.CONSTRAINT_SPECS["distance"]
        # (local0, local1, direction, halfSpace0, halfSpace1, length)
        self.assertEqual(spec.member_arities, (2,))
        self.assertEqual(
            spec.numeric_params(),
            ("direction", "halfSpace0", "halfSpace1", "length"),
        )

    def test_diameter_radius_are_loops_with_length(self):
        for name in ("diameter", "radius"):
            spec = tax.CONSTRAINT_SPECS[name]
            self.assertTrue(spec.allows_loop())
            self.assertEqual(spec.member_arities, (1,))
            self.assertIn("length", spec.numeric_params())

    def test_edge_classification(self):
        self.assertEqual(tax.classify_edge(1), "loop")
        self.assertEqual(tax.classify_edge(2), "edge")
        self.assertEqual(tax.classify_edge(3), "hyperedge")
        self.assertEqual(tax.classify_edge(5), "hyperedge")
        with self.assertRaises(ValueError):
            tax.classify_edge(0)

    def test_constraint_frequency_ordering(self):
        ranked = tax.constraints_by_frequency()
        self.assertEqual(ranked[0].name, "coincident")  # 42.17 %, dominant
        freqs = [s.frequency_pct for s in ranked]
        self.assertEqual(freqs, sorted(freqs, reverse=True))

    def test_unknown_constraint_raises(self):
        with self.assertRaises(KeyError):
            tax.constraint_dof("gravity")


if __name__ == "__main__":
    unittest.main()
