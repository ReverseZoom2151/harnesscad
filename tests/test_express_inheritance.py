"""Tests for spec.express_inheritance."""

import unittest

from spec.express_schema_parser import parse_schema
from spec.express_inheritance import (
    InheritanceError,
    build_inheritance,
    expand_select,
    flatten_attributes,
)


_SCHEMA = """
SCHEMA geom;
  ENTITY representation_item;
    name : label;
  END_ENTITY;
  ENTITY geometric_representation_item
    SUBTYPE OF (representation_item);
  END_ENTITY;
  ENTITY point SUBTYPE OF (geometric_representation_item);
  END_ENTITY;
  ENTITY cartesian_point SUBTYPE OF (point);
    coordinates : LIST [1:3] OF REAL;
  END_ENTITY;
  ENTITY placement
    SUPERTYPE OF (ONEOF(axis2_placement_3d))
    SUBTYPE OF (geometric_representation_item);
    location : cartesian_point;
  END_ENTITY;
  ENTITY axis2_placement_3d SUBTYPE OF (placement);
    axis : OPTIONAL direction;
    ref_direction : OPTIONAL direction;
  END_ENTITY;
  TYPE label = STRING; END_TYPE;
END_SCHEMA;
"""


class InheritanceGraphTest(unittest.TestCase):
    def setUp(self):
        self.schema = parse_schema(_SCHEMA)
        self.graph = build_inheritance(self.schema)

    def test_direct_edges(self):
        self.assertEqual(self.graph.supertypes["cartesian_point"], ["point"])
        self.assertIn("cartesian_point", self.graph.subtypes["point"])

    def test_supertype_of_declaration_adds_edge(self):
        # placement declared SUPERTYPE OF axis2_placement_3d
        self.assertIn("placement",
                      self.graph.supertypes["axis2_placement_3d"])
        self.assertIn("axis2_placement_3d",
                      self.graph.subtypes["placement"])

    def test_transitive_supertypes(self):
        self.assertEqual(
            self.graph.all_supertypes("cartesian_point"),
            ["point", "geometric_representation_item", "representation_item"])
        self.assertTrue(
            self.graph.is_subtype_of("cartesian_point", "representation_item"))
        self.assertFalse(
            self.graph.is_subtype_of("representation_item", "cartesian_point"))

    def test_roots_and_leaves(self):
        self.assertEqual(self.graph.roots(), ["representation_item"])
        self.assertIn("cartesian_point", self.graph.leaves())
        self.assertIn("axis2_placement_3d", self.graph.leaves())


class FlattenTest(unittest.TestCase):
    def setUp(self):
        self.graph = build_inheritance(parse_schema(_SCHEMA))

    def test_flatten_inherited_before_local(self):
        attrs = [a.name for a in flatten_attributes(self.graph,
                                                    "cartesian_point")]
        # representation_item.name is inherited first, then own coordinates
        self.assertEqual(attrs, ["name", "coordinates"])

    def test_flatten_multi_level(self):
        attrs = [a.name for a in flatten_attributes(self.graph,
                                                    "axis2_placement_3d")]
        # name (from representation_item), location (placement), then own two
        self.assertEqual(attrs,
                         ["name", "location", "axis", "ref_direction"])

    def test_diamond_dedup(self):
        src = """
        SCHEMA d;
          ENTITY base; b : REAL; END_ENTITY;
          ENTITY left SUBTYPE OF (base); l : REAL; END_ENTITY;
          ENTITY right SUBTYPE OF (base); r : REAL; END_ENTITY;
          ENTITY bottom SUBTYPE OF (left, right); x : REAL; END_ENTITY;
        END_SCHEMA;
        """
        graph = build_inheritance(parse_schema(src))
        attrs = [a.name for a in flatten_attributes(graph, "bottom")]
        # base.b appears once despite the diamond
        self.assertEqual(attrs.count("b"), 1)
        self.assertEqual(attrs, ["b", "l", "r", "x"])


class SelectAndCycleTest(unittest.TestCase):
    def test_expand_nested_select(self):
        src = """
        SCHEMA s;
          TYPE a = SELECT (p, q); END_TYPE;
          TYPE b = SELECT (a, r); END_TYPE;
          ENTITY p; END_ENTITY;
          ENTITY q; END_ENTITY;
          ENTITY r; END_ENTITY;
        END_SCHEMA;
        """
        schema = parse_schema(src)
        self.assertEqual(sorted(expand_select(schema, "b")), ["p", "q", "r"])

    def test_cycle_detected(self):
        src = """
        SCHEMA s;
          ENTITY a SUBTYPE OF (b); END_ENTITY;
          ENTITY b SUBTYPE OF (a); END_ENTITY;
        END_SCHEMA;
        """
        with self.assertRaises(InheritanceError):
            build_inheritance(parse_schema(src))


if __name__ == "__main__":
    unittest.main()
