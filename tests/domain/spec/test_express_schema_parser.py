"""Tests for spec.express_schema_parser (ISO 10303-11 EXPRESS parser)."""

import unittest

from harnesscad.domain.spec.express_schema_parser import (
    ExpressParseError,
    SuperTypeExpr,
    TypeRef,
    parse_schema,
    parse_type,
    supertype_leaf_names,
    tokenize,
    _Cursor,
)


def _type(src: str) -> TypeRef:
    return parse_type(_Cursor(tokenize(src), src))


class TokenizerTest(unittest.TestCase):
    def test_strips_both_comment_forms(self):
        src = "SCHEMA s; -- line\n(* block (* nested *) *) END_SCHEMA;"
        toks = [t.text for t in tokenize(src)]
        self.assertEqual(toks, ["SCHEMA", "s", ";", "END_SCHEMA", ";"])

    def test_string_and_number_tokens(self):
        toks = tokenize("'a''b' 1.5E-3")
        self.assertEqual(toks[0].kind, "str")
        self.assertEqual(toks[0].text, "a'b")
        self.assertEqual(toks[1].kind, "num")
        self.assertEqual(toks[1].text, "1.5E-3")


class TypeParsingTest(unittest.TestCase):
    def test_simple_and_named(self):
        self.assertEqual(_type("REAL"), TypeRef("simple", name="REAL"))
        self.assertEqual(_type("cartesian_point"),
                         TypeRef("named", name="cartesian_point"))

    def test_string_width_fixed(self):
        t = _type("STRING(10) FIXED")
        self.assertEqual(t.width, 10)
        self.assertTrue(t.fixed)

    def test_list_with_bounds_unique(self):
        t = _type("LIST [2:?] OF UNIQUE cartesian_point")
        self.assertEqual(t.kind, "list")
        self.assertEqual((t.lower, t.upper), ("2", "?"))
        self.assertTrue(t.unique)
        self.assertEqual(t.base, TypeRef("named", name="cartesian_point"))

    def test_array_optional(self):
        t = _type("ARRAY [0:3] OF OPTIONAL direction")
        self.assertEqual(t.kind, "array")
        self.assertTrue(t.optional)
        self.assertEqual((t.lower, t.upper), ("0", "3"))

    def test_enumeration(self):
        t = _type("ENUMERATION OF (up, down, left)")
        self.assertEqual(t.kind, "enum")
        self.assertEqual(t.items, ("up", "down", "left"))

    def test_select(self):
        t = _type("SELECT (a, b, c)")
        self.assertEqual(t.kind, "select")
        self.assertEqual(t.types, ("a", "b", "c"))


class EntityParsingTest(unittest.TestCase):
    def test_shared_attributes_and_optional(self):
        src = """
        SCHEMA geom;
          ENTITY point;
            x, y, z : REAL;
            label : OPTIONAL STRING;
          END_ENTITY;
        END_SCHEMA;
        """
        sch = parse_schema(src)
        p = sch.entity("point")
        self.assertEqual([a.name for a in p.attributes], ["x", "y", "z", "label"])
        self.assertEqual(p.arity, 4)
        self.assertFalse(p.attributes[0].optional)
        self.assertTrue(p.attributes[3].optional)
        self.assertEqual(p.attributes[0].type_ref, TypeRef("simple", name="REAL"))

    def test_subtype_of(self):
        src = """
        SCHEMA s;
          ENTITY placement; END_ENTITY;
          ENTITY axis2_placement_3d SUBTYPE OF (placement);
            axis : OPTIONAL direction;
          END_ENTITY;
        END_SCHEMA;
        """
        sch = parse_schema(src)
        self.assertEqual(sch.entity("axis2_placement_3d").supertypes,
                         ["placement"])

    def test_abstract_supertype_oneof(self):
        src = """
        SCHEMA s;
          ENTITY pet ABSTRACT SUPERTYPE OF (ONEOF(cat, rabbit, dog));
          END_ENTITY;
        END_SCHEMA;
        """
        ent = parse_schema(src).entity("pet")
        self.assertTrue(ent.is_abstract)
        self.assertEqual(ent.supertype_expr.op, "oneof")
        self.assertEqual(sorted(supertype_leaf_names(ent.supertype_expr)),
                         ["cat", "dog", "rabbit"])

    def test_supertype_andor_and_subtype_together(self):
        # from AP201 b_spline_curve shape
        src = """
        SCHEMA s;
          ENTITY b_spline_curve
            SUPERTYPE OF (ONEOF (uniform_curve, bezier_curve)
                          ANDOR rational_b_spline_curve)
            SUBTYPE OF (bounded_curve);
            degree : INTEGER;
            control_points_list : LIST [2:?] OF cartesian_point;
          END_ENTITY;
        END_SCHEMA;
        """
        ent = parse_schema(src).entity("b_spline_curve")
        self.assertEqual(ent.supertypes, ["bounded_curve"])
        self.assertEqual(ent.supertype_expr.op, "andor")
        self.assertEqual(ent.arity, 2)

    def test_where_derive_unique_inverse_clauses(self):
        src = """
        SCHEMA s;
          ENTITY drawing_revision SUBTYPE OF (presentation_set);
            revision_identifier : identifier;
            drawing_identifier  : drawing_definition;
            intended_scale      : OPTIONAL text;
          DERIVE
            SELF\\named_unit.dimensions : dimensional_exponents := foo(SELF.name);
          INVERSE
            opens : door FOR handle;
          UNIQUE
            ur1 : revision_identifier, drawing_identifier;
          WHERE
            wr1: SELF.degree > 0;
          END_ENTITY;
        END_SCHEMA;
        """
        ent = parse_schema(src).entity("drawing_revision")
        self.assertEqual(ent.arity, 3)
        self.assertEqual(len(ent.derived), 1)
        self.assertEqual(ent.derived[0].expr, "foo(SELF.name)")
        self.assertEqual(len(ent.inverse), 1)
        self.assertEqual(ent.inverse[0].name, "opens")
        self.assertEqual(ent.unique_rules[0].label, "ur1")
        self.assertEqual(ent.unique_rules[0].attributes,
                         ("revision_identifier", "drawing_identifier"))
        self.assertEqual(ent.where_rules[0].label, "wr1")
        self.assertEqual(ent.where_rules[0].expr, "SELF.degree > 0")


class TypeDeclTest(unittest.TestCase):
    def test_type_with_where(self):
        src = """
        SCHEMA s;
          TYPE dimension_count = INTEGER;
          WHERE
            wr1: SELF > 0;
          END_TYPE;
          TYPE label = STRING; END_TYPE;
        END_SCHEMA;
        """
        sch = parse_schema(src)
        self.assertEqual(sch.types["dimension_count"].underlying,
                         TypeRef("simple", name="INTEGER"))
        self.assertEqual(sch.types["dimension_count"].where_rules[0].expr,
                         "SELF > 0")
        self.assertEqual(sch.types["label"].underlying.name, "STRING")

    def test_enumeration_and_select_type_decls(self):
        src = """
        SCHEMA s;
          TYPE b_spline_curve_form = ENUMERATION OF
            (polyline_form, circular_arc, elliptic_arc);
          END_TYPE;
          TYPE unit = SELECT (named_unit, si_unit); END_TYPE;
        END_SCHEMA;
        """
        sch = parse_schema(src)
        self.assertEqual(sch.types["b_spline_curve_form"].underlying.items,
                         ("polyline_form", "circular_arc", "elliptic_arc"))
        self.assertEqual(sch.types["unit"].underlying.types,
                         ("named_unit", "si_unit"))


class SchemaBodyTest(unittest.TestCase):
    def test_skips_functions_and_use_from(self):
        src = """
        SCHEMA s;
          USE FROM other_schema;
          ENTITY a; v : REAL; END_ENTITY;
          FUNCTION dot(u : REAL; v : REAL) : REAL;
            RETURN (u * v);
          END_FUNCTION;
          ENTITY b; w : a; END_ENTITY;
        END_SCHEMA;
        """
        sch = parse_schema(src)
        self.assertEqual(sch.entity_order, ["a", "b"])
        self.assertEqual(sch.interfaces, [("USE", "other_schema")])

    def test_missing_schema_keyword_raises(self):
        with self.assertRaises(ExpressParseError):
            parse_schema("ENTITY a; END_ENTITY;")


if __name__ == "__main__":
    unittest.main()
