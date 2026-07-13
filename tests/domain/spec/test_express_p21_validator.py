"""Tests for spec.express_p21_validator (schema <-> part-21 validation)."""

import unittest

from harnesscad.io.formats.step import parse
from harnesscad.domain.spec.express_schema_parser import parse_schema
from harnesscad.domain.spec.express_inheritance import build_inheritance
from harnesscad.domain.spec.express_p21_validator import validate_data


_SCHEMA = """
SCHEMA geom;
  ENTITY representation_item;
    name : STRING;
  END_ENTITY;
  ENTITY geometric_representation_item
    SUBTYPE OF (representation_item);
  END_ENTITY;
  ENTITY cartesian_point SUBTYPE OF (geometric_representation_item);
    coordinates : LIST [1:3] OF REAL;
  END_ENTITY;
  ENTITY line SUBTYPE OF (geometric_representation_item);
    pnt : cartesian_point;
    dir : vector;
  END_ENTITY;
  ENTITY circle SUBTYPE OF (geometric_representation_item);
    position : axis2_placement;
    radius : REAL;
  END_ENTITY;
  ENTITY vector; END_ENTITY;
  ENTITY axis2_placement; END_ENTITY;
END_SCHEMA;
"""


class ValidatorTest(unittest.TestCase):
    def setUp(self):
        self.schema = parse_schema(_SCHEMA)
        self.graph = build_inheritance(self.schema)

    def _validate(self, data_body):
        text = ("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n"
                + data_body + "\nENDSEC;\nEND-ISO-10303-21;\n")
        step = parse(text)
        return validate_data(step, self.schema, self.graph)

    def test_valid_inherited_arity(self):
        # cartesian_point has inherited 'name' + own 'coordinates' = arity 2
        rep = self._validate("#1=CARTESIAN_POINT('origin',(0.,0.,0.));")
        self.assertTrue(rep.ok, rep.issues)
        self.assertEqual(rep.checked, 1)

    def test_wrong_arity_ignores_inheritance(self):
        # Supplying only the local attribute (missing inherited 'name') fails.
        rep = self._validate("#1=CARTESIAN_POINT((0.,0.,0.));")
        self.assertFalse(rep.ok)
        self.assertIn("expected 2 attributes", rep.issues[0].detail)

    def test_unknown_entity_flagged(self):
        rep = self._validate("#1=NURBS_SURFACE('x');")
        self.assertFalse(rep.ok)
        self.assertIn("unknown entity type", rep.issues[0].detail)

    def test_type_mismatch_flagged(self):
        # radius must be REAL, a string is incompatible.
        rep = self._validate(
            "#1=CIRCLE('c',#2,'not-a-real');\n#2=AXIS2_PLACEMENT();")
        self.assertFalse(rep.ok)
        self.assertTrue(any("radius" in i.detail for i in rep.issues))

    def test_aggregate_attribute_shape(self):
        # coordinates declared LIST OF REAL; a scalar there is wrong shape.
        rep = self._validate("#1=CARTESIAN_POINT('p',5.0);")
        self.assertFalse(rep.ok)
        self.assertTrue(any("coordinates" in i.detail for i in rep.issues))

    def test_valid_reference_and_multi_attr(self):
        rep = self._validate(
            "#1=LINE('l',#2,#3);\n#2=CARTESIAN_POINT('',(0.,0.,0.));\n"
            "#3=VECTOR();")
        self.assertTrue(rep.ok, rep.issues)

    def test_complex_instance_parts_checked(self):
        rep = self._validate(
            "#1=(REPRESENTATION_ITEM('n')GEOMETRIC_REPRESENTATION_ITEM());")
        self.assertTrue(rep.ok, rep.issues)
        rep2 = self._validate(
            "#1=(REPRESENTATION_ITEM('n')BOGUS_TYPE());")
        self.assertFalse(rep2.ok)

    def test_default_graph_built_when_omitted(self):
        text = ("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n"
                "#1=CARTESIAN_POINT('o',(1.,2.,3.));\n"
                "ENDSEC;\nEND-ISO-10303-21;\n")
        rep = validate_data(parse(text), self.schema)  # no graph passed
        self.assertTrue(rep.ok, rep.issues)


if __name__ == "__main__":
    unittest.main()
