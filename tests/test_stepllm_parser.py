import unittest

from formats.stepllm_parser import (
    DERIVED, Entity, Enum, ParseError, Real, Ref, StepFile, Typed, UNSET,
    entity_refs, iter_refs, parse, parse_expression, serialize,
    serialize_entity, serialize_value, split_statements, strip_comments,
)


SAMPLE = "\n".join([
    "ISO-10303-21;",
    "HEADER;",
    "FILE_DESCRIPTION((''),'2;1');",
    "FILE_NAME('part.step','2026-01-01',(''),(''),'','','');",
    "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));",
    "ENDSEC;",
    "DATA;",
    "#1=CARTESIAN_POINT('',(0.,0.,0.));",
    "#2=DIRECTION('',(0.,0.,1.));",
    "#3=DIRECTION('',(1.,0.,0.));",
    "#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);",
    "#5=PLANE('',#4);",
    "ENDSEC;",
    "END-ISO-10303-21;",
    "",
])


class TestScanner(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(parse_expression("#42"), Ref(42))

    def test_string_with_escaped_quote(self):
        self.assertEqual(parse_expression("'a''b'"), "a'b")

    def test_enum(self):
        self.assertEqual(parse_expression(".PLANE."), Enum("PLANE"))

    def test_integer_and_real(self):
        self.assertEqual(parse_expression("42"), 42)
        r = parse_expression("-1.5E-3")
        self.assertIsInstance(r, Real)
        self.assertAlmostEqual(r.value, -0.0015)

    def test_real_preserves_literal(self):
        self.assertEqual(parse_expression("0.").text, "0.")

    def test_unset_and_derived(self):
        self.assertIs(parse_expression("$"), UNSET)
        self.assertIs(parse_expression("*"), DERIVED)

    def test_list_nested(self):
        self.assertEqual(parse_expression("(1,(2,3),#4)"), [1, [2, 3], Ref(4)])

    def test_empty_list(self):
        self.assertEqual(parse_expression("()"), [])

    def test_typed_value(self):
        v = parse_expression("LENGTH_MEASURE(1.)")
        self.assertEqual(v.keyword, "LENGTH_MEASURE")
        self.assertEqual(v.params[0], Real("1."))

    def test_trailing_text_rejected(self):
        with self.assertRaises(ParseError):
            parse_expression("#1 junk")


class TestComments(unittest.TestCase):
    def test_strip_block_comment(self):
        self.assertEqual(strip_comments("A/* x */B"), "AB")

    def test_comment_inside_string_kept(self):
        self.assertEqual(strip_comments("'/* keep */'"), "'/* keep */'")

    def test_unterminated_comment(self):
        with self.assertRaises(ParseError):
            strip_comments("A/* x")


class TestSplitStatements(unittest.TestCase):
    def test_semicolon_in_string_not_split(self):
        stmts = list(split_statements("A('x;y');B;"))
        self.assertEqual(stmts, ["A('x;y')", "B"])

    def test_trailing_unterminated(self):
        with self.assertRaises(ParseError):
            list(split_statements("A;B"))


class TestParse(unittest.TestCase):
    def setUp(self):
        self.step = parse(SAMPLE)

    def test_header_records(self):
        keywords = [rec.keyword for rec in self.step.header]
        self.assertEqual(
            keywords, ["FILE_DESCRIPTION", "FILE_NAME", "FILE_SCHEMA"])

    def test_entity_count_and_order(self):
        self.assertEqual(self.step.order, [1, 2, 3, 4, 5])
        self.assertEqual(len(self.step.entities), 5)

    def test_entity_keyword_and_params(self):
        e4 = self.step.entities[4]
        self.assertEqual(e4.keyword, "AXIS2_PLACEMENT_3D")
        self.assertEqual(e4.params, ["", Ref(1), Ref(2), Ref(3)])

    def test_get_by_ref(self):
        self.assertEqual(self.step.get(Ref(5)).keyword, "PLANE")

    def test_requires_iso_header(self):
        with self.assertRaises(ParseError):
            parse("HEADER;\nENDSEC;\n")

    def test_requires_iso_footer(self):
        with self.assertRaises(ParseError):
            parse("ISO-10303-21;\nDATA;\nENDSEC;\n")

    def test_duplicate_id_rejected(self):
        bad = SAMPLE.replace("#2=DIRECTION", "#1=DIRECTION")
        with self.assertRaises(ValueError):
            parse(bad)


class TestComplexInstance(unittest.TestCase):
    def test_complex_instance_parsed(self):
        text = "\n".join([
            "ISO-10303-21;", "HEADER;", "ENDSEC;", "DATA;",
            "#1=(NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.)LENGTH_UNIT());",
            "ENDSEC;", "END-ISO-10303-21;", "",
        ])
        step = parse(text)
        e1 = step.entities[1]
        self.assertIsNone(e1.keyword)
        self.assertEqual([p.keyword for p in e1.params],
                         ["NAMED_UNIT", "SI_UNIT", "LENGTH_UNIT"])
        self.assertEqual(e1.params[0].params, (DERIVED,))


class TestSerialize(unittest.TestCase):
    def test_value_roundtrip_forms(self):
        self.assertEqual(serialize_value(Ref(7)), "#7")
        self.assertEqual(serialize_value(Enum("T")), ".T.")
        self.assertEqual(serialize_value(Real("1.5")), "1.5")
        self.assertEqual(serialize_value(3), "3")
        self.assertEqual(serialize_value("a'b"), "'a''b'")
        self.assertEqual(serialize_value(UNSET), "$")
        self.assertEqual(serialize_value(DERIVED), "*")
        self.assertEqual(serialize_value([1, 2]), "(1,2)")

    def test_bool_guard(self):
        self.assertEqual(serialize_value(True), ".T.")
        self.assertEqual(serialize_value(False), ".F.")

    def test_serialize_entity(self):
        e = Entity(4, "AXIS2_PLACEMENT_3D", ["", Ref(1), Ref(2), Ref(3)])
        self.assertEqual(
            serialize_entity(e), "#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);")

    def test_serialize_complex(self):
        e = Entity(1, None, [Typed("A", (1,)), Typed("B", (2,))])
        self.assertEqual(serialize_entity(e), "#1=(A(1)B(2));")

    def test_full_roundtrip_exact(self):
        step = parse(SAMPLE)
        self.assertEqual(serialize(step), SAMPLE)

    def test_roundtrip_idempotent(self):
        once = serialize(parse(SAMPLE))
        twice = serialize(parse(once))
        self.assertEqual(once, twice)


class TestRefUtils(unittest.TestCase):
    def test_iter_refs_nested(self):
        v = Typed("X", (Ref(1), [Ref(2), 3], Ref(1)))
        self.assertEqual([r.id for r in iter_refs(v)], [1, 2, 1])

    def test_entity_refs_dedup_ordered(self):
        e = Entity(9, "E", [Ref(2), Ref(1), Ref(2), "s"])
        self.assertEqual(entity_refs(e), [2, 1])


if __name__ == "__main__":
    unittest.main()
