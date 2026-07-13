import unittest

from harnesscad.data.dataengine.schemas.cot_trace import (
    INPUT_NATURAL,
    INPUT_STRUCTURED,
    CoTTrace,
    MultimodalIntent,
    check_conformance,
    is_conformant,
    parse_cot,
)


class TestIntent(unittest.TestCase):
    def test_flags(self):
        m = MultimodalIntent(natural_language="a bracket", reference_image="r.png")
        flags = m.modality_flags
        self.assertTrue(flags[INPUT_NATURAL])
        self.assertFalse(flags[INPUT_STRUCTURED])
        self.assertTrue(flags["reference_image"])

    def test_structured_only(self):
        m = MultimodalIntent(structured="Box(10,20,5)")
        self.assertTrue(m.modality_flags[INPUT_STRUCTURED])

    def test_requires_text(self):
        with self.assertRaises(ValueError):
            MultimodalIntent()


class TestParse(unittest.TestCase):
    def test_valid(self):
        trace = parse_cot("<Think>plan the box</Think>import cadquery")
        self.assertIsInstance(trace, CoTTrace)
        self.assertEqual(trace.think, "plan the box")
        self.assertEqual(trace.code, "import cadquery")

    def test_missing_delimiters(self):
        self.assertIsNone(parse_cot("just code"))

    def test_wrong_order(self):
        self.assertIsNone(parse_cot("</Think>code<Think>"))

    def test_duplicate_delimiters(self):
        self.assertIsNone(parse_cot("<Think>a</Think><Think>b</Think>c"))


class TestConformance(unittest.TestCase):
    def test_ok(self):
        rep = check_conformance("<Think>reason</Think>code")
        self.assertTrue(rep.conformant)
        self.assertEqual(rep.violations, ())

    def test_empty_reasoning(self):
        rep = check_conformance("<Think></Think>code")
        self.assertFalse(rep.conformant)
        self.assertIn("empty_reasoning", rep.violations)

    def test_empty_code(self):
        rep = check_conformance("<Think>reason</Think>   ")
        self.assertFalse(rep.conformant)
        self.assertIn("empty_code", rep.violations)

    def test_delimiter_count(self):
        rep = check_conformance("no delimiters here")
        self.assertFalse(rep.conformant)
        self.assertIn("delimiter_count", rep.violations)

    def test_convenience(self):
        self.assertTrue(is_conformant("<Think>x</Think>y"))
        self.assertFalse(is_conformant("bad"))


if __name__ == "__main__":
    unittest.main()
