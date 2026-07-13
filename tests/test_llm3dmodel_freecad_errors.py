import unittest

from harnesscad.eval.bench.llm3dmodel_freecad_errors import (
    SYNTAX, GEOMETRIC, EXECUTION, NONE, classify, tally, ErrorClassification)


class ClassifyTests(unittest.TestCase):
    def test_clean_run(self):
        c = classify("")
        self.assertEqual(c.family, NONE)
        c = classify("   \n ")
        self.assertEqual(c.family, NONE)

    def test_unsupported_api_is_execution(self):
        c = classify("module 'Part' has no attribute 'makeGear'")
        self.assertEqual(c.family, EXECUTION)
        self.assertEqual(c.signature, "has no attribute")

    def test_null_shape_is_geometric(self):
        c = classify("Exception while processing: Null shape")
        self.assertEqual(c.family, GEOMETRIC)

    def test_boolean_is_geometric(self):
        self.assertEqual(classify("Boolean operation failed").family, GEOMETRIC)

    def test_overconstraint_is_geometric(self):
        self.assertEqual(classify("sketch is overconstrained").family, GEOMETRIC)

    def test_syntax_error(self):
        self.assertEqual(classify("SyntaxError: invalid syntax").family, SYNTAX)
        self.assertEqual(classify("IndentationError: unexpected indent").family, SYNTAX)

    def test_unknown_is_execution_catch_all(self):
        c = classify("some totally unrecognised failure")
        self.assertEqual(c.family, EXECUTION)
        self.assertEqual(c.signature, "")

    def test_returns_classification(self):
        self.assertIsInstance(classify("Null shape"), ErrorClassification)

    def test_case_insensitive(self):
        self.assertEqual(classify("NULL SHAPE").family, GEOMETRIC)


class TallyTests(unittest.TestCase):
    def test_tally_counts(self):
        errs = ["", "Null shape", "has no attribute 'x'",
                "SyntaxError: bad", "Boolean failed"]
        t = tally(errs)
        self.assertEqual(t[NONE], 1)
        self.assertEqual(t[GEOMETRIC], 2)
        self.assertEqual(t[EXECUTION], 1)
        self.assertEqual(t[SYNTAX], 1)

    def test_tally_sums_to_len(self):
        errs = ["", "Null shape", "weird"]
        self.assertEqual(sum(tally(errs).values()), len(errs))


if __name__ == "__main__":
    unittest.main()
