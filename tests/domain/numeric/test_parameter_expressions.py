import math
import unittest

from harnesscad.domain.numeric.parameter_expressions import (
    CyclicParameterError,
    ExpressionError,
    ParameterTable,
    build_table,
    evaluate,
    extract_symbols,
)


class TestEvaluate(unittest.TestCase):
    def test_arithmetic_and_namespace(self):
        self.assertEqual(evaluate("2 * x + 1", {"x": 5}), 11.0)
        self.assertAlmostEqual(evaluate("sqrt(a**2 + b**2)", {"a": 3, "b": 4}), 5.0)

    def test_constants_and_functions(self):
        self.assertAlmostEqual(evaluate("degrees(pi)"), 180.0)
        self.assertAlmostEqual(evaluate("max(1, 2, 3) + min(4, 5)"), 7.0)
        self.assertAlmostEqual(evaluate("atan2(1, 1)"), math.pi / 4)

    def test_unary_and_precedence(self):
        self.assertEqual(evaluate("-3 + 2 * 4"), 5.0)

    def test_unknown_symbol(self):
        with self.assertRaises(ExpressionError):
            evaluate("width + 1")

    def test_rejects_attribute_access(self):
        with self.assertRaises(ExpressionError):
            evaluate("os.system", {})

    def test_rejects_disallowed_call(self):
        with self.assertRaises(ExpressionError):
            evaluate("eval('1')")

    def test_rejects_keyword_args_and_bad_syntax(self):
        with self.assertRaises(ExpressionError):
            evaluate("round(1.5, ndigits=1)")
        with self.assertRaises(ExpressionError):
            evaluate("2 +")

    def test_division_by_zero(self):
        with self.assertRaises(ExpressionError):
            evaluate("1 / 0")


class TestExtractSymbols(unittest.TestCase):
    def test_excludes_constants_and_functions(self):
        self.assertEqual(
            extract_symbols("sqrt(w * h) + pi * r"), {"w", "h", "r"}
        )

    def test_no_symbols(self):
        self.assertEqual(extract_symbols("1 + 2"), set())


class TestParameterTable(unittest.TestCase):
    def _table(self):
        table = ParameterTable()
        table.set("thickness", 2.0)
        table.set("clearance", 0.5)
        table.set_expr("wall", "thickness * 2 + clearance")
        table.set_expr("outer", "wall + 10")
        return table

    def test_evaluate_all_in_dependency_order(self):
        values = self._table().evaluate_all()
        self.assertAlmostEqual(values["wall"], 4.5)
        self.assertAlmostEqual(values["outer"], 14.5)

    def test_order_independent_of_insertion(self):
        table = ParameterTable()
        table.set_expr("outer", "wall + 10")
        table.set_expr("wall", "thickness * 2")
        table.set("thickness", 3.0)
        self.assertAlmostEqual(table.evaluate_all()["outer"], 16.0)
        self.assertEqual(table.evaluation_order()[0], "thickness")

    def test_dependencies_and_dependents(self):
        table = self._table()
        self.assertEqual(table.dependencies("outer"), {"wall", "thickness", "clearance"})
        self.assertEqual(table.dependents("thickness"), {"wall", "outer"})
        self.assertEqual(table.dependents("outer"), set())

    def test_cycle_detection(self):
        table = ParameterTable()
        table.set_expr("a", "b + 1")
        table.set_expr("b", "a + 1")
        with self.assertRaises(CyclicParameterError):
            table.evaluate_all()

    def test_self_reference(self):
        table = ParameterTable()
        table.set_expr("a", "a + 1")
        with self.assertRaises(CyclicParameterError):
            table.evaluation_order()

    def test_unknown_reference(self):
        table = ParameterTable()
        table.set_expr("a", "missing * 2")
        with self.assertRaises(ExpressionError):
            table.evaluate_all()

    def test_evaluate_one_and_extra_namespace(self):
        table = ParameterTable()
        table.set_expr("bore", "shaft + fit")
        table.set("fit", 0.1)
        self.assertAlmostEqual(table.evaluate_one("bore", {"shaft": 10.0}), 10.1)

    def test_build_table_helper_and_determinism(self):
        table = build_table([("d", 4.0), ("r", "d / 2"), ("area", "pi * r**2")])
        first = table.evaluate_all()
        second = table.evaluate_all()
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["area"], math.pi * 4.0)

    def test_missing_parameter_raises(self):
        with self.assertRaises(ExpressionError):
            ParameterTable().dependencies("nope")


if __name__ == "__main__":
    unittest.main()
