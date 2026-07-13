"""Tests for programs.paramgeom_classify (C1..C5 category classifier + tally)."""

import unittest

from harnesscad.domain.programs.expressions.classify import (
    Category,
    FormativeTally,
    classify_expression,
    classify_vector,
)


class ClassifyTest(unittest.TestCase):
    def test_c1_raw_number(self):
        self.assertEqual(classify_expression("4.0").category, Category.C1_RAW_NUMBER)

    def test_c1_negative_number(self):
        self.assertEqual(classify_expression("-3").category, Category.C1_RAW_NUMBER)

    def test_c1_constant_arithmetic(self):
        # 2+3 evaluates to a constant -> still a raw number
        self.assertEqual(classify_expression("2 + 3").category, Category.C1_RAW_NUMBER)

    def test_c2_one_variable(self):
        self.assertEqual(classify_expression("var1").category, Category.C2_ONE_VARIABLE)

    def test_c2_is_not_scaled_variable(self):
        # 2*var1 has coefficient != 1 -> linear combination, not "one variable"
        self.assertEqual(
            classify_expression("2*var1").category, Category.C3_LINEAR_COMBINATION
        )

    def test_c3_linear_combination(self):
        self.assertEqual(
            classify_expression("3 + 2*var1 - var2").category,
            Category.C3_LINEAR_COMBINATION,
        )

    def test_c3_paper_translate_example(self):
        # size_cube_a/2 + size_cube_b/2 (Listing 2)
        self.assertEqual(
            classify_expression("size_cube_a/2 + size_cube_b/2").category,
            Category.C3_LINEAR_COMBINATION,
        )

    def test_c4_polynomial_product(self):
        self.assertEqual(
            classify_expression("3 + 2*var1*var2").category, Category.C4_POLYNOMIAL
        )

    def test_c4_size_times_index(self):
        # translate([0,0,size_x*i]) -> C4 per the paper
        self.assertEqual(
            classify_expression("size_x*i").category, Category.C4_POLYNOMIAL
        )

    def test_c5_conditional(self):
        self.assertEqual(
            classify_expression("(var1>3)?1:2").category, Category.C5_OTHER
        )

    def test_c5_function_call(self):
        self.assertEqual(classify_expression("sin(x)").category, Category.C5_OTHER)

    def test_c5_unparseable(self):
        self.assertEqual(classify_expression("@@@").category, Category.C5_OTHER)

    def test_category_label(self):
        self.assertEqual(Category.C3_LINEAR_COMBINATION.label, "Linear combination")


class ClassifyVectorTest(unittest.TestCase):
    def test_mixed_vector(self):
        # cube(size = [5, size_y, size_z+3]) -> C1, C2, C3
        cats = [c.category for c in classify_vector(["5", "size_y", "size_z+3"])]
        self.assertEqual(
            cats,
            [
                Category.C1_RAW_NUMBER,
                Category.C2_ONE_VARIABLE,
                Category.C3_LINEAR_COMBINATION,
            ],
        )


class FormativeTallyTest(unittest.TestCase):
    def test_add_and_count(self):
        t = FormativeTally()
        t.add(Category.C1_RAW_NUMBER, "primitive", 196)
        t.add(Category.C3_LINEAR_COMBINATION, "translate", 234)
        self.assertEqual(t.count(Category.C1_RAW_NUMBER, "primitive"), 196)
        self.assertEqual(t.count(Category.C3_LINEAR_COMBINATION, "translate"), 234)

    def test_unknown_kind_raises(self):
        t = FormativeTally()
        with self.assertRaises(ValueError):
            t.add(Category.C1_RAW_NUMBER, "extrude")

    def test_negative_count_raises(self):
        t = FormativeTally()
        with self.assertRaises(ValueError):
            t.add(Category.C1_RAW_NUMBER, "primitive", -1)

    def test_totals_and_percentage(self):
        t = FormativeTally()
        t.add(Category.C1_RAW_NUMBER, "primitive", 3)
        t.add(Category.C2_ONE_VARIABLE, "translate", 1)
        self.assertEqual(t.grand_total(), 4)
        self.assertEqual(t.row_total(Category.C1_RAW_NUMBER), 3)
        self.assertEqual(t.column_total("primitive"), 3)
        self.assertAlmostEqual(t.percentage(Category.C1_RAW_NUMBER, "primitive"), 75.0)

    def test_classify_and_add(self):
        t = FormativeTally()
        t.classify_and_add("3 + 2*a - b", "translate")
        self.assertEqual(t.count(Category.C3_LINEAR_COMBINATION, "translate"), 1)

    def test_linear_share(self):
        # 71% linear (C1+C2+C3) as in the paper's headline finding
        t = FormativeTally()
        t.add(Category.C1_RAW_NUMBER, "primitive", 30)
        t.add(Category.C2_ONE_VARIABLE, "primitive", 20)
        t.add(Category.C3_LINEAR_COMBINATION, "translate", 21)
        t.add(Category.C4_POLYNOMIAL, "translate", 20)
        t.add(Category.C5_OTHER, "translate", 9)
        self.assertAlmostEqual(t.linear_share(), 71.0)

    def test_empty_tally_percentages_zero(self):
        t = FormativeTally()
        self.assertEqual(t.grand_total(), 0)
        self.assertEqual(t.percentage(Category.C1_RAW_NUMBER, "primitive"), 0.0)
        self.assertEqual(t.linear_share(), 0.0)


if __name__ == "__main__":
    unittest.main()
