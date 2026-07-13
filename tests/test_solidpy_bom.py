"""Tests for programs.solidpy_bom."""

import unittest

from harnesscad.domain.programs.runtime.solidpy_bom import (
    BOM_TRAIT,
    bill_of_materials,
    bom_part,
    bom_rows,
    bom_traits,
    table_string,
)
from harnesscad.domain.programs.emit.solidpy_scad_emit import cube, cylinder, translate, union


@bom_part("M3x16 Screw", 0.12, currency="US$", vendor="Acme")
def screw():
    return cylinder(r=1.5, h=16)


@bom_part("Bearing 608", 1.50, currency="US$", vendor="Bearings Inc")
def bearing():
    return cylinder(r=11, h=7)


@bom_part()
def plate():
    return cube([50, 50, 3])


@bom_part("Sticker")
def sticker():
    return cube([10, 10, 0.1])


def assembly():
    parts = [translate((10 * i, 0, 0))(screw()) for i in range(4)]
    return union()(plate(), bearing(), bearing(), sticker(), *parts)


class TestTraits(unittest.TestCase):
    def test_trait_attached(self):
        node = screw()
        trait = node.get_trait(BOM_TRAIT)
        self.assertEqual(trait["name"], "M3x16 Screw")
        self.assertEqual(trait["unit_price"], 0.12)
        self.assertEqual(trait["vendor"], "Acme")

    def test_name_defaults_to_function_name(self):
        self.assertEqual(plate().get_trait(BOM_TRAIT)["name"], "plate")

    def test_decorator_preserves_metadata(self):
        self.assertEqual(screw.__name__, "screw")

    def test_non_node_return_rejected(self):
        @bom_part("Bad")
        def bad():
            return 42

        with self.assertRaises(TypeError):
            bad()

    def test_traits_found_through_transforms(self):
        node = translate((1, 0, 0))(screw())
        self.assertEqual(len(bom_traits(node)), 1)

    def test_untagged_tree_is_empty(self):
        self.assertEqual(bom_traits(union()(cube(1))), [])


class TestRows(unittest.TestCase):
    def test_counts_aggregate_by_name(self):
        rows = {r["name"]: r["count"] for r in bom_rows(assembly())}
        self.assertEqual(rows["M3x16 Screw"], 4)
        self.assertEqual(rows["Bearing 608"], 2)
        self.assertEqual(rows["plate"], 1)

    def test_row_order_is_first_seen(self):
        names = [r["name"] for r in bom_rows(assembly())]
        self.assertEqual(names,
                         ["plate", "Bearing 608", "Sticker", "M3x16 Screw"])


class TestTable(unittest.TestCase):
    def test_table_string_aligns(self):
        out = table_string(["A", "BB"], [["x", "yyy"]])
        lines = out.splitlines()
        self.assertEqual(lines[0], "+---+-----+")
        self.assertEqual(lines[1], "| A | BB  |")
        self.assertEqual(lines[3], "| x | yyy |")

    def test_csv_is_tab_separated(self):
        out = table_string(["A", "B"], [[1, 2]], csv=True)
        self.assertEqual(out, "A\tB\n1\t2\n")


class TestBillOfMaterials(unittest.TestCase):
    def test_totals(self):
        out = bill_of_materials(assembly(), csv=True)
        lines = out.splitlines()
        self.assertEqual(lines[0].split("\t")[:4],
                         ["Description", "Count", "Unit Price", "Total Price"])
        screws = [line for line in lines if line.startswith("M3x16 Screw")][0]
        cells = screws.split("\t")
        self.assertEqual(cells[1], "4")
        self.assertEqual(cells[2], "US$ 0.12")
        self.assertEqual(cells[3], "US$ 0.48")
        total = [line for line in lines if line.startswith("Total Cost")][0]
        # 4 * 0.12 + 2 * 1.50 = 3.48; plate and sticker have no price
        self.assertIn("US$ 3.48", total)

    def test_unpriced_parts_have_blank_prices(self):
        out = bill_of_materials(assembly(), csv=True)
        plate_line = [line for line in out.splitlines()
                      if line.startswith("plate")][0]
        self.assertEqual(plate_line.split("\t"), ["plate", "1", "", ""])

    def test_extra_headers(self):
        out = bill_of_materials(assembly(), headers=["vendor"], csv=True)
        lines = out.splitlines()
        self.assertEqual(lines[0].split("\t")[-1], "vendor")
        screws = [line for line in lines if line.startswith("M3x16 Screw")][0]
        self.assertEqual(screws.split("\t")[-1], "Acme")
        plate_line = [line for line in lines if line.startswith("plate")][0]
        self.assertEqual(plate_line.split("\t")[-1], "")

    def test_multiple_currencies(self):
        @bom_part("Euro part", 2.0, currency="EUR")
        def euro_part():
            return cube(1)

        model = union()(screw(), euro_part())
        out = bill_of_materials(model, csv=True)
        self.assertIn("US$ 0.12", out)
        self.assertIn("EUR 2.00", out)
        totals = [line for line in out.splitlines() if line.startswith("Total Cost")]
        self.assertEqual(len(totals), 2)

    def test_empty_bom(self):
        out = bill_of_materials(union()(cube(1)), csv=True)
        self.assertEqual(out.splitlines(), [
            "Description\tCount\tUnit Price\tTotal Price"])

    def test_determinism(self):
        self.assertEqual(bill_of_materials(assembly()),
                         bill_of_materials(assembly()))

    def test_pretty_table_renders(self):
        out = bill_of_materials(assembly())
        self.assertTrue(out.startswith("+"))
        self.assertIn("| M3x16 Screw", out)


if __name__ == "__main__":
    unittest.main()
