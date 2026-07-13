import unittest

from harnesscad.domain.programs.review import cadreview_taxonomy as tax


class TestTaxonomy(unittest.TestCase):
    def test_eight_error_types(self):
        self.assertEqual(len(tax.ERROR_TYPES), 8)
        self.assertEqual(len(tax.ALL_TYPES), 9)  # + No error
        self.assertNotIn(tax.NO_ERROR, tax.ERROR_TYPES)

    def test_labels_match_paper(self):
        expected = {
            "Primitive error", "Rotation error", "Position error", "Size error",
            "Constant error", "Logic error", "Missing block", "Redundant block",
        }
        self.assertEqual(set(tax.labels()), expected)

    def test_unique_ids_and_labels(self):
        ids = tax.ids()
        self.assertEqual(len(ids), len(set(ids)))
        labs = tax.labels()
        self.assertEqual(len(labs), len(set(labs)))

    def test_by_id_and_from_label(self):
        self.assertIs(tax.by_id("rotation_error"), tax.ROTATION_ERROR)
        self.assertIsNone(tax.by_id("nope"))
        self.assertIs(tax.from_label("Missing block"), tax.MISSING_BLOCK)
        self.assertIs(tax.from_label("  missing   BLOCK "), tax.MISSING_BLOCK)
        self.assertIs(tax.from_label("No error"), tax.NO_ERROR)
        self.assertIsNone(tax.from_label("banana"))
        self.assertIsNone(tax.from_label(""))

    def test_each_type_has_fix_action(self):
        for t in tax.ERROR_TYPES:
            self.assertTrue(t.fix_action and t.fix_action != "none")
            self.assertIn("id", t.to_dict())


if __name__ == "__main__":
    unittest.main()
