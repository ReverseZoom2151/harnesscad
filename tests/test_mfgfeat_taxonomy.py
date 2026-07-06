"""Tests for fabrication/mfgfeat_taxonomy.py."""

from __future__ import annotations

import unittest

from fabrication import mfgfeat_taxonomy as tax


class TestHierarchy(unittest.TestCase):
    def test_primary_categories(self):
        self.assertEqual(
            tax.PRIMARY_CATEGORIES,
            ("machining", "extrusion", "freeform", "molding_casting",
             "sheet_metal"),
        )

    def test_leaf_features_unique_and_indexed(self):
        self.assertEqual(len(tax.LEAF_FEATURES), len(set(tax.LEAF_FEATURES)))
        # Every leaf maps to exactly one (category, subcategory).
        for leaf in tax.LEAF_FEATURES:
            self.assertTrue(tax.is_leaf(leaf))
            cat = tax.category_of(leaf)
            sub = tax.subcategory_of(leaf)
            self.assertIn(cat, tax.PRIMARY_CATEGORIES)
            self.assertIn(leaf, tax.leaves_of_category(cat))
            self.assertIsInstance(sub, str)

    def test_known_leaves_present(self):
        for leaf in ("hole", "slot", "step", "pocket", "chamfer", "fillet",
                     "thread", "gear_teeth", "neck", "pipe_tube", "boss",
                     "depression", "protrusion", "rib", "gusset", "draft",
                     "bend"):
            self.assertIn(leaf, tax.LEAF_FEATURES)

    def test_category_membership(self):
        self.assertEqual(tax.category_of("hole"), "machining")
        self.assertEqual(tax.category_of("boss"), "extrusion")
        self.assertEqual(tax.category_of("rib"), "molding_casting")
        self.assertEqual(tax.category_of("bend"), "sheet_metal")
        self.assertEqual(tax.subcategory_of("chamfer"), "edges_and_contours")

    def test_leaves_of_category_covers_all(self):
        collected = []
        for cat in tax.PRIMARY_CATEGORIES:
            collected.extend(tax.leaves_of_category(cat))
        self.assertEqual(sorted(collected), sorted(tax.LEAF_FEATURES))

    def test_unknown_category_raises(self):
        with self.assertRaises(KeyError):
            tax.leaves_of_category("nope")


class TestNormalisation(unittest.TestCase):
    def test_aliases_map_to_holes(self):
        for name in ("blind hole", "through hole", "countersink",
                     "counterbored hole", "tapered hole", "Holes", "bore"):
            self.assertEqual(tax.normalize_feature(name), "hole")

    def test_separator_variants(self):
        self.assertEqual(tax.normalize_feature("gear-teeth"), "gear_teeth")
        self.assertEqual(tax.normalize_feature("gear teeth"), "gear_teeth")
        self.assertEqual(tax.normalize_feature("pipe/tube"), "pipe_tube")
        self.assertEqual(tax.normalize_feature("PIPE TUBE"), "pipe_tube")

    def test_canonical_labels_roundtrip(self):
        for leaf in tax.LEAF_FEATURES:
            self.assertEqual(tax.normalize_feature(leaf), leaf)

    def test_unknown_raises(self):
        with self.assertRaises(KeyError):
            tax.normalize_feature("wormhole")

    def test_try_normalize_default(self):
        self.assertIsNone(tax.try_normalize("wormhole"))
        self.assertEqual(tax.try_normalize("wormhole", "?"), "?")
        self.assertEqual(tax.try_normalize("bevel"), "chamfer")


class TestAttributesAndSubtypes(unittest.TestCase):
    def test_attributes_defined_for_every_leaf(self):
        for leaf in tax.LEAF_FEATURES:
            self.assertIn(leaf, tax.FEATURE_ATTRIBUTES)
            self.assertGreater(len(tax.attributes_of(leaf)), 0)

    def test_hole_attributes(self):
        self.assertIn("diameter", tax.attributes_of("hole"))
        self.assertIn("subtype", tax.attributes_of("hole"))

    def test_hole_subtypes(self):
        for st in ("blind", "through", "countersink", "counterbore",
                   "tapered", "threaded", "simple"):
            self.assertTrue(tax.is_hole_subtype(st))
        self.assertFalse(tax.is_hole_subtype("banana"))


if __name__ == "__main__":
    unittest.main()
