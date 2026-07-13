import unittest

from harnesscad.domain.spec.intent_categories import (
    CATEGORIES,
    map_to_categories,
    normalize_category,
    validate_prompt,
)


class TestIntentCategories(unittest.TestCase):
    def test_direct_match(self):
        self.assertEqual(map_to_categories("a sturdy tower")[0], "tower")

    def test_synonym(self):
        self.assertIn("mug", map_to_categories("a coffee cup"))
        self.assertIn("sofa", map_to_categories("a soft couch"))

    def test_semantic_verb(self):
        cats = map_to_categories("something to sit on")
        self.assertIn("chair", cats)

    def test_top_k_limit(self):
        self.assertLessEqual(len(map_to_categories("furniture to sit and eat", top_k=3)), 3)

    def test_fallback_small(self):
        self.assertEqual(map_to_categories("a tiny thing"), ["basket", "bowl", "mug"])

    def test_fallback_default(self):
        self.assertEqual(map_to_categories("zxqw"), ["tower", "table", "chair"])

    def test_deterministic(self):
        self.assertEqual(map_to_categories("store books on a shelf"),
                         map_to_categories("store books on a shelf"))

    def test_normalize_known(self):
        self.assertEqual(normalize_category("Chair"), "chair")
        self.assertEqual(normalize_category("cup"), "mug")
        self.assertEqual(normalize_category("arm chair"), "chair")

    def test_normalize_unknown(self):
        self.assertEqual(normalize_category("spaceship"), "table")
        self.assertEqual(normalize_category(None), "table")

    def test_all_categories_normalize_to_self(self):
        for c in CATEGORIES:
            self.assertEqual(normalize_category(c), c)

    def test_validate_clean(self):
        v = validate_prompt("a rectangular table with four legs")
        self.assertTrue(v.valid)
        self.assertEqual(v.suggested_category, "table")

    def test_validate_out_of_scope(self):
        v = validate_prompt("a round curved organic floating blob")
        self.assertFalse(v.valid)
        self.assertGreater(len(v.warnings), 2)

    def test_validate_material_warning(self):
        v = validate_prompt("a wooden chair")
        self.assertTrue(any("brick" in w for w in v.warnings))


if __name__ == "__main__":
    unittest.main()
