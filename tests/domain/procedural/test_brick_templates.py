import unittest

from harnesscad.domain.procedural.brick_templates import (
    CATEGORY_TEMPLATES,
    generate_bottle,
    generate_car,
    generate_chair,
    generate_table,
    generate_tower,
    get_template,
)


class TestBrickTemplates(unittest.TestCase):
    def test_deterministic_same_seed(self):
        a = generate_table(6, 4, 3, seed=7)
        b = generate_table(6, 4, 3, seed=7)
        self.assertEqual(a, b)

    def test_no_global_rng_leak(self):
        import random
        random.seed(999)
        first = random.random()
        generate_table(8, 6, 3, seed=1)
        generate_chair(4, 4, 5, seed=2)
        random.seed(999)
        self.assertEqual(first, random.random())

    def test_total_bricks_matches_list(self):
        for cat in CATEGORY_TEMPLATES:
            m = get_template(cat, 6, 4, 4, seed=3)
            self.assertEqual(m["total_bricks"], len(m["bricks"]), cat)
            self.assertGreater(m["total_bricks"], 0, cat)

    def test_ids_unique_and_contiguous(self):
        m = generate_tower(4, 4, 8, seed=5)
        ids = [b["id"] for b in m["bricks"]]
        self.assertEqual(ids, list(range(len(ids))))

    def test_within_bounds(self):
        m = generate_chair(6, 6, 6, seed=11)
        for b in m["bricks"]:
            self.assertTrue(0 <= b["x"] < m["width"])
            self.assertTrue(0 <= b["y"] < m["depth"])
            self.assertTrue(0 <= b["z"] < m["height"])

    def test_dimension_clamping(self):
        # Oversized request is clamped into the valid table range.
        m = generate_table(100, 100, 100, seed=1)
        self.assertLessEqual(m["width"], 12)
        self.assertLessEqual(m["depth"], 8)
        self.assertLessEqual(m["height"], 4)

    def test_layer_equals_z_plus_one(self):
        m = generate_car(6, 3, 2, seed=1)
        for b in m["bricks"]:
            self.assertEqual(b["layer"], b["z"] + 1)

    def test_tower_base_solid(self):
        m = generate_tower(5, 5, 6, seed=0)
        base = [b for b in m["bricks"] if b["z"] == 0]
        self.assertEqual(len(base), m["width"] * m["depth"])

    def test_bottle_narrows(self):
        m = generate_bottle(3, 3, 9, seed=0)
        base_cells = {(b["x"], b["y"]) for b in m["bricks"] if b["z"] == 0}
        top_cells = {(b["x"], b["y"]) for b in m["bricks"] if b["z"] == m["height"] - 1}
        self.assertLess(len(top_cells), len(base_cells))

    def test_unknown_category_falls_back(self):
        m = get_template("nonsense", 4, 4, 6, seed=1)
        self.assertEqual(m["object_type"], "tower")


if __name__ == "__main__":
    unittest.main()
