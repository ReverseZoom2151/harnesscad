import unittest

from harnesscad.domain.procedural.brick_templates import generate_car, generate_tower
from harnesscad.domain.procedural.voxel_compose import compose, mutate_dims


class TestCompose(unittest.TestCase):
    def test_ids_unique_after_compose(self):
        a = generate_car(6, 3, 2, seed=1)
        b = generate_tower(4, 4, 6, seed=2)
        c = compose([a, b])
        ids = [br["id"] for br in c["bricks"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(c["total_bricks"], a["total_bricks"] + b["total_bricks"])

    def test_no_x_overlap(self):
        a = generate_car(6, 3, 2, seed=1)
        b = generate_car(6, 3, 2, seed=1)
        c = compose([a, b], spacing=2)
        a_max_x = max(br["x"] for br in c["bricks"][: a["total_bricks"]])
        b_min_x = min(br["x"] for br in c["bricks"][a["total_bricks"]:])
        self.assertGreater(b_min_x, a_max_x)

    def test_width_spans_arrangement(self):
        a = generate_car(6, 3, 2, seed=1)
        b = generate_tower(4, 4, 6, seed=1)
        c = compose([a, b], spacing=2)
        # 6 + 2 gap + 4 = 12
        self.assertEqual(c["width"], 12)
        self.assertEqual(c["height"], 6)

    def test_empty_compose(self):
        c = compose([])
        self.assertEqual(c["total_bricks"], 0)

    def test_deterministic(self):
        a = generate_car(6, 3, 2, seed=1)
        self.assertEqual(compose([a, a]), compose([a, a]))


class TestMutate(unittest.TestCase):
    def test_no_mutation(self):
        self.assertEqual(mutate_dims(6, 4, 3, None), (6, 4, 3))
        self.assertEqual(mutate_dims(6, 4, 3, "spin"), (6, 4, 3))

    def test_taller(self):
        _, _, h = mutate_dims(6, 4, 10, "taller")
        self.assertEqual(h, 13)

    def test_wider_clamped(self):
        w, _, _ = mutate_dims(12, 4, 3, "wider")
        self.assertEqual(w, 12)

    def test_shorter_floor(self):
        _, _, h = mutate_dims(6, 4, 2, "shorter")
        self.assertEqual(h, 2)

    def test_deeper(self):
        _, d, _ = mutate_dims(6, 4, 3, "deeper")
        self.assertEqual(d, 5)

    def test_alias_signs(self):
        self.assertEqual(mutate_dims(6, 4, 10, "+height"),
                         mutate_dims(6, 4, 10, "taller"))


if __name__ == "__main__":
    unittest.main()
