import unittest
from random import Random

from harnesscad.domain.programs.runtime.show_object import (
    DEFAULT_COLOR_SEED,
    infer_object_name,
    collect_shown_objects,
    rand_color,
    color_sequence,
)


class TestInferObjectName(unittest.TestCase):
    def test_finds_bound_name(self):
        box = ["a shape"]
        ns = {"box": box, "other": 5}
        self.assertEqual(infer_object_name(box, ns), "box")

    def test_first_matching_name(self):
        v = 42
        ns = {"a": 42, "b": 99}
        self.assertEqual(infer_object_name(v, ns), "a")

    def test_fallback_to_id_string(self):
        obj = object()
        name = infer_object_name(obj, {"x": 1})
        self.assertEqual(name, str(id(obj)))

    def test_explicit_fallback(self):
        obj = object()
        self.assertEqual(infer_object_name(obj, {"x": 1}, fallback="anon"), "anon")


class TestCollectShownObjects(unittest.TestCase):
    def test_filters_by_predicate(self):
        ns = {"a": 1, "b": "str", "c": 2}
        result = collect_shown_objects(ns, lambda v: isinstance(v, int))
        self.assertEqual(result, {"a": 1, "c": 2})

    def test_skips_private_names(self):
        ns = {"_hidden": 1, "shown": 2}
        result = collect_shown_objects(ns, lambda v: True)
        self.assertEqual(result, {"shown": 2})

    def test_preserves_order(self):
        ns = {"z": 1, "y": 1, "x": 1}
        result = collect_shown_objects(ns, lambda v: True)
        self.assertEqual(list(result.keys()), ["z", "y", "x"])


class TestRandColor(unittest.TestCase):
    def test_default_is_reproducible(self):
        self.assertEqual(rand_color(), rand_color())

    def test_dict_form_bounds(self):
        c = rand_color(alpha=0.3)
        self.assertEqual(c["alpha"], 0.3)
        for ch in c["color"]:
            self.assertTrue(10 <= ch <= 100)
            self.assertIsInstance(ch, int)

    def test_float_form(self):
        r, g, b, a = rand_color(alpha=0.5, cfloat=True)
        self.assertEqual(a, 0.5)
        for ch in (r, g, b):
            self.assertTrue(10 / 255 <= ch <= 100 / 255)

    def test_explicit_rng_matches_reference(self):
        rng = Random(DEFAULT_COLOR_SEED)
        ref = Random(DEFAULT_COLOR_SEED)
        c = rand_color(rng=rng)
        expected = (
            ref.randint(10, 100),
            ref.randint(10, 100),
            ref.randint(10, 100),
        )
        self.assertEqual(c["color"], expected)

    def test_shared_rng_advances(self):
        rng = Random(DEFAULT_COLOR_SEED)
        first = rand_color(rng=rng)
        second = rand_color(rng=rng)
        self.assertNotEqual(first, second)


class TestColorSequence(unittest.TestCase):
    def test_length(self):
        self.assertEqual(len(color_sequence(5)), 5)

    def test_reproducible(self):
        self.assertEqual(color_sequence(4), color_sequence(4))

    def test_seed_changes_output(self):
        self.assertNotEqual(color_sequence(3, seed=1), color_sequence(3, seed=2))

    def test_zero(self):
        self.assertEqual(color_sequence(0), [])

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            color_sequence(-1)


if __name__ == "__main__":
    unittest.main()
