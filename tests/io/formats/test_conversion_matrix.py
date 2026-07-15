"""Tests for the format-conversion matrix."""

import unittest

from harnesscad.io.formats.conversion_matrix import (
    ConversionError,
    FORMATS,
    can_convert,
    conversion_matrix,
    conversion_path,
    convert,
    direct_targets,
    kind_can_lower,
)


class TestKindLattice(unittest.TestCase):
    def test_lower_forward_allowed(self):
        self.assertTrue(kind_can_lower("brep", "mesh"))
        self.assertTrue(kind_can_lower("csg", "mesh"))
        self.assertTrue(kind_can_lower("program", "mesh"))
        self.assertTrue(kind_can_lower("brep", "drawing"))

    def test_raise_backward_forbidden(self):
        self.assertFalse(kind_can_lower("mesh", "brep"))
        self.assertFalse(kind_can_lower("mesh", "csg"))
        self.assertFalse(kind_can_lower("drawing", "brep"))

    def test_reflexive(self):
        for k in ("mesh", "brep", "csg", "drawing", "image", "program"):
            self.assertTrue(kind_can_lower(k, k))


class TestDirectConversion(unittest.TestCase):
    def test_mesh_to_mesh(self):
        self.assertTrue(can_convert("stl", "obj"))
        self.assertTrue(can_convert("obj", "glb"))

    def test_brep_to_mesh(self):
        self.assertTrue(can_convert("step", "stl"))

    def test_mesh_cannot_become_brep(self):
        self.assertFalse(can_convert("stl", "step"))

    def test_write_only_target_ok_source_not(self):
        # svg is write-only: nothing converts *from* it, but brep converts *to* it.
        self.assertTrue(can_convert("step", "svg"))
        self.assertFalse(can_convert("svg", "dxf"))

    def test_unknown_format_raises(self):
        with self.assertRaises(KeyError):
            can_convert("stl", "nope")

    def test_direct_targets_sorted_and_excludes_self(self):
        tg = direct_targets("step")
        self.assertNotIn("step", tg)
        self.assertEqual(list(tg), sorted(tg))
        self.assertIn("stl", tg)
        self.assertIn("svg", tg)


class TestPaths(unittest.TestCase):
    def test_same_format_is_singleton_path(self):
        self.assertEqual(conversion_path("stl", "stl"), ["stl"])

    def test_direct_path_length_two(self):
        p = conversion_path("step", "stl")
        self.assertEqual(p[0], "step")
        self.assertEqual(p[-1], "stl")
        self.assertEqual(len(p), 2)

    def test_multi_hop_program_to_image(self):
        # program -> ... -> image requires lowering through mesh/drawing.
        p = conversion_path("kcl", "png")
        self.assertIsNotNone(p)
        self.assertEqual(p[0], "kcl")
        self.assertEqual(p[-1], "png")

    def test_no_path_returns_none(self):
        # nothing can be read from png (write-only) nor raised to brep.
        self.assertIsNone(conversion_path("stl", "step"))

    def test_path_is_deterministic(self):
        self.assertEqual(conversion_path("kcl", "png"), conversion_path("kcl", "png"))


class TestMatrix(unittest.TestCase):
    def test_matrix_is_square_over_all_formats(self):
        m = conversion_matrix()
        self.assertEqual(set(m), set(FORMATS))
        for row in m.values():
            self.assertEqual(set(row), set(FORMATS))

    def test_diagonal_true(self):
        m = conversion_matrix()
        for f in FORMATS:
            self.assertTrue(m[f][f])

    def test_matrix_matches_can_convert(self):
        m = conversion_matrix()
        self.assertEqual(m["step"]["stl"], can_convert("step", "stl"))
        self.assertEqual(m["stl"]["step"], can_convert("stl", "step"))


class TestConvertDispatch(unittest.TestCase):
    def test_dispatch_applies_handlers_in_order(self):
        handlers = {
            ("step", "stl"): lambda v: v + "|step2stl",
        }
        out = convert("part", "step", "stl", handlers)
        self.assertEqual(out, "part|step2stl")

    def test_multi_hop_dispatch(self):
        path = conversion_path("kcl", "png")
        handlers = {(a, b): (lambda a=a, b=b: (lambda v: v + f"|{a}>{b}"))()
                    for a, b in zip(path, path[1:])}
        out = convert("x", "kcl", "png", handlers)
        self.assertTrue(out.startswith("x|"))
        self.assertEqual(out.count("|"), len(path) - 1)

    def test_missing_handler_raises(self):
        with self.assertRaises(ConversionError):
            convert("x", "step", "stl", handlers={})

    def test_no_path_raises(self):
        with self.assertRaises(ConversionError):
            convert("x", "stl", "step", handlers={})


if __name__ == "__main__":
    unittest.main()
