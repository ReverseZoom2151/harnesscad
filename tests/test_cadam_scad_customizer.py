"""Tests for programs.cadam_scad_customizer (OpenSCAD Customizer parser)."""

import unittest

from programs.cadam_scad_customizer import parse_parameters, Parameter


def by_name(params):
    return {p.name: p for p in params}


class TestBasicTypes(unittest.TestCase):
    def test_number_boolean_string(self):
        src = (
            "width = 10;\n"
            "enabled = true;\n"
            'label = "hello";\n'
        )
        p = by_name(parse_parameters(src))
        self.assertEqual(p["width"].type, "number")
        self.assertEqual(p["width"].value, 10.0)
        self.assertEqual(p["enabled"].type, "boolean")
        self.assertIs(p["enabled"].value, True)
        self.assertEqual(p["label"].type, "string")
        self.assertEqual(p["label"].value, "hello")

    def test_negative_and_float(self):
        p = by_name(parse_parameters("offset = -3.5;"))
        self.assertEqual(p["offset"].value, -3.5)

    def test_expression_skipped(self):
        # References another variable -> not a constant, skipped.
        src = "a = 10;\nb = a + 1;\n"
        p = by_name(parse_parameters(src))
        self.assertIn("a", p)
        self.assertNotIn("b", p)


class TestRangesAndOptions(unittest.TestCase):
    def test_min_max(self):
        p = by_name(parse_parameters("w = 10; // [1:50]"))
        self.assertEqual(p["w"].range, {"min": 1.0, "max": 50.0})

    def test_min_step_max(self):
        p = by_name(parse_parameters("w = 10; // [1:1:50]"))
        self.assertEqual(p["w"].range, {"min": 1.0, "step": 1.0, "max": 50.0})

    def test_bare_number_is_step_for_numeric(self):
        p = by_name(parse_parameters("w = 10; // 5"))
        self.assertEqual(p["w"].range, {"step": 5.0})

    def test_bare_number_is_maxlength_for_string(self):
        p = by_name(parse_parameters('name = "x"; // 20'))
        self.assertEqual(p["name"].range, {"max": 20.0})

    def test_enum_string_options(self):
        p = by_name(parse_parameters('c = "red"; // [red, green, blue]'))
        vals = [o.value for o in p["c"].options]
        self.assertEqual(vals, ["red", "green", "blue"])

    def test_enum_value_label_pairs(self):
        p = by_name(parse_parameters('c = "a"; // [a:Apple, b:Banana]'))
        opts = p["c"].options
        self.assertEqual(opts[0].value, "a")
        self.assertEqual(opts[0].label, "Apple")
        self.assertEqual(opts[1].label, "Banana")

    def test_numeric_enum_values_parsed(self):
        p = by_name(parse_parameters("n = 2; // [1, 2, 4]"))
        vals = [o.value for o in p["n"].options]
        self.assertEqual(vals, [1.0, 2.0, 4.0])


class TestGroupsAndDescriptions(unittest.TestCase):
    def test_groups(self):
        src = (
            "/* [Geometry] */\n"
            "w = 10;\n"
            "/* [Appearance] */\n"
            'color = "red";\n'
        )
        p = by_name(parse_parameters(src))
        self.assertEqual(p["w"].group, "Geometry")
        self.assertEqual(p["color"].group, "Appearance")

    def test_description_from_line_above(self):
        src = "// The overall width\nwidth = 10;\n"
        p = by_name(parse_parameters(src))
        self.assertEqual(p["width"].description, "The overall width")

    def test_no_description(self):
        p = by_name(parse_parameters("width = 10;\n"))
        self.assertIsNone(p["width"].description)


class TestDisplayNamesAndModules(unittest.TestCase):
    def test_snake_case_to_title(self):
        p = by_name(parse_parameters("root_chord = 120;"))
        self.assertEqual(p["root_chord"].display_name, "Root Chord")

    def test_fn_special_name(self):
        p = by_name(parse_parameters("$fn = 64;"))
        self.assertEqual(p["$fn"].display_name, "Resolution")

    def test_declarations_below_module_ignored(self):
        src = (
            "width = 10;\n"
            "module part() {\n"
            "  internal = 5;\n"
            "}\n"
        )
        p = by_name(parse_parameters(src))
        self.assertIn("width", p)
        self.assertNotIn("internal", p)

    def test_leading_underscore_name(self):
        # Must not crash on empty tokens from leading/trailing underscores.
        p = by_name(parse_parameters("__width = 3;"))
        self.assertIn("__width", p)
        self.assertEqual(p["__width"].display_name, "Width")


class TestVectorFlattening(unittest.TestCase):
    def test_number_vector_flattened_xyz(self):
        p = by_name(parse_parameters("pos = [1, 2, 3];"))
        self.assertIn("pos[0]", p)
        self.assertIn("pos[2]", p)
        self.assertEqual(p["pos[0]"].value, 1.0)
        self.assertEqual(p["pos[0]"].display_name, "Pos X")
        self.assertEqual(p["pos[2]"].display_name, "Pos Z")

    def test_size_vector_uses_wdh_labels(self):
        p = by_name(parse_parameters("body_size = [10, 20, 30];"))
        # "Size" suffix dropped, replaced with axis words.
        self.assertEqual(p["body_size[0]"].display_name, "Body Width")
        self.assertEqual(p["body_size[1]"].display_name, "Body Depth")

    def test_two_element_vector_xy(self):
        p = by_name(parse_parameters("uv = [0.5, 0.25];"))
        self.assertEqual(p["uv[0]"].display_name, "Uv X")
        self.assertEqual(p["uv[1]"].display_name, "Uv Y")


class TestIntegrationAndDeterminism(unittest.TestCase):
    SRC = (
        "// Tapered Wing\n"
        "/* [Wing Geometry] */\n"
        "root_chord = 120;  // [50:10:300]\n"
        "tip_chord = 80;    // [30:10:200]\n"
        "/* [Appearance] */\n"
        '// Wing colour\n'
        'wing_color = "SteelBlue"; // [SteelBlue, Silver, Orange]\n'
        "$fn = 64;\n"
        "function naca_camber(x) = x;\n"
        "hidden = 99;\n"
    )

    def test_end_to_end(self):
        params = parse_parameters(self.SRC)
        p = by_name(params)
        self.assertEqual(set(p), {"root_chord", "tip_chord", "wing_color", "$fn"})
        self.assertEqual(p["root_chord"].group, "Wing Geometry")
        self.assertEqual(p["root_chord"].range, {"min": 50.0, "step": 10.0, "max": 300.0})
        self.assertEqual(p["wing_color"].group, "Appearance")
        self.assertEqual(p["wing_color"].description, "Wing colour")
        self.assertEqual(len(p["wing_color"].options), 3)
        self.assertNotIn("hidden", p)

    def test_deterministic(self):
        a = parse_parameters(self.SRC)
        b = parse_parameters(self.SRC)
        self.assertEqual([x.name for x in a], [x.name for x in b])
        self.assertEqual([x.value for x in a], [x.value for x in b])

    def test_order_preserved(self):
        names = [x.name for x in parse_parameters(self.SRC)]
        self.assertEqual(names, ["root_chord", "tip_chord", "wing_color", "$fn"])

    def test_returns_parameter_instances(self):
        for x in parse_parameters(self.SRC):
            self.assertIsInstance(x, Parameter)


if __name__ == "__main__":
    unittest.main()
