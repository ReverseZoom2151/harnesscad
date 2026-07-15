import unittest

from harnesscad.domain.spec.scad_parameters import parse_parameters


class TestScadParameters(unittest.TestCase):
    def test_number_with_range(self):
        params = parse_parameters("width = 10; // [1:50]\n")
        self.assertEqual(len(params), 1)
        p = params[0]
        self.assertEqual(p.name, "width")
        self.assertEqual(p.display_name, "Width")
        self.assertEqual(p.type, "number")
        self.assertEqual(p.value, 10.0)
        self.assertEqual(p.range, {"min": 1.0, "max": 50.0})

    def test_min_step_max(self):
        params = parse_parameters("h = 10; // [1:1:50]\n")
        self.assertEqual(params[0].range, {"min": 1.0, "max": 50.0, "step": 1.0})

    def test_enum_options_with_labels(self):
        params = parse_parameters('color = "red"; // [red, green, blue]\n')
        p = params[0]
        self.assertEqual(p.type, "string")
        self.assertEqual([o.value for o in p.options], ["red", "green", "blue"])

    def test_bare_number_is_step_for_numeric(self):
        params = parse_parameters("n = 3; // 2\n")
        self.assertEqual(params[0].range, {"step": 2.0})

    def test_boolean(self):
        params = parse_parameters("flag = true;\n")
        self.assertEqual(params[0].type, "boolean")
        self.assertIs(params[0].value, True)

    def test_description_from_line_above(self):
        src = "// The overall width\nwidth = 10; // [1:50]\n"
        params = parse_parameters(src)
        self.assertEqual(params[0].description, "The overall width")

    def test_group_sections(self):
        src = "/* [Sizes] */\nwidth = 10;\nheight = 20;\n"
        params = parse_parameters(src)
        self.assertTrue(all(p.group == "Sizes" for p in params))

    def test_numeric_vector_flattened(self):
        params = parse_parameters("size = [10, 20, 30];\n")
        self.assertEqual([p.name for p in params], ["size[0]", "size[1]", "size[2]"])
        self.assertEqual([p.display_name for p in params],
                         ["Size Width", "Size Depth", "Size Height"])
        self.assertEqual([p.value for p in params], [10.0, 20.0, 30.0])

    def test_implementation_after_module_excluded(self):
        src = "width = 10;\nmodule body() { internal = 5; }\n"
        params = parse_parameters(src)
        self.assertEqual([p.name for p in params], ["width"])

    def test_expression_reference_skipped(self):
        src = "base = 10;\nderived = base * 2;\n"
        params = parse_parameters(src)
        self.assertEqual([p.name for p in params], ["base"])

    def test_fn_resolution_name(self):
        params = parse_parameters("$fn = 64;\n")
        self.assertEqual(params[0].display_name, "Resolution")

    def test_determinism(self):
        src = "// w\nwidth = 10; // [1:50]\n/* [G] */\ncolor = \"red\"; // [red, blue]\n"
        a = parse_parameters(src)
        b = parse_parameters(src)
        self.assertEqual([(p.name, p.value, p.group, p.range) for p in a],
                         [(p.name, p.value, p.group, p.range) for p in b])


if __name__ == "__main__":
    unittest.main()
