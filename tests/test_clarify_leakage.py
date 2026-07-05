import unittest

from spec.clarify_leakage import check_leakage, style_warnings


class TestHardFails(unittest.TestCase):
    def test_import_leak(self):
        r = check_leakage("First import cadquery as cq, then sketch.")
        self.assertTrue(r.contains_code)
        self.assertTrue(any("import cadquery" in s for s in
                            r.detected_code_snippets))

    def test_method_chain_leak(self):
        r = check_leakage("Use wp.extrude(0.75) to make the solid.")
        self.assertTrue(r.contains_code)
        self.assertTrue(any(".extrude(" in s for s in r.detected_code_snippets))

    def test_workplane_call_leak(self):
        r = check_leakage("Start with cq.Workplane('XY').")
        self.assertTrue(r.contains_code)

    def test_python_keyword_leak(self):
        r = check_leakage("def build(): return solid")
        self.assertTrue(r.contains_code)

    def test_identifier_reuse_leak(self):
        r = check_leakage("Set r_out to 5 and w0 accordingly.",
                          original_identifiers=["r_out", "w0"])
        self.assertTrue(r.contains_code)
        self.assertIn("r_out", r.detected_code_snippets)


class TestAllowed(unittest.TestCase):
    def test_plain_geometry_is_clean(self):
        text = ("Sketch a rectangle 200 by 150 on the XY plane with origin "
                "moved to (-4, -100, -75). Extrude 7 in the positive normal "
                "direction.")
        r = check_leakage(text)
        self.assertFalse(r.contains_code)
        self.assertEqual(r.detected_code_snippets, ())

    def test_origin_word_allowed(self):
        r = check_leakage("Use the XY workplane; the origin moved to (0,0,0).")
        self.assertFalse(r.contains_code)

    def test_spec_assignment_allowed(self):
        r = check_leakage("radius = 10, origin = (-100, 0, -12).")
        self.assertFalse(r.contains_code)

    def test_coordinate_tuples_allowed(self):
        r = check_leakage("Vertices at (0,0), (0,200), (36,200), (91,0).")
        self.assertFalse(r.contains_code)

    def test_identifier_origin_not_flagged(self):
        r = check_leakage("the origin is shifted",
                          original_identifiers=["origin", "workplane"])
        self.assertFalse(r.contains_code)


class TestJsonAndStyle(unittest.TestCase):
    def test_json_shape(self):
        j = check_leakage("plain text").to_json()
        self.assertIn("contains_code", j)
        self.assertIn("detected_code_snippets", j)
        self.assertIn("explanation", j)

    def test_style_warning(self):
        self.assertTrue(style_warnings("place at ZX @ (-64, 9, -36)"))
        self.assertFalse(style_warnings("place on the ZX plane"))


if __name__ == "__main__":
    unittest.main()
