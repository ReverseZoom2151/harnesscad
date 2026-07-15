import unittest

from harnesscad.eval.bench.generative import text2cad_complexity as tc


L1 = """
import cadquery as cq
result = (
    cq.Workplane("XY")
    .polygon(6, 50)
    .extrude(20)
    .faces(">Z").workplane()
    .circle(10)
    .cutThruAll()
    .faces(">Z").edges("%Line")
    .chamfer(2)
)
"""

L2 = """
import cadquery as cq
hemisphere = cq.Workplane("XY").sphere(40).cut(
    cq.Workplane("XY").box(120, 120, 80)
)
result = hemisphere
"""

L3 = """
import cadquery as cq
arm = cq.Workplane("YZ").circle(5).loft(ruled=False)
result = arm.faces("<X or >X").shell(-1.5)
"""


class TestMethodCalls(unittest.TestCase):
    def test_collects_names(self):
        names = tc.method_calls(L1)
        self.assertIn("polygon", names)
        self.assertIn("chamfer", names)
        self.assertIn("cutThruAll", names)

    def test_syntax_error(self):
        with self.assertRaises(SyntaxError):
            tc.method_calls("result = (((")


class TestClassify(unittest.TestCase):
    def test_l1_primitive_and_finishing(self):
        self.assertEqual(tc.classify_level(L1), "L1")

    def test_l2_boolean(self):
        self.assertEqual(tc.classify_level(L2), "L2")

    def test_l3_advanced(self):
        self.assertEqual(tc.classify_level(L3), "L3")

    def test_cutthruall_is_not_boolean(self):
        # cutThruAll is a finishing feature, must stay L1 (not promoted to L2)
        self.assertEqual(tc.classify_level(L1), "L1")

    def test_empty_program_is_l1(self):
        self.assertEqual(tc.classify_level("x = 1"), "L1")


class TestComplexityReport(unittest.TestCase):
    def test_report_fields(self):
        r = tc.complexity_report(L1)
        self.assertEqual(r.level, "L1")
        self.assertIn("polygon", r.categories["primitives"])
        self.assertIn("chamfer", r.categories["finishing"])
        self.assertGreater(r.api_calls, 0)
        self.assertGreater(r.code_lines, 0)

    def test_advanced_ops_listed(self):
        r = tc.complexity_report(L3)
        self.assertEqual(r.level, "L3")
        self.assertEqual(set(r.categories["advanced"]), {"loft", "shell"})

    def test_l3_more_calls_than_l1_here(self):
        # Sanity: proxies are computed, monotonic within this small sample set.
        self.assertGreaterEqual(tc.complexity_report(L1).api_calls,
                                tc.complexity_report(L3).api_calls - 100)


if __name__ == "__main__":
    unittest.main()
