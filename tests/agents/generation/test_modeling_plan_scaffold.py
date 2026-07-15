import unittest

from harnesscad.agents.generation import modeling_plan_scaffold as mps


class TestClassify(unittest.TestCase):
    def test_global_parameter(self):
        self.assertEqual(
            mps.classify_line("Outer radius Rout = 0.4, wall thickness 0.1"),
            "global_parameters")

    def test_primary_geometry(self):
        self.assertEqual(
            mps.classify_line("On the XY workplane sketch a circle and extrude"),
            "primary_geometry")

    def test_secondary_feature(self):
        self.assertEqual(
            mps.classify_line("A base plate is integrated via Boolean union"),
            "secondary_features")

    def test_pattern_logic(self):
        self.assertEqual(
            mps.classify_line("Arranged in a 2x2 linear array along X and Y"),
            "pattern_logic")

    def test_topological_refinement(self):
        self.assertEqual(
            mps.classify_line("Apply a fillet radius of 0.03 to the base edges"),
            "topological_refinement")

    def test_refinement_wins_over_dimension(self):
        # mentions a radius (param cue) but is really a chamfer refinement line
        self.assertEqual(
            mps.classify_line("Chamfer the top edges with width 2mm"),
            "topological_refinement")

    def test_unclassified(self):
        self.assertEqual(mps.classify_line("this is a nice part"), "")


class TestStructure(unittest.TestCase):
    def setUp(self):
        self.lines = [
            "Global radius R = 0.5 and height H = 0.8",
            "On the XY plane sketch a circle then extrude",
            "Cut a central hole through the body",
            "Mirror the feature across the YZ plane",
            "Fillet the four vertical edges",
            "",
            "some prose with no cue",
        ]

    def test_all_sections_present(self):
        s = mps.structure_description(self.lines)
        for key in mps.SECTIONS:
            self.assertIn(key, s)
        self.assertTrue(s["global_parameters"])
        self.assertTrue(s["primary_geometry"])
        self.assertTrue(s["secondary_features"])
        self.assertTrue(s["pattern_logic"])
        self.assertTrue(s["topological_refinement"])

    def test_unclassified_collected(self):
        s = mps.structure_description(self.lines)
        self.assertEqual(len(s[""]), 1)

    def test_coverage_full(self):
        s = mps.structure_description(self.lines)
        cov = mps.coverage(s)
        self.assertTrue(all(cov.values()))

    def test_coverage_partial(self):
        s = mps.structure_description(["Fillet the edges"])
        cov = mps.coverage(s)
        self.assertTrue(cov["topological_refinement"])
        self.assertFalse(cov["primary_geometry"])


class TestScaffoldRender(unittest.TestCase):
    def test_empty_scaffold_order(self):
        sc = mps.empty_scaffold()
        self.assertEqual([s.key for s in sc], list(mps.SECTIONS))
        self.assertEqual(sc[0].title, "Global Parameters and Constants")

    def test_render_has_all_titles_and_placeholder(self):
        s = mps.structure_description(["Fillet the edges"])
        text = mps.render(s)
        self.assertIn("Topological Refinement", text)
        self.assertIn("Global Parameters and Constants", text)
        self.assertIn("(none)", text)  # empty sections get a placeholder

    def test_render_deterministic(self):
        s = mps.structure_description(["On XY plane sketch circle extrude"])
        self.assertEqual(mps.render(s), mps.render(s))


if __name__ == "__main__":
    unittest.main()
