import unittest

from harnesscad.data.dataengine.text2cad_prompt_generator import (
    PromptGeneratorError,
    extract_aspects,
    generate_all_levels,
    generate_prompt,
)
from harnesscad.data.dataengine.text2cad_prompt_levels import classify_prompt_level
from harnesscad.domain.reconstruction.deepcad_command_spec import command


def _two_circle_extrude():
    # Two concentric circles in one loop, then an extrusion.
    return [
        command("SOL"),
        command("Circle", x=0.5, y=0.5, r=0.4),
        command("Circle", x=0.5, y=0.5, r=0.2),
        command("Ext", theta=0.0, phi=0.0, gamma=0.0, px=0.0, py=0.0, pz=0.0,
                s=1.0, e1=0.25, e2=0.0, b=0, u=0),
    ]


def _rectangle_extrude():
    return [
        command("SOL"),
        command("Line", x=0.0, y=0.0),
        command("Line", x=1.0, y=0.0),
        command("Line", x=1.0, y=0.5),
        command("Line", x=0.0, y=0.5),
        command("Ext", theta=0.0, phi=0.0, gamma=0.0, px=0.0, py=0.0, pz=0.0,
                s=1.0, e1=0.3, e2=0.1, b=1, u=0),
    ]


class ExtractAspectsTests(unittest.TestCase):
    def test_circle_counts(self):
        a = extract_aspects(_two_circle_extrude())
        self.assertEqual(a.circles, 2)
        self.assertEqual(a.lines, 0)
        self.assertEqual(a.n_loops, 1)
        self.assertEqual(a.n_extrusions, 1)
        self.assertEqual(a.extrude_distances, ((0.25, 0.0),))
        self.assertEqual(a.booleans, ("new body",))

    def test_rectangle_counts(self):
        a = extract_aspects(_rectangle_extrude())
        self.assertEqual(a.lines, 4)
        self.assertEqual(a.circles, 0)
        self.assertEqual(a.n_loops, 1)
        self.assertEqual(a.booleans, ("cut",))

    def test_two_loops(self):
        cmds = [
            command("SOL"), command("Circle", x=0.5, y=0.5, r=0.4),
            command("SOL"), command("Circle", x=0.5, y=0.5, r=0.2),
            command("Ext", theta=0, phi=0, gamma=0, px=0, py=0, pz=0,
                    s=1, e1=0.2, e2=0, b=0, u=0),
        ]
        a = extract_aspects(cmds)
        self.assertEqual(a.n_loops, 2)
        self.assertEqual(a.circles, 2)


class GeneratePromptTests(unittest.TestCase):
    def setUp(self):
        self.cmds = _two_circle_extrude()

    def test_l0_is_shape_verbatim(self):
        p = generate_prompt(self.cmds, "L0", shape_description="two concentric cylinders")
        self.assertEqual(p, "two concentric cylinders")

    def test_l0_requires_shape(self):
        with self.assertRaises(PromptGeneratorError):
            generate_prompt(self.cmds, "L0")

    def test_l1_has_shape_and_simple_steps_no_numbers(self):
        p = generate_prompt(self.cmds, "L1", shape_description="a ring")
        self.assertIn("a ring", p)
        self.assertIn("circle", p)
        # Beginner text must not leak the extrude distance value.
        self.assertNotIn("0.25", p)

    def test_l2_mentions_extrusion_generally(self):
        p = generate_prompt(self.cmds, "L2", shape_description="a ring")
        self.assertIn("extrude", p.lower())
        self.assertNotIn("0.25", p)

    def test_l3_has_coordinate_system_and_precise_value(self):
        p = generate_prompt(self.cmds, "L3")
        self.assertIn("coordinate system", p.lower())
        self.assertIn("0.25", p)

    def test_levels_classify_back_correctly(self):
        # Round-trip: generated prompts should be recognised at (near) their level.
        self.assertEqual(classify_prompt_level(generate_prompt(
            self.cmds, "L3")), "L3")
        self.assertEqual(classify_prompt_level(generate_prompt(
            self.cmds, "L2", shape_description="a ring")), "L2")

    def test_deterministic(self):
        a = generate_prompt(self.cmds, "L3")
        b = generate_prompt(self.cmds, "L3")
        self.assertEqual(a, b)


class GenerateAllLevelsTests(unittest.TestCase):
    def test_all_four_levels(self):
        out = generate_all_levels(_rectangle_extrude(), "a flat rectangular plate")
        self.assertEqual(set(out), {"L0", "L1", "L2", "L3"})
        # Detail should grow: L3 longer than L0.
        self.assertGreater(len(out["L3"]), len(out["L0"]))
        # Cut operation should surface at expert level.
        self.assertIn("cut", out["L3"].lower())


if __name__ == "__main__":
    unittest.main()
