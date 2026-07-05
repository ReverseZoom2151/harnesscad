import unittest

from spec.clarify_scaling import (
    BuildStep, parse_step, parse_steps, detect_scaling, rewrite_steps,
    SKETCH, SCALE, EXTRUDE,
)


def steps_with_scale():
    return parse_steps([
        "On the XY workplane sketch a circle with radius 0.1293.",
        "Apply a scaling factor of 0.2586 to the entire sketch.",
        "Extrude the sketch by 0.75 along the normal direction.",
    ])


class TestParse(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(parse_step("sketch a circle").kind, SKETCH)
        self.assertEqual(parse_step("extrude by 5").kind, EXTRUDE)
        s = parse_step("apply a scaling factor of 0.2586 to the sketch")
        self.assertEqual(s.kind, SCALE)
        self.assertAlmostEqual(s.factor, 0.2586)


class TestDetect(unittest.TestCase):
    def test_pre_extrude_scale_flagged(self):
        issues = detect_scaling(steps_with_scale())
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].index, 1)
        self.assertTrue(issues[0].hallucination_risk)
        self.assertEqual(issues[0].interpretation, "unknown")

    def test_post_extrude_scale_not_flagged(self):
        steps = parse_steps([
            "sketch a circle radius 5",
            "extrude 10",
            "scale the solid by 2",
        ])
        self.assertEqual(detect_scaling(steps), [])

    def test_literal_interpretation(self):
        # sketch extent 0.1293, factor 0.2586, final 0.03345 -> literal.
        issues = detect_scaling(steps_with_scale(),
                                sketch_extent=0.1293, final_extent=0.033448)
        self.assertEqual(issues[0].interpretation, "literal")

    def test_redundant_interpretation(self):
        # sketch already equals final extent -> scale is redundant.
        issues = detect_scaling(steps_with_scale(),
                                sketch_extent=0.2586, final_extent=0.2586)
        self.assertEqual(issues[0].interpretation, "redundant")


class TestRewrite(unittest.TestCase):
    def test_redundant_scale_dropped(self):
        out = rewrite_steps(steps_with_scale(),
                            sketch_extent=0.2586, final_extent=0.2586)
        self.assertEqual([s.kind for s in out], [SKETCH, EXTRUDE])
        self.assertFalse(any(s.kind == SCALE for s in out))

    def test_literal_scale_folded_into_sketch(self):
        out = rewrite_steps(steps_with_scale(),
                            sketch_extent=0.1293, final_extent=0.033448)
        self.assertFalse(any(s.kind == SCALE for s in out))
        # a sketch step now carries the fold-in factor.
        self.assertTrue(any(s.kind == SKETCH and s.factor == 0.2586
                            for s in out))

    def test_no_scale_unchanged(self):
        steps = parse_steps(["sketch a rect 200 by 150", "extrude 7"])
        out = rewrite_steps(steps)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
