import unittest

from harnesscad.data.dataengine.annotation import design_rules as k


CYL = [k.ADD_SKETCH, k.ADD_CIRCLE, k.ADD_EXTRUDE]
TRI = [k.ADD_SKETCH, k.ADD_LINE, k.ADD_LINE, k.ADD_LINE, k.ADD_EXTRUDE]


class TestSignatureAndTemplate(unittest.TestCase):
    def test_signature_strips_markers(self):
        seq = [k.SOP, k.ADD_SKETCH, k.ADD_CIRCLE, k.ADD_EXTRUDE, k.EOP, k.EOP]
        self.assertEqual(k.op_type_signature(seq), tuple(CYL))

    def test_infer_cylinder(self):
        self.assertEqual(k.infer_template(CYL), "cylinder")

    def test_infer_triangular_prism(self):
        self.assertEqual(k.infer_template(TRI), "triangular_prism")

    def test_infer_unknown(self):
        self.assertIsNone(k.infer_template([k.ADD_SKETCH, k.ADD_EXTRUDE]))


class TestPartAttributes(unittest.TestCase):
    def test_counts_triprism(self):
        attrs = k.part_attributes([k.SOP] + TRI + [k.EOP])
        self.assertEqual(attrs.n_sketches, 1)
        self.assertEqual(attrs.n_lines, 3)
        self.assertEqual(attrs.n_extrudes, 1)
        self.assertEqual(attrs.n_curves, 3)
        self.assertEqual(attrs.template, "triangular_prism")

    def test_counts_cylinder_curves(self):
        attrs = k.part_attributes(CYL)
        self.assertEqual(attrs.n_circles, 1)
        self.assertEqual(attrs.n_curves, 1)


class TestDesignRules(unittest.TestCase):
    def setUp(self):
        # program: sketch, circle(center-x, r), extrude(d)
        self.program = [
            {"t": k.ADD_SKETCH},
            {"t": k.ADD_CIRCLE, "x": 0.4, "y": 0.0, "r": 0.5},
            {"t": k.ADD_EXTRUDE, "d": 0.0},
        ]
        # rule: extrude depth = 2 * circle center-x  (paper's example)
        self.rule = k.DesignRule(target=(2, "d"), source=(1, "x"),
                                 scale=2.0, name="depth_from_center")

    def test_apply_rules_sets_target(self):
        out = k.apply_rules(self.program, [self.rule])
        self.assertAlmostEqual(out[2]["d"], 0.8)
        # original untouched (copy semantics)
        self.assertEqual(self.program[2]["d"], 0.0)

    def test_check_rules_detects_violation(self):
        rc = k.check_rules(self.program, [self.rule])
        self.assertFalse(rc["per_rule"]["depth_from_center"])
        self.assertEqual(rc["fidelity"], 0.0)

    def test_check_rules_after_apply(self):
        out = k.apply_rules(self.program, [self.rule])
        rc = k.check_rules(out, [self.rule])
        self.assertTrue(rc["per_rule"]["depth_from_center"])
        self.assertEqual(rc["fidelity"], 1.0)

    def test_rule_evaluate_and_holds(self):
        self.assertAlmostEqual(self.rule.evaluate(self.program), 0.8)
        self.assertFalse(self.rule.holds(self.program))


class TestSynthesis(unittest.TestCase):
    def test_deterministic_with_seed(self):
        a = k.synthesize_program(CYL, seed=7)
        b = k.synthesize_program(CYL, seed=7)
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        a = k.synthesize_program(CYL, seed=1)
        b = k.synthesize_program(CYL, seed=2)
        self.assertNotEqual(a, b)

    def test_parameters_within_ranges(self):
        prog = k.synthesize_program(CYL, seed=3)
        circle = prog[1]
        self.assertTrue(-1.0 <= circle["x"] <= 1.0)
        self.assertTrue(0.0 < circle["r"] <= 1.0)

    def test_rule_based_synthesis_satisfies_rules(self):
        rule = k.DesignRule(target=(2, "d"), source=(1, "x"), scale=1.0,
                            name="r")
        prog = k.synthesize_program(CYL, seed=5, rules=[rule])
        self.assertTrue(k.check_rules(prog, [rule])["per_rule"]["r"])


class TestKnowledgeRecord(unittest.TestCase):
    def test_infer_knowledge_without_rules(self):
        rec = k.infer_knowledge(CYL)
        self.assertEqual(rec.template, "cylinder")
        self.assertEqual(rec.rule_fidelity, 1.0)
        self.assertEqual(rec.rules_total, 0)

    def test_infer_knowledge_with_rules(self):
        rule = k.DesignRule(target=(2, "d"), source=(1, "x"), name="r")
        prog = k.synthesize_program(CYL, seed=9, rules=[rule])
        rec = k.infer_knowledge(CYL, program=prog, rules=[rule])
        self.assertEqual(rec.rules_total, 1)
        self.assertEqual(rec.rules_satisfied, 1)
        self.assertEqual(rec.rule_fidelity, 1.0)


if __name__ == "__main__":
    unittest.main()
