import unittest

from harnesscad.agents.agent.assembly_trace import (
    AssemblyStep,
    AssemblyTrace,
    build_trace,
    component_numeracy,
    rationale_alignment,
    trace_stability,
)


class BuildTraceTests(unittest.TestCase):
    def test_respects_dependencies(self):
        trace = build_trace(
            ["base", "leg1", "leg2", "top"],
            {"leg1": ["base"], "leg2": ["base"], "top": ["leg1", "leg2"]},
        )
        self.assertEqual(trace.steps[0].parts, ("base",))
        self.assertEqual(set(trace.steps[1].parts), {"leg1", "leg2"})
        self.assertEqual(trace.steps[2].parts, ("top",))
        self.assertEqual(trace.part_count(), 4)

    def test_no_dependencies_single_step(self):
        trace = build_trace(["a", "b", "c"])
        self.assertEqual(len(trace.steps), 1)
        self.assertEqual(trace.steps[0].parts, ("a", "b", "c"))

    def test_cycle_raises(self):
        with self.assertRaises(ValueError):
            build_trace(["a", "b"], {"a": ["b"], "b": ["a"]})

    def test_unknown_dependency_raises(self):
        with self.assertRaises(ValueError):
            build_trace(["a"], {"a": ["ghost"]})

    def test_monotone_trace_is_stable(self):
        trace = build_trace(["base", "leg"], {"leg": ["base"]})
        self.assertTrue(trace_stability(trace))


class GraderTests(unittest.TestCase):
    def test_component_numeracy_exact(self):
        trace = build_trace(["a", "b", "c"])
        self.assertEqual(component_numeracy(trace, 3), 1.0)

    def test_component_numeracy_undercount(self):
        trace = build_trace(["a", "b"])
        self.assertAlmostEqual(component_numeracy(trace, 4), 0.5)

    def test_component_numeracy_requires_positive(self):
        with self.assertRaises(ValueError):
            component_numeracy(build_trace(["a"]), 0)

    def test_trace_stability_rejects_empty_step(self):
        trace = AssemblyTrace((AssemblyStep(("a",)), AssemblyStep(())))
        self.assertFalse(trace_stability(trace))

    def test_trace_stability_rejects_reused_part(self):
        trace = AssemblyTrace((AssemblyStep(("a",)), AssemblyStep(("a",))))
        self.assertFalse(trace_stability(trace))

    def test_rationale_alignment_default_is_full(self):
        trace = build_trace(["base", "leg"], {"leg": ["base"]})
        self.assertEqual(rationale_alignment(trace), 1.0)

    def test_rationale_alignment_partial(self):
        trace = AssemblyTrace((
            AssemblyStep(("base",), "add the base plate"),
            AssemblyStep(("leg",), "attach the support"),  # 'leg' not mentioned
        ))
        self.assertAlmostEqual(rationale_alignment(trace), 0.5)

    def test_rationale_alignment_empty(self):
        self.assertEqual(rationale_alignment(AssemblyTrace(())), 0.0)


if __name__ == "__main__":
    unittest.main()
