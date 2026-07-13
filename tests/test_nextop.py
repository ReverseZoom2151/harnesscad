import unittest

from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.cisp.ops import AddCircle, AddInstance, Extrude, NewSketch
from harnesscad.eval.quality.nextop import (
    NextOperationRanker,
    reciprocal_rank,
    rank_next_operations,
    top_k_accuracy,
)
from harnesscad.core.state.opdag import OpDAG
from harnesscad.eval.verifiers.verify import Diagnostic, Severity


def apply(dag, backend, op):
    result = backend.apply(op)
    assert result.ok, result.diagnostics
    dag.append(op)


class NextOperationRankerTests(unittest.TestCase):
    def setUp(self):
        self.dag = OpDAG()
        self.backend = StubBackend()

    def test_empty_design_starts_with_sketch(self):
        ranked = rank_next_operations(self.dag, self.backend)
        self.assertEqual("new_sketch", ranked[0].op)
        self.assertEqual(1.0, ranked[0].confidence)

    def test_sketch_prefers_geometry(self):
        apply(self.dag, self.backend, NewSketch())
        ranked = NextOperationRanker().rank(self.dag, self.backend, top_k=4)
        self.assertEqual(
            ["add_circle", "add_line", "add_point", "add_rectangle"],
            [item.op for item in ranked],
        )

    def test_profile_suggests_constraint_then_solid_features(self):
        apply(self.dag, self.backend, NewSketch())
        apply(self.dag, self.backend, AddCircle(sketch="sk1", r=5))
        ranked = rank_next_operations(self.dag, self.backend, top_k=8)
        tags = [item.op for item in ranked]
        self.assertEqual("constrain", tags[0])
        self.assertLess(tags.index("extrude"), tags.index("revolve"))
        self.assertNotIn("hole", tags)

    def test_solid_only_exposes_solid_valid_operations(self):
        apply(self.dag, self.backend, NewSketch())
        apply(self.dag, self.backend, AddCircle(sketch="sk1", r=5))
        apply(self.dag, self.backend, Extrude(sketch="sk1", distance=10))
        ranked = rank_next_operations(self.dag, self.backend, top_k=20)
        tags = [item.op for item in ranked]
        self.assertIn("hole", tags)
        self.assertIn("fillet", tags)
        self.assertIn("add_instance", tags)
        self.assertNotIn("boolean", tags)  # cannot prove a second body exists

    def test_two_instances_enable_mate(self):
        apply(self.dag, self.backend, NewSketch())
        apply(self.dag, self.backend, AddCircle(sketch="sk1", r=5))
        apply(self.dag, self.backend, Extrude(sketch="sk1", distance=10))
        apply(self.dag, self.backend, AddInstance(part="solid"))
        apply(self.dag, self.backend, AddInstance(part="solid", x=20))
        ranked = rank_next_operations(self.dag, self.backend, top_k=20)
        self.assertIn("mate", [item.op for item in ranked])

    def test_error_prioritizes_repair_and_accepts_dict_diagnostic(self):
        apply(self.dag, self.backend, NewSketch())
        ranked = rank_next_operations(
            self.dag,
            self.backend,
            [{"severity": "error", "code": "bad-value"}],
        )
        self.assertEqual("set_param", ranked[0].op)

    def test_bad_reference_prioritizes_prerequisite(self):
        apply(self.dag, self.backend, NewSketch())
        diagnostic = Diagnostic(Severity.ERROR, "bad-ref", "missing")
        ranked = rank_next_operations(self.dag, self.backend, [diagnostic], top_k=2)
        self.assertEqual(["set_param", "new_sketch"], [item.op for item in ranked])

    def test_deterministic_and_json_safe(self):
        first = rank_next_operations(self.dag, self.backend)
        second = rank_next_operations(self.dag, self.backend)
        self.assertEqual(first, second)
        self.assertEqual("new_sketch", first[0].to_dict()["op"])

    def test_top_k_validation(self):
        self.assertEqual([], rank_next_operations(self.dag, self.backend, top_k=0))
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                rank_next_operations(self.dag, self.backend, top_k=value)

    def test_metrics(self):
        ranked = rank_next_operations(self.dag, self.backend)
        self.assertEqual(1.0, top_k_accuracy(ranked, "new_sketch", k=1))
        self.assertEqual(0.0, top_k_accuracy(ranked, "extrude", k=1))
        self.assertEqual(1.0, reciprocal_rank(ranked, "new_sketch"))
        self.assertEqual(0.0, reciprocal_rank(ranked, "extrude"))


if __name__ == "__main__":
    unittest.main()
