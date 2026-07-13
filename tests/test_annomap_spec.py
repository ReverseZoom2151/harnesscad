import unittest

from harnesscad.domain.drawings.annomap_parser import CADFeature, parse_callout
from harnesscad.domain.drawings.annomap_scoring import assign_features
from harnesscad.domain.drawings.annomap_spec import (
    METHOD_DETERMINISTIC,
    METHOD_HUMAN,
    MappingMetrics,
    UnifiedSpec,
    apply_human_edit,
    build_spec,
    evaluate_mapping,
    macro_average,
)


def _feat(ftype, params=None, fid="F1"):
    return CADFeature(ftype, params or {}, 1.0, feature_id=fid)


class BuildSpecTests(unittest.TestCase):
    def test_basic_mapping_and_unmapped(self):
        feats = [_feat("hole", {"diameter": 10.0}, "F1"),
                 _feat("fillet", {"radius": 3.0}, "F2")]
        ents = [parse_callout("Ø10", entity_id="E1"),
                parse_callout("R3", entity_id="E2"),
                parse_callout("NOTE HERE", entity_id="E3")]
        assigns = assign_features(feats, ents)
        spec = build_spec(assigns, ["E1", "E2", "E3"])
        pairs = spec.pairs()
        self.assertIn(("F1", "E1"), pairs)
        self.assertIn(("F2", "E2"), pairs)
        self.assertIn("E3", spec.unmapped_entities)

    def test_unmapped_feature_recorded(self):
        feats = [_feat("fillet", {"radius": 3.0}, "F1")]
        ents = [parse_callout("Ø10", entity_id="E1")]  # incompatible
        assigns = assign_features(feats, ents)
        spec = build_spec(assigns, ["E1"])
        self.assertEqual(spec.unmapped_features, ["F1"])
        self.assertIn("E1", spec.unmapped_entities)

    def test_provenance_method(self):
        feats = [_feat("hole", {"diameter": 10.0}, "F1")]
        ents = [parse_callout("Ø10", entity_id="E1")]
        assigns = assign_features(feats, ents)
        spec = build_spec(assigns, ["E1"], method=METHOD_DETERMINISTIC)
        self.assertEqual(spec.mappings[0].method, METHOD_DETERMINISTIC)
        self.assertTrue(spec.mappings[0].rationale)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            build_spec([], [], method="magic")

    def test_to_dict(self):
        spec = UnifiedSpec()
        self.assertIn("mappings", spec.to_dict())


class HumanEditTests(unittest.TestCase):
    def test_add_binding(self):
        spec = UnifiedSpec(unmapped_entities=["E9"])
        edited = apply_human_edit(spec, add=[("F5", "E9")],
                                  all_entity_ids=["E9"])
        self.assertIn(("F5", "E9"), edited.pairs())
        m = next(m for m in edited.mappings if m.entity_id == "E9")
        self.assertEqual(m.method, METHOD_HUMAN)
        self.assertTrue(m.human_edited)
        self.assertNotIn("E9", edited.unmapped_entities)

    def test_remove_binding(self):
        feats = [_feat("hole", {"diameter": 10.0}, "F1")]
        ents = [parse_callout("Ø10", entity_id="E1")]
        spec = build_spec(assign_features(feats, ents), ["E1"])
        edited = apply_human_edit(spec, remove=[("F1", "E1")],
                                  all_entity_ids=["E1"])
        self.assertNotIn(("F1", "E1"), edited.pairs())
        self.assertIn("E1", edited.unmapped_entities)

    def test_original_not_mutated(self):
        spec = UnifiedSpec(unmapped_entities=["E1"])
        apply_human_edit(spec, add=[("F1", "E1")], all_entity_ids=["E1"])
        self.assertEqual(spec.mappings, [])


class EvaluationTests(unittest.TestCase):
    def test_perfect_match(self):
        pred = [("F1", "E1"), ("F2", "E2")]
        gt = [("F1", "E1"), ("F2", "E2")]
        m = evaluate_mapping(pred, gt)
        self.assertAlmostEqual(m.precision, 1.0)
        self.assertAlmostEqual(m.recall, 1.0)
        self.assertAlmostEqual(m.f1, 1.0)
        self.assertAlmostEqual(m.exact_match_rate, 1.0)

    def test_partial(self):
        pred = [("F1", "E1"), ("F1", "E3")]
        gt = [("F1", "E1"), ("F1", "E2")]
        m = evaluate_mapping(pred, gt)
        self.assertAlmostEqual(m.precision, 0.5)
        self.assertAlmostEqual(m.recall, 0.5)
        # F1 predicted {E1,E3} vs gt {E1,E2}: not exact, but partial (E1 shared)
        self.assertAlmostEqual(m.exact_match_rate, 0.0)
        self.assertAlmostEqual(m.partial_match_rate, 1.0)

    def test_empty_pred(self):
        m = evaluate_mapping([], [("F1", "E1")])
        self.assertAlmostEqual(m.precision, 1.0)  # vacuous
        self.assertAlmostEqual(m.recall, 0.0)

    def test_f1_zero_when_disjoint(self):
        m = evaluate_mapping([("F1", "E1")], [("F2", "E2")])
        self.assertAlmostEqual(m.f1, 0.0)

    def test_macro_average(self):
        m1 = MappingMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 2)
        m2 = MappingMetrics(0.5, 0.5, 0.5, 0.0, 1.0, 2)
        avg = macro_average([m1, m2])
        self.assertAlmostEqual(avg.precision, 0.75)
        self.assertAlmostEqual(avg.f1, 0.75)
        self.assertEqual(avg.n_features, 4)

    def test_macro_average_empty(self):
        avg = macro_average([])
        self.assertEqual(avg.n_features, 0)


class IntegrationTests(unittest.TestCase):
    def test_end_to_end_scoring_to_metrics(self):
        feats = [_feat("hole", {"diameter": 10.0}, "F1"),
                 _feat("fillet", {"radius": 5.0}, "F2")]
        ents = [parse_callout("Ø10", entity_id="E1"),
                parse_callout("R5", entity_id="E2")]
        assigns = assign_features(feats, ents)
        spec = build_spec(assigns, ["E1", "E2"])
        gt = [("F1", "E1"), ("F2", "E2")]
        m = evaluate_mapping(spec.pairs(), gt)
        self.assertAlmostEqual(m.f1, 1.0)


if __name__ == "__main__":
    unittest.main()
