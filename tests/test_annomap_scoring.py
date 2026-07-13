import unittest

from harnesscad.domain.drawings.annomap_parser import CADFeature, parse_callout
from harnesscad.domain.drawings.annomap_scoring import (
    EPSILON,
    RHO,
    THETA_CAND,
    W_TYPE,
    Assignment,
    assign_features,
    context_consistency,
    dimensional_agreement,
    engineering_heuristics,
    score_pair,
    type_compatibility,
)


def _feat(ftype, params=None, fid="F1", conf=1.0):
    return CADFeature(ftype, params or {}, conf, feature_id=fid)


class TypeCompatibilityTests(unittest.TestCase):
    def test_exact_match(self):
        e = parse_callout("Ø10", entity_id="E1")   # target hole
        s, _ = type_compatibility(_feat("hole", {"diameter": 10.0}), e)
        self.assertEqual(s, 1.0)

    def test_semantic_group(self):
        e = parse_callout("Ø10", entity_id="E1")   # target hole
        s, _ = type_compatibility(_feat("bore", {"diameter": 10.0}), e)
        self.assertEqual(s, 0.9)

    def test_incompatible_hard_gate(self):
        e = parse_callout("R5", entity_id="E1")     # target fillet
        s, _ = type_compatibility(_feat("hole", {"diameter": 10.0}), e)
        self.assertEqual(s, 0.0)

    def test_linear_matches_any(self):
        e = parse_callout("25", entity_id="E1")     # no target
        s, _ = type_compatibility(_feat("slot", {"length": 25.0}), e)
        self.assertEqual(s, 0.9)


class DimensionalAgreementTests(unittest.TestCase):
    def test_exact_within_epsilon(self):
        e = parse_callout("Ø10", entity_id="E1")
        s, has, _ = dimensional_agreement(_feat("hole", {"diameter": 10.05}), e)
        self.assertEqual(s, 1.0)
        self.assertTrue(has)

    def test_stepped_within_2epsilon(self):
        e = parse_callout("Ø10", entity_id="E1")
        s, _, _ = dimensional_agreement(_feat("hole", {"diameter": 10.15}), e)
        self.assertEqual(s, 0.7)

    def test_mismatch_zero(self):
        e = parse_callout("Ø10", entity_id="E1")
        s, has, _ = dimensional_agreement(_feat("hole", {"diameter": 12.0}), e)
        self.assertEqual(s, 0.0)
        self.assertTrue(has)

    def test_radius_matches_half_diameter(self):
        e = parse_callout("R5", entity_id="E1")
        s, _, _ = dimensional_agreement(_feat("fillet", {"diameter": 10.0}), e)
        self.assertEqual(s, 1.0)

    def test_no_numeric(self):
        e = parse_callout("DATUM A", entity_id="E1")
        s, has, _ = dimensional_agreement(_feat("plane"), e)
        self.assertFalse(has)


class ContextTests(unittest.TestCase):
    def test_neutral_without_cues(self):
        e = parse_callout("Ø10", entity_id="E1")
        s, _ = context_consistency(e)
        self.assertEqual(s, 0.5)

    def test_vlm_confidence(self):
        e = parse_callout("Ø10", entity_id="E1")
        s, _ = context_consistency(e, vlm_confidence=0.8)
        self.assertAlmostEqual(s, 0.8)


class HeuristicTests(unittest.TestCase):
    def test_diameter_symbol_bonus(self):
        e = parse_callout("Ø10", entity_id="E1")
        h, mult, _ = engineering_heuristics(_feat("hole"), e)
        self.assertAlmostEqual(h, 0.1)
        self.assertEqual(mult, 1.0)

    def test_diameter_without_symbol_penalty(self):
        # A diameter-typed entity lacking the Ø symbol -> multiplicative penalty.
        e = parse_callout("DIA 10", entity_id="E1")
        e.symbol = ""
        e.raw_text = "10"
        h, mult, _ = engineering_heuristics(_feat("hole"), e)
        self.assertAlmostEqual(mult, 0.7)

    def test_thread_on_non_cylindrical_rejected(self):
        e = parse_callout("M8", entity_id="E1")
        h, mult, _ = engineering_heuristics(_feat("slot"), e)
        self.assertEqual(mult, 0.0)

    def test_runout_prior(self):
        e = parse_callout("CIRCULAR RUNOUT 0.1 A", entity_id="E1")
        h, mult, _ = engineering_heuristics(_feat("cylinder"), e)
        self.assertAlmostEqual(h, 0.1)


class ScorePairTests(unittest.TestCase):
    def test_hard_gate_zero(self):
        e = parse_callout("R5", entity_id="E1")
        b = score_pair(_feat("hole", {"diameter": 10.0}), e)
        self.assertEqual(b.composite, 0.0)

    def test_full_score_hole(self):
        e = parse_callout("Ø10", entity_id="E1")
        b = score_pair(_feat("hole", {"diameter": 10.0}), e)
        # w_t*1 + w_d*1 + w_c*0.5 + 0.1 = 0.4+0.4+0.1+0.1 = 1.0
        self.assertAlmostEqual(b.composite, 1.0)

    def test_dim_mismatch_suppression(self):
        e = parse_callout("Ø10", entity_id="E1")
        b = score_pair(_feat("hole", {"diameter": 20.0}), e)
        # numeric dim present, S_dim=0 -> composite *0.3
        # base = 0.4*1 + 0 + 0.2*0.5 + 0.1 = 0.6; *0.3 = 0.18
        self.assertAlmostEqual(b.composite, 0.18)

    def test_breakdown_dict(self):
        e = parse_callout("Ø10", entity_id="E1")
        b = score_pair(_feat("hole", {"diameter": 10.0}), e)
        self.assertIn("composite", b.to_dict())
        self.assertTrue(b.rationale)


class AssignmentTests(unittest.TestCase):
    def test_near_tie_keeps_multiple(self):
        # Two identical holes dimensioned by one Ø10 entity across views.
        feats = [_feat("hole", {"diameter": 10.0}, "F1")]
        ents = [parse_callout("Ø10", entity_id="E1"),
                parse_callout("Ø10", entity_id="E2")]
        assigns = assign_features(feats, ents)
        self.assertEqual(len(assigns), 1)
        self.assertEqual(set(assigns[0].entity_ids), {"E1", "E2"})

    def test_low_score_dropped(self):
        feats = [_feat("hole", {"diameter": 10.0}, "F1")]
        ents = [parse_callout("Ø10", entity_id="E1"),
                parse_callout("Ø25", entity_id="E2")]  # mismatch -> suppressed
        assigns = assign_features(feats, ents)
        self.assertEqual(assigns[0].entity_ids, ["E1"])

    def test_unmapped_feature(self):
        feats = [_feat("fillet", {"radius": 3.0}, "F1")]
        ents = [parse_callout("Ø10", entity_id="E1")]  # incompatible type
        assigns = assign_features(feats, ents)
        self.assertEqual(assigns[0].entity_ids, [])

    def test_deterministic_order(self):
        feats = [_feat("hole", {"diameter": 10.0}, "F1")]
        ents = [parse_callout("Ø10", entity_id="E2"),
                parse_callout("Ø10", entity_id="E1")]
        a1 = assign_features(feats, ents)
        a2 = assign_features(feats, ents)
        self.assertEqual(a1[0].entity_ids, a2[0].entity_ids)
        # sorted by (score desc, id asc) -> E1 before E2
        self.assertEqual(a1[0].entity_ids, ["E1", "E2"])

    def test_constants_present(self):
        self.assertAlmostEqual(W_TYPE, 0.4)
        self.assertAlmostEqual(EPSILON, 0.1)
        self.assertAlmostEqual(RHO, 0.9)
        self.assertAlmostEqual(THETA_CAND, 0.3)


if __name__ == "__main__":
    unittest.main()
