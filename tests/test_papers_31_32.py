"""Tests for SPCC structure, CAD complexity, hierarchical annotation, domain
shift, prefix completion, and sketch-sequence metrics.

Rewritten from a single bare ``test_all`` pytest-style function (never
collected by ``python -m unittest``) into focused unittest.TestCase methods.
"""

import unittest

from harnesscad.eval.bench.data.cad_domain_shift import audit
from harnesscad.eval.bench.harness.prefix_completion import auc, cuts
from harnesscad.eval.bench.sketch.sketch_sequence_metrics import metrics
from harnesscad.data.dataengine.annotation.hierarchical_cad_annotation import validate
from harnesscad.eval.quality.sequence.cad_complexity import classify
from harnesscad.eval.quality.sequence.spcc_structure import Component, collapse, expand


class SPCCStructureTest(unittest.TestCase):
    def test_collapse_then_expand_round_trips(self):
        component = Component((1,))
        original = (component,) * 4
        self.assertEqual(expand(collapse(original)), original)


class CADComplexityTest(unittest.TestCase):
    def test_minimal_part_still_classifies_at_level_one_or_above(self):
        result = classify(components=1, loops=1, curves=1,
                          type_diversity=1, feature_depth=1)
        self.assertGreaterEqual(result["level"], 1)


class HierarchicalAnnotationTest(unittest.TestCase):
    def test_validate_rejects_inconsistent_annotation(self):
        self.assertFalse(validate({"a"}, {"a": "x"}, {"a"}))


class DomainShiftTest(unittest.TestCase):
    def test_audit_reports_categories_unseen_in_training(self):
        self.assertEqual(audit([{"a"}], [{"b"}])["unseen"], ("b",))


class PrefixCompletionTest(unittest.TestCase):
    def test_cuts_produces_four_prefix_points(self):
        self.assertEqual(len(cuts(range(10))), 4)

    def test_auc_of_the_diagonal_is_one_half(self):
        self.assertEqual(auc(((0, 0), (1, 1))), .5)


class SketchSequenceMetricsTest(unittest.TestCase):
    def test_identical_sequences_score_perfect_f1(self):
        self.assertEqual(metrics({"s": (1,)}, {"s": (1,)})["f1"], 1)


if __name__ == "__main__":
    unittest.main()
