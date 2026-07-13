import unittest

from harnesscad.eval.bench.protocols.criteria import (
    Criterion, Dimension, Modality, Subdimension, aggregate,
    render_failure_rate, route_and_evaluate, syntax_failure_rate,
)
from harnesscad.eval.bench.data.splits import SplitEntry, audit_splits
from harnesscad.data.dataengine.annotation.review_workflow import (
    AnnotationItem, adjudicate, decision_distribution, qc_sample,
)
from harnesscad.governance.research.agreement import cohen_kappa


class BlenderLLMEvaluationTests(unittest.TestCase):
    def criteria(self):
        return (
            Criterion("shape-1", Dimension.ATTRIBUTE, Subdimension.SHAPE,
                      Modality.IMAGE, "object is a chair"),
            Criterion("shape-2", Dimension.ATTRIBUTE, Subdimension.SHAPE,
                      Modality.IMAGE, "seat is round"),
            Criterion("size-1", Dimension.ATTRIBUTE, Subdimension.SIZE,
                      Modality.SCRIPT, "legs are 35 cm"),
            Criterion("space-1", Dimension.SPATIAL, Subdimension.SPACE,
                      Modality.IMAGE, "legs support seat"),
            Criterion("execute-1", Dimension.INSTRUCTION, Subdimension.EXECUTE,
                      Modality.SCRIPT, "all attributes represented"),
        )

    def test_modality_routing_and_paper_aggregation(self):
        calls = []
        results = route_and_evaluate(
            self.criteria(),
            image_evaluator=lambda c: calls.append(("image", c.id)) or c.id != "shape-2",
            script_evaluator=lambda c: calls.append(("script", c.id)) or True,
        )
        self.assertEqual([name for name, _ in calls],
                         ["image", "image", "script", "image", "script"])
        score = aggregate(results)
        # Attribute = mean(shape=.5, size=1)=.75, spatial=1, instruction=1.
        self.assertAlmostEqual(score.dimension_scores["attribute"], .75)
        self.assertAlmostEqual(score.overall, (0.75 + 1 + 1) / 3)
        self.assertGreater(score.standard_deviation, 0)

    def test_invalid_dimension_subdimension_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            Criterion("bad", Dimension.SPATIAL, Subdimension.SIZE,
                      Modality.SCRIPT, "size")

    def test_failure_rates(self):
        self.assertEqual(syntax_failure_rate([True, False, True, False]), .5)
        self.assertEqual(render_failure_rate([True, True, False]), 1 / 3)
        self.assertIsNone(render_failure_rate([]))

    def test_split_audit_detects_quota_duplicate_and_cross_split_leakage(self):
        entries = (
            SplitEntry("1", "Make a chair", "sim", "generator", "furn", "look", "short"),
            SplitEntry("2", " make  A   CHAIR ", "wild", "forum", "furn", "look", "short"),
        )
        report = audit_splits(entries, quotas={"sim": 2, "wild": 1})
        self.assertFalse(report.ok)
        self.assertEqual(len(report.duplicate_prompts), 1)
        self.assertEqual(report.duplicate_prompts, report.cross_split_leakage)
        self.assertEqual(report.quota_shortfalls, {"sim": 1})

    def test_clean_source_aware_manifest(self):
        entries = (
            SplitEntry("1", "plate", "sim", "generator", "tools", "design", "short"),
            SplitEntry("2", "lamp help", "wild", "forum", "home", "use", "medium"),
        )
        self.assertTrue(audit_splits(entries, quotas={"sim": 1, "wild": 1}).ok)


class AnnotationGovernanceTests(unittest.TestCase):
    def test_two_votes_and_third_adjudication(self):
        items = [AnnotationItem("b", 2), AnnotationItem("a", 1)]
        decisions = adjudicate(
            items,
            first_review=lambda item: item.payload > 0,
            second_review=lambda item: item.payload == 1,
            third_review=lambda item, first, second: False,
        )
        self.assertEqual([d.item_id for d in decisions], ["a", "b"])
        self.assertFalse(decisions[0].adjudicated)
        self.assertTrue(decisions[1].adjudicated)
        self.assertEqual(decision_distribution(decisions), {"False": 1, "True": 1})

    def test_qc_checksum_sample_is_exact_and_order_independent(self):
        items = tuple(AnnotationItem(str(i), {"x": i}) for i in range(10))
        a = qc_sample(items, fraction=.3, salt="run")
        b = qc_sample(reversed(items), fraction=.3, salt="run")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)

    def test_cohen_kappa_and_confusion(self):
        perfect = cohen_kappa(["yes", "no"], ["yes", "no"])
        self.assertEqual(perfect.kappa, 1)
        report = cohen_kappa(["yes", "yes", "no", "no"],
                             ["yes", "no", "no", "no"])
        self.assertEqual(report.confusion["yes"]["no"], 1)
        self.assertAlmostEqual(report.observed, .75)
        self.assertGreater(report.kappa, 0)
        with self.assertRaises(ValueError):
            cohen_kappa([1], [1, 2])


if __name__ == "__main__":
    unittest.main()
