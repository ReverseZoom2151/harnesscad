import unittest

from harnesscad.eval.bench.anomaly_splits import (
    few_shot, group_safe_split, normal_only_train, open_set, synthetic_transfer,
)
from harnesscad.eval.bench.task_interaction import efficiency, interaction_report
from harnesscad.eval.bench.vision_metrics import (
    average_precision, box_iou, classification_metrics, detection_at_threshold,
    mask_iou, mean_average_precision, mean_iou, slice_metric, top_k_accuracy,
)
from harnesscad.data.dataengine.anomaly_distribution import audit_anomaly_distribution
from harnesscad.data.dataengine.anomaly_schema import (
    AnomalyAsset, Box, Mask, VisionTask, validate_hierarchy,
)
from harnesscad.data.dataengine.cross_task_consistency import validate_asset
from harnesscad.data.dataengine.task_suitability import route_tasks
from harnesscad.data.dataengine.visual_qc import inspect_visual
from harnesscad.data.datagen.anomaly_pairs import compose_pair
from harnesscad.governance.security.image_privacy import PrivacyRegion, release_gate


def asset(i, *, kind="real", anomaly="crack", normal=False, group=""):
    return AnomalyAsset(
        str(i), "product", "assembly", "panel",
        "normal" if normal else anomaly,
        frozenset({VisionTask.CLASSIFICATION}), 100, 80, "source",
        source_kind=kind, normal=normal,
        labels=() if normal else (anomaly,), group_id=group,
    )


class VisualAnomalyGovernanceTests(unittest.TestCase):
    def test_schema_hierarchy_and_linked_annotations(self):
        a = AnomalyAsset(
            "x", "product", "assembly", "panel", "crack",
            frozenset({VisionTask.CLASSIFICATION, VisionTask.DETECTION,
                       VisionTask.SEGMENTATION}),
            100, 80, "camera", labels=("crack",),
            boxes=(Box(10, 10, 30, 40, "crack"),),
            masks=(Mask("crack", ((12, 12), (25, 12), (20, 30))),),
        )
        self.assertEqual(validate_asset(a), ())
        self.assertEqual(validate_hierarchy(
            a, {"assembly": "product", "panel": "assembly"}), ())
        bad = AnomalyAsset(
            "bad", "product", "assembly", "panel", "crack",
            frozenset({VisionTask.DETECTION}), 10, 10, "s",
            labels=("dent",), boxes=(Box(-1, 0, 5, 5, "crack"),),
        )
        self.assertEqual(validate_asset(bad), ("box_bounds:0", "box_label:0"))

    def test_classification_detection_and_segmentation_metrics(self):
        score = classification_metrics(("a", "b", "b"), ("a", "a", "b"))
        self.assertAlmostEqual(score["accuracy"], 2/3)
        self.assertEqual(top_k_accuracy(("a", "b"), (("a", "c"), ("a", "b")), 2), 1)
        self.assertEqual(box_iou((0, 0, 2, 2), (0, 0, 2, 2)), 1)
        det = detection_at_threshold(
            (("x", (0, 0, 2, 2)),),
            (("x", (0, 0, 2, 2), .8), ("x", (5, 5, 6, 6), .9)),
        )
        self.assertEqual((det["tp"], det["fp"]), (1, 1))
        perfect = (("x", (0, 0, 2, 2), .8),)
        self.assertEqual(average_precision((("x", (0, 0, 2, 2)),), perfect), 1)
        self.assertEqual(mean_average_precision(
            (("x", (0, 0, 2, 2)),), perfect, thresholds=(.5, .75)), 1)
        self.assertEqual(mask_iou({1, 2}, {2, 3}), 1/3)
        self.assertEqual(mean_iou({"a": {1}}, {"a": {1}}), 1)
        self.assertEqual(slice_metric([("b",1),("a",2)], lambda x:x[0],
                                      lambda rows:len(rows)), {"a":1,"b":1})

    def test_split_regimes_are_group_safe_and_source_aware(self):
        values = (
            asset(1, group="vehicle-a"), asset(2, group="vehicle-a"),
            asset(3, kind="synthetic", anomaly="dent"),
            asset(4, normal=True),
        )
        split = group_safe_split(values)
        locations = [set(split.train), set(split.validation), set(split.test)]
        self.assertTrue(any({"1", "2"} <= bucket for bucket in locations))
        self.assertEqual(synthetic_transfer(values).train, ("3",))
        self.assertEqual(normal_only_train(values), ("4",))
        self.assertEqual(few_shot(values, 1), ("1", "3", "4"))
        self.assertEqual(open_set(values, {"dent"}).test, ("3",))

    def test_distribution_pair_qc_privacy_and_routing(self):
        values=(asset(1),asset(2,kind="synthetic",anomaly="dent"))
        report=audit_anomaly_distribution(values,rarity=2,targets={"real":.5})
        self.assertEqual(report["rare_anomalies"], ("crack","dent"))
        pair=compose_pair("clean","scratch",7,
                          lambda normal, anomaly, seed:("changed",{(2,3),(3,3)}))
        self.assertEqual(pair.box,(2,3,4,4))
        self.assertTrue(inspect_visual("x",b"ok",decode=lambda p:(100,80),
                                      reviewers=("a","b")).passed)
        denied=release_gate([PrivacyRegion("face",False)],manually_verified=True)
        self.assertFalse(denied.releasable)
        allowed=release_gate([PrivacyRegion("face",True)],manually_verified=True)
        self.assertTrue(allowed.releasable)
        self.assertEqual(route_tasks(visibility=.8,has_boxes=True,has_masks=True).tasks,
                         frozenset(VisionTask))

    def test_task_interaction_reports_negative_transfer(self):
        report=interaction_report({"cls":.9,"seg":.6},{"cls":.9,"seg":.55})
        self.assertEqual(report["negative_transfer"],("seg",))
        self.assertFalse(report["pareto_dominates"])
        self.assertEqual(efficiency(10,2),.2)


if __name__ == "__main__":
    unittest.main()
