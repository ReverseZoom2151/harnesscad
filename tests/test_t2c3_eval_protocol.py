"""Tests for the Text2CAD evaluation protocol (bench.t2c3_eval_protocol)."""

import unittest

from harnesscad.eval.bench.t2c3_eval_protocol import (
    NULL_LABEL,
    CurveScore,
    EvalProtocolError,
    aggregate_reports,
    confusion_matrix,
    curve_bbox,
    curve_distance,
    curve_scores,
    evaluate_model,
    extrusion_report,
    label_streams,
    loop_bbox,
    loop_distance,
    macro_average,
    match_loops,
    match_model_curves,
    match_primitives,
    micro_average,
    type_accuracy,
)


def _line(x0, y0, x1, y1):
    return {"type": "line", "start": (x0, y0), "end": (x1, y1)}


def _square(offset=0.0, size=10.0):
    o = offset
    s = size
    return [
        _line(o, o, o + s, o),
        _line(o + s, o, o + s, o + s),
        _line(o + s, o + s, o, o + s),
        _line(o, o + s, o, o),
    ]


def _ext(**kw):
    base = {
        "extent_one": 0.75, "extent_two": 0.0,
        "origin": (0.0, 0.0, 0.0), "euler": (0.0, 0.0, 0.0),
        "sketch_size": 0.75, "boolean": 0,
    }
    base.update(kw)
    return base


def _part(loops, **kw):
    return {"sketch": [[lp] for lp in loops], "extrusion": _ext(**kw)}


class TestGeometry(unittest.TestCase):
    def test_line_bbox(self):
        self.assertEqual(curve_bbox(_line(3, 4, 1, 0)), ((1, 0), (3, 4)))

    def test_arc_bbox_uses_three_points(self):
        arc = {"type": "arc", "start": (0, 0), "mid": (5, 9), "end": (10, 0)}
        self.assertEqual(curve_bbox(arc), ((0, 0), (10, 9)))

    def test_circle_bbox(self):
        circ = {"type": "circle", "center": (5, 5), "radius": 2}
        self.assertEqual(curve_bbox(circ), ((3, 3), (7, 7)))

    def test_unknown_curve(self):
        with self.assertRaises(EvalProtocolError):
            curve_bbox({"type": "nurbs"})

    def test_loop_bbox_union(self):
        self.assertEqual(loop_bbox(_square()), ((0.0, 0.0), (10.0, 10.0)))

    def test_identical_loops_have_zero_distance(self):
        self.assertEqual(loop_distance(_square(), _square(), 1.0), 0.0)

    def test_distance_scales(self):
        d1 = loop_distance(_square(), _square(offset=1.0), 1.0)
        d2 = loop_distance(_square(), _square(offset=1.0), 2.0)
        self.assertAlmostEqual(d2, 2 * d1)

    def test_curve_distance_symmetric(self):
        a, b = _line(0, 0, 1, 1), _line(0, 0, 2, 2)
        self.assertAlmostEqual(curve_distance(a, b, 1.0), curve_distance(b, a, 1.0))


class TestMatching(unittest.TestCase):
    def test_loops_matched_to_nearest_bbox(self):
        gt = [_square(offset=0.0), _square(offset=100.0)]
        pred = [_square(offset=100.2), _square(offset=0.1)]   # reversed order
        pairs = match_loops(gt, pred, 1.0)
        self.assertEqual(len(pairs), 2)
        # gt loop 0 (near origin) must match the pred loop near the origin
        self.assertAlmostEqual(pairs[0][1][0]["start"][0], 0.1)
        self.assertAlmostEqual(pairs[1][1][0]["start"][0], 100.2)

    def test_unmatched_loop_pairs_with_none(self):
        pairs = match_loops([_square()], [], 1.0)
        self.assertEqual(len(pairs), 1)
        self.assertIsNone(pairs[0][1])

    def test_primitives_padded_with_none(self):
        gt_loop = _square()
        pred_loop = _square()[:2]
        pairs = match_primitives(gt_loop, pred_loop, 1.0)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(sum(1 for _, p in pairs if p is None), 2)

    def test_match_model_curves_positional_sketches(self):
        gt = [_part([_square()])]
        pred = [_part([_square()]), _part([_square()])]
        pairs = match_model_curves(gt, pred, 1.0)
        # 4 gt curves matched + 4 curves of the extra predicted sketch vs None
        self.assertEqual(len(pairs), 8)
        self.assertEqual(sum(1 for g, _ in pairs if g is None), 4)


class TestLabelsAndScores(unittest.TestCase):
    def test_null_label_for_unmatched(self):
        y_true, y_pred = label_streams([(_line(0, 0, 1, 1), None),
                                        (None, {"type": "circle", "center": (0, 0), "radius": 1})])
        self.assertEqual(y_true, [0, NULL_LABEL])
        self.assertEqual(y_pred, [NULL_LABEL, 2])

    def test_perfect_match_scores_one(self):
        model = [_part([_square()])]
        report = evaluate_model(model, model)
        self.assertEqual(report.curves["line"].f1, 1.0)
        self.assertEqual(report.curves["line"].correct, 4)
        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.extrusion.f1, 1.0)
        self.assertEqual(report.extrusion.boolean_correct, 1)

    def test_hallucinated_curves_hurt_precision(self):
        gt = [_part([_square()])]
        pred = [_part([_square(), _square(offset=50.0)])]
        report = evaluate_model(gt, pred)
        line = report.curves["line"]
        self.assertEqual(line.total_gt, 4)
        self.assertEqual(line.total_pred, 8)
        self.assertEqual(line.recall, 1.0)
        self.assertAlmostEqual(line.precision, 0.5)
        self.assertAlmostEqual(line.f1, 2 / 3)

    def test_missing_curves_hurt_recall(self):
        gt = [_part([_square(), _square(offset=50.0)])]
        pred = [_part([_square()])]
        line = evaluate_model(gt, pred).curves["line"]
        self.assertAlmostEqual(line.recall, 0.5)
        self.assertEqual(line.precision, 1.0)

    def test_wrong_type_confusion(self):
        gt = [_part([_square()])]
        wrong = [{"type": "arc", "start": (0, 0), "mid": (5, 9), "end": (10, 0)}] + _square()[1:]
        pred = [_part([wrong])]
        report = evaluate_model(gt, pred)
        self.assertEqual(report.curves["line"].correct, 3)
        self.assertEqual(report.curves["arc"].total_pred, 1)
        self.assertEqual(report.curves["arc"].total_gt, 0)
        self.assertEqual(report.curves["arc"].f1, 0.0)

    def test_confusion_matrix_shape(self):
        cm = confusion_matrix([0, 1, 3], [0, 2, 3])
        self.assertEqual(len(cm), 4)
        self.assertEqual(cm[0][0], 1)
        self.assertEqual(cm[1][2], 1)
        self.assertEqual(cm[3][3], 1)

    def test_scores_from_confusion(self):
        cm = confusion_matrix([0, 0, 1], [0, 1, 1])
        scores = curve_scores(cm)
        self.assertIsInstance(scores["line"], CurveScore)
        self.assertAlmostEqual(scores["line"].recall, 0.5)
        self.assertEqual(scores["line"].precision, 1.0)
        self.assertAlmostEqual(scores["arc"].precision, 0.5)

    def test_macro_and_micro(self):
        y_true, y_pred = [0, 0, 1], [0, 1, 1]
        cm = confusion_matrix(y_true, y_pred)
        micro = micro_average(y_true, y_pred)
        self.assertAlmostEqual(micro["f1"], 2 / 3)
        self.assertAlmostEqual(micro["f1"], type_accuracy(y_true, y_pred))
        macro = macro_average(cm, y_true, y_pred)
        # labels present: {0, 1}; f1(line)=2/3, f1(arc)=2/3
        self.assertAlmostEqual(macro["f1"], 2 / 3)

    def test_empty_streams(self):
        self.assertEqual(type_accuracy([], []), 0.0)
        self.assertEqual(micro_average([], [])["f1"], 0.0)


class TestExtrusion(unittest.TestCase):
    def test_counts_and_f1(self):
        gt = [_part([_square()]), _part([_square()])]
        pred = [_part([_square()])]
        score = extrusion_report(gt, pred)
        self.assertEqual((score.num_gt, score.num_pred, score.num_matched), (2, 1, 1))
        self.assertAlmostEqual(score.recall, 0.5)
        self.assertEqual(score.precision, 1.0)
        self.assertAlmostEqual(score.f1, 2 / 3)

    def test_parameters_rescaled_by_norm_factor(self):
        gt = [_part([_square()], extent_one=0.75)]
        pred = [_part([_square()], extent_one=0.0)]
        score = extrusion_report(gt, pred)
        # 0.75 / 0.75 = 1.0 after rescaling
        self.assertAlmostEqual(score.parameter_l1["dist"], 1.0)

    def test_angles_are_not_rescaled(self):
        gt = [_part([_square()], euler=(0.75, 0.0, 0.0))]
        pred = [_part([_square()], euler=(0.0, 0.0, 0.0))]
        score = extrusion_report(gt, pred)
        self.assertAlmostEqual(score.parameter_l1["theta"], 0.75)

    def test_boolean_mismatch(self):
        gt = [_part([_square()], boolean=0)]
        pred = [_part([_square()], boolean=2)]
        self.assertEqual(extrusion_report(gt, pred).boolean_correct, 0)

    def test_zero_scaling_rejected(self):
        with self.assertRaises(EvalProtocolError):
            extrusion_report([_part([_square()])], [_part([_square()])], scaling_factor=0)


class TestAggregate(unittest.TestCase):
    def test_means_over_samples_with_gt_primitive(self):
        model = [_part([_square()])]
        circle_model = [_part([[{"type": "circle", "center": (0, 0), "radius": 1}]])]
        reports = [evaluate_model(model, model), evaluate_model(circle_model, circle_model)]
        agg = aggregate_reports(reports)
        self.assertAlmostEqual(agg["line"]["f1"], 100.0)
        self.assertAlmostEqual(agg["circle"]["f1"], 100.0)
        self.assertEqual(agg["arc"]["f1"], 0.0)
        self.assertAlmostEqual(agg["extrusion"]["f1"], 100.0)

    def test_empty_aggregate(self):
        agg = aggregate_reports([])
        self.assertEqual(agg["line"]["precision"], 0.0)
        self.assertEqual(agg["extrusion"]["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
