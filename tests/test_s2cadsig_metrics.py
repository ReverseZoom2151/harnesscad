import unittest

from reconstruction.s2cadsig_metrics import (
    MetricError,
    curve_class_metrics,
    face_heatmap_error,
    foreground_background_curve_error,
    masked_curve_error,
    mean_iou,
    operation_report,
    stroke_mask,
)


class TestFaceError(unittest.TestCase):
    def test_perfect(self):
        e = face_heatmap_error([0.0, 1.0, 0.5], [0.0, 1.0, 0.5])
        self.assertEqual(e.mse, 0.0)
        self.assertEqual(e.mae, 0.0)

    def test_known_values(self):
        e = face_heatmap_error([1.0, 0.0], [0.0, 0.0])
        self.assertAlmostEqual(e.mse, 0.5)
        self.assertAlmostEqual(e.mae, 0.5)

    def test_errors(self):
        with self.assertRaises(MetricError):
            face_heatmap_error([1.0], [1.0, 2.0])
        with self.assertRaises(MetricError):
            face_heatmap_error([], [])


class TestMaskedCurve(unittest.TestCase):
    def test_stroke_pixels_are_free(self):
        # pixel 0 is a stroke pixel: any prediction there is ignored
        pred = [99.0, 1.0]
        truth = [0.0, 1.0]
        stroke = [1.0, 0.0]
        e = masked_curve_error(pred, truth, stroke)
        self.assertAlmostEqual(e.mse, 0.0)
        self.assertAlmostEqual(e.mae, 0.0)

    def test_known_value(self):
        # both pixels scored; pred 0.0 vs truth 1.0 on one of them
        e = masked_curve_error([1.0, 0.0], [1.0, 1.0], [0.0, 0.0])
        self.assertAlmostEqual(e.mse, 0.5)
        self.assertAlmostEqual(e.mae, 0.5)

    def test_stroke_mask_helper(self):
        self.assertEqual(stroke_mask([1.0, 0.0]), [0.0, 1.0])
        with self.assertRaises(MetricError):
            stroke_mask([])

    def test_all_stroke(self):
        with self.assertRaises(MetricError):
            masked_curve_error([1.0], [1.0], [1.0])


class TestForegroundBackground(unittest.TestCase):
    def test_perfect_prediction(self):
        # curve on pixel 1; predictor outputs exactly the mask
        pred = [0.0, 1.0, 0.0]
        curve = [0.0, 1.0, 0.0]
        stroke = [0.0, 0.0, 0.0]
        e = foreground_background_curve_error(pred, curve, stroke)
        self.assertAlmostEqual(e.mse, 0.0)
        self.assertAlmostEqual(e.mae, 0.0)

    def test_background_false_positive_penalised(self):
        pred = [1.0, 1.0, 0.0]
        curve = [0.0, 1.0, 0.0]
        stroke = [0.0, 0.0, 0.0]
        e = foreground_background_curve_error(pred, curve, stroke)
        # one background pixel predicted 1.0 -> squared sum 1.0 over 3 stroke pixels
        self.assertAlmostEqual(e.mse, 1.0 / 3.0)
        self.assertAlmostEqual(e.mae, 1.0 / 3.0)

    def test_missed_curve_penalised(self):
        e = foreground_background_curve_error([0.0, 0.0], [0.0, 1.0], [0.0, 0.0])
        self.assertAlmostEqual(e.mse, 0.5)

    def test_normalised_by_stroke_pixels(self):
        # a stroke pixel shrinks the denominator
        e = foreground_background_curve_error(
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]
        )
        self.assertAlmostEqual(e.mse, 0.5)

    def test_all_stroke(self):
        with self.assertRaises(MetricError):
            foreground_background_curve_error([1.0], [1.0], [1.0])


class TestCurveClass(unittest.TestCase):
    def test_scores(self):
        pred = [0, 1, 2, 1]
        truth = [0, 1, 1, 1]
        mask = [1.0, 1.0, 1.0, 0.0]  # last pixel not a curve pixel -> skipped
        s = curve_class_metrics(pred, truth, mask)
        self.assertEqual(s["base"].tp, 1)
        self.assertEqual(s["offset"].tp, 1)
        self.assertEqual(s["offset"].fn, 1)
        self.assertEqual(s["profile"].fp, 1)
        self.assertAlmostEqual(s["base"].precision, 1.0)
        self.assertAlmostEqual(s["offset"].recall, 0.5)
        self.assertAlmostEqual(s["offset"].f1, 2.0 / 3.0)
        self.assertAlmostEqual(s["offset"].iou, 0.5)
        self.assertAlmostEqual(s["profile"].iou, 0.0)

    def test_mean_iou_uses_present_classes(self):
        s = curve_class_metrics([0, 1], [0, 1], [1.0, 1.0])
        self.assertAlmostEqual(mean_iou(s), 1.0)
        self.assertAlmostEqual(mean_iou({}), 0.0)

    def test_out_of_range(self):
        with self.assertRaises(MetricError):
            curve_class_metrics([5], [0], [1.0])

    def test_size_mismatch(self):
        with self.assertRaises(MetricError):
            curve_class_metrics([0, 1], [0], [1.0, 1.0])


class TestReport(unittest.TestCase):
    def test_regression_head(self):
        rep = operation_report(
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]
        )
        self.assertAlmostEqual(rep["face_mse"], 0.0)
        self.assertAlmostEqual(rep["curve_mse"], 0.0)
        self.assertAlmostEqual(rep["total_loss"], 0.0)

    def test_heatmap_head(self):
        rep = operation_report(
            [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            curve_head="heatmap",
        )
        self.assertAlmostEqual(rep["curve_mse"], 0.5)

    def test_bad_head(self):
        with self.assertRaises(MetricError):
            operation_report([0.0], [0.0], [0.0], [0.0], [0.0], curve_head="nope")


if __name__ == "__main__":
    unittest.main()
