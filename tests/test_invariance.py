import json
import math
import unittest

from quality.invariance import (
    ContractMetadata,
    InvarianceContract,
    PerturbationCase,
    rotate_points_2d,
    scale_expected,
    scale_points,
    translate_points,
)


def polygon_area(points):
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    ) / 2


class InvarianceContractTests(unittest.TestCase):
    square = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))

    def test_translation_invariance(self):
        contract = InvarianceContract(
            ContractMetadata(
                "area-translation",
                "translation",
                "invariant",
                "polygon_area",
            ),
            transform=translate_points,
            measure=polygon_area,
        )
        report = contract.evaluate(
            self.square,
            (
                PerturbationCase("positive", (12.0, -4.0)),
                PerturbationCase("negative", (-3.5, -8.0)),
            ),
        )
        self.assertTrue(report.passed)
        self.assertEqual([r.expected for r in report.results], [4.0, 4.0])

    def test_rotation_invariance(self):
        contract = InvarianceContract(
            ContractMetadata("area-rotation", "rotation", "invariant", "area"),
            transform=rotate_points_2d,
            measure=polygon_area,
        )
        report = contract.evaluate(
            self.square,
            [PerturbationCase("quarter-turn", math.pi / 2)],
        )
        self.assertTrue(report.passed)
        self.assertLess(report.results[0].absolute_error, 1e-12)

    def test_scale_equivariance(self):
        contract = InvarianceContract(
            ContractMetadata(
                "area-scale",
                "scale",
                "equivariant",
                "area",
                scale_exponent=2,
            ),
            transform=scale_points,
            measure=polygon_area,
            expected=scale_expected(2),
        )
        report = contract.evaluate(
            self.square,
            [PerturbationCase("double", 2.0), PerturbationCase("half", 0.5)],
        )
        self.assertTrue(report.passed)
        self.assertEqual([r.actual for r in report.results], [16.0, 1.0])

    def test_detects_inconsistent_measurement(self):
        contract = InvarianceContract(
            ContractMetadata("centroid-x", "translation", "invariant", "centroid_x"),
            transform=translate_points,
            measure=lambda points: sum(p[0] for p in points) / len(points),
        )
        report = contract.evaluate(
            self.square, [PerturbationCase("shift-x", (10.0, 0.0))]
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.results[0].absolute_error, 10.0)

    def test_custom_comparator_supports_structured_measurements(self):
        contract = InvarianceContract(
            ContractMetadata("labels", "custom", "invariant", "labels"),
            transform=lambda values, _parameter: tuple(reversed(values)),
            measure=lambda values: tuple(values),
            comparator=lambda actual, expected: sorted(actual) == sorted(expected),
        )
        self.assertTrue(
            contract.evaluate((1, 2, 3), [PerturbationCase("reverse", None)]).passed
        )

    def test_report_metadata_and_results_are_json_serializable(self):
        contract = InvarianceContract(
            ContractMetadata(
                "area-scale", "scale", "equivariant", "area", scale_exponent=2
            ),
            transform=scale_points,
            measure=polygon_area,
            expected=scale_expected(2),
        )
        payload = contract.evaluate(
            self.square, [PerturbationCase("double", 2.0)]
        ).to_dict()
        self.assertEqual(payload["metadata"]["scale_exponent"], 2)
        self.assertTrue(payload["results"][0]["passed"])
        json.dumps(payload)

    def test_invalid_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            ContractMetadata("bad", "shear", "invariant", "area")
        with self.assertRaises(ValueError):
            ContractMetadata("bad", "scale", "invariant", "area", scale_exponent=2)

    def test_transform_dimension_and_scale_validation(self):
        with self.assertRaises(ValueError):
            translate_points(((1.0, 2.0),), (1.0,))
        with self.assertRaises(ValueError):
            rotate_points_2d(((1.0, 2.0, 3.0),), 1.0)
        with self.assertRaises(ValueError):
            scale_points(self.square, 0)

    def test_empty_translation_is_supported(self):
        self.assertEqual(translate_points((), (1.0, 2.0)), ())


if __name__ == "__main__":
    unittest.main()
