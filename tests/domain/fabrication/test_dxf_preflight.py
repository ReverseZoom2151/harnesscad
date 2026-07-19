"""Tests for the neutral-DXF structural manufacturing preflight."""

import unittest

from harnesscad.domain.fabrication.dxf_preflight import dxf_preflight
from harnesscad.io.formats.dxf import DxfDocument, Entity, Layer


def document(units="mm", **entities):
    return DxfDocument(units, (Layer("cut"),), {
        name: Entity(kind, values, layer="cut")
        for name, (kind, values) in entities.items()
    })


class TestDxfPreflightMeasurements(unittest.TestCase):
    def test_resolves_declared_document_units_to_millimetres(self):
        report = dxf_preflight(document(
            "cm", circle=("CIRCLE", {"center": (1, 1), "radius": 0.1}),
        ))
        self.assertEqual(report.verdict, "PASS")
        self.assertEqual(report.metrics.circle_diameters_mm, (2.0,))
        self.assertEqual(report.metrics.bbox_mm, (9.0, 9.0, 11.0, 11.0))

    def test_polyline_measurement_includes_closed_final_segment(self):
        report = dxf_preflight(document(
            "mm", outline=("LWPOLYLINE", {
                "points": ((0, 0), (3, 0), (3, 4)), "closed": True,
            }),
        ))
        self.assertAlmostEqual(report.metrics.total_path_length_mm, 12.0)
        self.assertEqual(report.metrics.zero_length_segments, 0)

    def test_arc_bounds_include_cardinal_extrema_inside_sweep(self):
        report = dxf_preflight(document(
            "mm", arc=("ARC", {"center": (0, 0), "radius": 2,
                                  "start_angle_deg": 45, "end_angle_deg": 135}),
        ))
        for actual, expected in zip(report.metrics.bbox_mm,
                                    (-math_sqrt2(), math_sqrt2(), math_sqrt2(), 2.0)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(report.metrics.arc_radii_mm, (2.0,))


def math_sqrt2():
    # Exact rounded source values keep this test focused on extrema selection,
    # while the report keeps full floating-point measurements.
    return 2 ** 0.5


class TestDxfPreflightVerdicts(unittest.TestCase):
    def test_degenerate_line_fails(self):
        report = dxf_preflight(document(
            "mm", zero=("LINE", {"start": (2, 2), "end": (2, 2)}),
        ))
        self.assertEqual(report.verdict, "FAIL")
        self.assertEqual(report.metrics.zero_length_segments, 1)
        self.assertIn("DXF_ZERO_LENGTH", {f.code for f in report.findings})

    def test_non_positive_radius_is_not_normalized_into_a_plausible_circle(self):
        report = dxf_preflight(document(
            "mm", bad_circle=("CIRCLE", {"center": (0, 0), "radius": -2}),
        ))
        self.assertEqual(report.verdict, "REVIEW")
        self.assertEqual(report.metrics.circle_diameters_mm, ())
        self.assertIn("DXF_ENTITY_UNMEASURABLE", {f.code for f in report.findings})

    def test_policy_thresholds_are_explicit_not_source_defaults(self):
        doc = document("mm", circle=("CIRCLE", {"center": (0, 0), "radius": 1}))
        self.assertEqual(dxf_preflight(doc).verdict, "PASS")
        report = dxf_preflight(doc, min_circle_diameter_mm=2.1)
        self.assertEqual(report.verdict, "FAIL")
        self.assertIn("DXF_CIRCLE_BELOW_MIN", {f.code for f in report.findings})

    def test_small_arc_is_review_not_a_claim_of_3d_failure(self):
        report = dxf_preflight(document(
            "mm", arc=("ARC", {"center": (0, 0), "radius": 0.9,
                                  "start_angle_deg": 0, "end_angle_deg": 90}),
        ), min_arc_radius_mm=1.0)
        self.assertEqual(report.verdict, "REVIEW")
        finding = next(f for f in report.findings if f.code == "DXF_ARC_BELOW_MIN")
        self.assertEqual(finding.severity, "warning")

    def test_unknown_entities_prevent_density_from_becoming_green(self):
        report = dxf_preflight(document(
            "mm", line=("LINE", {"start": (0, 0), "end": (10, 0)}),
            spline=("SPLINE", {}),
        ), max_entity_density_per_mm2=0.1)
        self.assertEqual(report.verdict, "REVIEW")
        self.assertIsNone(report.metrics.density_per_mm2)
        self.assertEqual(report.metrics.unmeasured_entity_ids, ("spline",))
        self.assertTrue({"DXF_ENTITY_UNMEASURABLE", "DXF_DENSITY_UNCERTIFIED"}
                        <= {f.code for f in report.findings})

    def test_empty_document_is_review(self):
        report = dxf_preflight(document("mm"))
        self.assertEqual(report.verdict, "REVIEW")
        self.assertIn("DXF_EMPTY", {f.code for f in report.findings})

    def test_threshold_validation_rejects_non_positive_policy(self):
        with self.assertRaisesRegex(ValueError, "min_circle_diameter_mm must be > 0"):
            dxf_preflight(document("mm"), min_circle_diameter_mm=0)


if __name__ == "__main__":
    unittest.main()
