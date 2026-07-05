import unittest

from generation.cadsmith_kernel_metrics import (
    KernelMetrics, hard_kernel_gate, compare_to_plan, discrepancy_feedback,
)
from generation.cadsmith_design_plan import DesignPlan, Component, GeometricConstraints


def _metrics(**kw):
    base = dict(volume=1000.0, bbox_mm=(50.0, 50.0, 60.0),
                center_of_mass=(0.0, 0.0, 30.0), face_count=12,
                edge_count=24, vertex_count=16, is_valid=True)
    base.update(kw)
    return KernelMetrics(**base)


def _plan():
    return DesignPlan((Component("body"),), (50.0, 50.0, 60.0),
                      GeometricConstraints(hole_count=6))


class TestSerialization(unittest.TestCase):
    def test_round_trip(self):
        m = _metrics()
        self.assertEqual(KernelMetrics.from_dict(m.to_dict()), m)

    def test_json_stable(self):
        m = _metrics()
        self.assertEqual(m.to_json(), m.to_json())


class TestHardGate(unittest.TestCase):
    def test_valid_passes(self):
        self.assertTrue(hard_kernel_gate(_metrics()).passed)

    def test_invalid_solid_fails(self):
        r = hard_kernel_gate(_metrics(is_valid=False))
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "solid-not-valid")

    def test_zero_volume_fails(self):
        r = hard_kernel_gate(_metrics(volume=0.0))
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "non-positive-volume")


class TestCompare(unittest.TestCase):
    def test_within_tolerance(self):
        cmp = compare_to_plan(_metrics(), _plan())
        self.assertTrue(cmp.all_within_tol)
        self.assertEqual(cmp.out_of_tol, ())

    def test_bbox_deviation(self):
        cmp = compare_to_plan(_metrics(bbox_mm=(55.0, 50.0, 60.0)), _plan())
        self.assertFalse(cmp.all_within_tol)
        codes = {d.field for d in cmp.out_of_tol}
        self.assertIn("bbox_x", codes)

    def test_signed_delta(self):
        cmp = compare_to_plan(_metrics(bbox_mm=(55.0, 50.0, 60.0)), _plan())
        dx = [d for d in cmp.discrepancies if d.field == "bbox_x"][0]
        self.assertAlmostEqual(dx.delta, 5.0)

    def test_negative_tol_rejected(self):
        with self.assertRaises(ValueError):
            compare_to_plan(_metrics(), _plan(), bbox_tol_mm=-1.0)

    def test_feedback_only_lists_failures(self):
        cmp = compare_to_plan(_metrics(bbox_mm=(55.0, 50.0, 60.0)), _plan())
        fb = discrepancy_feedback(cmp)
        self.assertIn("bbox_x", fb)
        self.assertNotIn("bbox_y", fb)

    def test_feedback_empty_when_ok(self):
        self.assertEqual(discrepancy_feedback(compare_to_plan(_metrics(), _plan())), "")


if __name__ == "__main__":
    unittest.main()
