import unittest

from harnesscad.eval.verifiers.kernel_preflight import (
    BoundingBox,
    ErrorCode,
    ShapeInfo,
    check_bbox_overlap,
    check_containment,
    check_fillet_radius,
    check_manifold,
    check_nonzero_volume,
    check_shell_thickness,
    contains,
    overlap_volume,
    preflight_boolean,
    preflight_fillet,
    preflight_shell,
)

BOX = ShapeInfo(
    id="box",
    bbox=BoundingBox(0, 0, 0, 10, 10, 10),
    volume=1000.0,
    manifold=True,
)
OVERLAPPING = ShapeInfo(
    id="tool",
    bbox=BoundingBox(5, 5, 5, 15, 15, 15),
    volume=1000.0,
)
DISJOINT = ShapeInfo(
    id="far",
    bbox=BoundingBox(50, 50, 50, 60, 60, 60),
    volume=1000.0,
)


class TestGeometryHelpers(unittest.TestCase):
    def test_overlap_volume(self):
        self.assertAlmostEqual(overlap_volume(BOX.bbox, OVERLAPPING.bbox), 125.0)
        self.assertEqual(overlap_volume(BOX.bbox, DISJOINT.bbox), 0.0)

    def test_contains(self):
        inner = BoundingBox(1, 1, 1, 2, 2, 2)
        self.assertTrue(contains(BOX.bbox, inner))
        self.assertFalse(contains(inner, BOX.bbox))

    def test_min_extent(self):
        self.assertAlmostEqual(BoundingBox(0, 0, 0, 10, 4, 6).min_extent(), 4.0)


class TestIndividualChecks(unittest.TestCase):
    def test_zero_volume(self):
        failure = check_nonzero_volume(ShapeInfo("s", BOX.bbox, volume=0.0))
        self.assertEqual(failure.code, ErrorCode.ZERO_VOLUME)
        self.assertEqual(failure.failed_check, "nonzero_volume")
        self.assertIn("dimensions", failure.suggestion)

    def test_nonzero_volume_passes(self):
        self.assertIsNone(check_nonzero_volume(BOX))

    def test_manifold(self):
        bad = ShapeInfo("s", BOX.bbox, volume=5.0, manifold=False)
        self.assertEqual(check_manifold(bad).code, ErrorCode.NON_MANIFOLD)
        self.assertIsNone(check_manifold(BOX))

    def test_bbox_no_overlap(self):
        failure = check_bbox_overlap(BOX, DISJOINT)
        self.assertEqual(failure.code, ErrorCode.BBOX_NO_OVERLAP)

    def test_bbox_near_tangent(self):
        touching = ShapeInfo("t", BoundingBox(10, 0, 0, 20, 10, 10), volume=1000.0)
        failure = check_bbox_overlap(BOX, touching, tolerance=1e-6)
        self.assertEqual(failure.code, ErrorCode.BBOX_NO_OVERLAP)
        sliver = ShapeInfo(
            "t2", BoundingBox(10 - 1e-12, 0, 0, 20, 10, 10), volume=1000.0
        )
        failure = check_bbox_overlap(BOX, sliver, tolerance=1e-6)
        self.assertEqual(failure.code, ErrorCode.BBOX_NEAR_TANGENT)

    def test_overlapping_boxes_pass(self):
        self.assertIsNone(check_bbox_overlap(BOX, OVERLAPPING))

    def test_containment_cut_empty_result(self):
        big = ShapeInfo("big", BoundingBox(-5, -5, -5, 15, 15, 15), volume=8000.0)
        failure = check_containment(BOX, big, "cut")
        self.assertEqual(failure.code, ErrorCode.EMPTY_RESULT)

    def test_containment_union_redundant(self):
        small = ShapeInfo("small", BoundingBox(1, 1, 1, 2, 2, 2), volume=1.0)
        failure = check_containment(BOX, small, "union")
        self.assertEqual(failure.code, ErrorCode.REDUNDANT_OPERAND)

    def test_containment_ok(self):
        self.assertIsNone(check_containment(BOX, OVERLAPPING, "cut"))

    def test_fillet_radius(self):
        self.assertIsNone(check_fillet_radius(BOX, 2.0))
        too_big = check_fillet_radius(BOX, 6.0)
        self.assertEqual(too_big.code, ErrorCode.RADIUS_TOO_LARGE)
        self.assertEqual(
            check_fillet_radius(BOX, -1.0).code, ErrorCode.INVALID_INPUT
        )

    def test_fillet_radius_boundary_is_the_degenerate_limit(self):
        # The 50x30x6 plate from the pressure report. The constraint is
        # r < half the smallest extent (3 mm). The boundary r == 3 is the
        # degenerate limit -- the two fillets meet and the face vanishes -- so
        # it MUST fire; the old strict `>` let exactly that case through while
        # its own suggestion said "reduce below 3".
        plate = ShapeInfo("plate", BoundingBox(0, 0, 0, 50, 30, 6), volume=9000.0)
        self.assertEqual(
            check_fillet_radius(plate, 3.0).code, ErrorCode.RADIUS_TOO_LARGE
        )
        self.assertEqual(
            check_fillet_radius(plate, 3.1).code, ErrorCode.RADIUS_TOO_LARGE
        )
        self.assertIsNone(check_fillet_radius(plate, 2.99))
        self.assertIn("3", check_fillet_radius(plate, 3.0).suggestion)

    def test_shell_thickness_boundary_still_fires(self):
        # The harness's genuine win (trap_shell_too_thick) must survive: a 9 mm
        # shell in 5 mm of stock, and the exact boundary t == half the extent.
        tray = ShapeInfo("tray", BoundingBox(0, 0, 0, 60, 40, 5), volume=12000.0)
        fail = check_shell_thickness(tray, 9.0)
        self.assertEqual(fail.code, ErrorCode.THICKNESS_TOO_LARGE)
        self.assertIn("leaves no cavity", fail.message)
        self.assertEqual(
            check_shell_thickness(tray, 2.5).code, ErrorCode.THICKNESS_TOO_LARGE
        )
        self.assertIsNone(check_shell_thickness(tray, 2.4))

    def test_fillet_radius_uses_min_edge_length(self):
        thin = ShapeInfo("t", BOX.bbox, volume=1000.0, min_edge_length=1.0)
        self.assertEqual(
            check_fillet_radius(thin, 0.9).code, ErrorCode.RADIUS_TOO_LARGE
        )
        self.assertIsNone(check_fillet_radius(thin, 0.4))

    def test_shell_thickness(self):
        self.assertIsNone(check_shell_thickness(BOX, 1.0))
        self.assertEqual(
            check_shell_thickness(BOX, 5.0).code, ErrorCode.THICKNESS_TOO_LARGE
        )
        self.assertEqual(
            check_shell_thickness(BOX, 0.0).code, ErrorCode.INVALID_INPUT
        )


class TestBatteries(unittest.TestCase):
    def test_boolean_ok(self):
        self.assertIsNone(preflight_boolean(BOX, OVERLAPPING, "cut"))

    def test_boolean_unknown_operation(self):
        failure = preflight_boolean(BOX, OVERLAPPING, "xor")
        self.assertEqual(failure.code, ErrorCode.INVALID_INPUT)

    def test_boolean_disjoint_cut_fails_but_union_allowed(self):
        self.assertEqual(
            preflight_boolean(BOX, DISJOINT, "cut").code, ErrorCode.BBOX_NO_OVERLAP
        )
        self.assertIsNone(preflight_boolean(BOX, DISJOINT, "union"))

    def test_boolean_checks_both_operands(self):
        empty = ShapeInfo("empty", OVERLAPPING.bbox, volume=0.0)
        failure = preflight_boolean(BOX, empty, "union")
        self.assertEqual(failure.code, ErrorCode.ZERO_VOLUME)
        self.assertIn("empty", failure.message)

    def test_fillet_and_shell_batteries(self):
        self.assertIsNone(preflight_fillet(BOX, 1.0))
        self.assertEqual(
            preflight_fillet(BOX, 20.0).code, ErrorCode.RADIUS_TOO_LARGE
        )
        self.assertIsNone(preflight_shell(BOX, 1.0))
        self.assertEqual(
            preflight_shell(
                ShapeInfo("bad", BOX.bbox, volume=10.0, manifold=False), 1.0
            ).code,
            ErrorCode.NON_MANIFOLD,
        )

    def test_failure_as_dict_is_serialisable(self):
        payload = preflight_fillet(BOX, 20.0).as_dict()
        self.assertEqual(
            sorted(payload), ["code", "failed_check", "message", "suggestion"]
        )

    def test_deterministic(self):
        first = preflight_boolean(BOX, DISJOINT, "intersection")
        second = preflight_boolean(BOX, DISJOINT, "intersection")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
