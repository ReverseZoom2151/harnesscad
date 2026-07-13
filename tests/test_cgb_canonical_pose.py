"""Tests for the canonical-pose contract."""
import unittest

from harnesscad.eval.quality.geometry.cgb_canonical_pose import (
    ambiguity_flags,
    axes_are_ordered,
    bounding_box,
    canonicalize,
    is_centered,
    pose_report,
    reference_face_seated,
)


def _box_corners(lx, ly, lz, origin=(0.0, 0.0, 0.0)):
    ox, oy, oz = origin
    return [
        (ox + x * lx, oy + y * ly, oz + z * lz)
        for x in (0, 1)
        for y in (0, 1)
        for z in (0, 1)
    ]


class TestBoundingBox(unittest.TestCase):
    def test_extents_and_center(self):
        box = bounding_box(_box_corners(4, 2, 1, origin=(10, 10, 10)))
        self.assertEqual(box.extents, (4, 2, 1))
        self.assertEqual(box.center, (12, 11, 10.5))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            bounding_box([])


class TestCanonicalize(unittest.TestCase):
    def test_centers_and_orders(self):
        # Extents (1, 4, 2) off in the corner: must come back as (4, 2, 1) centred.
        pts = canonicalize(_box_corners(1, 4, 2, origin=(7, -3, 11)))
        box = bounding_box(pts)
        self.assertEqual(box.extents, (4, 2, 1))
        self.assertTrue(is_centered(pts))
        self.assertTrue(axes_are_ordered(box.extents))

    def test_already_canonical_is_a_fixed_point(self):
        pts = canonicalize(_box_corners(4, 2, 1, origin=(-2, -1, -0.5)))
        self.assertTrue(pose_report(pts).compliant)
        self.assertEqual(bounding_box(canonicalize(pts)).extents, (4, 2, 1))

    def test_rotation_is_proper_not_a_reflection(self):
        # Extents (2, 4, 1) sort to axis order (1, 0, 2): an odd permutation.
        # A chirality marker must survive with the same handedness.
        pts = _box_corners(2, 4, 1)
        marker = (2.0, 0.0, 0.0)  # along the original x
        out = canonicalize(list(pts) + [marker])
        # The signed volume of the frame is preserved: no mirroring.
        a, b, c = out[1], out[2], out[4]
        o = out[0]
        u = [a[k] - o[k] for k in range(3)]
        v = [b[k] - o[k] for k in range(3)]
        w = [c[k] - o[k] for k in range(3)]
        det = (
            u[0] * (v[1] * w[2] - v[2] * w[1])
            - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])
        )
        pts_o = pts[0]
        u0 = [pts[1][k] - pts_o[k] for k in range(3)]
        v0 = [pts[2][k] - pts_o[k] for k in range(3)]
        w0 = [pts[4][k] - pts_o[k] for k in range(3)]
        det0 = (
            u0[0] * (v0[1] * w0[2] - v0[2] * w0[1])
            - u0[1] * (v0[0] * w0[2] - v0[2] * w0[0])
            + u0[2] * (v0[0] * w0[1] - v0[1] * w0[0])
        )
        self.assertGreater(det * det0, 0.0)

    def test_deterministic_on_a_cube(self):
        pts = _box_corners(2, 2, 2, origin=(1, 1, 1))
        self.assertEqual(canonicalize(pts), canonicalize(pts))


class TestChecks(unittest.TestCase):
    def test_axes_ordered(self):
        self.assertTrue(axes_are_ordered((4, 2, 1)))
        self.assertFalse(axes_are_ordered((1, 2, 4)))

    def test_reference_face_seated(self):
        self.assertTrue(reference_face_seated(-0.5, (4, 2, 1)))
        self.assertFalse(reference_face_seated(0.5, (4, 2, 1)))

    def test_ambiguity_flag_on_square_plate(self):
        flags = ambiguity_flags((4.0, 4.0, 1.0))
        self.assertEqual(len(flags), 1)
        self.assertIn("pose ambiguous", flags[0])

    def test_no_flag_on_distinct_extents(self):
        self.assertEqual(ambiguity_flags((4.0, 2.0, 1.0)), [])

    def test_cube_flags_all_three_pairs(self):
        self.assertEqual(len(ambiguity_flags((2.0, 2.0, 2.0))), 3)


class TestPoseReport(unittest.TestCase):
    def test_non_compliant_offset_part(self):
        report = pose_report(_box_corners(4, 2, 1, origin=(5, 5, 5)))
        self.assertFalse(report.compliant)
        self.assertFalse(report.centered)
        self.assertTrue(report.axes_ordered)
        self.assertIsNone(report.reference_face_seated)

    def test_wrong_axis_order(self):
        report = pose_report(_box_corners(1, 2, 4, origin=(-0.5, -1, -2)))
        self.assertTrue(report.centered)
        self.assertFalse(report.axes_ordered)
        self.assertFalse(report.compliant)

    def test_reference_face_check(self):
        pts = canonicalize(_box_corners(4, 2, 1))
        good = pose_report(pts, reference_face_z=-0.5)
        bad = pose_report(pts, reference_face_z=0.5)
        self.assertTrue(good.compliant)
        self.assertTrue(good.reference_face_seated)
        self.assertFalse(bad.compliant)
        self.assertFalse(bad.reference_face_seated)

    def test_to_dict(self):
        payload = pose_report(canonicalize(_box_corners(4, 2, 1))).to_dict()
        self.assertTrue(payload["compliant"])
        self.assertEqual(payload["extents"], [4, 2, 1])


if __name__ == "__main__":
    unittest.main()
