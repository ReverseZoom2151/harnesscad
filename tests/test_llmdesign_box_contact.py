"""Tests for :mod:`geometry.llmdesign_box_contact`.

Covers the paper's "collision problem" (Makatura et al., sec. 4.1.2 / L.1):
the tri-state SEPARATED / TOUCHING / OVERLAPPING classification, the
face-contact axis/area report, edge/corner degeneracies, and the
floating-tabletop should-touch auditor plus the all-pairs protrusion scan.
"""

import unittest

from geometry.llmdesign_box_contact import (
    Box,
    audit_should_touch,
    classify_boxes,
    classify_interval,
    scan_protrusions,
)


class TestBoxContact(unittest.TestCase):
    # -- box basics ---------------------------------------------------------
    def test_aabb_conversion(self):
        b = Box(0.0, 0.0, 0.0, 2.0, 4.0, 6.0)
        lo, hi = b.aabb()
        self.assertEqual(lo, (-1.0, -2.0, -3.0))
        self.assertEqual(hi, (1.0, 2.0, 3.0))
        self.assertEqual(b.interval(1), (-2.0, 2.0))

    def test_non_positive_size_raises(self):
        with self.assertRaises(ValueError):
            Box(0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            Box(0.0, 0.0, 0.0, 1.0, -2.0, 1.0)

    # -- per-axis interval relation ----------------------------------------
    def test_interval_relation(self):
        # Overlapping.
        r = classify_interval((0.0, 2.0), (1.0, 3.0))
        self.assertEqual(r.kind, "overlapping")
        self.assertAlmostEqual(r.overlap, 1.0)
        self.assertEqual(r.gap, 0.0)
        # Touching (edges coincide).
        r = classify_interval((0.0, 2.0), (2.0, 4.0))
        self.assertEqual(r.kind, "touching")
        self.assertAlmostEqual(r.overlap, 0.0)
        # Separated (gap).
        r = classify_interval((0.0, 2.0), (5.0, 7.0))
        self.assertEqual(r.kind, "separated")
        self.assertAlmostEqual(r.gap, 3.0)

    # -- overlap volume -----------------------------------------------------
    def test_overlap_volume(self):
        a = Box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)   # [-1,1]^3
        b = Box(1.0, 1.0, 1.0, 2.0, 2.0, 2.0)   # [0,2]^3
        rel = classify_boxes(a, b)
        self.assertEqual(rel.classification, "OVERLAPPING")
        self.assertTrue(rel.is_overlapping)
        self.assertTrue(rel.is_protruding)
        self.assertAlmostEqual(rel.overlap_volume, 1.0)  # unit cube overlap

    # -- three top-level classifications -----------------------------------
    def test_face_contact_axis_and_area(self):
        # Two 2x2x2 boxes stacked along z, sharing the z=1 face.
        a = Box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)   # z in [-1,1]
        b = Box(0.0, 0.0, 2.0, 2.0, 2.0, 2.0)   # z in [1,3]
        rel = classify_boxes(a, b)
        self.assertEqual(rel.classification, "TOUCHING")
        self.assertTrue(rel.is_face_contact)
        self.assertEqual(rel.contact_axis, (2,))
        self.assertEqual(rel.contact_axis_names, ("z",))
        self.assertAlmostEqual(rel.contact_area, 4.0)  # 2 x 2 face
        self.assertAlmostEqual(rel.overlap_volume, 0.0)

    def test_separated_reports_gap_and_axis(self):
        a = Box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)   # z in [-1,1]
        b = Box(0.0, 0.0, 5.0, 2.0, 2.0, 2.0)   # z in [4,6]
        rel = classify_boxes(a, b)
        self.assertEqual(rel.classification, "SEPARATED")
        self.assertTrue(rel.is_separated)
        self.assertAlmostEqual(rel.separation_gap, 3.0)
        self.assertEqual(rel.separation_axis_name, "z")

    # -- degenerate edge / corner contact ----------------------------------
    def test_edge_and_corner_contact(self):
        base = Box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)  # [-1,1]^3
        # Edge contact: touching on x and z, overlapping on y.
        edge = Box(2.0, 0.0, 2.0, 2.0, 2.0, 2.0)  # x[1,3], y[-1,1], z[1,3]
        rel = classify_boxes(base, edge)
        self.assertEqual(rel.classification, "TOUCHING")
        self.assertFalse(rel.is_face_contact)
        self.assertEqual(set(rel.contact_axis), {0, 2})
        self.assertAlmostEqual(rel.contact_area, 0.0)
        # Corner contact: touching on all three axes.
        corner = Box(2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        rel = classify_boxes(base, corner)
        self.assertEqual(rel.classification, "TOUCHING")
        self.assertFalse(rel.is_face_contact)
        self.assertEqual(set(rel.contact_axis), {0, 1, 2})
        self.assertAlmostEqual(rel.contact_area, 0.0)

    # -- tolerance behaviour ------------------------------------------------
    def test_tolerance_behaviour(self):
        # A 1e-6 gap: with the default tiny tol it is SEPARATED...
        a = Box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)         # z in [-1,1]
        b = Box(0.0, 0.0, 2.0 + 1e-6, 2.0, 2.0, 2.0)  # z in [1+1e-6, 3+1e-6]
        rel = classify_boxes(a, b)
        self.assertEqual(rel.classification, "SEPARATED")
        # ...but with a looser tol the tiny gap is treated as face contact.
        rel = classify_boxes(a, b, tol=1e-3)
        self.assertEqual(rel.classification, "TOUCHING")
        self.assertTrue(rel.is_face_contact)

    # -- floating tabletop (should_touch auditor) --------------------------
    def test_floating_tabletop_then_contact(self):
        # A tabletop 10x10x1 that should rest on legs whose tops are at z=4.
        # Leg: height 4, centred so its top face is at z = 4.
        leg = Box(0.0, 0.0, 2.0, 1.0, 1.0, 4.0)  # z in [0,4]
        # Floating top: bottom face at z = 5, leaving a 1.0 gap above the leg.
        floating_top = Box(0.0, 0.0, 5.5, 10.0, 10.0, 1.0)  # z in [5,6]
        boxes = [("leg", leg), ("top", floating_top)]
        audits = audit_should_touch(boxes, [("top", "leg")])
        self.assertEqual(len(audits), 1)
        self.assertTrue(audits[0].is_floating)
        self.assertEqual(audits[0].status, "floating")
        self.assertAlmostEqual(audits[0].gap, 1.0)
        self.assertEqual(audits[0].gap_axis, "z")
        self.assertFalse(audits[0].ok)

        # Lower the top so its bottom face sits exactly on the leg top (z=4).
        seated_top = Box(0.0, 0.0, 4.5, 10.0, 10.0, 1.0)  # z in [4,5]
        boxes = [("leg", leg), ("top", seated_top)]
        audits = audit_should_touch(boxes, [("top", "leg")])
        self.assertEqual(audits[0].status, "face_contact")
        self.assertTrue(audits[0].ok)
        self.assertTrue(audits[0].relation.is_face_contact)
        self.assertAlmostEqual(audits[0].relation.contact_area, 1.0)  # 1 x 1 leg

    def test_should_touch_flags_protrusion(self):
        # A leg sunk into the tabletop -> protruding error, not contact.
        leg = Box(0.0, 0.0, 2.0, 1.0, 1.0, 4.0)      # z in [0,4]
        sunk_top = Box(0.0, 0.0, 3.5, 10.0, 10.0, 1.0)  # z in [3,4] -> overlaps
        boxes = [("leg", leg), ("top", sunk_top)]
        audits = audit_should_touch(boxes, [("top", "leg")])
        self.assertTrue(audits[0].is_protruding)
        self.assertGreater(audits[0].overlap_volume, 0.0)

    # -- all-pairs protrusion scan -----------------------------------------
    def test_scan_protrusions(self):
        a = Box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)        # [-1,1]^3
        b = Box(1.0, 1.0, 1.0, 2.0, 2.0, 2.0)        # overlaps a (vol 1)
        c = Box(0.5, 0.5, 0.5, 2.0, 2.0, 2.0)        # overlaps a & b (bigger)
        far = Box(100.0, 100.0, 100.0, 2.0, 2.0, 2.0)  # isolated
        boxes = [("a", a), ("b", b), ("c", c), ("far", far)]
        viols = scan_protrusions(boxes)
        # far participates in no protrusion.
        names = {(v.name_a, v.name_b) for v in viols}
        for pair in names:
            self.assertNotIn("far", pair)
        # Sorted worst-first by overlap volume.
        vols = [v.overlap_volume for v in viols]
        self.assertEqual(vols, sorted(vols, reverse=True))
        self.assertTrue(all(v.overlap_volume > 0.0 for v in viols))


if __name__ == "__main__":
    unittest.main()
