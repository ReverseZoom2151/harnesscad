"""Tests for OpenECAD Algorithm 2 reference-plane finding."""

import unittest

from programs import openecad_script as oe
from reconstruction import openecad_refplane as rp


class TestVectorPredicates(unittest.TestCase):
    def test_parallel(self):
        self.assertTrue(rp.is_parallel((0, 0, 1), (0, 0, -2)))
        self.assertFalse(rp.is_parallel((0, 0, 1), (1, 0, 0)))

    def test_perpendicular(self):
        self.assertTrue(rp.is_perpendicular((0, 0, 1), (1, 0, 0)))
        self.assertFalse(rp.is_perpendicular((0, 0, 1), (0, 0, 1)))


class TestFindReferencePlane(unittest.TestCase):
    def setUp(self):
        # A box extruded from z=0 up to z=10 along +z.
        self.box = rp.ExtrudeFeature(
            normal=(0.0, 0.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            extent=10.0,
            lines=(
                ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)),   # bottom edge, y=0 face
                ((5.0, 0.0, 0.0), (5.0, 5.0, 0.0)),   # x=5 face
            ),
        )

    def test_sameplane_base_face(self):
        res = rp.find_reference_plane((0.0, 0.0, 1.0), (2.0, 2.0, 0.0), [self.box])
        self.assertTrue(res.found)
        self.assertEqual(res.ref_type, oe.REF_SAMEPLANE)
        self.assertEqual(res.extrude_index, 0)

    def test_topface(self):
        res = rp.find_reference_plane((0.0, 0.0, 1.0), (2.0, 2.0, 10.0), [self.box])
        self.assertTrue(res.found)
        self.assertEqual(res.ref_type, oe.REF_TOPFACE)

    def test_parallel_but_no_match(self):
        res = rp.find_reference_plane((0.0, 0.0, 1.0), (2.0, 2.0, 4.0), [self.box])
        self.assertFalse(res.found)

    def test_sideface_via_boundary_line(self):
        # Target plane y=0 (normal +y) is perpendicular to the box normal and
        # contains the first boundary line (both endpoints at y=0).
        res = rp.find_reference_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), [self.box])
        self.assertTrue(res.found)
        self.assertEqual(res.ref_type, oe.REF_SIDEFACE)
        self.assertEqual(res.line_index, 0)

    def test_sideface_selects_correct_line(self):
        # Target plane x=5 contains only the second boundary line.
        res = rp.find_reference_plane((1.0, 0.0, 0.0), (5.0, 0.0, 0.0), [self.box])
        self.assertTrue(res.found)
        self.assertEqual(res.ref_type, oe.REF_SIDEFACE)
        self.assertEqual(res.line_index, 1)

    def test_no_extrudes(self):
        self.assertFalse(
            rp.find_reference_plane((0, 0, 1), (0, 0, 0), []).found)

    def test_returns_first_match(self):
        second = rp.ExtrudeFeature((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 10.0)
        res = rp.find_reference_plane(
            (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), [self.box, second])
        self.assertEqual(res.extrude_index, 0)

    def test_as_call_kwargs(self):
        res = rp.find_reference_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), [self.box])
        kw = res.as_call_kwargs()
        self.assertEqual(kw["type"], oe.REF_SIDEFACE)
        self.assertEqual(kw["line_index"], 0)


if __name__ == "__main__":
    unittest.main()
