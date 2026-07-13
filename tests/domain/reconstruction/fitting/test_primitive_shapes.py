"""Tests for the Sketch2CAD architectural shape geometry library."""

import math
import unittest

from harnesscad.domain.reconstruction.fitting import primitive_shapes as sh
from harnesscad.domain.reconstruction.tokens import sketch2cad_scene as sd


class TestBuilders(unittest.TestCase):
    def test_all_seven_shapes_build(self):
        for shape in sh.SHAPE_TYPES:
            mesh = sh.build_shape(shape)
            self.assertGreaterEqual(len(mesh.vertices), 4)
            self.assertGreaterEqual(len(mesh.edges), 3)
            self.assertGreaterEqual(len(mesh.faces), 2)

    def test_unknown_shape(self):
        with self.assertRaises(ValueError):
            sh.build_shape("igloo")

    def test_edge_indices_valid(self):
        for shape in sh.SHAPE_TYPES:
            mesh = sh.build_shape(shape)
            n = len(mesh.vertices)
            for (a, b) in mesh.edges:
                self.assertTrue(0 <= a < n and 0 <= b < n)
            for face in mesh.faces:
                for idx in face:
                    self.assertTrue(0 <= idx < n)

    def test_cube_is_unit_box(self):
        mesh = sh.build_shape("cube", size=(2.0, 4.0, 6.0))
        lo, hi = mesh.bounding_box()
        self.assertAlmostEqual(hi[0] - lo[0], 2.0)
        self.assertAlmostEqual(hi[1] - lo[1], 4.0)
        self.assertAlmostEqual(hi[2] - lo[2], 6.0)
        self.assertEqual(len(mesh.vertices), 8)

    def test_pyramid_has_apex(self):
        mesh = sh.build_shape("pyramid", size=(2.0, 2.0, 3.0))
        lo, hi = mesh.bounding_box()
        self.assertAlmostEqual(hi[2], 3.0)  # apex height
        self.assertEqual(len(mesh.vertices), 5)

    def test_aframe_is_prism(self):
        mesh = sh.build_shape("aframe")
        self.assertEqual(len(mesh.vertices), 6)
        # ridge at z=1
        lo, hi = mesh.bounding_box()
        self.assertAlmostEqual(hi[2], 1.0)


class TestPlacement(unittest.TestCase):
    def test_translation(self):
        mesh = sh.build_shape("cube", position=(10.0, 20.0, 5.0))
        lo, hi = mesh.bounding_box()
        # base centre footprint shifts by position; z base at 5
        self.assertAlmostEqual(lo[2], 5.0)
        cx = (lo[0] + hi[0]) / 2
        cy = (lo[1] + hi[1]) / 2
        self.assertAlmostEqual(cx, 10.0)
        self.assertAlmostEqual(cy, 20.0)

    def test_yaw_90_swaps_footprint(self):
        mesh = sh.build_shape("cube", rotation=(90.0, 0.0), size=(4.0, 2.0, 1.0))
        lo, hi = mesh.bounding_box()
        # after 90deg yaw the 4-wide x becomes 4-wide y
        self.assertAlmostEqual(hi[0] - lo[0], 2.0, places=5)
        self.assertAlmostEqual(hi[1] - lo[1], 4.0, places=5)

    def test_yaw_360_is_identity(self):
        a = sh.build_shape("hip", rotation=(0.0, 0.0))
        b = sh.build_shape("hip", rotation=(360.0, 0.0))
        for va, vb in zip(a.vertices, b.vertices):
            for ca, cb in zip(va, vb):
                self.assertAlmostEqual(ca, cb, places=5)


class TestSceneObjectBridge(unittest.TestCase):
    def test_build_from_object(self):
        obj = sd.SceneObject("shed", (1.0, 2.0, 0.0), (0.0, 0.0), (3.0, 3.0, 4.0))
        mesh = sh.build_from_object(obj)
        lo, hi = mesh.bounding_box()
        self.assertAlmostEqual(hi[2], 4.0)  # ridge height = sz

    def test_roof_shapes_have_ridge_above_walls(self):
        for shape in ("shed", "hip", "mansard"):
            mesh = sh.build_shape(shape, size=(2.0, 2.0, 10.0))
            zs = sorted({round(v[2], 6) for v in mesh.vertices})
            # at least three distinct heights: base, eave, ridge/deck
            self.assertGreaterEqual(len(zs), 3)
            self.assertAlmostEqual(max(zs), 10.0)


if __name__ == "__main__":
    unittest.main()
