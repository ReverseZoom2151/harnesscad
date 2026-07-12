"""Tests for geometry.shapegramm_geometry_sharing (dedup + transform compression)."""

import math
import unittest

from geometry.shapegramm_geometry_sharing import (
    geometry_signature, share_geometries, compress_transforms, CompressedGroup,
)


def _unit_cube():
    return [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ]


def _translate(verts, t):
    return [(x + t[0], y + t[1], z + t[2]) for x, y, z in verts]


def _scale(verts, s):
    return [(x * s, y * s, z * s) for x, y, z in verts]


def _rotate_z(verts, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return [(x * c - y * s, x * s + y * c, z) for x, y, z in verts]


class SignatureTest(unittest.TestCase):
    def test_translation_invariant(self):
        a = geometry_signature(_unit_cube())
        b = geometry_signature(_translate(_unit_cube(), (10, -5, 3)))
        self.assertEqual(a, b)

    def test_uniform_scale_invariant(self):
        a = geometry_signature(_unit_cube())
        b = geometry_signature(_scale(_unit_cube(), 4.0))
        self.assertEqual(a, b)

    def test_rotation_invariant(self):
        a = geometry_signature(_unit_cube())
        b = geometry_signature(_rotate_z(_unit_cube(), 37.0))
        self.assertEqual(a, b)

    def test_ordering_invariant(self):
        cube = _unit_cube()
        a = geometry_signature(cube)
        b = geometry_signature(list(reversed(cube)))
        self.assertEqual(a, b)

    def test_different_shapes_differ(self):
        cube = geometry_signature(_unit_cube())
        tetra = geometry_signature([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        self.assertNotEqual(cube, tetra)

    def test_empty_and_single(self):
        self.assertEqual(geometry_signature([]), (0, ()))
        self.assertEqual(geometry_signature([(1, 2, 3)]), (1, ()))


class ShareGeometriesTest(unittest.TestCase):
    def test_clusters_affine_copies(self):
        objects = [
            ("a", _unit_cube()),
            ("b", _translate(_unit_cube(), (5, 0, 0))),
            ("c", _scale(_rotate_z(_unit_cube(), 20), 2.0)),
            ("d", [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),  # tetra, unique
        ]
        assignment, stats = share_geometries(objects)
        self.assertEqual(assignment["a"], assignment["b"])
        self.assertEqual(assignment["a"], assignment["c"])
        self.assertNotEqual(assignment["a"], assignment["d"])
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["unique"], 2)
        self.assertEqual(stats["redundant"], 2)
        self.assertAlmostEqual(stats["redundant_fraction"], 0.5)

    def test_geometry_ids_stable_order(self):
        objects = [("x", _unit_cube()), ("y", _unit_cube())]
        assignment, _ = share_geometries(objects)
        self.assertEqual(assignment["x"], "geom_0")
        self.assertEqual(assignment["y"], "geom_0")

    def test_empty(self):
        assignment, stats = share_geometries([])
        self.assertEqual(assignment, {})
        self.assertEqual(stats["redundant_fraction"], 0.0)


class CompressTransformsTest(unittest.TestCase):
    def _beam_row(self):
        # a row of beams: same scale/rotation/color, varying translation
        return [
            {"translation": (i * 2.0, 0, 0), "scale": (1, 1, 4),
             "rotation": (0, 0, 0), "color": (0.5, 0.5, 0.5)}
            for i in range(10)
        ]

    def test_coherent_row_compresses_to_one_group(self):
        groups, stats = compress_transforms(self._beam_row())
        self.assertEqual(len(groups), 1)
        self.assertIsInstance(groups[0], CompressedGroup)
        self.assertEqual(len(groups[0].translations), 10)
        self.assertLess(stats["compression_ratio"], 1.0)

    def test_distinct_transforms_not_merged(self):
        objs = [
            {"translation": (0, 0, 0), "scale": (1, 1, 1),
             "rotation": (0, 0, 0), "color": (1, 0, 0)},
            {"translation": (1, 0, 0), "scale": (2, 2, 2),
             "rotation": (0, 0, 0), "color": (1, 0, 0)},
        ]
        groups, stats = compress_transforms(objs)
        self.assertEqual(len(groups), 2)
        self.assertEqual(stats["groups"], 2)

    def test_footprint_estimate(self):
        groups, stats = compress_transforms(self._beam_row())
        # 10 objects * 12 = 120 uncompressed; 1 group: 9 + 3*10 = 39
        self.assertEqual(stats["uncompressed_components"], 120)
        self.assertEqual(stats["compressed_components"], 39)

    def test_deterministic(self):
        a = compress_transforms(self._beam_row())
        b = compress_transforms(self._beam_row())
        self.assertEqual(a, b)

    def test_empty(self):
        groups, stats = compress_transforms([])
        self.assertEqual(groups, ())
        self.assertEqual(stats["compression_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
