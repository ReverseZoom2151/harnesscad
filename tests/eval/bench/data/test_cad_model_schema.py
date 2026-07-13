"""Tests for bench.cadtests_model (CADTESTS structured B-rep schema)."""

import unittest

from harnesscad.eval.bench.data.cad_model_schema import (
    CADModel,
    Edge,
    Face,
    similarity_augmentations,
)


def _box_model():
    """A unit-ish rectangular block: 6 planar faces, 12 line edges, 8 vertices."""
    faces = tuple(Face("plane", area) for area in (2.0, 2.0, 3.0, 3.0, 6.0, 6.0))
    edges = tuple(Edge("line", 1.0) for _ in range(12))
    return CADModel(
        faces=faces,
        edges=edges,
        vertices=8,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_size=(2.0, 3.0, 1.0),
        volume=6.0,
        center_of_mass=(1.0, 1.5, 0.5),
        solids=1,
        shells=1,
    )


def _cyl_hole_model():
    """A block with a cylindrical through-hole: 2 cylinder faces, circle edges."""
    faces = (Face("plane", 4.0), Face("plane", 4.0), Face("cylinder", 3.0),
             Face("cylinder", 3.0), Face("plane", 2.0), Face("plane", 2.0))
    edges = (Edge("circle", 3.14), Edge("circle", 3.14), Edge("line", 1.0))
    return CADModel(
        faces=faces, edges=edges, vertices=8,
        bbox_min=(0.0, 0.0, 0.0), bbox_size=(4.0, 4.0, 2.0),
        volume=28.0, center_of_mass=(2.0, 2.0, 1.0),
    )


class TestCounts(unittest.TestCase):
    def test_topology_counts(self):
        m = _box_model()
        self.assertEqual(m.num_faces(), 6)
        self.assertEqual(m.num_edges(), 12)
        self.assertEqual(m.num_vertices(), 8)
        self.assertEqual(m.num_solids(), 1)
        self.assertEqual(m.num_shells(), 1)

    def test_typed_counts(self):
        m = _cyl_hole_model()
        self.assertEqual(m.count_faces_of_type("cylinder"), 2)
        self.assertEqual(m.count_faces_of_type("plane"), 4)
        self.assertEqual(m.count_edges_of_type("circle"), 2)
        self.assertTrue(m.has_face_type("cylinder"))
        self.assertFalse(m.has_face_type("sphere"))
        self.assertTrue(m.has_edge_type("circle"))
        self.assertFalse(m.has_edge_type("spline"))

    def test_bad_type_raises(self):
        with self.assertRaises(ValueError):
            Face("blob", 1.0)
        with self.assertRaises(ValueError):
            Edge("wiggle", 1.0)
        with self.assertRaises(ValueError):
            _box_model().count_faces_of_type("blob")


class TestDimensions(unittest.TestCase):
    def test_dimension_and_axes(self):
        m = _box_model()
        self.assertEqual(m.dimension("x"), 2.0)
        self.assertEqual(m.dimension(1), 3.0)
        self.assertEqual(m.dimension("z"), 1.0)
        self.assertEqual(m.largest_axis(), 1)
        self.assertEqual(m.smallest_axis(), 2)

    def test_aspect_ratio(self):
        m = _box_model()
        self.assertAlmostEqual(m.aspect_ratio("y", "x"), 1.5)
        with self.assertRaises(ValueError):
            m.dimension("w")

    def test_bbox_center(self):
        m = _box_model()
        self.assertEqual(m.bbox_center(), (1.0, 1.5, 0.5))

    def test_areas(self):
        m = _box_model()
        self.assertEqual(m.face_area(4), 6.0)
        self.assertEqual(m.total_face_area(), 22.0)


class TestVolumetric(unittest.TestCase):
    def test_volume_and_fill(self):
        m = _box_model()
        self.assertEqual(m.get_volume(), 6.0)
        self.assertEqual(m.bbox_volume(), 6.0)
        self.assertAlmostEqual(m.fill_factor(), 1.0)

    def test_hole_fill_factor_below_one(self):
        m = _cyl_hole_model()
        self.assertLess(m.fill_factor(), 1.0)

    def test_com(self):
        self.assertEqual(_box_model().get_center_of_mass(), (1.0, 1.5, 0.5))


class TestValidity(unittest.TestCase):
    def test_valid_solid(self):
        self.assertTrue(_box_model().is_valid_solid())

    def test_two_solids_invalid(self):
        m = _box_model()
        bad = CADModel(m.faces, m.edges, m.vertices, m.bbox_min, m.bbox_size,
                       m.volume, m.center_of_mass, solids=2, shells=2)
        self.assertFalse(bad.is_valid_solid())

    def test_zero_volume_invalid(self):
        m = _box_model()
        bad = CADModel(m.faces, m.edges, m.vertices, m.bbox_min,
                       (2.0, 3.0, 0.0), 0.0, m.center_of_mass)
        self.assertFalse(bad.is_valid_solid())


class TestSimilarityTransform(unittest.TestCase):
    def test_scale_effects(self):
        m = _box_model()
        t = m.transformed(scale=2.0)
        # linear dims * 2, volume * 8, area * 4
        self.assertEqual(t.bbox_size, (4.0, 6.0, 2.0))
        self.assertEqual(t.get_volume(), 48.0)
        self.assertEqual(t.face_area(4), 24.0)
        # types and counts preserved
        self.assertEqual(t.num_faces(), 6)
        self.assertEqual(t.count_faces_of_type("plane"), 6)

    def test_rotation_permutes_dims(self):
        m = _box_model()
        # rotate 90 about z swaps x and y extents
        t = m.transformed(rotate_axis="z")
        self.assertEqual(t.bbox_size, (3.0, 2.0, 1.0))
        self.assertEqual(t.get_volume(), m.get_volume())

    def test_translation_shifts_com(self):
        m = _box_model()
        t = m.transformed(translate=(10.0, 0.0, 0.0))
        self.assertEqual(t.get_center_of_mass()[0], 11.0)
        self.assertEqual(t.bbox_size, m.bbox_size)

    def test_scale_must_be_positive(self):
        with self.assertRaises(ValueError):
            _box_model().transformed(scale=0.0)

    def test_augmentation_set(self):
        m = _box_model()
        variants = similarity_augmentations(
            m, scales=(1.0, 2.0), rotations=(None, "z"),
            translations=((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)))
        # first is the reference itself
        self.assertIs(variants[0], m)
        self.assertGreater(len(variants), 1)

    def test_detailed_omits_scale(self):
        m = _box_model()
        variants = similarity_augmentations(
            m, scales=(2.0,), rotations=(None,),
            translations=((0.0, 0.0, 0.0),), include_scale=False)
        # scaling omitted -> only the reference remains
        self.assertEqual(len(variants), 1)


if __name__ == "__main__":
    unittest.main()
