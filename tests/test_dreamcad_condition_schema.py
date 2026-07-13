"""Tests for reconstruction.dreamcad_condition_schema."""

import unittest
from math import sqrt

from harnesscad.domain.reconstruction.fitting.dreamcad_condition_schema import (
    ConditionSchema,
    Modality,
)


class TestConditionSchema(unittest.TestCase):
    def setUp(self):
        self.schema = ConditionSchema(feature_dim=32)

    def test_length(self):
        self.assertEqual(self.schema.length, 36)

    def test_text_deterministic_and_normalised(self):
        a = self.schema.encode_text("a mounting plate with holes")
        b = self.schema.encode_text("a mounting plate with holes")
        self.assertEqual(a, b)
        block = a[4:]
        self.assertAlmostEqual(sqrt(sum(x * x for x in block)), 1.0, places=9)
        self.assertEqual(self.schema.modality_of(a), Modality.TEXT)

    def test_text_differs_by_content(self):
        a = self.schema.encode_text("bolt")
        b = self.schema.encode_text("washer")
        self.assertNotEqual(a, b)

    def test_points_encoding(self):
        pts = [(-0.4, -0.4, -0.4), (0.4, 0.4, 0.4), (0.0, 0.0, 0.0)]
        vec = self.schema.encode_points(pts, grid=2)
        self.assertEqual(len(vec), self.schema.length)
        self.assertEqual(self.schema.modality_of(vec), Modality.POINT)

    def test_points_empty_rejected(self):
        with self.assertRaises(ValueError):
            self.schema.encode_points([])

    def test_points_grid_too_large(self):
        small = ConditionSchema(feature_dim=8)
        with self.assertRaises(ValueError):
            small.encode_points([(0.0, 0.0, 0.0)], grid=4)

    def test_voxel_feature_layout(self):
        vec = self.schema.encode_voxel_feature(
            visual=[0.1, 0.2], normal=[0.0, 0.0, 1.0],
            center=(0.5, 0.5, 0.5), sdf=0.01)
        self.assertEqual(len(vec), self.schema.length)
        self.assertEqual(self.schema.modality_of(vec), Modality.VOXEL)

    def test_voxel_center_validation(self):
        with self.assertRaises(ValueError):
            self.schema.encode_voxel_feature([], [], (0.0, 0.0), 0.0)

    def test_modality_roundtrip_all(self):
        vecs = {
            Modality.TEXT: self.schema.encode_text("plate"),
            Modality.POINT: self.schema.encode_points(
                [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1)], grid=2),
            Modality.VOXEL: self.schema.encode_voxel_feature(
                [1.0], [0.0], (0.0, 0.0, 0.0), 0.0),
        }
        for modality, vec in vecs.items():
            self.assertEqual(self.schema.modality_of(vec), modality)


if __name__ == "__main__":
    unittest.main()
