"""Tests for reconstruction.joinable_entity_features."""

import math
import unittest

from bench.joinable_joint_metrics import hit_at_top_k
from reconstruction.joinable_entity_features import (
    CONVEXITY_TYPES,
    ENTITY_TYPES,
    EntityFeatureError,
    LABEL_TYPES,
    bounding_box,
    candidate_label_matrix,
    candidate_pairs,
    common_scale,
    entity_area,
    entity_convexity,
    entity_dihedral_angle,
    entity_feature_names,
    entity_feature_vector,
    entity_length,
    entity_radius,
    entity_reversed_flags,
    entity_size,
    entity_type_index,
    entity_type_name,
    is_face,
    is_positive_label,
    label_index,
    one_hot,
    scale_features,
)


def _plane():
    return {"surface_type": "PlaneSurfaceType", "area": 4.0, "reversed": False}


def _cylinder():
    return {"surface_type": "CylinderSurfaceType", "area": 12.0,
            "reversed": True, "radius": 2.0}


def _circle():
    return {"curve_type": "Circle3DCurveType", "length": 6.0,
            "convexity": "Convex", "dihedral_angle": math.pi / 2,
            "radius": 1.0, "reversed": False}


class VocabularyTests(unittest.TestCase):
    def test_entity_type_count(self):
        self.assertEqual(len(ENTITY_TYPES), 16)
        self.assertEqual(ENTITY_TYPES[0], "PlaneSurfaceType")

    def test_convexity_and_label_vocabulary(self):
        self.assertEqual(len(CONVEXITY_TYPES), 6)
        self.assertEqual(len(LABEL_TYPES), 7)
        self.assertEqual(label_index("Joint"), 1)
        self.assertTrue(is_positive_label("JointEquivalent"))
        self.assertFalse(is_positive_label("Hole"))
        with self.assertRaises(EntityFeatureError):
            label_index("Nope")

    def test_one_hot(self):
        self.assertEqual(one_hot(1, 3), [0.0, 1.0, 0.0])
        with self.assertRaises(EntityFeatureError):
            one_hot(5, 3)


class EntityAccessorTests(unittest.TestCase):
    def test_is_face(self):
        self.assertTrue(is_face(_plane()))
        self.assertFalse(is_face(_circle()))

    def test_type_name_and_index(self):
        self.assertEqual(entity_type_name(_cylinder()), "CylinderSurfaceType")
        self.assertEqual(entity_type_index(_cylinder()), 1)
        self.assertEqual(entity_type_name(_circle()), "Circle3DCurveType")

    def test_degenerate_edge_type(self):
        edge = {"curve_type": "Line3DCurveType", "is_degenerate": True}
        self.assertEqual(entity_type_name(edge), "Degenerate3DCurveType")
        self.assertEqual(entity_convexity(edge), "Degenerate")

    def test_unknown_entity(self):
        with self.assertRaises(EntityFeatureError):
            entity_type_name({"area": 1.0})
        with self.assertRaises(EntityFeatureError):
            entity_type_name({"surface_type": "MysterySurfaceType"})

    def test_area_and_length_are_exclusive(self):
        self.assertEqual(entity_area(_plane()), 4.0)
        self.assertEqual(entity_length(_plane()), 0.0)
        self.assertEqual(entity_area(_circle()), 0.0)
        self.assertEqual(entity_length(_circle()), 6.0)
        self.assertEqual(entity_size(_plane()), 4.0)
        self.assertEqual(entity_size(_circle()), 6.0)

    def test_reversed_flags(self):
        self.assertEqual(entity_reversed_flags(_cylinder()), (1, 0, 1))
        edge = dict(_circle())
        edge["reversed"] = True
        self.assertEqual(entity_reversed_flags(edge), (0, 1, 1))
        self.assertEqual(entity_reversed_flags(_plane()), (0, 0, 0))

    def test_convexity_of_face_is_none(self):
        self.assertEqual(entity_convexity(_plane()), "None")
        self.assertEqual(entity_convexity(_circle()), "Convex")
        with self.assertRaises(EntityFeatureError):
            entity_convexity({"curve_type": "Line3DCurveType",
                              "convexity": "Bendy"})

    def test_dihedral_and_radius_defaults(self):
        self.assertEqual(entity_dihedral_angle(_plane()), 0.0)
        self.assertAlmostEqual(entity_dihedral_angle(_circle()), math.pi / 2)
        self.assertEqual(entity_radius(_plane()), -1.0)
        self.assertEqual(entity_radius(_cylinder()), 2.0)


class FeatureVectorTests(unittest.TestCase):
    def test_vector_length_matches_names(self):
        vector = entity_feature_vector(_plane())
        self.assertEqual(len(vector), len(entity_feature_names()))
        self.assertEqual(len(vector), 16 + 6 + 6 + 2)

    def test_plane_vector_contents(self):
        names = entity_feature_names()
        vector = entity_feature_vector(_plane())
        lookup = dict(zip(names, vector))
        self.assertEqual(lookup["entity_type::PlaneSurfaceType"], 1.0)
        self.assertEqual(lookup["entity_type::CylinderSurfaceType"], 0.0)
        self.assertEqual(lookup["is_face"], 1.0)
        self.assertEqual(lookup["area"], 4.0)
        self.assertEqual(lookup["length"], 0.0)
        self.assertEqual(lookup["convexity::None"], 1.0)
        self.assertEqual(lookup["radius"], -1.0)

    def test_edge_vector_contents(self):
        lookup = dict(zip(entity_feature_names(),
                          entity_feature_vector(_circle())))
        self.assertEqual(lookup["is_face"], 0.0)
        self.assertEqual(lookup["length"], 6.0)
        self.assertEqual(lookup["convexity::Convex"], 1.0)
        self.assertEqual(lookup["radius"], 1.0)

    def test_vector_is_deterministic(self):
        self.assertEqual(entity_feature_vector(_cylinder()),
                         entity_feature_vector(_cylinder()))

    def test_one_hot_blocks_sum_to_one(self):
        names = entity_feature_names()
        vector = entity_feature_vector(_circle())
        type_block = [v for n, v in zip(names, vector)
                      if n.startswith("entity_type::")]
        convex_block = [v for n, v in zip(names, vector)
                        if n.startswith("convexity::")]
        self.assertEqual(sum(type_block), 1.0)
        self.assertEqual(sum(convex_block), 1.0)


class ScalingTests(unittest.TestCase):
    def test_bounding_box(self):
        box = bounding_box([(0.0, 1.0, 2.0), (-1.0, 5.0, 0.0)])
        self.assertEqual(box, ((-1.0, 1.0, 0.0), (0.0, 5.0, 2.0)))
        with self.assertRaises(EntityFeatureError):
            bounding_box([])

    def test_common_scale_brings_into_unit_box(self):
        pts1 = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
        pts2 = [(0.0, 0.0, 0.0), (0.0, 10.0, 0.0)]
        scale = common_scale(pts1, pts2)
        scaled, _, _ = scale_features(scale, points=pts1 + pts2)
        self.assertTrue(all(abs(c) <= 1.0 for p in scaled for c in p))
        self.assertAlmostEqual(max(abs(c) for p in scaled for c in p), 0.999999)

    def test_common_scale_shared_between_bodies(self):
        pts1 = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        pts2 = [(0.0, 0.0, 0.0), (0.0, 4.0, 0.0)]
        scale = common_scale(pts1, pts2)
        s1, _, _ = scale_features(scale, points=pts1)
        s2, _, _ = scale_features(scale, points=pts2)
        # Relative proportion 2:4 preserved.
        self.assertAlmostEqual(s2[1][1] / s1[1][0], 2.0)

    def test_degenerate_scale(self):
        p = [(0.0, 0.0, 0.0)]
        self.assertEqual(common_scale(p, p), 1.0)

    def test_scale_features_scales_areas_and_lengths(self):
        _, areas, lengths = scale_features(0.5, areas=[4.0], lengths=[6.0])
        self.assertEqual(areas, [2.0])
        self.assertEqual(lengths, [3.0])


class CandidatePairTests(unittest.TestCase):
    def test_candidate_pairs_row_major(self):
        pairs = candidate_pairs([1, 2], [1, 2, 3])
        self.assertEqual(len(pairs), 6)
        self.assertEqual(pairs[0], (0, 0))
        self.assertEqual(pairs[3], (1, 0))

    def test_label_matrix(self):
        matrix = candidate_label_matrix([1, 2], [1, 2, 3], [(1, 2)])
        self.assertEqual(matrix, [[0, 0, 0], [0, 0, 1]])

    def test_label_matrix_out_of_range(self):
        with self.assertRaises(EntityFeatureError):
            candidate_label_matrix([1], [1], [(0, 5)])

    def test_label_matrix_feeds_topk_metric(self):
        entities1 = [_plane(), _cylinder()]
        entities2 = [_plane(), _circle()]
        labels = candidate_label_matrix(entities1, entities2, [(1, 1)])
        scores = [[0.1, 0.2], [0.3, 0.9]]
        self.assertTrue(hit_at_top_k(scores, labels, k=1))


if __name__ == "__main__":
    unittest.main()
