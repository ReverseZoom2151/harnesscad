"""Tests for the normal-consistency (NC) metric.

NC is the third RLCAD term (arXiv:2503.18549); these assert real computed
values, not "returns a float".
"""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.pointcloud.normal_consistency import (
    cosine_similarity,
    nearest_index,
    nearest_indices,
    normal_consistency,
    symmetric_normal_consistency,
    vector_norm,
)


def unit_box(scale=1.0, offset=(0.0, 0.0, 0.0), per_face=3):
    """Axis-aligned box surface samples as ``(point, outward normal)`` pairs.

    ``per_face`` x ``per_face`` samples on each of the 6 faces of a box of edge
    ``scale`` centred on ``offset``.
    """
    step = [(-0.5 + (i + 0.5) / per_face) for i in range(per_face)]
    out = []
    for axis in range(3):
        for side in (-1.0, 1.0):
            for u in step:
                for v in step:
                    p = [0.0, 0.0, 0.0]
                    p[axis] = side * 0.5
                    p[(axis + 1) % 3] = u
                    p[(axis + 2) % 3] = v
                    n = [0.0, 0.0, 0.0]
                    n[axis] = side
                    out.append((
                        tuple(p[d] * scale + offset[d] for d in range(3)),
                        tuple(n),
                    ))
    return out


class TestVectorHelpers(unittest.TestCase):
    def test_vector_norm(self):
        self.assertEqual(vector_norm((3.0, 4.0)), 5.0)
        self.assertEqual(vector_norm((0.0, 0.0, 0.0)), 0.0)

    def test_cosine_of_identical_directions_is_one(self):
        self.assertAlmostEqual(cosine_similarity((0, 0, 2), (0, 0, 7)), 1.0)

    def test_cosine_of_opposite_directions_is_minus_one(self):
        self.assertAlmostEqual(cosine_similarity((1, 0, 0), (-3, 0, 0)), -1.0)

    def test_cosine_of_perpendicular_is_zero(self):
        self.assertAlmostEqual(cosine_similarity((1, 0, 0), (0, 1, 0)), 0.0)

    def test_cosine_at_45_degrees(self):
        self.assertAlmostEqual(
            cosine_similarity((1, 0, 0), (1, 1, 0)), math.sqrt(0.5))

    def test_zero_normal_is_an_error_not_a_middling_score(self):
        with self.assertRaises(ValueError):
            cosine_similarity((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    def test_mismatched_length_is_an_error(self):
        with self.assertRaises(ValueError):
            cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0))


class TestNearest(unittest.TestCase):
    def test_nearest_index_picks_the_closest(self):
        pts = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.assertEqual(nearest_index((1.2, 0.0, 0.0), pts), 2)

    def test_nearest_index_ties_go_to_the_lowest_index(self):
        pts = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]
        self.assertEqual(nearest_index((0.0, 0.0, 0.0), pts), 0)

    def test_nearest_indices_returns_every_tie(self):
        pts = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (9.0, 0.0, 0.0)]
        self.assertEqual(nearest_indices((0.0, 0.0, 0.0), pts), [0, 1])

    def test_nearest_indices_on_a_unique_winner(self):
        pts = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)]
        self.assertEqual(nearest_indices((0.1, 0.0, 0.0), pts), [0])

    def test_empty_candidate_set_is_an_error(self):
        with self.assertRaises(ValueError):
            nearest_index((0.0, 0.0, 0.0), [])


class TestNormalConsistency(unittest.TestCase):
    def test_a_cloud_against_itself_is_exactly_one(self):
        box = unit_box()
        self.assertAlmostEqual(normal_consistency(box, box), 1.0)

    def test_self_consistency_survives_shared_edge_positions(self):
        # per_face=2 with corner samples puts several faces' samples on the same
        # positions; the tie-break by normal agreement is what keeps this at 1.0.
        box = [
            ((0.5, 0.5, 0.5), (1.0, 0.0, 0.0)),
            ((0.5, 0.5, 0.5), (0.0, 1.0, 0.0)),
            ((0.5, 0.5, 0.5), (0.0, 0.0, 1.0)),
        ]
        self.assertAlmostEqual(normal_consistency(box, box), 1.0)

    def test_a_globally_flipped_cloud_scores_one_unoriented_and_minus_one_signed(self):
        box = unit_box()
        flipped = [(p, tuple(-c for c in n)) for p, n in box]
        self.assertAlmostEqual(normal_consistency(box, flipped), 1.0)
        self.assertAlmostEqual(
            normal_consistency(box, flipped, unoriented=False), -1.0)

    def test_a_uniform_90_degree_rotation_of_normals_scores_zero(self):
        ref = [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
               ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
        cand = [((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
        self.assertAlmostEqual(normal_consistency(ref, cand), 0.0)

    def test_half_the_normals_wrong_scores_one_half(self):
        ref = [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
               ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
        cand = [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
        self.assertAlmostEqual(normal_consistency(ref, cand), 0.5)

    def test_a_45_degree_tilt_scores_the_cosine(self):
        ref = [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))]
        cand = [((0.0, 0.0, 0.0), (0.0, 1.0, 1.0))]
        self.assertAlmostEqual(normal_consistency(ref, cand), math.sqrt(0.5))

    def test_correspondence_follows_position_not_order(self):
        # Reversing the candidate order must not change the score: the pairing is
        # by position, so this is the invariant that proves it is not zip().
        box = unit_box()
        self.assertAlmostEqual(
            normal_consistency(box, list(reversed(box))), 1.0)

    def test_max_correspondence_is_an_optimistic_upper_bound(self):
        # A single candidate sample with the right normal, sitting nowhere near
        # the reference, scores 1.0 under "max" and much less under "nearest".
        ref = [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
        cand = [((99.0, 99.0, 99.0), (1.0, 0.0, 0.0)),
                ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
        self.assertAlmostEqual(
            normal_consistency(ref, cand, correspondence="max"), 1.0)
        self.assertAlmostEqual(
            normal_consistency(ref, cand, correspondence="nearest"), 0.0)

    def test_unknown_correspondence_is_rejected(self):
        box = unit_box()
        with self.assertRaises(ValueError):
            normal_consistency(box, box, correspondence="hopeful")

    def test_empty_cloud_is_an_error_not_a_free_one(self):
        box = unit_box()
        with self.assertRaises(ValueError):
            normal_consistency([], box)
        with self.assertRaises(ValueError):
            normal_consistency(box, [])

    def test_malformed_entries_are_rejected(self):
        with self.assertRaises(ValueError):
            normal_consistency([(1.0, 2.0, 3.0)], unit_box())


class TestSymmetricNormalConsistency(unittest.TestCase):
    def test_self_symmetric_score_is_one(self):
        box = unit_box()
        self.assertAlmostEqual(symmetric_normal_consistency(box, box), 1.0)

    def test_symmetric_is_the_mean_of_both_directions(self):
        ref = [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
               ((4.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
        cand = [((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
        fwd = normal_consistency(ref, cand)
        bwd = normal_consistency(cand, ref)
        self.assertAlmostEqual(
            symmetric_normal_consistency(ref, cand), (fwd + bwd) / 2.0)

    def test_symmetric_penalises_unreferenced_extra_surface(self):
        # The directed score ignores extra candidate geometry; symmetrising it
        # does not.  A tilted extra face drags the symmetric score below 1.0.
        ref = [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))]
        cand = [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                ((9.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
        self.assertAlmostEqual(normal_consistency(ref, cand), 1.0)
        # backward pass: the good sample scores 1, the stray one 0 -> 0.5;
        # symmetric mean of 1.0 and 0.5 is 0.75.
        self.assertAlmostEqual(normal_consistency(cand, ref), 0.5)
        self.assertAlmostEqual(symmetric_normal_consistency(ref, cand), 0.75)


if __name__ == "__main__":
    unittest.main()
