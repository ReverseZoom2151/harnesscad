"""Tests for exploration.blockdecomp_state."""

import random
import unittest

from harnesscad.domain.geometry.blockdecomp_domain import Shape
from harnesscad.domain.geometry.blockdecomp_cut import CutAction
from harnesscad.agents.exploration.blockdecomp_state import (
    DecompositionState,
    LocalObservation,
    observe,
    observe_all,
    select_vertex_index,
)


class TestLocalObservation(unittest.TestCase):
    def setUp(self):
        self.sq = Shape.from_rectangles([(0, 0, 4, 2)])

    def test_observe_all_count(self):
        obs = observe_all(self.sq)
        self.assertEqual(len(obs), 4)

    def test_vector_length_is_nine(self):
        obs = observe_all(self.sq)[0]
        self.assertEqual(len(obs.to_vector()), 9)

    def test_right_angle_type_on_rectangle(self):
        for o in observe_all(self.sq):
            self.assertEqual(o.angle_type, "right")
            self.assertAlmostEqual(o.angle, 90.0)

    def test_aspect_ratio_in_observation(self):
        self.assertAlmostEqual(observe_all(self.sq)[0].aspect_ratio, 2.0)

    def test_centroid_vector_points_inward(self):
        # From corner (0,0) the centroid (2,1) is up-and-right.
        o = [o for o in observe_all(self.sq) if o.vertex == (0.0, 0.0)][0]
        self.assertGreater(o.to_centroid[0], 0.0)
        self.assertGreater(o.to_centroid[1], 0.0)


class TestReentrantObservation(unittest.TestCase):
    def test_reentrant_angle_type(self):
        l = Shape.from_rectangles([(0, 0, 2, 1), (0, 0, 1, 2)])
        obs = observe_all(l)
        reentrant = [o for o in obs if o.angle_type == "reentrant"]
        self.assertEqual(len(reentrant), 1)
        self.assertAlmostEqual(reentrant[0].angle, 270.0)


class TestDecompositionState(unittest.TestCase):
    def test_initial_quad_is_terminal(self):
        st = DecompositionState.initial(Shape.from_rectangles([(0, 0, 2, 2)]))
        self.assertTrue(st.is_terminal)
        self.assertEqual(len(st.all_blocks()), 1)

    def test_initial_non_quad_not_terminal(self):
        l = Shape.from_rectangles([(0, 0, 2, 1), (0, 0, 1, 2)])
        st = DecompositionState.initial(l)
        self.assertFalse(st.is_terminal)
        self.assertIsNotNone(st.current())

    def test_effective_cut_completes_l_shape(self):
        l = Shape.from_rectangles([(0, 0, 2, 1), (0, 0, 1, 2)])
        st = DecompositionState.initial(l)
        st2 = st.apply(CutAction(vertex=(1.0, 1.0), direction="y"))
        self.assertTrue(st2.is_terminal)
        self.assertEqual(len(st2.all_blocks()), 2)
        self.assertEqual(st2.steps, 1)

    def test_ineffective_cut_keeps_part(self):
        l = Shape.from_rectangles([(0, 0, 2, 1), (0, 0, 1, 2)])
        st = DecompositionState.initial(l)
        # Cut along a boundary side: ineffective.
        st2 = st.apply(CutAction(vertex=(0.0, 0.0), direction="x"))
        self.assertFalse(st2.is_terminal)
        self.assertEqual(st2.steps, 1)

    def test_legal_actions_nonempty_for_non_quad(self):
        l = Shape.from_rectangles([(0, 0, 2, 1), (0, 0, 1, 2)])
        st = DecompositionState.initial(l)
        self.assertGreater(len(st.legal_actions()), 0)


class TestSelectVertex(unittest.TestCase):
    def test_deterministic_argmax(self):
        self.assertEqual(select_vertex_index(None, [0.1, 0.9, 0.3]), 1)

    def test_argmax_ties_lowest_index(self):
        self.assertEqual(select_vertex_index(None, [0.5, 0.5]), 0)

    def test_stochastic_is_seed_reproducible(self):
        a = select_vertex_index(None, [1.0, 1.0, 1.0], rng=random.Random(7))
        b = select_vertex_index(None, [1.0, 1.0, 1.0], rng=random.Random(7))
        self.assertEqual(a, b)
        self.assertIn(a, (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
