"""Tests that edge_convexity can return discrete JoinABLe Convexity ids.

These prove the wiring: edge_convexity's new discrete path maps its continuous
three-way sign onto the authoritative ``brep_entity_ids.Convexity`` enum via that
module's ``EDGE_CONVEXITY_TO_ID`` bridge (not a second mapping), while the
existing string API and its return type stay unchanged.
"""

import unittest

from harnesscad.domain.geometry.topology import brep_entity_ids
from harnesscad.domain.geometry.topology.brep_entity_ids import Convexity
from harnesscad.domain.geometry.topology.edge_convexity import (
    CONCAVE,
    CONVEX,
    SMOOTH,
    classify_edge_convexity,
    classify_edge_convexity_id,
    discrete_convexity,
)


class DiscreteConvexityTest(unittest.TestCase):
    def test_string_api_return_type_unchanged(self):
        # The existing classifier still returns a plain string label.
        label = classify_edge_convexity((1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.assertIsInstance(label, str)
        self.assertEqual(label, CONVEX)

    def test_label_lift_uses_bridge_table(self):
        for label in (CONVEX, CONCAVE, SMOOTH):
            self.assertEqual(
                int(discrete_convexity(label)),
                brep_entity_ids.EDGE_CONVEXITY_TO_ID[label],
            )

    def test_convex_edge_to_discrete_id(self):
        # A convex edge -> the discrete Convexity.Convex id, through edge_convexity.
        cvx = classify_edge_convexity_id((1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.assertIs(cvx, Convexity.Convex)
        self.assertEqual(int(cvx), 1)

    def test_concave_edge_to_discrete_id(self):
        cvx = classify_edge_convexity_id((1, 0, 0), (0, 1, 0), (0, 0, -1))
        self.assertIs(cvx, Convexity.Concave)
        self.assertEqual(int(cvx), 2)

    def test_smooth_edge_to_discrete_id(self):
        cvx = classify_edge_convexity_id((1, 0, 0), (1, 0, 0), (0, 0, 1))
        self.assertIs(cvx, Convexity.Smooth)
        self.assertEqual(int(cvx), 3)

    def test_discrete_agrees_with_string_classifier(self):
        # The discrete path is exactly the string path lifted through the bridge.
        args = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        label = classify_edge_convexity(*args)
        self.assertIs(
            classify_edge_convexity_id(*args), discrete_convexity(label)
        )

    def test_forward_flag_flips_discrete_id(self):
        args = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.assertIs(classify_edge_convexity_id(*args, forward=True), Convexity.Convex)
        self.assertIs(
            classify_edge_convexity_id(*args, forward=False), Convexity.Concave
        )

    def test_new_states_available_from_enum(self):
        # The three states the continuous sign cannot express remain reachable.
        self.assertIs(brep_entity_ids.classify("None"), Convexity.NoneType)
        self.assertIs(brep_entity_ids.classify("Non-manifold"), Convexity.Nonmanifold)
        self.assertIs(brep_entity_ids.classify("Degenerate"), Convexity.Degenerate)

    def test_unknown_label_raises(self):
        with self.assertRaises(KeyError):
            discrete_convexity("not-a-label")


if __name__ == "__main__":
    unittest.main()
