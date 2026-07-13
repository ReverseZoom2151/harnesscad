"""Tests for reconstruction.vitruvion_constraint_tokens."""

import unittest

from geometry.vitruvion_sketch_norm import VCircle, entity_from_params
from reconstruction.vitruvion_constraint_tokens import (
    CONSTRAINT_COORD_TOKENS,
    ConstraintEdge,
    ConstraintToken,
    constraints_from_tokens,
    reference_token,
    reference_vocabulary_size,
    tokenize_constraints,
)
from reconstruction.vitruvion_primitive_tokens import NON_COORD_TOKEN, Token, tokenize_sketch


class TestVocabulary(unittest.TestCase):
    def test_constraint_vocabulary(self):
        self.assertEqual(len(ConstraintToken), 16)
        self.assertEqual(int(ConstraintToken.Coincident), 3)
        self.assertEqual(int(ConstraintToken.Vertical), 15)

    def test_reference_tokens_are_offset_by_the_vocabulary(self):
        self.assertEqual(reference_token(0), 16)
        self.assertEqual(reference_token(7), 23)
        self.assertEqual(reference_vocabulary_size(130), 146)

    def test_only_two_argument_slots(self):
        self.assertEqual(CONSTRAINT_COORD_TOKENS, [2, 3])


class TestTokenizeConstraints(unittest.TestCase):
    def setUp(self):
        self.entities = [
            VCircle(xCenter=0.0, yCenter=0.0, radius=0.25),
            entity_from_params([-0.25, -0.25, 0.25, 0.25]),
        ]
        self.streams, self.gather = tokenize_sketch(self.entities, 64)
        # gather: [0 external, 1 circle, 2 circle.centre, 6 line, 7 line.start,
        #          9 line.end]

    def test_gather_layout(self):
        self.assertEqual(self.gather, [0, 1, 2, 6, 7, 9])

    def test_edge_encoding(self):
        edges = [ConstraintEdge("Coincident", (2, 4))]
        out = tokenize_constraints(edges, self.gather)
        self.assertEqual(
            out["val"],
            [
                int(ConstraintToken.Start),
                int(ConstraintToken.Coincident),
                reference_token(self.gather[2]),
                reference_token(self.gather[4]),
                int(ConstraintToken.Stop),
            ],
        )
        self.assertEqual(out["coord"], [NON_COORD_TOKEN, NON_COORD_TOKEN, 2, 3, NON_COORD_TOKEN])
        # The type token and both references share one group index.
        self.assertEqual(out["pos"], [1, 2, 2, 2, 3])

    def test_reference_tokens_point_into_the_primitive_stream(self):
        edges = [ConstraintEdge("Concentric", (1, 3))]
        out = tokenize_constraints(edges, self.gather)
        position = out["val"][2] - len(ConstraintToken)
        self.assertEqual(self.streams["val"][position], int(Token.Circle))
        position = out["val"][3] - len(ConstraintToken)
        self.assertEqual(self.streams["val"][position], int(Token.Line))

    def test_unary_constraint_uses_one_slot(self):
        out = tokenize_constraints([ConstraintEdge("Fix", (1,))], self.gather)
        self.assertEqual(out["coord"], [NON_COORD_TOKEN, NON_COORD_TOKEN, 2, NON_COORD_TOKEN])
        self.assertEqual(out["pos"], [1, 2, 2, 3])

    def test_references_are_sorted_not_argument_ordered(self):
        forward = tokenize_constraints([ConstraintEdge("Midpoint", (4, 2))], self.gather)
        reverse = tokenize_constraints([ConstraintEdge("Midpoint", (2, 4))], self.gather)
        self.assertEqual(forward["val"], reverse["val"])

    def test_external_constraints_are_dropped(self):
        out = tokenize_constraints([ConstraintEdge("Coincident", (0, 2))], self.gather)
        self.assertEqual(out["val"], [int(ConstraintToken.Start), int(ConstraintToken.Stop)])

    def test_unknown_label_is_dropped(self):
        out = tokenize_constraints([ConstraintEdge("Distance", (1, 3))], self.gather)
        self.assertEqual(out["val"], [int(ConstraintToken.Start), int(ConstraintToken.Stop)])

    def test_higher_arity_raises(self):
        with self.assertRaises(ValueError):
            tokenize_constraints([ConstraintEdge("Equal", (1, 2, 3))], self.gather)

    def test_unknown_reference_raises(self):
        with self.assertRaises(ValueError):
            tokenize_constraints([ConstraintEdge("Equal", (1, 99))], self.gather)

    def test_padding(self):
        out = tokenize_constraints(
            [ConstraintEdge("Coincident", (2, 4))], self.gather, max_length=12
        )
        self.assertEqual(len(out["val"]), 12)
        self.assertEqual(len(out["coord"]), 12)
        self.assertEqual(out["val"][-1], int(ConstraintToken.Pad))

    def test_multiple_edges_increment_the_group_index(self):
        edges = [
            ConstraintEdge("Coincident", (2, 4)),
            ConstraintEdge("Horizontal", (3,)),
        ]
        out = tokenize_constraints(edges, self.gather)
        self.assertEqual(out["pos"], [1, 2, 2, 2, 3, 3, 4])


class TestDecoding(unittest.TestCase):
    def setUp(self):
        self.entities = [
            VCircle(xCenter=0.0, yCenter=0.0, radius=0.25),
            entity_from_params([-0.25, -0.25, 0.25, 0.25]),
        ]
        _, self.gather = tokenize_sketch(self.entities, 64)

    def test_roundtrip(self):
        edges = [
            ConstraintEdge("Coincident", (2, 4)),
            ConstraintEdge("Fix", (3,)),
            ConstraintEdge("Vertical", (3,)),
        ]
        out = tokenize_constraints(edges, self.gather, max_length=32)
        decoded = constraints_from_tokens(out["val"], self.gather)
        self.assertEqual(decoded, edges)

    def test_empty_stream(self):
        out = tokenize_constraints([], self.gather)
        self.assertEqual(constraints_from_tokens(out["val"], self.gather), [])

    def test_bad_reference_token_raises(self):
        bad = [int(ConstraintToken.Start), int(ConstraintToken.Fix), reference_token(4)]
        with self.assertRaises(ValueError):
            constraints_from_tokens(bad, self.gather)


if __name__ == "__main__":
    unittest.main()
