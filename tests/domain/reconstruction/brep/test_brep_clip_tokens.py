"""Tests for domain.reconstruction.brep.brep_clip_tokens."""

import unittest

from harnesscad.domain.reconstruction.brep.brep_clip_tokens import (
    CURVE_VOCAB,
    SURFACE_VOCAB,
    brep_descriptor,
    descriptor_similarity,
    edge_token,
    face_token,
)


class TokenTest(unittest.TestCase):
    def test_face_token_onehot(self):
        tok = face_token("cylinder", 0.0)
        self.assertEqual(tok[:len(SURFACE_VOCAB)], [0, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(tok[-1], 0.0)  # log1p(0)

    def test_edge_token_unknown(self):
        with self.assertRaises(ValueError):
            edge_token("hyperbola", 1.0)

    def test_curve_vocab_len(self):
        self.assertEqual(len(edge_token("line", 2.0)), len(CURVE_VOCAB) + 1)


class DescriptorTest(unittest.TestCase):
    def test_fixed_length(self):
        d = brep_descriptor(
            [{"type": "plane", "area": 4.0}],
            [{"type": "line", "length": 2.0}],
        )
        self.assertEqual(len(d), (len(SURFACE_VOCAB) + 1) + (len(CURVE_VOCAB) + 1))

    def test_count_invariant_length(self):
        d1 = brep_descriptor(
            [{"type": "plane", "area": 1.0}],
            [{"type": "line", "length": 1.0}],
        )
        d2 = brep_descriptor(
            [{"type": "plane", "area": 1.0}] * 5,
            [{"type": "line", "length": 1.0}] * 9,
        )
        self.assertEqual(len(d1), len(d2))

    def test_identical_similarity_is_one(self):
        faces = [{"type": "cylinder", "area": 3.0}]
        edges = [{"type": "circle", "length": 5.0}]
        d = brep_descriptor(faces, edges)
        self.assertAlmostEqual(descriptor_similarity(d, d), 1.0)

    def test_different_types_less_similar(self):
        a = brep_descriptor([{"type": "plane", "area": 1.0}], [{"type": "line", "length": 1.0}])
        b = brep_descriptor([{"type": "sphere", "area": 1.0}], [{"type": "bspline", "length": 1.0}])
        c = brep_descriptor([{"type": "plane", "area": 1.0}], [{"type": "line", "length": 1.0}])
        self.assertLess(descriptor_similarity(a, b), descriptor_similarity(a, c))


if __name__ == "__main__":
    unittest.main()
