"""Tests for the AutoBrep tokenisation vocabulary and hierarchical serialisation."""

import unittest

from harnesscad.domain.reconstruction.tokens import autobrep_serialize as ab


class VocabularyLayoutTest(unittest.TestCase):
    def setUp(self):
        self.v = ab.AutoBrepVocabulary()  # shipped config

    def test_default_block_sizes(self):
        self.assertEqual(self.v.flag_pad, 21)
        self.assertEqual(self.v.pos_pad, 1024)
        self.assertEqual(self.v.id_pad, 100)
        self.assertEqual(self.v.face_z_pad, 21 + 1024 + 100)
        self.assertEqual(self.v.edge_offset, self.v.face_z_pad + 10000)
        self.assertEqual(self.v.num_tokens, self.v.face_z_pad + 10000 + 10000)

    def test_ranges_are_contiguous_and_non_overlapping(self):
        r = self.v.ranges()
        ordered = [
            r[ab.TokenKind.SPECIAL],
            r[ab.TokenKind.POSITION],
            r[ab.TokenKind.ID],
            r[ab.TokenKind.SURFACE],
            r[ab.TokenKind.EDGE],
        ]
        self.assertEqual(ordered[0][0], 0)
        for (lo, hi), (nlo, nhi) in zip(ordered, ordered[1:]):
            self.assertEqual(hi, nlo)  # contiguous
        self.assertEqual(ordered[-1][1], self.v.num_tokens)

    def test_encode_classify_roundtrip(self):
        self.assertEqual(self.v.classify(self.v.pos_token(500)), (ab.TokenKind.POSITION, 500))
        self.assertEqual(self.v.classify(self.v.id_token(7)), (ab.TokenKind.ID, 7))
        self.assertEqual(self.v.classify(self.v.surf_token(1234)), (ab.TokenKind.SURFACE, 1234))
        self.assertEqual(self.v.classify(self.v.edge_token(4321)), (ab.TokenKind.EDGE, 4321))
        self.assertEqual(self.v.classify(int(ab.MMTokenIndex.BOF)), (ab.TokenKind.SPECIAL, 8))

    def test_out_of_range_encode_raises(self):
        with self.assertRaises(ValueError):
            self.v.id_token(self.v.max_face)
        with self.assertRaises(ValueError):
            self.v.pos_token(self.v.pos_pad)

    def test_classify_rejects_out_of_vocab(self):
        with self.assertRaises(ValueError):
            self.v.classify(self.v.num_tokens)


class QuantizeTest(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(ab.quantize_coord(-1.0, bit=10), 0)
        self.assertEqual(ab.quantize_coord(1.0, bit=10), 1023)

    def test_clip(self):
        self.assertEqual(ab.quantize_coord(-5.0, bit=10), 0)
        self.assertEqual(ab.quantize_coord(5.0, bit=10), 1023)

    def test_roundtrip_close(self):
        for x in (-0.5, 0.0, 0.25, 0.9):
            q = ab.quantize_coord(x, bit=10)
            self.assertAlmostEqual(ab.dequantize_coord(q, bit=10), x, places=2)


class SerializeParseTest(unittest.TestCase):
    def setUp(self):
        self.v = ab.AutoBrepVocabulary()
        e0 = ab.Edge(edge_id=0, pos=(10, 20), code=5)
        e1 = ab.Edge(edge_id=1, pos=(30, 40, 50), code=6)
        f0 = ab.Face(face_id=0, pos=(100, 200, 300), code=11, edges=(e0, e1))
        f1 = ab.Face(face_id=1, pos=(1, 2, 3), code=22, edges=())
        self.brep = ab.BrepSequence(levels=(ab.Level(faces=(f0, f1)),))

    def test_roundtrip(self):
        toks = ab.serialize(self.brep, self.v)
        back = ab.parse(toks, self.v)
        self.assertEqual(back, self.brep)

    def test_stream_starts_and_ends_with_markers(self):
        toks = ab.serialize(self.brep, self.v)
        self.assertEqual(toks[0], int(ab.MMTokenIndex.BOC))
        self.assertEqual(toks[-1], int(ab.MMTokenIndex.EOC))

    def test_all_tokens_in_vocab_and_validate(self):
        toks = ab.serialize(self.brep, self.v)
        self.assertTrue(ab.validate_tokens(toks, self.v))

    def test_empty_face_no_edges(self):
        brep = ab.BrepSequence(levels=(ab.Level(faces=(ab.Face(0, (5,), 3),)),))
        self.assertEqual(ab.parse(ab.serialize(brep, self.v), self.v), brep)

    def test_multi_level(self):
        f = ab.Face(face_id=2, pos=(7,), code=1)
        brep = ab.BrepSequence(levels=(ab.Level((f,)), ab.Level((f,))))
        self.assertEqual(ab.parse(ab.serialize(brep, self.v), self.v), brep)

    def test_custom_vocab_config(self):
        v = ab.AutoBrepVocabulary(bit=8, max_face=16, surf_codebook_size=64, edge_codebook_size=64)
        self.assertEqual(v.pos_pad, 256)
        self.assertEqual(v.face_z_pad, 21 + 256 + 16)
        brep = ab.BrepSequence(levels=(ab.Level((ab.Face(3, (5, 9), 10),)),))
        self.assertEqual(ab.parse(ab.serialize(brep, v), v), brep)


if __name__ == "__main__":
    unittest.main()
