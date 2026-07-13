import unittest

from harnesscad.io.ingest.cadvlm_entity_sequence import (
    ENTITY, TOKEN, ParsedSequence, build_sequence, entity_segments,
    flat_sequence, parse_sequence,
)


class CadVLMEntitySequenceTests(unittest.TestCase):
    def setUp(self):
        # entity token tuples in the codec's (kind, *coords) form.
        self.entities = (
            ("line", 1, 32, 64, 32),
            ("circle", 48, 32, 32, 48, 16, 32, 32, 16),
        )

    def test_entity_segments_prefixed(self):
        segs = entity_segments(self.entities)
        self.assertTrue(all(seg[0] == ENTITY for seg in segs))
        self.assertEqual(segs[0][1:], self.entities[0])

    def test_flat_sequence_concatenates(self):
        flat = flat_sequence(self.entities)
        self.assertEqual(flat, self.entities[0] + self.entities[1])

    def test_build_has_token_delimiter_and_layout(self):
        seq = build_sequence(self.entities)
        self.assertIn(TOKEN, seq)
        split = seq.index(TOKEN)
        self.assertEqual(seq[0], ENTITY)
        self.assertEqual(seq[split + 1:], flat_sequence(self.entities))
        self.assertEqual(seq[:split].count(ENTITY), len(self.entities))

    def test_roundtrip(self):
        parsed = parse_sequence(build_sequence(self.entities))
        self.assertIsInstance(parsed, ParsedSequence)
        self.assertEqual(parsed.entities, self.entities)
        self.assertEqual(parsed.entity_count, 2)
        self.assertEqual(parsed.flat, flat_sequence(self.entities))

    def test_rejects_reserved_tokens_in_entities(self):
        with self.assertRaises(ValueError):
            build_sequence(((ENTITY, 1, 2),))
        with self.assertRaises(ValueError):
            build_sequence(((),))

    def test_parse_rejects_malformed(self):
        with self.assertRaises(ValueError):
            parse_sequence((ENTITY, "line", 1))          # no <TOKEN>
        with self.assertRaises(ValueError):
            parse_sequence(("line", TOKEN, "line"))      # head not <ENTITY>-led
        # tail disagreeing with head
        with self.assertRaises(ValueError):
            parse_sequence((ENTITY, "line", TOKEN, "circle"))


if __name__ == "__main__":
    unittest.main()
