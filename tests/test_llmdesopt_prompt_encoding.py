"""Tests for text-prompt design-variable encodings."""
import unittest

from harnesscad.agents.exploration.llmdesopt_prompt_encoding import (
    WordSlot,
    BagOfWordsPrompt,
    TokenisationPrompt,
    clamp_tokens,
    DEFAULT_VOCAB_SIZE,
)


class WordSlotTests(unittest.TestCase):
    def test_decode_nearest_value(self):
        slot = WordSlot({"fast": 1.0, "slow": 0.2, "medium": 0.6})
        self.assertEqual(slot.decode(0.95), "fast")
        self.assertEqual(slot.decode(0.25), "slow")
        self.assertEqual(slot.decode(0.58), "medium")

    def test_decode_tie_breaks_alphabetically(self):
        slot = WordSlot({"beta": 0.5, "alpha": 0.5})
        self.assertEqual(slot.decode(0.5), "alpha")

    def test_value_of(self):
        slot = WordSlot({"wing": 0.9})
        self.assertEqual(slot.value_of("wing"), 0.9)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            WordSlot({})


class BagOfWordsTests(unittest.TestCase):
    def setUp(self):
        self.prompt = BagOfWordsPrompt(
            template="A {adjective} car in the shape of {noun}",
            slots={
                "adjective": WordSlot({"fast": 1.0, "compact": 0.4}),
                "noun": WordSlot({"wing": 1.0, "box": 0.3}),
            },
            slot_order=["adjective", "noun"],
        )

    def test_decode_full_prompt(self):
        txt = self.prompt.decode([0.95, 0.35])
        self.assertEqual(txt, "A fast car in the shape of box")

    def test_encode_inverse(self):
        vals = self.prompt.encode({"adjective": "fast", "noun": "wing"})
        self.assertEqual(vals, [1.0, 1.0])
        # round-trip: decoding the encoded values recovers the same prompt
        self.assertEqual(self.prompt.decode(vals),
                         "A fast car in the shape of wing")

    def test_wrong_design_length_raises(self):
        with self.assertRaises(ValueError):
            self.prompt.decode([0.5])

    def test_slot_order_mismatch_raises(self):
        with self.assertRaises(ValueError):
            BagOfWordsPrompt("A {a}", {"a": WordSlot({"x": 1.0})}, ["b"])


class ClampTokensTests(unittest.TestCase):
    def test_round_and_clamp(self):
        self.assertEqual(clamp_tokens([2.4, 2.6, -5.0]), [2, 3, 0])

    def test_upper_clamp(self):
        self.assertEqual(clamp_tokens([1e9], vocab_size=100), [99])

    def test_full_range_default(self):
        toks = clamp_tokens([DEFAULT_VOCAB_SIZE + 10.0])
        self.assertEqual(toks, [DEFAULT_VOCAB_SIZE - 1])

    def test_bad_vocab_size(self):
        with self.assertRaises(ValueError):
            clamp_tokens([1.0], vocab_size=0)


class TokenisationPromptTests(unittest.TestCase):
    def test_default_codec_reversible(self):
        p = TokenisationPrompt()
        self.assertEqual(p.tokens([10.2, 20.9]), [10, 21])
        self.assertEqual(p.decode([10.2, 20.9]),
                         "A car in the shape of 10 21")

    def test_custom_codec(self):
        p = TokenisationPrompt(codec=lambda ts: "".join(chr(65 + t) for t in ts))
        self.assertEqual(p.decode([0.0, 1.0, 2.0]),
                         "A car in the shape of ABC")

    def test_template_requires_placeholder(self):
        with self.assertRaises(ValueError):
            TokenisationPrompt(template="no placeholder here")


if __name__ == "__main__":
    unittest.main()
