import base64
import os
import unittest

from harnesscad.io.formats.t2cblender_base64data import (
    Base64DecodeError,
    canonical_padding,
    decode_base64data,
    encode_base64data,
    is_urlsafe_alphabet,
    normalise_padding,
)


class CanonicalPaddingTest(unittest.TestCase):
    def test_multiples_of_four_need_no_padding(self):
        self.assertEqual(canonical_padding(0), 0)
        self.assertEqual(canonical_padding(4), 0)
        self.assertEqual(canonical_padding(8), 0)

    def test_remainders(self):
        self.assertEqual(canonical_padding(2), 2)
        self.assertEqual(canonical_padding(3), 1)
        self.assertEqual(canonical_padding(6), 2)
        self.assertEqual(canonical_padding(7), 1)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            canonical_padding(-1)


class NormalisePaddingTest(unittest.TestCase):
    def test_recomputes_padding(self):
        # 22 chars -> needs 2 pad chars.
        raw = b"any carnal pleasure."
        enc = base64.urlsafe_b64encode(raw).decode()
        stripped = enc.rstrip("=")
        self.assertEqual(normalise_padding(stripped), enc)

    def test_strips_whitespace(self):
        raw = os.urandom(30)
        enc = base64.urlsafe_b64encode(raw).decode()
        folded = enc[:10] + "\n" + enc[10:]
        self.assertEqual(normalise_padding(folded), enc)


class DecodeBase64DataTest(unittest.TestCase):
    def test_roundtrip_all_lengths(self):
        # exercise every length modulo 3 to cover padding cases.
        for n in range(0, 40):
            raw = bytes(range(n))
            enc = base64.urlsafe_b64encode(raw).decode()
            self.assertEqual(decode_base64data(enc), raw)

    def test_matches_kittycad_quirk(self):
        # the addon does urlsafe_b64decode(s.strip("=") + "===").
        raw = os.urandom(23)
        enc = base64.urlsafe_b64encode(raw).decode()
        quirk = base64.urlsafe_b64decode(enc.strip("=") + "===")
        self.assertEqual(decode_base64data(enc), quirk)
        self.assertEqual(decode_base64data(enc.strip("=")), raw)

    def test_extra_and_missing_padding_recovered(self):
        raw = os.urandom(17)
        enc = base64.urlsafe_b64encode(raw).decode()
        self.assertEqual(decode_base64data(enc + "==="), raw)
        self.assertEqual(decode_base64data(enc.rstrip("=")), raw)

    def test_standard_alphabet_autodetected(self):
        raw = bytes([251, 252, 253, 254, 255, 250])
        std = base64.b64encode(raw).decode()
        self.assertTrue("+" in std or "/" in std)
        self.assertFalse(is_urlsafe_alphabet(std))
        self.assertEqual(decode_base64data(std), raw)

    def test_impossible_length_rejected(self):
        with self.assertRaises(Base64DecodeError):
            decode_base64data("A")  # len%4==1 after stripping

    def test_whitespace_folded_payload(self):
        raw = os.urandom(64)
        enc = base64.urlsafe_b64encode(raw).decode()
        folded = "\n".join(enc[i : i + 16] for i in range(0, len(enc), 16))
        self.assertEqual(decode_base64data(folded), raw)


class EncodeBase64DataTest(unittest.TestCase):
    def test_encode_strip_roundtrip(self):
        raw = os.urandom(50)
        transport = encode_base64data(raw, urlsafe=True, strip=True)
        self.assertNotIn("=", transport)
        self.assertEqual(decode_base64data(transport), raw)

    def test_encode_std(self):
        raw = os.urandom(9)
        enc = encode_base64data(raw, urlsafe=False)
        self.assertEqual(base64.b64decode(enc), raw)


if __name__ == "__main__":
    unittest.main()
