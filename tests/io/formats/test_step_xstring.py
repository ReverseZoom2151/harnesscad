"""Tests for formats.step_p21_xstring (part-21 string X-encoding codec)."""

import unittest

from harnesscad.io.formats.step_xstring import (
    XStringError,
    decode,
    encode,
    round_trip,
)

BS = "\\"  # single backslash, to avoid trailing-backslash raw-string issues


class DecodeTest(unittest.TestCase):
    def test_plain_ascii(self):
        self.assertEqual(decode("hello world"), "hello world")

    def test_literal_backslash(self):
        self.assertEqual(decode("C:" + BS + BS + "temp"), "C:" + BS + "temp")

    def test_x_single_octet(self):
        # \X\E9 -> Latin-1 e-acute
        self.assertEqual(decode("caf" + BS + "X" + BS + "E9"), "café")

    def test_x2_run(self):
        # \X2\00E9\X0\ -> single UCS-2 code unit
        self.assertEqual(
            decode("caf" + BS + "X2" + BS + "00E9" + BS + "X0" + BS), "café")

    def test_x2_multi_code_units(self):
        self.assertEqual(
            decode(BS + "X2" + BS + "03B103B2" + BS + "X0" + BS), "αβ")

    def test_x4_run(self):
        # U+1F600 grinning face
        self.assertEqual(
            decode(BS + "X4" + BS + "0001F600" + BS + "X0" + BS),
            "\U0001f600")

    def test_s_shift(self):
        # \S\A -> code point of 'A' (0x41) + 128
        self.assertEqual(decode(BS + "S" + BS + "A"), chr(0x41 + 128))

    def test_bad_hex_raises(self):
        with self.assertRaises(XStringError):
            decode(BS + "X" + BS + "ZZ")

    def test_dangling_solidus_raises(self):
        with self.assertRaises(XStringError):
            decode("abc" + BS)


class EncodeTest(unittest.TestCase):
    def test_ascii_verbatim(self):
        self.assertEqual(encode("hello"), "hello")

    def test_backslash(self):
        self.assertEqual(encode("a" + BS + "b"), "a" + BS + BS + "b")

    def test_latin1_octet(self):
        self.assertEqual(encode("café"), "caf" + BS + "X" + BS + "E9")

    def test_wide_run_grouped(self):
        # two BMP code points share one \X2\...\X0\ block
        self.assertEqual(encode("αβ"),
                         BS + "X2" + BS + "03B103B2" + BS + "X0" + BS)

    def test_astral_uses_x4(self):
        self.assertEqual(encode("\U0001f600"),
                         BS + "X4" + BS + "0001F600" + BS + "X0" + BS)


class RoundTripTest(unittest.TestCase):
    def test_round_trip_mixed(self):
        for s in ["plain", "café", "αxβ",
                  "emoji \U0001f600 end", "back" + BS + "slash",
                  "aéα\U0001f4a9"]:
            self.assertEqual(round_trip(s), s, msg=s)


if __name__ == "__main__":
    unittest.main()
