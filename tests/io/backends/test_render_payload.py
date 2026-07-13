"""Tests for backends.cadhub_concat_payload."""

import gzip
import unittest

from harnesscad.io.backends.render_payload import (
    SENTINEL,
    TYPE_STL,
    PayloadError,
    decode,
    encode,
    encode_metadata,
    gzip_bytes,
    metadata_of,
    roundtrip,
    split,
)

META = {
    "type": TYPE_STL,
    "consoleMessage": "Geometries in cache: 3",
    "customizerParams": [{"name": "w", "type": "number", "initial": 10}],
}


class TestEncodeDecode(unittest.TestCase):
    def test_roundtrip(self):
        payload = roundtrip(b"solid mesh\x00\x01\xff", META)
        self.assertEqual(payload.artifact, b"solid mesh\x00\x01\xff")
        self.assertEqual(payload.metadata["type"], TYPE_STL)
        self.assertEqual(payload.console_message, "Geometries in cache: 3")
        self.assertEqual(payload.customizer_params[0]["name"], "w")
        self.assertTrue(payload.is_mesh())

    def test_uncompressed_layout(self):
        raw = encode(b"AB", {"type": "png"}, compress=False)
        self.assertTrue(raw.startswith(b"AB" + SENTINEL))
        artifact, tail = split(raw)
        self.assertEqual(artifact, b"AB")
        self.assertEqual(tail, b'{"type":"png"}')

    def test_gzip_is_detected_and_reproducible(self):
        blob1 = encode(b"AB", {"type": "png"})
        blob2 = encode(b"AB", {"type": "png"})
        self.assertEqual(blob1, blob2)  # mtime=0 -> content-addressable
        self.assertEqual(blob1[:2], b"\x1f\x8b")
        self.assertEqual(decode(blob1).artifact, b"AB")

    def test_metadata_key_sorted(self):
        self.assertEqual(encode_metadata({"b": 1, "a": 2}), b'{"a":2,"b":1}')

    def test_empty_artifact(self):
        self.assertEqual(roundtrip(b"", {"type": "png"}).artifact, b"")


class TestBinarySafety(unittest.TestCase):
    def test_sentinel_inside_binary_artifact(self):
        artifact = b"\x00\x01" + SENTINEL + b"\x02\x03"  # a chance collision
        payload = roundtrip(artifact, {"type": TYPE_STL})
        self.assertEqual(payload.artifact, artifact)  # last-occurrence split wins
        self.assertEqual(payload.metadata["type"], TYPE_STL)

    def test_sentinel_in_metadata_rejected(self):
        with self.assertRaises(PayloadError):
            encode(b"A", {"consoleMessage": SENTINEL.decode()})


class TestErrors(unittest.TestCase):
    def test_missing_sentinel_strict(self):
        with self.assertRaises(PayloadError):
            decode(gzip_bytes(b"just an stl"))

    def test_missing_sentinel_lenient(self):
        payload = decode(gzip_bytes(b"just an stl"), strict=False)
        self.assertEqual(payload.artifact, b"just an stl")
        self.assertEqual(payload.metadata, {})
        self.assertIsNone(payload.artifact_type)

    def test_bad_json_lenient(self):
        blob = gzip_bytes(b"AB" + SENTINEL + b"{not json")
        self.assertEqual(metadata_of(blob), {})
        with self.assertRaises(PayloadError):
            decode(blob)

    def test_non_object_json(self):
        blob = gzip_bytes(b"AB" + SENTINEL + b"[1,2]")
        with self.assertRaises(PayloadError):
            decode(blob)

    def test_non_bytes_artifact(self):
        with self.assertRaises(PayloadError):
            encode("text", {})

    def test_plain_bytes_accepted_by_decode(self):
        raw = encode(b"AB", {"type": "png"}, compress=False)
        self.assertEqual(decode(raw).artifact_type, "png")

    def test_gzip_helper_is_valid_gzip(self):
        self.assertEqual(gzip.decompress(gzip_bytes(b"xyz")), b"xyz")


if __name__ == "__main__":
    unittest.main()
