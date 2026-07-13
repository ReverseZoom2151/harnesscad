"""Tests for formats.sgraphs2_flat_array."""

import json
import os
import tempfile
import unittest

from harnesscad.io.formats.flat_array import (
    ARRAY_MAGIC,
    DICT_MAGIC,
    FlatArray,
    load_array,
    load_dict,
    merge_arrays,
    pack_array,
    pack_dict,
    save_array,
    save_dict,
    unpack_array,
    unpack_dict,
)


class TestPackArray(unittest.TestCase):
    def setUp(self):
        self.items = [b"alpha", b"", b"gamma-payload", b"d"]
        self.blob = pack_array(self.items)

    def test_magic(self):
        self.assertTrue(self.blob.startswith(ARRAY_MAGIC))

    def test_round_trip(self):
        self.assertEqual(unpack_array(self.blob), self.items)

    def test_empty_element_preserved(self):
        self.assertEqual(unpack_array(self.blob)[1], b"")

    def test_empty_array(self):
        blob = pack_array([])
        self.assertEqual(unpack_array(blob), [])
        self.assertEqual(len(FlatArray(blob)), 0)

    def test_deterministic(self):
        self.assertEqual(pack_array(self.items), pack_array(list(self.items)))

    def test_accepts_bytearray_and_memoryview(self):
        blob = pack_array([bytearray(b"ab"), memoryview(b"cd")])
        self.assertEqual(unpack_array(blob), [b"ab", b"cd"])

    def test_payload_is_contiguous_and_unencoded(self):
        # The payload region is the raw concatenation -- nothing is escaped.
        self.assertIn(b"alphagamma-payloadd", self.blob)


class TestFlatArray(unittest.TestCase):
    def setUp(self):
        self.items = [b"one", b"two", b"three"]
        self.array = FlatArray(pack_array(self.items))

    def test_len_and_index(self):
        self.assertEqual(len(self.array), 3)
        self.assertEqual(self.array[0], b"one")
        self.assertEqual(self.array[2], b"three")

    def test_negative_index(self):
        self.assertEqual(self.array[-1], b"three")

    def test_slice(self):
        self.assertEqual(self.array[1:], [b"two", b"three"])

    def test_out_of_range(self):
        with self.assertRaises(IndexError):
            _ = self.array[3]
        with self.assertRaises(IndexError):
            _ = self.array[-4]

    def test_iteration(self):
        self.assertEqual(list(self.array), self.items)

    def test_random_access_does_not_decode_neighbours(self):
        # Element 1 is retrievable from a corpus whose other payloads are junk
        # that no decoder could parse -- the container never touches them.
        blob = pack_array([b"\xff\xfe", b"good", b"\x00\x01"])
        self.assertEqual(FlatArray(blob)[1], b"good")


class TestArrayValidation(unittest.TestCase):
    def test_bad_magic(self):
        with self.assertRaises(ValueError):
            FlatArray(b"NOTAFLAT" + b"\x00" * 24)

    def test_truncated_magic(self):
        with self.assertRaises(ValueError):
            FlatArray(b"SGF")

    def test_bad_version(self):
        blob = bytearray(pack_array([b"x"]))
        blob[len(ARRAY_MAGIC)] = 9  # version byte
        with self.assertRaises(ValueError):
            FlatArray(bytes(blob))

    def test_truncated_payload(self):
        blob = pack_array([b"abcdef"])
        with self.assertRaises(ValueError):
            FlatArray(blob[:-3])


class TestMerge(unittest.TestCase):
    def test_merge_shards(self):
        a = pack_array([b"a1", b"a2"])
        b = pack_array([b"b1"])
        c = pack_array([])
        merged = merge_arrays([a, b, c])
        self.assertEqual(unpack_array(merged), [b"a1", b"a2", b"b1"])

    def test_merge_equals_direct_pack(self):
        items_a = [b"x", b"", b"yy"]
        items_b = [b"zzz"]
        merged = merge_arrays([pack_array(items_a), pack_array(items_b)])
        self.assertEqual(merged, pack_array(items_a + items_b))

    def test_merge_of_nothing(self):
        self.assertEqual(unpack_array(merge_arrays([])), [])

    def test_merge_is_associative(self):
        a, b, c = pack_array([b"a"]), pack_array([b"b"]), pack_array([b"c"])
        left = merge_arrays([merge_arrays([a, b]), c])
        right = merge_arrays([a, merge_arrays([b, c])])
        self.assertEqual(left, right)


class TestDict(unittest.TestCase):
    def setUp(self):
        self.mapping = {
            "sketches": pack_array([b"s0", b"s1"]),
            "stats": json.dumps({"count": 2}).encode("utf-8"),
            "empty": b"",
        }
        self.blob = pack_dict(self.mapping)

    def test_magic(self):
        self.assertTrue(self.blob.startswith(DICT_MAGIC))

    def test_round_trip(self):
        self.assertEqual(unpack_dict(self.blob), self.mapping)

    def test_nested_array_is_usable(self):
        inner = unpack_dict(self.blob)["sketches"]
        self.assertEqual(unpack_array(inner), [b"s0", b"s1"])

    def test_deterministic_regardless_of_insertion_order(self):
        reordered = {k: self.mapping[k] for k in reversed(list(self.mapping))}
        self.assertEqual(pack_dict(reordered), self.blob)

    def test_empty_dict(self):
        self.assertEqual(unpack_dict(pack_dict({})), {})

    def test_bad_magic(self):
        with self.assertRaises(ValueError):
            unpack_dict(b"BADMAGIC" + b"\x00" * 16)

    def test_truncated_entry(self):
        with self.assertRaises(ValueError):
            unpack_dict(self.blob[:-4])


class TestFileHelpers(unittest.TestCase):
    def test_array_file_round_trip(self):
        items = [b"p", b"qq", b""]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "corpus.sgf")
            save_array(path, items)
            self.assertEqual(list(load_array(path)), items)

    def test_dict_file_round_trip(self):
        mapping = {"a": b"1", "b": b"22"}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "corpus.sgd")
            save_dict(path, mapping)
            self.assertEqual(load_dict(path), mapping)

    def test_saved_file_is_byte_identical_across_runs(self):
        items = [b"x", b"y"]
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "a.sgf")
            second = os.path.join(tmp, "b.sgf")
            save_array(first, items)
            save_array(second, items)
            with open(first, "rb") as fa, open(second, "rb") as fb:
                self.assertEqual(fa.read(), fb.read())


if __name__ == "__main__":
    unittest.main()
