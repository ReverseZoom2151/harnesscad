"""Tests for data.datagen.sequence_packing (mesh-transformer-jax packer)."""

import unittest

from harnesscad.data.datagen.sequence_packing import (
    append_separator,
    arrays_to_sequences,
    chunk_and_finalize,
    enforce_min_unique,
    eot_split,
    pack_sequences,
    split_list,
)


class TestSplitList(unittest.TestCase):
    def test_even(self):
        self.assertEqual(split_list([1, 2, 3, 4], 2), [[1, 2], [3, 4]])

    def test_uneven(self):
        self.assertEqual(split_list([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])

    def test_bad_size(self):
        with self.assertRaises(ValueError):
            split_list([1, 2], 0)


class TestEotSplit(unittest.TestCase):
    def test_splits_and_drops_empty(self):
        # eot = 0; leading/trailing/double eot produce no empty pieces
        docs = [[1, 2, 0, 3, 4, 0, 0, 5]]
        self.assertEqual(list(eot_split(docs, 0)), [[1, 2], [3, 4], [5]])

    def test_no_eot(self):
        self.assertEqual(list(eot_split([[1, 2, 3]], 0)), [[1, 2, 3]])


class TestAppendSeparator(unittest.TestCase):
    def test_appends(self):
        self.assertEqual(list(append_separator([[1, 2], [3]], 9)), [[1, 2, 9], [3, 9]])


class TestArraysToSequences(unittest.TestCase):
    def test_concat_and_chunk(self):
        arrs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        out = list(arrays_to_sequences(arrs, sequence_length=4))
        # 9 tokens -> [1,2,3,4],[5,6,7,8],[9]
        self.assertEqual(out, [[1, 2, 3, 4], [5, 6, 7, 8], [9]])

    def test_bad_length(self):
        with self.assertRaises(ValueError):
            list(arrays_to_sequences([[1]], 0))


class TestEnforceMinUnique(unittest.TestCase):
    def test_filters(self):
        seqs = [[1, 1, 1], [1, 2, 3]]
        self.assertEqual(list(enforce_min_unique(seqs, 3)), [[1, 2, 3]])


class TestChunkAndFinalize(unittest.TestCase):
    def test_trailing_peeled(self):
        arrs = [list(range(10))]
        full, trailing = chunk_and_finalize(arrs, sequence_length=4)
        self.assertEqual(full, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(trailing, [8, 9])

    def test_empty(self):
        self.assertEqual(chunk_and_finalize([], 4), ([], []))


class TestPackSequences(unittest.TestCase):
    def test_basic_deterministic(self):
        docs = [list(range(5)), list(range(5, 12))]
        seqs, dropped = pack_sequences(docs, sequence_length=4)
        # 12 tokens -> 3 full windows of 4, trailing empty? last window peeled.
        # sequences: [0,1,2,3],[4,5,6,7],[8,9,10,11] -> last peeled as trailing
        self.assertEqual(seqs, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(dropped, 4)

    def test_eos_separator_counts(self):
        docs = [[1, 2], [3, 4]]
        seqs, _ = pack_sequences(docs, sequence_length=3, eos_token=0)
        # -> [1,2,0,3,4,0] chunked by 3: [1,2,0],[3,4,0]; last peeled
        self.assertEqual(seqs, [[1, 2, 0]])

    def test_eot_splitting(self):
        docs = [[1, 2, 7, 3, 4]]
        seqs, dropped = pack_sequences(docs, sequence_length=2, eot_token=7)
        # eot split -> [1,2],[3,4]; concat 4 -> [1,2],[3,4]; last peeled
        self.assertEqual(seqs, [[1, 2]])

    def test_repack_epochs_preserve_order(self):
        docs = [list(range(10))]
        seqs, _ = pack_sequences(docs, sequence_length=4, n_repack_epochs=2)
        # epoch1: [0,1,2,3],[4,5,6,7] trailing [8,9]
        # epoch2: prefix [8,9]+windows -> [8,9,0,1],[2,3,4,5] trailing [6,7]
        self.assertEqual(seqs[0], [0, 1, 2, 3])
        self.assertEqual(seqs[2], [8, 9, 0, 1])

    def test_shuffle_reproducible(self):
        docs = [[i] * 3 for i in range(20)]
        a, _ = pack_sequences(docs, sequence_length=4, preserve_data_order=False, seed=1)
        b, _ = pack_sequences(docs, sequence_length=4, preserve_data_order=False, seed=1)
        self.assertEqual(a, b)

    def test_bad_epochs(self):
        with self.assertRaises(ValueError):
            pack_sequences([[1]], 4, n_repack_epochs=0)


if __name__ == "__main__":
    unittest.main()
