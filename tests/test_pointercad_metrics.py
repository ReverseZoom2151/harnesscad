import unittest

from harnesscad.domain.reconstruction.brep import pointercad_indexing as idx
from harnesscad.domain.reconstruction.evaluate import pointercad_metrics as m
from harnesscad.domain.reconstruction.brep.pointercad_indexing import EdgeRecord, FaceRecord
from harnesscad.domain.reconstruction.sequences.pointercad_pointer import CHAMFER, SKETCH, PointerCommand


class PointerAccuracyTest(unittest.TestCase):
    def test_hit_in_valid_set(self):
        self.assertTrue(m.pointer_hit(4, {2, 4, 6}))
        self.assertFalse(m.pointer_hit(3, {2, 4, 6}))

    def test_batch_accuracy_counts_set_membership(self):
        preds = [4, 3, 7]
        valids = [{4, 5}, {1, 2}, {7}]  # hit, miss, hit
        acc = m.pointer_accuracy(preds, valids)
        self.assertEqual(acc.hits, 2)
        self.assertEqual(acc.total, 3)
        self.assertAlmostEqual(acc.accuracy, 2 / 3)

    def test_empty_batch(self):
        acc = m.pointer_accuracy([], [])
        self.assertEqual(acc.accuracy, 0.0)

    def test_length_mismatch(self):
        with self.assertRaises(m.PointerMetricError):
            m.pointer_accuracy([1], [{1}, {2}])

    def test_coplanar_equivalent_choice_counts_as_hit(self):
        # ground truth pointer set = a coplanar group; picking any member is correct
        valid = {3, 4, 5}
        self.assertTrue(m.pointer_hit(5, valid))


class CosineMatchTest(unittest.TestCase):
    def test_match_selects_highest_cosine(self):
        pred = [1.0, 0.0, 0.0]
        cands = [[0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [-1.0, 0.0, 0.0]]
        self.assertEqual(m.match_pointer(pred, cands), 1)

    def test_cosine_of_identical_is_one(self):
        self.assertAlmostEqual(m.cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)

    def test_tie_breaks_to_lowest_index(self):
        pred = [1.0, 0.0]
        cands = [[1.0, 0.0], [2.0, 0.0]]  # both cosine 1.0
        self.assertEqual(m.match_pointer(pred, cands), 0)

    def test_zero_norm_rejected(self):
        with self.assertRaises(m.PointerMetricError):
            m.cosine_similarity([0, 0], [1, 1])

    def test_no_candidates(self):
        with self.assertRaises(m.PointerMetricError):
            m.match_pointer([1.0], [])


class InvalidityRatioTest(unittest.TestCase):
    def test_ir_formula(self):
        self.assertAlmostEqual(m.invalidity_ratio(100, 85), 0.15)

    def test_all_valid_is_zero(self):
        self.assertEqual(m.invalidity_ratio(50, 50), 0.0)

    def test_bad_inputs(self):
        with self.assertRaises(m.PointerMetricError):
            m.invalidity_ratio(0, 0)
        with self.assertRaises(m.PointerMetricError):
            m.invalidity_ratio(10, 20)


class DanglingRatioTest(unittest.TestCase):
    def _index(self):
        return idx.build_index([FaceRecord(key="a"), FaceRecord(key="b")],
                               [EdgeRecord(key="e", face_keys=("a", "b"))])

    def test_all_valid_is_sound(self):
        ix = self._index()
        cmds = [PointerCommand(kind=SKETCH, face_pointers=(0,)),
                PointerCommand(kind=CHAMFER, edge_pointers=(0,), param=0.1)]
        rep = m.dangling_pointer_ratio(cmds, ix)
        self.assertTrue(rep.is_sound)
        self.assertEqual(rep.ratio, 0.0)
        self.assertEqual(rep.total_pointers, 2)

    def test_counts_dangling(self):
        ix = self._index()
        cmds = [PointerCommand(kind=SKETCH, face_pointers=(99,)),   # dangling
                PointerCommand(kind=CHAMFER, edge_pointers=(0, 5), param=0.1)]  # one dangling
        rep = m.dangling_pointer_ratio(cmds, ix)
        self.assertEqual(rep.total_pointers, 3)
        self.assertEqual(rep.dangling_pointers, 2)
        self.assertFalse(rep.is_sound)
        self.assertAlmostEqual(rep.ratio, 2 / 3)


if __name__ == "__main__":
    unittest.main()
