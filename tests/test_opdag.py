"""Tests for the content-hashed, checkpointable ops-DAG."""

import hashlib
import unittest

from cisp.ops import NewSketch, AddRectangle, AddCircle, Constrain
from state.opdag import OpDAG

_GENESIS = hashlib.sha256(b"harnesscad-genesis-v0").hexdigest()


def _seq():
    return [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", w=10.0, h=5.0),
        Constrain(kind="distance", a="e1", value=10.0),
    ]


class TestOpDAG(unittest.TestCase):
    def test_empty_dag_head_is_genesis(self):
        dag = OpDAG()
        self.assertEqual(dag.head_hash, _GENESIS)
        self.assertEqual(len(dag), 0)

    def test_identical_sequences_yield_identical_head_hash(self):
        a, b = OpDAG(), OpDAG()
        for op in _seq():
            a.append(op)
        for op in _seq():
            b.append(op)
        self.assertEqual(a.head_hash, b.head_hash)

    def test_different_op_changes_head_hash(self):
        a, b = OpDAG(), OpDAG()
        for op in _seq():
            a.append(op)
        seq = _seq()
        seq[1] = AddCircle(sketch="sk1", r=2.0)  # differ at second op
        for op in seq:
            b.append(op)
        self.assertNotEqual(a.head_hash, b.head_hash)

    def test_checkpoint_then_truncate_restores_length(self):
        dag = OpDAG()
        for op in _seq():
            dag.append(op)
        dag.checkpoint("cp")
        idx = dag.index_of("cp")
        self.assertEqual(idx, 3)
        head_at_cp = dag.head_hash
        # append more, then truncate back to the checkpoint length.
        dag.append(NewSketch(plane="YZ"))
        self.assertEqual(len(dag), 4)
        self.assertNotEqual(dag.head_hash, head_at_cp)
        dag.truncate(idx)
        self.assertEqual(len(dag), 3)
        self.assertEqual(dag.head_hash, head_at_cp)

    def test_rollback_restores_length_and_head(self):
        dag = OpDAG()
        for op in _seq():
            dag.append(op)
        dag.checkpoint("cp")
        head_at_cp = dag.head_hash
        dag.append(NewSketch(plane="YZ"))
        dag.append(AddCircle(sketch="sk2", r=1.0))
        self.assertEqual(len(dag), 5)
        dag.rollback("cp")
        self.assertEqual(len(dag), 3)
        self.assertEqual(dag.head_hash, head_at_cp)

    def test_truncate_drops_stale_checkpoints(self):
        dag = OpDAG()
        for op in _seq():
            dag.append(op)
        dag.checkpoint("early")     # at index 3
        dag.append(NewSketch(plane="YZ"))
        dag.checkpoint("late")      # at index 4
        self.assertEqual(dag.index_of("late"), 4)
        dag.truncate(3)
        # "early" (<= 3) survives; "late" (> 3) is dropped.
        self.assertEqual(dag.index_of("early"), 3)
        with self.assertRaises(KeyError):
            dag.index_of("late")

    def test_replayed_sequence_after_rollback_matches(self):
        # Re-appending the original tail reproduces the original head hash.
        dag = OpDAG()
        for op in _seq():
            dag.append(op)
        full_head = dag.head_hash
        dag.checkpoint("cp")
        dag.append(AddCircle(sketch="sk2", r=1.0))
        dag.rollback("cp")
        self.assertEqual(dag.head_hash, full_head)


if __name__ == "__main__":
    unittest.main()
