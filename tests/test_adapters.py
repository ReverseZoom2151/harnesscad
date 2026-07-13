import unittest

from harnesscad.io.adapters.base import (
    CADAdapter,
    Capability,
    CapabilityError,
    IdempotencyConflict,
    TransactionStateError,
    VerificationRequired,
    WriteCommand,
)
from harnesscad.io.adapters.memory import MemoryCADAdapter


class MemoryCADAdapterTests(unittest.TestCase):
    def test_implements_runtime_protocol_and_discovers_capabilities(self):
        adapter = MemoryCADAdapter()
        self.assertIsInstance(adapter, CADAdapter)
        caps = adapter.capabilities()
        self.assertEqual("memory", caps.host)
        self.assertTrue(caps.supports(Capability.TRANSACTIONS))
        self.assertTrue(caps.supports(Capability.IDEMPOTENCY))

    def test_create_verify_commit(self):
        adapter = MemoryCADAdapter()
        self.assertEqual("tx-import", adapter.begin("tx-import"))
        receipt = adapter.apply(
            WriteCommand("create", "part-1", {"kind": "plate", "valid": True}),
            idempotency_key="create-part-1",
        )
        self.assertFalse(receipt.replayed)
        self.assertTrue(adapter.verify().ok)
        committed = adapter.commit()
        self.assertEqual("tx-import", committed.transaction_id)
        self.assertEqual(("create-part-1",), committed.applied_keys)
        self.assertEqual("plate", adapter.read("part-1")["kind"])

    def test_update_and_delete(self):
        adapter = MemoryCADAdapter({"p": {"mass": 10}, "old": {}})
        adapter.begin()
        adapter.apply(WriteCommand("update", "p", {"mass": 8}), idempotency_key="u")
        adapter.apply(WriteCommand("delete", "old"), idempotency_key="d")
        self.assertEqual(8, adapter.read("p")["mass"])
        self.assertNotIn("old", adapter.read())
        adapter.verify()
        adapter.commit()

    def test_rollback_restores_snapshot(self):
        adapter = MemoryCADAdapter({"p": {"mass": 10}})
        before = adapter.revision()
        txid = adapter.begin()
        adapter.apply(WriteCommand("update", "p", {"mass": 1}), idempotency_key="u")
        self.assertEqual(txid, adapter.rollback())
        self.assertEqual(10, adapter.read("p")["mass"])
        self.assertEqual(before, adapter.revision())

    def test_commit_requires_verification(self):
        adapter = MemoryCADAdapter()
        adapter.begin()
        adapter.apply(WriteCommand("create", "p"), idempotency_key="c")
        with self.assertRaises(VerificationRequired):
            adapter.commit()

    def test_mutation_after_verify_invalidates_verification(self):
        adapter = MemoryCADAdapter()
        adapter.begin()
        adapter.apply(WriteCommand("create", "p"), idempotency_key="c")
        self.assertTrue(adapter.verify().ok)
        adapter.apply(WriteCommand("update", "p", {"x": 1}), idempotency_key="u")
        with self.assertRaises(VerificationRequired):
            adapter.commit()

    def test_failed_verification_cannot_commit(self):
        adapter = MemoryCADAdapter()
        adapter.begin()
        adapter.apply(
            WriteCommand("create", "bad", {"valid": False}), idempotency_key="bad"
        )
        report = adapter.verify()
        self.assertFalse(report.ok)
        self.assertEqual("host-invalid", report.issues[0].code)
        with self.assertRaises(VerificationRequired):
            adapter.commit()

    def test_idempotent_replay_does_not_apply_twice(self):
        adapter = MemoryCADAdapter()
        adapter.begin()
        command = WriteCommand("create", "p", {"x": 1})
        first = adapter.apply(command, idempotency_key="same")
        second = adapter.apply(command, idempotency_key="same")
        self.assertFalse(first.replayed)
        self.assertTrue(second.replayed)
        self.assertEqual(first.staged_revision, second.staged_revision)

    def test_idempotency_survives_commit(self):
        adapter = MemoryCADAdapter()
        command = WriteCommand("create", "p", {"x": 1})
        adapter.begin()
        adapter.apply(command, idempotency_key="same")
        adapter.verify()
        adapter.commit()
        adapter.begin()
        replay = adapter.apply(command, idempotency_key="same")
        self.assertTrue(replay.replayed)
        self.assertEqual({"p": {"x": 1}}, adapter.read())

    def test_conflicting_idempotency_key_is_rejected(self):
        adapter = MemoryCADAdapter()
        adapter.begin()
        adapter.apply(WriteCommand("create", "p"), idempotency_key="same")
        with self.assertRaises(IdempotencyConflict):
            adapter.apply(WriteCommand("create", "q"), idempotency_key="same")

    def test_transaction_state_errors(self):
        adapter = MemoryCADAdapter()
        with self.assertRaises(TransactionStateError):
            adapter.apply(WriteCommand("create", "p"), idempotency_key="c")
        adapter.begin()
        with self.assertRaises(TransactionStateError):
            adapter.begin()

    def test_read_is_defensive_copy(self):
        adapter = MemoryCADAdapter({"p": {"nested": {"x": 1}}})
        value = adapter.read("p")
        value["nested"]["x"] = 99
        self.assertEqual(1, adapter.read("p")["nested"]["x"])

    def test_unsupported_action_and_missing_entity(self):
        adapter = MemoryCADAdapter()
        adapter.begin()
        with self.assertRaises(CapabilityError):
            adapter.apply(WriteCommand("explode", "p"), idempotency_key="x")
        with self.assertRaises(KeyError):
            adapter.apply(WriteCommand("update", "missing"), idempotency_key="u")

    def test_deterministic_revision(self):
        left = MemoryCADAdapter({"a": {"x": 1}, "b": {"x": 2}})
        right = MemoryCADAdapter({"b": {"x": 2}, "a": {"x": 1}})
        self.assertEqual(left.revision(), right.revision())


if __name__ == "__main__":
    unittest.main()
