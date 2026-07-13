import unittest

from harnesscad.agents.agent.edit_session import EditSession
from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.cisp.ops import AddRectangle, Extrude, Fillet, NewSketch
from harnesscad.core.loop import HarnessSession


class EditSessionTests(unittest.TestCase):
    def setUp(self):
        self.harness = HarnessSession(StubBackend())
        self.edit = EditSession(self.harness)

    def test_preview_requires_approval_and_does_not_mutate(self):
        before = self.edit.current_digest
        proposal = self.edit.propose(
            "make a plate",
            [NewSketch(), AddRectangle("sk1", 0, 0, 20, 10), Extrude("sk1", 3)],
        )
        self.assertEqual(before, self.edit.current_digest)
        self.assertEqual(0, len(self.harness.opdag))
        self.assertTrue(proposal.preview()["requires_approval"])
        self.assertIn("+1 new_sketch", proposal.preview()["summary"])

        result = self.edit.approve(proposal.id)
        self.assertTrue(result.ok)
        self.assertEqual("applied", proposal.status)
        self.assertEqual(3, len(self.harness.opdag))

    def test_rejection_preserves_design_and_history(self):
        proposal = self.edit.propose("do not apply this", [NewSketch()])
        self.edit.reject(proposal.id, "wrong plane")
        self.assertEqual(0, len(self.harness.opdag))
        self.assertEqual("rejected", proposal.status)
        self.assertEqual("wrong plane", self.edit.turns[-1].content)
        with self.assertRaises(ValueError):
            self.edit.approve(proposal.id)

    def test_partial_batch_failure_rolls_back_atomically(self):
        seed = self.edit.propose(
            "base",
            [NewSketch(), AddRectangle("sk1", 0, 0, 20, 10), Extrude("sk1", 3)],
        )
        self.edit.approve(seed.id)
        before_digest = self.edit.current_digest
        before_ops = self.edit.current_ops

        # First fillet succeeds; the second is rejected. Both must disappear.
        proposal = self.edit.propose(
            "two fillets",
            [Fillet(("edge1",), 1), Fillet(("edge1",), -1)],
        )
        result = self.edit.approve(proposal.id)
        self.assertFalse(result.ok)
        self.assertEqual("failed", proposal.status)
        self.assertEqual(before_digest, self.edit.current_digest)
        self.assertEqual(before_ops, self.edit.current_ops)

    def test_stale_proposal_cannot_overwrite_newer_state(self):
        proposal = self.edit.propose("candidate", [NewSketch()])
        self.harness.apply_ops([NewSketch("XZ")])
        with self.assertRaises(RuntimeError):
            self.edit.approve(proposal.id)
        self.assertEqual("stale", proposal.status)
        self.assertEqual(1, len(self.harness.opdag))

    def test_history_is_serializable_shape(self):
        proposal = self.edit.propose("add sketch", [NewSketch()])
        history = self.edit.history()
        self.assertEqual(proposal.id, history["proposals"][0]["id"])
        self.assertEqual(["user", "assistant"], [t["role"] for t in history["turns"]])


if __name__ == "__main__":
    unittest.main()
