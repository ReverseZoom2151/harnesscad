"""End-to-end behaviour tests for the HarnessSession spine."""

import unittest

from cisp.ops import NewSketch, AddRectangle, Constrain, Extrude
from backends.stub import StubBackend
from loop import HarnessSession


def _rect_setup():
    """Ops that leave a fresh session with one sketch 'sk1' + rectangle 'e1'."""
    return [NewSketch(), AddRectangle(sketch="sk1")]


def _four_constraints():
    return [Constrain(kind="distance", a="e1", value=10.0) for _ in range(4)]


class TestValidBatch(unittest.TestCase):
    def test_valid_batch_produces_solid(self):
        session = HarnessSession(StubBackend())
        ops = (
            [NewSketch(), AddRectangle(sketch="sk1")]
            + _four_constraints()
            + [Extrude(sketch="sk1", distance=5.0)]
        )
        res = session.apply_ops(ops)
        self.assertTrue(res.ok)
        self.assertEqual(res.applied, len(ops))  # 1 + 1 + 4 + 1 == 7
        self.assertIsNone(res.rejected)
        summary = session.summary()
        self.assertTrue(summary["solid_present"])
        self.assertEqual(summary["feature_count"], 1)


class TestDeterministicReplay(unittest.TestCase):
    def test_same_ops_same_digest(self):
        ops = (
            [NewSketch(), AddRectangle(sketch="sk1")]
            + _four_constraints()
            + [Extrude(sketch="sk1", distance=5.0)]
        )
        s1 = HarnessSession(StubBackend())
        s2 = HarnessSession(StubBackend())
        r1 = s1.apply_ops(list(ops))
        r2 = s2.apply_ops(list(ops))
        self.assertTrue(r1.ok and r2.ok)
        self.assertEqual(s1.digest(), s2.digest())
        self.assertEqual(r1.digest, r2.digest)


class TestBlockAndCorrect(unittest.TestCase):
    def test_bad_reference_does_not_mutate(self):
        session = HarnessSession(StubBackend())
        empty_digest = session.digest()
        res = session.apply_ops([Extrude(sketch="nope", distance=5.0)])
        self.assertFalse(res.ok)
        self.assertIsNotNone(res.rejected)
        self.assertEqual(res.rejected["op"], "extrude")
        self.assertEqual(res.applied, 0)
        # State unchanged: block-and-correct never mutated the model.
        self.assertEqual(session.digest(), empty_digest)


class TestTransactionalVerifyRollback(unittest.TestCase):
    def test_over_constraint_rolls_back_to_last_good(self):
        # Reference session: rectangle + exactly 4 good constraints (dof 0).
        good = HarnessSession(StubBackend())
        good_res = good.apply_ops(_rect_setup() + _four_constraints())
        self.assertTrue(good_res.ok)
        good_digest = good.digest()

        # Test session: same setup but a 5th distance constraint over-constrains.
        session = HarnessSession(StubBackend())
        ops = _rect_setup() + _four_constraints() + [
            Constrain(kind="distance", a="e1", value=10.0)
        ]
        res = session.apply_ops(ops)
        self.assertFalse(res.ok)
        self.assertIsNotNone(res.rejected)
        self.assertEqual(res.rejected["op"], "constrain")
        # Only the 6 good ops (2 setup + 4 constraints) applied; 5th rolled back.
        self.assertEqual(res.applied, 6)
        # Last-good state preserved: digest equals the 4-good-constraint state.
        self.assertEqual(session.digest(), good_digest)


class TestCheckpointRollback(unittest.TestCase):
    def test_checkpoint_and_rollback_restore_digest(self):
        session = HarnessSession(StubBackend())
        session.apply_ops(_rect_setup() + _four_constraints())
        session.checkpoint("cp")
        cp_digest = session.digest()

        # Apply more ops; digest must change.
        more = session.apply_ops([Extrude(sketch="sk1", distance=5.0)])
        self.assertTrue(more.ok)
        self.assertNotEqual(session.digest(), cp_digest)

        # Roll back to the checkpoint; digest must be restored.
        session.rollback("cp")
        self.assertEqual(session.digest(), cp_digest)


if __name__ == "__main__":
    unittest.main()
