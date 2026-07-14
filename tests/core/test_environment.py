"""The Environment protocol, and GeometryBackend as one honest implementation."""

import unittest

from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch
from harnesscad.core.environment import (
    BackendEnvironment, CapabilityError, Capabilities, Environment,
    KERNEL_CAPABILITIES, Observation, StepResult, coerce_ops, require,
    supported_subset,
)
from harnesscad.io.backends.stub import StubBackend


class TestCapabilities(unittest.TestCase):
    def test_kernel_declares_the_four_contracts(self):
        c = KERNEL_CAPABILITIES
        self.assertTrue(c.content_digest)
        self.assertTrue(c.nonmutating_reject)
        self.assertTrue(c.synchronous_read)
        self.assertTrue(c.deterministic_replay)

    def test_empty_supported_ops_means_all(self):
        self.assertTrue(Capabilities().supports("extrude"))

    def test_unsupported_op_carries_its_reason(self):
        caps = Capabilities(supported_ops=("extrude",),
                            unsupported_ops={"fillet": "needs an edge pick"})
        self.assertTrue(caps.supports("extrude"))
        self.assertFalse(caps.supports("fillet"))
        self.assertIn("edge pick", caps.why_not("fillet"))
        self.assertFalse(caps.supports("revolve"))
        self.assertIn("not in this environment", caps.why_not("revolve"))

    def test_subset_split(self):
        caps = Capabilities(supported_ops=("new_sketch",))
        ops = [NewSketch(), Extrude(sketch="sk1")]
        ok, reasons = supported_subset(caps, ops)
        self.assertEqual([type(o).OP for o in ok], ["new_sketch"])
        self.assertEqual(len(reasons), 1)


class TestBackendEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = BackendEnvironment(StubBackend())

    def test_is_an_environment(self):
        self.assertIsInstance(self.env, Environment)

    def test_declares_the_full_kernel_contract(self):
        caps = self.env.capabilities()
        self.assertTrue(caps.content_digest)
        self.assertTrue(caps.nonmutating_reject)
        self.assertTrue(caps.synchronous_read)
        self.assertFalse(caps.resolve_before_act)   # no UI to resolve against
        require(self.env, "content_digest", "synchronous_read")

    def test_step_is_verified_by_the_digest_moving(self):
        obs = self.env.reset()
        self.assertIsInstance(obs, Observation)
        before = obs.digest
        result = self.env.step(NewSketch(plane="XY"))
        self.assertIsInstance(result, StepResult)
        self.assertTrue(result.ok)
        self.assertTrue(result.verified)
        self.assertNotEqual(result.observation.digest, before)

    def test_rejected_op_does_not_mutate(self):
        self.env.reset()
        before = self.env.state_digest()
        result = self.env.step(Extrude(sketch="nope", distance=1.0))
        self.assertFalse(result.ok)
        self.assertFalse(result.verified)
        self.assertEqual(self.env.state_digest(), before)   # non-mutating reject

    def test_deterministic_replay(self):
        ops = [NewSketch(plane="XY"),
               AddRectangle(sketch="sk1", x=0, y=0, w=4, h=3),
               Extrude(sketch="sk1", distance=2)]
        self.env.reset()
        for op in ops:
            self.env.step(op)
        first = self.env.state_digest()
        self.env.reset()
        for op in ops:
            self.env.step(op)
        self.assertEqual(self.env.state_digest(), first)

    def test_six_backends_are_untouched(self):
        """The adapter reads a declaration; it does not require one. A backend with
        no CAPABILITIES attribute still works, which is what keeps the six shipped
        backends byte-for-byte unchanged."""
        self.assertFalse(hasattr(StubBackend, "CAPABILITIES"))
        self.assertEqual(self.env.capabilities().name, "StubBackend")

    def test_capability_error_when_a_flag_is_false(self):
        class Blind(StubBackend):
            CAPABILITIES = Capabilities(name="blind", content_digest=False)

        env = BackendEnvironment(Blind())
        with self.assertRaises(CapabilityError):
            env.state_digest()
        with self.assertRaises(CapabilityError):
            require(env, "content_digest")


class TestCoerceOps(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(coerce_ops(None), [])
        self.assertEqual(len(coerce_ops(NewSketch())), 1)
        self.assertEqual(len(coerce_ops({"op": "new_sketch", "plane": "XY"})), 1)
        self.assertEqual(len(coerce_ops(("new_sketch", {"plane": "XY"}))), 1)
        self.assertEqual(len(coerce_ops([NewSketch(), NewSketch()])), 2)
        with self.assertRaises(TypeError):
            coerce_ops(3.14)


if __name__ == "__main__":
    unittest.main()
