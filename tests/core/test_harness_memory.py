"""AgentHarness + HarnessMemory: retrieval in the prompt, oracle on the writes.

These are the tests that fail on the pre-memory harness. Everything is offline
and deterministic: a scripted planner, a real HarnessSession over StubBackend,
an injected oracle so the gate's geometry opinion is not what is under test.
"""

import unittest

from harnesscad.agents.agent.planner import Planner
from harnesscad.agents.llm.base import CompletionResult
from harnesscad.agents.llm.structured import ParsedOps
from harnesscad.agents.memory.harness_memory import HarnessMemory, OracleVerdict
from harnesscad.core.cisp.ops import AddRectangle, Constrain, Extrude, NewSketch
from harnesscad.core.harness import AgentHarness
from harnesscad.core.loop import HarnessSession
from harnesscad.io.backends.stub import StubBackend


def good_ops():
    return (
        [NewSketch(), AddRectangle(sketch="sk1")]
        + [Constrain(kind="distance", a="e1", value=10.0) for _ in range(4)]
        + [Extrude(sketch="sk1", distance=5.0)]
    )


PASS = OracleVerdict(True, (), "test")
FAIL = OracleVerdict(False, ("empty-solid",), "test")


class ScriptedPlanner:
    """Returns the same ops every call; records every prompt it was asked for."""

    def __init__(self, ops, memory=None):
        self.ops = ops
        self.memory = memory
        self.briefs = []

    def plan_parsed(self, brief, state_summary=None, diagnostics=None):
        self.briefs.append(brief)
        return ParsedOps(list(self.ops))


class EchoLLM:
    """An LLM that returns a fixed op array and stores the messages it was sent."""

    def __init__(self, ops_json):
        self.ops_json = ops_json
        self.seen = []

    def complete(self, messages, tools=None):
        self.seen.append(messages)
        return CompletionResult(text=self.ops_json)


class TestRetrievalIsPartOfPromptComposition(unittest.TestCase):
    def test_planner_with_no_memory_prompt_is_unchanged(self):
        """The OFF arm must be byte-identical to the pre-memory harness."""
        llm = EchoLLM("[]")
        p = Planner(llm)
        msgs = p.build_messages("a 20mm plate")
        self.assertNotIn("MEMORY", msgs[-1].content)
        self.assertTrue(msgs[-1].content.startswith("DESIGN BRIEF:"))

    def test_verified_episode_appears_in_the_prompt(self):
        m = HarnessMemory(min_similarity=0.1)
        m.commit("a 20mm square plate 5mm thick", good_ops(), PASS, digest="d")
        p = Planner(EchoLLM("[]"), memory=m)
        content = p.build_messages("a 20mm square plate 6mm thick")[-1].content
        self.assertIn("MEMORY", content)
        self.assertIn("VERIFIED PRIOR SOLUTIONS", content)
        self.assertIn("new_sketch", content)
        # Memory is placed at the HEAD, before the brief.
        self.assertLess(content.index("MEMORY"), content.index("DESIGN BRIEF"))

    def test_refused_episode_never_appears_in_the_prompt(self):
        m = HarnessMemory(min_similarity=0.1)
        m.commit("a 20mm square plate 5mm thick", good_ops(), FAIL, digest="d")
        p = Planner(EchoLLM("[]"), memory=m)
        content = p.build_messages("a 20mm square plate 6mm thick")[-1].content
        self.assertNotIn("VERIFIED PRIOR SOLUTIONS", content)

    def test_a_broken_memory_degrades_to_no_memory(self):
        class Boom:
            def recall(self, brief):
                raise RuntimeError("store corrupt")

        p = Planner(EchoLLM("[]"), memory=Boom())
        content = p.build_messages("a plate")[-1].content   # must not raise
        self.assertIn("DESIGN BRIEF", content)


class TestHarnessWritesAreOracleGated(unittest.TestCase):
    def _harness(self, memory, oracle):
        session = HarnessSession(StubBackend())
        planner = ScriptedPlanner(good_ops(), memory=memory)
        return AgentHarness(session, planner, memory=memory, oracle=oracle,
                            max_iterations=1, gated=False)

    def test_converged_run_with_passing_oracle_is_remembered(self):
        m = HarnessMemory()
        run = self._harness(m, lambda ops: PASS).run("a 20mm plate")
        self.assertTrue(run.ok)
        self.assertEqual(len(m.store.episodic), 1)
        self.assertEqual(m.store.episodic[0].outcome, "ok")
        self.assertTrue(run.memory_writes[0]["admitted"])

    def test_converged_run_with_a_REFUSING_oracle_is_NOT_remembered(self):
        """The harness said it converged. The oracle measured it and said no.
        The oracle wins. This is the entire thesis of the module."""
        m = HarnessMemory()
        run = self._harness(m, lambda ops: FAIL).run("a 20mm plate")
        self.assertTrue(run.ok)              # the loop still converged...
        self.assertEqual(m.store.episodic, [])   # ...and memory refused it.
        self.assertFalse(run.memory_writes[0]["admitted"])

    def test_harness_adopts_the_planners_memory(self):
        m = HarnessMemory()
        session = HarnessSession(StubBackend())
        h = AgentHarness(session, ScriptedPlanner(good_ops(), memory=m),
                         oracle=lambda ops: PASS, max_iterations=1, gated=False)
        self.assertIs(h.memory, m)

    def test_no_memory_means_no_memory_writes(self):
        session = HarnessSession(StubBackend())
        run = AgentHarness(session, ScriptedPlanner(good_ops()),
                           max_iterations=1, gated=False).run("a plate")
        self.assertEqual(run.memory_writes, [])

    def test_default_oracle_measures_a_fresh_rebuild(self):
        """With no oracle injected the harness rebuilds and calls io/gate.py."""
        m = HarnessMemory()
        session = HarnessSession(StubBackend())
        h = AgentHarness(session, ScriptedPlanner(good_ops()), memory=m,
                         max_iterations=1, gated=False)
        run = h.run("a 20mm plate")
        self.assertEqual(len(run.memory_writes), 1)
        self.assertEqual(run.memory_writes[0]["oracle"]["source"], "gate")

    def test_second_run_of_the_same_brief_recalls_the_first(self):
        """The point of the exercise: the agent stops starting from zero."""
        m = HarnessMemory(min_similarity=0.1)
        p = Planner(EchoLLM("[]"), memory=m)
        m.commit("a 20mm square plate", good_ops(), PASS, digest="d")
        AgentHarness(HarnessSession(StubBackend()), p, memory=m,
                     oracle=lambda ops: PASS, max_iterations=1,
                     gated=False).run("a 20mm square plate, 5mm thick")
        # The LLM saw the prior verified solution in its context.
        self.assertIn("VERIFIED PRIOR SOLUTIONS", p.llm.seen[0][-1].content)


class TestFalsePositiveIsCapturedByTheHarness(unittest.TestCase):
    def test_fleet_error_plus_passing_oracle_records_a_false_positive(self):
        """A rule rejects a part; the gate rebuilds it and measures it correct.
        THAT is the record the fleet audit never had."""
        m = HarnessMemory()

        class NoisyVerifier:
            """A harness-level verifier that always cries wolf."""

            def check(self, backend, opdag):
                from harnesscad.eval.verifiers.verify import (
                    Diagnostic, Severity, VerifyReport)
                return VerifyReport([Diagnostic(
                    Severity.ERROR, "hole-oversize", "hole exceeds thickness")])

        session = HarnessSession(StubBackend())
        h = AgentHarness(session, ScriptedPlanner(good_ops()), memory=m,
                         oracle=lambda ops: PASS, verifiers=[NoisyVerifier()],
                         max_iterations=2, gated=False)
        run = h.run("an 80mm disc, 8mm thick, with a 30mm bore")

        self.assertFalse(run.ok)                       # the fleet blocked it
        self.assertGreaterEqual(m.false_positive_counts()["hole-oversize"], 1)
        self.assertTrue(m.store.episodic)              # the part was real
        self.assertTrue(any(w["false_positive"] for w in run.memory_writes))


if __name__ == "__main__":
    unittest.main()
