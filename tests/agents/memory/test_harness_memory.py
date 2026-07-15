"""Behaviour tests for HarnessMemory — the oracle-gated memory facade.

The load-bearing assertions:

  * a FAILED verdict writes NOTHING to episodic memory (the Agent-S failure);
  * a PASSED verdict writes an exemplar that later recall returns;
  * fleet-said-broken + gate-said-fine is recorded as a verifier FALSE POSITIVE;
  * decay is driven by the LOGICAL clock (ticks), never a wall clock;
  * two identical runs produce byte-identical memory.
"""

import json
import unittest

from harnesscad.agents.memory.harness_memory import (
    FALSE_POSITIVE_PREFIX,
    HarnessMemory,
    OracleVerdict,
    gate_oracle,
)
from harnesscad.agents.memory.skills import build_default_library
from harnesscad.core.cisp.ops import AddRectangle, Constrain, Extrude, NewSketch
from harnesscad.core.loop import HarnessSession
from harnesscad.io.backends.stub import StubBackend


def good_ops():
    return (
        [NewSketch(), AddRectangle(sketch="sk1")]
        + [Constrain(kind="distance", a="e1", value=10.0) for _ in range(4)]
        + [Extrude(sketch="sk1", distance=5.0)]
    )


PASS = OracleVerdict(True, (), "test")
FAIL = OracleVerdict(False, ("not-watertight",), "test")


class TestOracleGateOnWrites(unittest.TestCase):
    def test_failed_verdict_writes_no_episode(self):
        """A memory of a wrong answer is worse than no memory."""
        m = HarnessMemory()
        m.commit("a 20mm plate", good_ops(), FAIL)
        self.assertEqual(m.store.episodic, [])
        self.assertEqual(m.stats["commits_admitted"], 0)
        self.assertEqual(m.stats["commits_refused"], 1)
        # ... and recall of the very same brief offers no exemplar to copy.
        self.assertEqual(m.recall("a 20mm plate").episodes, [])

    def test_failed_verdict_still_writes_a_lesson(self):
        """A lesson is a statement ABOUT a failure, not an exemplar OF success."""
        m = HarnessMemory()
        w = m.commit("a 20mm plate", good_ops(), FAIL,
                     fleet_diagnostics=[{"severity": "error",
                                         "code": "over-constrained",
                                         "message": "dof < 0"}])
        self.assertIn("over-constrained", w["insight"])
        self.assertEqual(m.store.episodic, [])   # still no exemplar
        self.assertIn(w["insight"], m.recall("a 20mm plate").insights)

    def test_passed_verdict_is_admitted_and_recalled(self):
        m = HarnessMemory(min_similarity=0.1)
        m.commit("a 20mm square plate 5mm thick", good_ops(), PASS, digest="d1")
        self.assertEqual(m.stats["commits_admitted"], 1)
        r = m.recall("a 20mm square plate 6mm thick")
        self.assertEqual(len(r.episodes), 1)
        self.assertEqual(r.episodes[0].outcome, "ok")
        self.assertIn("VERIFIED PRIOR SOLUTIONS", r.prompt_block())
        self.assertIn("new_sketch", r.prompt_block())

    def test_dissimilar_brief_recalls_nothing(self):
        """A far-fetched exemplar is a distractor, and distractors are how
        Agent-S's memory went net-negative."""
        m = HarnessMemory(min_similarity=0.5)
        m.commit("a 20mm square plate 5mm thick", good_ops(), PASS, digest="d1")
        self.assertEqual(m.recall("a helical compression spring").episodes, [])


class TestVerifierFalsePositive(unittest.TestCase):
    """The memory that would have paid for itself: the washer."""

    WASHER = "an 80mm disc, 8mm thick, with a 30mm bore"

    def test_fleet_wrong_gate_right_is_recorded(self):
        m = HarnessMemory(min_similarity=0.1)
        w = m.commit(
            self.WASHER, good_ops(), PASS, digest="d1",
            fleet_diagnostics=[
                {"severity": "error", "code": "hole-oversize",
                 "message": "hole diameter exceeds plate thickness"},
                {"severity": "warning", "code": "under-constrained",
                 "message": "1 dof"},   # not an error -> not part of the claim
            ])
        self.assertTrue(w["admitted"])
        fp = w["false_positive"]
        self.assertIsNotNone(fp)
        self.assertEqual(fp["wrong_answer"], ["hole-oversize"])
        self.assertTrue(fp["insight"].startswith(FALSE_POSITIVE_PREFIX))
        self.assertEqual(m.false_positive_counts(), {"hole-oversize": 1})

    def test_repeated_false_positive_accumulates_the_pattern(self):
        """Forty rejections of one washer by one bad rule should be VISIBLE."""
        m = HarnessMemory()
        for i in range(40):
            m.commit(f"{self.WASHER} (attempt {i})", good_ops(), PASS,
                     fleet_diagnostics=[{"severity": "error",
                                         "code": "hole-oversize",
                                         "message": "x"}])
        self.assertEqual(m.false_positive_counts()["hole-oversize"], 40)
        self.assertEqual(len(m.false_positive_records()), 40)

    def test_no_error_diagnostics_means_no_false_positive(self):
        m = HarnessMemory()
        w = m.commit(self.WASHER, good_ops(), PASS, fleet_diagnostics=[])
        self.assertTrue(w["admitted"])
        self.assertIsNone(w["false_positive"])
        self.assertEqual(m.false_positive_counts(), {})

    def test_false_positive_is_recalled_into_the_prompt(self):
        m = HarnessMemory(min_similarity=0.1)
        m.commit(self.WASHER, good_ops(), PASS,
                 fleet_diagnostics=[{"severity": "error", "code": "hole-oversize",
                                     "message": "x"}])
        r = m.recall("an 80mm disc 8mm thick with a 30mm bore, chamfered")
        self.assertTrue(r.false_positives)
        self.assertIn("KNOWN VERIFIER FALSE POSITIVES", r.prompt_block())
        self.assertIn("hole-oversize", r.prompt_block())

    def test_a_failed_part_never_becomes_a_false_positive(self):
        """If the gate REFUSES the part, the fleet was right. No FP record."""
        m = HarnessMemory()
        w = m.commit(self.WASHER, good_ops(), FAIL,
                     fleet_diagnostics=[{"severity": "error",
                                         "code": "hole-oversize", "message": "x"}])
        self.assertIsNone(w["false_positive"])
        self.assertEqual(m.false_positive_counts(), {})


class TestLogicalClockDecay(unittest.TestCase):
    def test_tick_advances_only_on_commit(self):
        m = HarnessMemory()
        self.assertEqual(m.tick, 0)
        m.recall("anything")
        self.assertEqual(m.tick, 0)          # retrieval does not age the store
        m.commit("b", good_ops(), PASS)
        self.assertEqual(m.tick, 1)

    def test_recall_reinforces_and_stale_episodes_fade(self):
        m = HarnessMemory(min_similarity=0.05, tau=1.0, forget_threshold=0.5,
                          keep_min=0, k_episodes=1)
        m.commit("plate alpha", good_ops(), PASS, digest="alpha")
        # Age the store by committing other, dissimilar work.
        for i in range(6):
            m.commit(f"unrelated widget {i}", good_ops(), PASS, digest=f"u{i}")
        # 'plate alpha' was never recalled again; with tau=1 it has faded out.
        briefs = [e.brief for e in m.store.episodic]
        self.assertNotIn("plate alpha", briefs)
        self.assertGreater(m.stats["forgotten"], 0)

    def test_reinforced_episode_survives(self):
        m = HarnessMemory(min_similarity=0.05, tau=1.0, forget_threshold=0.5,
                          keep_min=0, k_episodes=1)
        m.commit("plate alpha", good_ops(), PASS, digest="alpha")
        for i in range(6):
            m.recall("plate alpha")          # keep it warm
            m.commit(f"unrelated widget {i}", good_ops(), PASS, digest=f"u{i}")
        self.assertIn("plate alpha", [e.brief for e in m.store.episodic])

    def test_determinism_two_identical_runs_byte_identical(self):
        def build():
            m = HarnessMemory()
            for b in ("plate a", "plate b", "bracket c"):
                m.recall(b)
                m.commit(b, good_ops(), PASS, digest=b[-1])
            return json.dumps(m.to_dict(), sort_keys=True)
        self.assertEqual(build(), build())


class TestPersistence(unittest.TestCase):
    def test_round_trip(self):
        m = HarnessMemory(min_similarity=0.1)
        m.commit("a 20mm plate", good_ops(), PASS, digest="d",
                 fleet_diagnostics=[{"severity": "error", "code": "bogus",
                                     "message": "x"}])
        back = HarnessMemory.from_dict(json.loads(json.dumps(m.to_dict())))
        self.assertEqual(back.tick, m.tick)
        self.assertEqual(len(back.store.episodic), 1)
        self.assertEqual(back.false_positive_counts(), {"bogus": 1})
        self.assertTrue(back.recall("a 20mm plate"))


class TestGateOracleAndSeeding(unittest.TestCase):
    def test_gate_oracle_passes_a_real_solid(self):
        session = HarnessSession(StubBackend())
        result = session.apply_ops(good_ops())
        self.assertTrue(result.ok)
        verdict = gate_oracle(session, good_ops())
        self.assertEqual(verdict.source, "gate")
        self.assertIsInstance(verdict.ok, bool)

    def test_gate_oracle_refuses_an_empty_model(self):
        session = HarnessSession(StubBackend())
        self.assertFalse(gate_oracle(session, []).ok)

    def test_seed_from_skills_gates_each_skill_on_the_oracle(self):
        """SkillLibrary's own gate is apply_ops(); ours is the measured gate.
        Seeding re-measures every skill rather than trusting the weaker one."""
        m = HarnessMemory(min_similarity=0.1)
        lib = build_default_library(lambda: HarnessSession(StubBackend()))
        admitted = m.seed_from_skills(
            lib, lambda: HarnessSession(StubBackend()),
            oracle=lambda s, ops: PASS)
        self.assertEqual(admitted, ["bracket", "plate"])
        self.assertEqual(len(m.store.episodic), 2)
        self.assertTrue(m.recall("a flat rectangular plate").episodes)

    def test_seed_admits_nothing_when_the_oracle_refuses(self):
        m = HarnessMemory()
        lib = build_default_library(lambda: HarnessSession(StubBackend()))
        admitted = m.seed_from_skills(
            lib, lambda: HarnessSession(StubBackend()),
            oracle=lambda s, ops: FAIL)
        self.assertEqual(admitted, [])
        self.assertEqual(m.store.episodic, [])


class TestHonesty(unittest.TestCase):
    def test_limits_are_stated_in_code(self):
        limits = HarnessMemory.limits()
        self.assertTrue(any("REFERENCE-FREE" in s for s in limits))
        self.assertTrue(any("lexical" in s.lower() for s in limits))


if __name__ == "__main__":
    unittest.main()
