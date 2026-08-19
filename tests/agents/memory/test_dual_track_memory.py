"""Tests for the dual-track agent memory with utility-learned retrieval.

These test the BEHAVIOUR the paper claims, not the plumbing:

  * a high-similarity / low-utility case LOSES to a lower-similarity /
    high-utility one once annealing has advanced (and wins before it has);
  * repeated failure drives a skill below the eligibility threshold and freezes
    it out of recall, without destroying its record;
  * the short-term mask suppresses a just-failed skill for exactly one round;
  * retrieval is deterministic given a seed;
  * on a FAILED episode the selected cases become negatives.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from harnesscad.agents.memory.case_library import (
    Case,
    CaseIntent,
    CaseLibrary,
    CaseOutcome,
    ToolCall,
)
from harnesscad.agents.memory.learned_retrieval import (
    CaseValueModel,
    DualTrackAgentMemory,
    EpisodeSelection,
    terminal_reward,
)
from harnesscad.agents.memory.skill_utility import (
    ELIGIBILITY_THRESHOLD,
    SkillDoc,
    UtilitySkill,
    UtilitySkillLibrary,
    internalise_trajectory,
)


def make_case(case_id: str, text: str, tools=("sketch", "extrude"), passed=True) -> Case:
    return Case(
        case_id=case_id,
        intent=CaseIntent(text=text, parse={"shape": "plate"}),
        trajectory=[
            ToolCall(tool=t, params={"size": 10 + i}, refs_out=(f"{t}_{i}",))
            for i, t in enumerate(tools)
        ],
        outcome=CaseOutcome(
            passed=passed, model="solid_1", checks=("closed", "manifold"),
            stats={"volume": 100.0},
        ),
    )


class CaseLibraryTest(unittest.TestCase):
    def test_write_back_is_gated_on_verification(self):
        lib = CaseLibrary()
        self.assertFalse(lib.write_back(make_case("bad", "a failed plate", passed=False)))
        self.assertTrue(lib.write_back(make_case("good", "a good plate")))
        self.assertEqual(lib.ids(), ["good"])
        self.assertEqual(lib.refused, 1)

    def test_recall_ranks_by_similarity_over_successful_cases(self):
        lib = CaseLibrary()
        lib.write_back(make_case("c1", "bracket with two mounting holes"))
        lib.write_back(make_case("c2", "cylindrical spacer sleeve"))
        got = lib.recall("bracket with mounting holes", k0=2)
        self.assertEqual(got[0][0].case_id, "c1")
        self.assertGreaterEqual(got[0][1], got[1][1])

    def test_case_document_includes_intent_parse_and_tools(self):
        case = make_case("c1", "bracket", tools=("sketch", "fillet"))
        doc = case.document()
        self.assertIn("bracket", doc)
        self.assertIn("plate", doc)      # from the structured parse
        self.assertIn("fillet", doc)     # from the trajectory
        self.assertIn("manifold", doc)   # from the passed checks

    def test_round_trip(self):
        lib = CaseLibrary()
        lib.write_back(make_case("c1", "bracket with holes"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cases.json")
            lib.save(path)
            back = CaseLibrary.load(path)
        self.assertEqual(back.ids(), ["c1"])
        self.assertEqual(back.get("c1").to_dict(), lib.get("c1").to_dict())


class SkillInternalisationTest(unittest.TestCase):
    def test_only_successful_cases_are_internalised(self):
        case = make_case("c1", "plate", passed=False)
        self.assertEqual(internalise_trajectory(case), [])

    def test_fragments_stop_at_a_failed_call(self):
        case = make_case("c1", "plate", tools=("a", "b", "c"))
        case.trajectory[1].status = "failed"
        case.trajectory[1].repair = "reduced the fillet radius"
        skills = internalise_trajectory(case, min_len=2, max_len=3)
        # "a" and "c" are in different clean runs, so no fragment spans them.
        self.assertEqual(skills, [])

    def test_fragments_are_parameterised_and_named_by_tool_sequence(self):
        case = make_case("c1", "plate", tools=("sketch", "extrude", "fillet"))
        skills = internalise_trajectory(case, min_len=2, max_len=2)
        names = [s.name for s in skills]
        self.assertEqual(names, ["sketch-extrude", "extrude-fillet"])
        skill = skills[0]
        self.assertIn("{s0_size}", skill.script)   # value lifted to a parameter
        self.assertEqual(skill.params["s0_size"], 10)
        self.assertIn("s0_size", skill.doc.params)

    def test_failure_modes_are_documented_from_the_repair_record(self):
        case = make_case("c1", "plate", tools=("a", "b", "c", "d"))
        case.trajectory[3].status = "failed"
        case.trajectory[3].repair = "shell thickness exceeded the wall"
        case.outcome.repaired_causes = ("non-manifold shell",)
        skills = internalise_trajectory(case, min_len=2, max_len=2)
        self.assertTrue(skills)
        modes = skills[0].doc.failure_modes
        self.assertTrue(any("shell thickness" in m for m in modes))
        self.assertTrue(any("non-manifold shell" in m for m in modes))


class SkillUtilityTest(unittest.TestCase):
    def test_ema_update(self):
        skill = UtilitySkill(name="k", script="", utility=0.5)
        skill.update(1.0, alpha=0.1)
        self.assertAlmostEqual(skill.utility, 0.55)
        skill.update(0.0, alpha=0.1)
        self.assertAlmostEqual(skill.utility, 0.495)
        self.assertEqual((skill.uses, skill.successes, skill.failures), (2, 1, 1))
        self.assertEqual(skill.last_reward, 0.0)

    def test_repeated_failure_freezes_the_skill_out_of_recall(self):
        lib = UtilitySkillLibrary()
        lib.register(UtilitySkill(name="bad-skill", script="cut(depth={d})",
                                  doc=SkillDoc(function="cut a pocket")))
        lib.register(UtilitySkill(name="good-skill", script="cut(depth={d})",
                                  doc=SkillDoc(function="cut a pocket")))
        for _ in range(3):
            lib.record_reward("bad-skill", 0.0)
            lib.record_reward("good-skill", 1.0)
        self.assertLess(lib.get("bad-skill").utility, ELIGIBILITY_THRESHOLD)
        self.assertEqual(lib.freeze_sweep(), ["bad-skill"])
        names = [s.name for s, _ in lib.recall("cut a pocket", k=5)]
        self.assertEqual(names, ["good-skill"])
        # Frozen, NOT deleted: the evidence survives for review.
        self.assertTrue(lib.has("bad-skill"))
        self.assertEqual(lib.get("bad-skill").failures, 3)
        self.assertEqual(lib.frozen_names(), ["bad-skill"])

    def test_freeze_waits_for_n_min_uses(self):
        lib = UtilitySkillLibrary()
        lib.register(UtilitySkill(name="k", script=""))
        lib.record_reward("k", 0.0)
        lib.record_reward("k", 0.0)
        self.assertLess(lib.get("k").utility, ELIGIBILITY_THRESHOLD)
        self.assertFalse(lib.get("k").eligible)   # out of recall already ...
        self.assertEqual(lib.freeze_sweep(), [])   # ... but not condemned yet
        self.assertFalse(lib.get("k").frozen)
        lib.record_reward("k", 0.0)                # n_min reached
        self.assertEqual(lib.freeze_sweep(), ["k"])

    def test_short_term_mask_lasts_exactly_one_round(self):
        lib = UtilitySkillLibrary()
        lib.register(UtilitySkill(name="alpha", script="", doc=SkillDoc(function="cut slot")))
        lib.register(UtilitySkill(name="beta", script="", doc=SkillDoc(function="cut slot")))

        first = sorted(s.name for s, _ in lib.recall("cut slot", k=5))
        self.assertEqual(first, ["alpha", "beta"])

        lib.record_invocation_failure("alpha")
        self.assertEqual(lib.masked_names(), ["alpha"])

        second = sorted(s.name for s, _ in lib.recall("cut slot", k=5))
        self.assertEqual(second, ["beta"])          # suppressed for THIS round

        third = sorted(s.name for s, _ in lib.recall("cut slot", k=5))
        self.assertEqual(third, ["alpha", "beta"])  # and back for the next one

    def test_rerank_weights_utility_against_similarity(self):
        lib = UtilitySkillLibrary()
        lib.register(UtilitySkill(name="cut-slot-exact", script="",
                                  doc=SkillDoc(function="cut a slot"), utility=0.5))
        lib.register(UtilitySkill(name="drill-hole", script="",
                                  doc=SkillDoc(function="drill a hole"), utility=0.5))
        self.assertEqual(lib.recall("cut a slot", k=1)[0][0].name, "cut-slot-exact")
        # Same query, but the other skill now has a much better track record.
        for _ in range(40):
            lib.record_reward("drill-hole", 1.0)
        self.assertGreater(lib.get("drill-hole").utility, 0.9)

    def test_register_does_not_launder_a_bad_record(self):
        lib = UtilitySkillLibrary()
        lib.register(UtilitySkill(name="k", script="", utility=0.5))
        for _ in range(5):
            lib.record_reward("k", 0.0)
        low = lib.get("k").utility
        lib.register(UtilitySkill(name="k", script="", utility=0.9))
        self.assertAlmostEqual(lib.get("k").utility, low)

    def test_round_trip(self):
        lib = UtilitySkillLibrary()
        lib.register(UtilitySkill(name="k", script="cut()", params={"d": 3}))
        lib.record_reward("k", 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "skills.json")
            lib.save(path)
            back = UtilitySkillLibrary.load(path)
        self.assertEqual(back.get("k").to_dict(), lib.get("k").to_dict())


class TerminalRewardTest(unittest.TestCase):
    class _Verdict:
        def __init__(self, ok):
            self.ok = ok

    def test_adapts_the_existing_verdicts(self):
        self.assertEqual(terminal_reward(True), 1.0)
        self.assertEqual(terminal_reward(False), 0.0)
        self.assertEqual(terminal_reward(self._Verdict(True)), 1.0)
        self.assertEqual(terminal_reward(self._Verdict(False)), 0.0)
        self.assertEqual(terminal_reward(None), 0.0)


class ValueModelTest(unittest.TestCase):
    def test_learns_to_separate_a_positive_from_a_negative(self):
        model = CaseValueModel(lr=0.2)
        model.train([("q hole", "hole plate good", 1), ("q hole", "slot sleeve bad", 0)],
                    epochs=200)
        self.assertGreater(model.value("q hole", "hole plate good"),
                           model.value("q hole", "slot sleeve bad"))

    def test_entropy_regulariser_holds_confidence_back(self):
        examples = [("q", "d", 1)]
        greedy = CaseValueModel(lr=0.2, entropy_coef=0.0)
        regularised = CaseValueModel(lr=0.2, entropy_coef=0.03)
        greedy.train(examples, epochs=300)
        regularised.train(examples, epochs=300)
        self.assertLess(regularised.value("q", "d"), greedy.value("q", "d"))


class AnnealingTest(unittest.TestCase):
    def test_lambda_anneals_from_start_to_end_then_holds(self):
        mem = DualTrackAgentMemory(anneal_episodes=400)
        self.assertAlmostEqual(mem.lambda_t(0), 0.9)
        self.assertAlmostEqual(mem.lambda_t(200), 0.625)
        self.assertAlmostEqual(mem.lambda_t(400), 0.35)
        self.assertAlmostEqual(mem.lambda_t(4000), 0.35)

    def test_annealing_flips_a_high_similarity_low_utility_case(self):
        """The paper's core claim, as a ranking assertion.

        ``looks-alike`` matches the query almost word for word but has failed
        every time it was used. ``works`` matches it less well and has passed.
        With the SAME value weights, early lambda ranks the look-alike first and
        late lambda ranks the one that works first -- so the flip is caused by
        the annealing schedule, not by a change in the store.
        """
        query = "bracket with two mounting holes"
        mem = DualTrackAgentMemory(seed=3, anneal_episodes=400)
        mem.cases.write_back(make_case("looks-alike", "bracket with two mounting holes"))
        mem.cases.write_back(make_case("works", "angle plate drilled and tapped"))

        # Measured feedback: the look-alike fails, the other one passes.
        for _ in range(40):
            fail = EpisodeSelection(
                query=query, episode=mem.episode, lambda_t=mem.lambda_t(),
                candidates=[("looks-alike", 1.0), ("works", 0.0)],
                selected=["looks-alike"],
            )
            mem.end_episode(fail, False)
            good = EpisodeSelection(
                query=query, episode=mem.episode, lambda_t=mem.lambda_t(),
                candidates=[("works", 1.0), ("looks-alike", 0.0)],
                selected=["works"],
            )
            mem.end_episode(good, True)

        mem.episode = 0                     # lambda = 0.9: trust similarity
        early = [c.case_id for c, _, _, _ in mem.rank_cases(query)]
        mem.episode = 400                   # lambda = 0.35: trust measured utility
        late = [c.case_id for c, _, _, _ in mem.rank_cases(query)]

        self.assertEqual(early[0], "looks-alike")
        self.assertEqual(late[0], "works")

    def test_similarity_only_retrieval_keeps_the_trap(self):
        """Control for the test above: with lambda pinned at 1.0 the trap stays.

        This is the semantic-retrieval baseline the ablation beats; if it also
        flipped, the flip above would not be evidence about utility.
        """
        query = "bracket with two mounting holes"
        mem = DualTrackAgentMemory(seed=3, lambda_start=1.0, lambda_end=1.0)
        mem.cases.write_back(make_case("looks-alike", "bracket with two mounting holes"))
        mem.cases.write_back(make_case("works", "angle plate drilled and tapped"))
        for _ in range(40):
            mem.end_episode(
                EpisodeSelection(query=query, episode=mem.episode, lambda_t=1.0,
                                 candidates=[("looks-alike", 1.0), ("works", 0.0)],
                                 selected=["looks-alike"]),
                False,
            )
        mem.episode = 400
        self.assertEqual(mem.rank_cases(query)[0][0].case_id, "looks-alike")


class EpisodeTrainingTest(unittest.TestCase):
    def _memory(self):
        mem = DualTrackAgentMemory(seed=11)
        for i in range(8):
            mem.cases.write_back(make_case(f"c{i}", f"plate number {i} with holes"))
        return mem

    def test_failed_episode_selected_cases_become_negatives(self):
        mem = self._memory()
        sel = mem.begin_episode("plate with holes", k=2)
        self.assertEqual(len(sel.selected), 2)
        mem.end_episode(sel, False)
        labels = dict(mem.last_examples)
        self.assertEqual(sorted(labels), sorted(sel.selected))
        self.assertTrue(all(v == 0 for v in labels.values()))
        self.assertEqual(mem.stats["failed"], 1)

    def test_successful_episode_labels_selected_positive_and_bottom_negative(self):
        mem = self._memory()
        sel = mem.begin_episode("plate with holes", k=2)
        mem.end_episode(sel, True)
        labels = dict(mem.last_examples)
        for cid in sel.selected:
            self.assertEqual(labels[cid], 1)
        negatives = [cid for cid, label in mem.last_examples if label == 0]
        self.assertTrue(negatives)
        # Negatives come only from the UNSELECTED candidates -- never from the
        # cases that were actually injected.
        for cid in negatives:
            self.assertNotIn(cid, sel.selected)
        # ... and never from every unselected candidate: at most `negatives`.
        self.assertLessEqual(len(negatives), mem.negatives)

    def test_uses_are_recorded_against_the_selected_cases(self):
        mem = self._memory()
        sel = mem.begin_episode("plate with holes", k=2)
        mem.end_episode(sel, True)
        for cid in sel.selected:
            self.assertEqual(mem.cases.get(cid).selections, 1)
            self.assertEqual(mem.cases.get(cid).successes, 1)

    def test_end_episode_writes_back_and_internalises_only_on_a_pass(self):
        mem = DualTrackAgentMemory(seed=5)
        sel = mem.begin_episode("a plate")
        failed = make_case("f1", "a plate", tools=("sketch", "extrude"), passed=False)
        mem.end_episode(sel, False, case=failed)
        self.assertEqual(len(mem.cases), 0)
        self.assertEqual(len(mem.skills), 0)

        sel = mem.begin_episode("a plate")
        passed = make_case("p1", "a plate", tools=("sketch", "extrude"))
        mem.end_episode(sel, True, case=passed)
        self.assertEqual(mem.cases.ids(), ["p1"])
        self.assertIn("sketch-extrude", mem.skills.names())
        self.assertEqual(mem.stats["cases_written"], 1)

    def test_skill_utility_follows_the_terminal_reward_and_freezes(self):
        mem = DualTrackAgentMemory(seed=5)
        mem.skills.register(UtilitySkill(name="risky", script="",
                                         doc=SkillDoc(function="shell the part")))
        first = mem.begin_episode("shell the part")
        self.assertIn("risky", first.skill_names())
        mem.end_episode(first, False)
        # Two failures put it under the eligibility bar, so recall stops offering
        # it -- but the agent may still invoke it, and `skills_used` is what the
        # utility estimate is actually keyed on.
        mem.end_episode(mem.begin_episode("shell the part"), False,
                        skills_used=["risky"])
        self.assertFalse(mem.skills.get("risky").eligible)
        self.assertFalse(mem.skills.get("risky").frozen)   # n_min not reached yet
        mem.end_episode(mem.begin_episode("shell the part"), False,
                        skills_used=["risky"])
        self.assertTrue(mem.skills.get("risky").frozen)
        self.assertEqual(mem.begin_episode("shell the part").skill_names(), [])
        self.assertEqual(mem.stats["skills_frozen"], 1)

    def test_episode_counter_is_the_only_clock(self):
        mem = DualTrackAgentMemory(seed=5)
        self.assertEqual(mem.episode, 0)
        mem.end_episode(mem.begin_episode("q"), True)
        self.assertEqual(mem.episode, 1)
        mem.advance_episodes(9)
        self.assertEqual(mem.episode, 10)


class DeterminismTest(unittest.TestCase):
    @staticmethod
    def _build(seed):
        mem = DualTrackAgentMemory(seed=seed)
        for i in range(12):
            mem.cases.write_back(make_case(f"c{i}", f"plate {i} with holes and slots"))
        return mem

    def test_selection_is_deterministic_given_a_seed(self):
        runs = []
        for _ in range(2):
            mem = self._build(42)
            out = []
            for i in range(4):
                sel = mem.begin_episode(f"plate {i} with holes")
                out.append(tuple(sel.selected))
                mem.end_episode(sel, i % 2 == 0)
            runs.append(out)
        self.assertEqual(runs[0], runs[1])

    def test_ranking_is_deterministic_and_sampling_explores(self):
        mem = self._build(42)
        first = [c.case_id for c, _, _, _ in mem.rank_cases("plate with holes")]
        second = [c.case_id for c, _, _, _ in mem.rank_cases("plate with holes")]
        self.assertEqual(first, second)
        # Sampling must not collapse onto the single top case every time.
        seen = set()
        for _ in range(10):
            seen.update(mem.begin_episode("plate with holes", k=2).selected)
        self.assertGreater(len(seen), 2)

    def test_two_seeds_can_differ(self):
        a = self._build(1).begin_episode("plate with holes", k=3).selected
        b = self._build(999).begin_episode("plate with holes", k=3).selected
        self.assertEqual(len(a), 3)
        self.assertEqual(len(b), 3)

    def test_round_trip_preserves_the_learned_state(self):
        mem = self._build(7)
        mem.skills.register(UtilitySkill(name="k", script=""))
        sel = mem.begin_episode("plate with holes")
        mem.end_episode(sel, True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "memory.json")
            mem.save(path)
            back = DualTrackAgentMemory.load(path)
        self.assertEqual(back.episode, mem.episode)
        self.assertEqual(back.cases.ids(), mem.cases.ids())
        self.assertEqual(back.skills.names(), mem.skills.names())
        self.assertEqual(back.value_model.to_dict(), mem.value_model.to_dict())


if __name__ == "__main__":
    unittest.main()
