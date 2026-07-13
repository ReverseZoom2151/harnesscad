"""Tests for the memory + skill-library layer (blueprint sec.8).

Covers: episodic write+recall by similarity, semantic set/get + JSON round-trip,
procedural round-trip, and the Voyager execution-verified SkillLibrary
(add_verified admits a good `plate`, REJECTS a broken skill; expand + find work).
"""

import os
import tempfile
import unittest

from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.cisp.ops import NewSketch, Extrude, AddRectangle
from harnesscad.core.loop import HarnessSession

from harnesscad.agents.memory.store import MemoryStore, TokenOverlapSimilarity
from harnesscad.agents.memory.skills import (
    Skill, SkillLibrary, plate_skill, bracket_skill, plate_ops,
    default_expanders, build_default_library,
)


def _session_factory():
    return HarnessSession(StubBackend())


# ---------------------------------------------------------------------------
# Episodic
# ---------------------------------------------------------------------------
class TestEpisodic(unittest.TestCase):
    def _seeded(self) -> MemoryStore:
        store = MemoryStore()
        store.add_episodic(
            "a rectangular mounting plate 20mm wide with two holes",
            plate_ops(20, 20, 2), outcome="ok", digest="deadbeef")
        store.add_episodic(
            "a cylindrical shaft 100mm long", [NewSketch()], outcome="ok")
        store.add_episodic(
            "a hex nut M8 threaded", [NewSketch()], outcome="ok")
        return store

    def test_similar_brief_retrieves_right_attempt(self):
        store = self._seeded()
        hits = store.recall_episodic(
            "mounting plate with holes, rectangular", k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("mounting plate", hits[0].brief)
        # The ops of the recalled attempt are preserved (serialised).
        self.assertEqual(hits[0].ops[0]["op"], "new_sketch")
        self.assertEqual(hits[0].digest, "deadbeef")

    def test_recall_k_and_ordering(self):
        store = self._seeded()
        hits = store.recall_episodic("shaft cylinder", k=2)
        self.assertEqual(len(hits), 2)
        # Most-similar first: the shaft episode outranks the others.
        self.assertIn("shaft", hits[0].brief)

    def test_recall_filters_by_outcome(self):
        store = MemoryStore()
        store.add_episodic("failed plate attempt", [NewSketch()], outcome="failed")
        store.add_episodic("good plate attempt", plate_ops(), outcome="ok")
        hits = store.recall_episodic("plate", k=5, outcome="ok")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].outcome, "ok")

    def test_episodic_json_round_trip(self):
        store = self._seeded()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mem.json")
            store.save(path)
            loaded = MemoryStore.load(path)
        self.assertEqual(len(loaded.episodic), 3)
        hits = loaded.recall_episodic("rectangular plate holes", k=1)
        self.assertIn("mounting plate", hits[0].brief)


# ---------------------------------------------------------------------------
# Semantic + procedural
# ---------------------------------------------------------------------------
class TestSemanticProcedural(unittest.TestCase):
    def test_semantic_set_get(self):
        store = MemoryStore()
        store.set_semantic("aluminium.density", 2700)
        store.set_semantic("user.pref.units", "mm")
        self.assertEqual(store.get_semantic("aluminium.density"), 2700)
        self.assertEqual(store.get_semantic("user.pref.units"), "mm")
        self.assertIsNone(store.get_semantic("missing"))
        self.assertEqual(store.get_semantic("missing", 0), 0)

    def test_semantic_json_round_trip(self):
        store = MemoryStore()
        store.set_semantic("steel.density", 7850)
        store.set_procedural(
            "extrude_rule", "always fully constrain the sketch before extrude")
        store.note("current_task", "make a plate")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mem.json")
            store.save(path)
            loaded = MemoryStore.load(path)
        self.assertEqual(loaded.get_semantic("steel.density"), 7850)
        self.assertEqual(
            loaded.get_procedural("extrude_rule"),
            "always fully constrain the sketch before extrude")
        self.assertEqual(loaded.get_note("current_task"), "make a plate")


# ---------------------------------------------------------------------------
# SkillLibrary — Voyager execution-verified gate
# ---------------------------------------------------------------------------
def _broken_skill() -> Skill:
    """Expands to an extrude of a sketch with no profile -> backend rejects it."""
    def _bad(**_):
        return [NewSketch(), Extrude(sketch="sk1", distance=2.0)]
    return Skill(name="broken", description="extrudes an empty sketch",
                 template=_bad, sample_params={})


class TestSkillLibrary(unittest.TestCase):
    def test_add_verified_admits_good_plate(self):
        lib = SkillLibrary()
        ok = lib.add_verified(plate_skill(), _session_factory)
        self.assertTrue(ok)
        self.assertIn("plate", lib)
        self.assertTrue(lib.get("plate").verified)

    def test_add_verified_rejects_broken_skill(self):
        lib = SkillLibrary()
        ok = lib.add_verified(_broken_skill(), _session_factory)
        self.assertFalse(ok)
        self.assertNotIn("broken", lib)   # monotonic: never admitted

    def test_expanded_good_skill_actually_applies(self):
        # Independent confirmation that plate's ops verify on the harness.
        res = _session_factory().apply_ops(plate_skill().expand())
        self.assertTrue(res.ok)
        self.assertEqual(res.rejected, None)

    def test_expand_with_params(self):
        lib = SkillLibrary()
        lib.add_verified(plate_skill(), _session_factory)
        ops = lib.expand("plate", w=50.0, h=30.0, thickness=5.0)
        rect = [o for o in ops if isinstance(o, AddRectangle)][0]
        self.assertEqual(rect.w, 50.0)
        self.assertEqual(rect.h, 30.0)
        # And the custom-param expansion still verifies.
        self.assertTrue(_session_factory().apply_ops(ops).ok)

    def test_bracket_skill_verifies(self):
        lib = SkillLibrary()
        self.assertTrue(lib.add_verified(bracket_skill(), _session_factory))

    def test_find_by_similarity(self):
        lib = build_default_library(_session_factory)
        hits = lib.find("flat rectangular plate", k=1)
        self.assertEqual(hits[0].name, "plate")
        hits = lib.find("bracket with a mounting hole", k=1)
        self.assertEqual(hits[0].name, "bracket")

    def test_library_json_round_trip(self):
        lib = build_default_library(_session_factory)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "skills.json")
            lib.save(path)
            loaded = SkillLibrary.load(path, default_expanders())
        self.assertIn("plate", loaded)
        self.assertIn("bracket", loaded)
        self.assertTrue(loaded.get("plate").verified)
        # Re-attached expander works after load.
        self.assertTrue(_session_factory().apply_ops(loaded.expand("plate")).ok)

    def test_loaded_skill_missing_expander_raises(self):
        lib = build_default_library(_session_factory)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "skills.json")
            lib.save(path)
            loaded = SkillLibrary.load(path, {})   # no expanders re-attached
        with self.assertRaises(RuntimeError):
            loaded.expand("plate")


if __name__ == "__main__":
    unittest.main()
