"""Tests for the Agent-S experience store (dialog->feature memory + narrative)."""

import os
import tempfile
import unittest

from harnesscad.agents.cua.experience import (
    DialogFeatureMemory, DialogRecipe, ExperienceStore, NarrativeEntry,
    summarize_trajectory,
)


def _pad_recipe(**kw) -> DialogRecipe:
    base = dict(op_tag="pad", tier="semantic_gui", dialog="PadDialog",
                command="Part_Pad", fields={"boxLength": "length"})
    base.update(kw)
    return DialogRecipe(**base)


class TestDialogRecipe(unittest.TestCase):
    def test_confidence_zero_without_attempts(self):
        self.assertEqual(_pad_recipe().confidence, 0.0)

    def test_record_accumulates(self):
        r = _pad_recipe()
        r.record(True)
        r.record(False)
        r.record(True)
        self.assertEqual(r.attempts, 3)
        self.assertEqual(r.successes, 2)
        self.assertAlmostEqual(r.confidence, 2 / 3)

    def test_dict_roundtrip(self):
        r = _pad_recipe(notes="canonical")
        r.record(True)
        back = DialogRecipe.from_dict(r.to_dict())
        self.assertEqual(back.to_dict(), r.to_dict())


class TestDialogFeatureMemory(unittest.TestCase):
    def test_identical_recipes_merge_onto_one_row(self):
        m = DialogFeatureMemory()
        m.learn(_pad_recipe(), ok=True)
        m.learn(_pad_recipe(), ok=True)
        m.learn(_pad_recipe(), ok=False)
        recipes = m.recipes("pad")
        self.assertEqual(len(recipes), 1)
        self.assertEqual(recipes[0].attempts, 3)
        self.assertEqual(recipes[0].successes, 2)

    def test_field_difference_makes_a_distinct_recipe(self):
        m = DialogFeatureMemory()
        m.learn(_pad_recipe(fields={"boxLength": "length"}), ok=True)
        m.learn(_pad_recipe(fields={"boxLength": "height"}), ok=True)
        self.assertEqual(len(m.recipes("pad")), 2)

    def test_recall_prefers_confidence_then_attempts(self):
        m = DialogFeatureMemory()
        # a: 1/1 = conf 1.0; b: 3/4 = conf 0.75 -> a wins on confidence.
        m.learn(_pad_recipe(dialog="A"), ok=True)
        for ok in (True, True, True, False):
            m.learn(_pad_recipe(dialog="B"), ok=ok)
        self.assertEqual(m.recall("pad").dialog, "A")

    def test_recall_none_for_unknown_op(self):
        self.assertIsNone(DialogFeatureMemory().recall("fillet"))

    def test_known_ops_sorted(self):
        m = DialogFeatureMemory()
        m.learn(_pad_recipe(op_tag="pad"), ok=True)
        m.learn(_pad_recipe(op_tag="box"), ok=True)
        self.assertEqual(m.known_ops(), ["box", "pad"])

    def test_dict_roundtrip(self):
        m = DialogFeatureMemory()
        m.learn(_pad_recipe(), ok=True)
        back = DialogFeatureMemory.from_dict(m.to_dict())
        self.assertEqual(back.to_dict(), m.to_dict())


class TestSummarize(unittest.TestCase):
    def test_deterministic_and_reports_outcome(self):
        a = summarize_trajectory("build a block", ["box"], True,
                                 tier_counts={"semantic_gui": 2})
        b = summarize_trajectory("build a block", ["box"], True,
                                 tier_counts={"semantic_gui": 2})
        self.assertEqual(a, b)
        self.assertIn("solved", a)
        self.assertIn("box", a)

    def test_unsolved_lists_misses(self):
        s = summarize_trajectory("b", ["box"], False, misses=["volume off"])
        self.assertIn("did NOT solve", s)
        self.assertIn("volume off", s)


class TestExperienceStore(unittest.TestCase):
    def test_ingest_writes_both_stores(self):
        s = ExperienceStore()
        entry = s.ingest("make a 30mm block", ["box"], solved=True,
                         recipes=[_pad_recipe(op_tag="box")])
        self.assertIsInstance(entry, NarrativeEntry)
        self.assertEqual(entry.outcome, "solved")
        self.assertEqual(len(s.narrative), 1)
        self.assertIsNotNone(s.recipe_for("box"))

    def test_recipe_outcome_follows_trajectory(self):
        s = ExperienceStore()
        s.ingest("b", ["box"], solved=False, recipes=[_pad_recipe(op_tag="box")])
        self.assertEqual(s.recipe_for("box").confidence, 0.0)

    def test_retrieve_most_similar_solved_only(self):
        s = ExperienceStore()
        s.ingest("design a rectangular block 30 mm long", ["box"], solved=True)
        s.ingest("model a cylinder of radius 5", ["cyl"], solved=True)
        s.ingest("design a rectangular block 40 mm long", ["box"], solved=False)
        hits = s.retrieve("design a rectangular block 35 mm long", k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("block", hits[0].brief)
        self.assertEqual(hits[0].outcome, "solved")

    def test_retrieve_can_include_failures(self):
        s = ExperienceStore()
        s.ingest("a failing block brief", ["box"], solved=False)
        self.assertEqual(s.retrieve("block", solved_only=True), [])
        self.assertEqual(len(s.retrieve("block", solved_only=False)), 1)

    def test_save_load_roundtrip(self):
        s = ExperienceStore()
        s.ingest("make a block", ["box"], solved=True,
                 recipes=[_pad_recipe(op_tag="box")])
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            s.save(path)
            back = ExperienceStore.load(path)
        finally:
            os.remove(path)
        self.assertEqual(back.to_dict(), s.to_dict())


if __name__ == "__main__":
    unittest.main()
