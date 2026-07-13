"""Tests for the Error Notebook-guided training-free part-retrieval modules.

Covers:
  - memory/errornotebook_store.py   (Error Notebook memory + persistence)
  - reliability/errornotebook_gc.py (GC verifier + corrected-trajectory build)
  - rag/partretr_rerank.py          (Error-Notebook re-ranking policy)
  - bench/partretr_eval.py          (recall@k, MRR, accuracy, F1 harness)
"""

import os
import tempfile
import unittest

from harnesscad.agents.memory.error_notebook import (
    ErrorNotebook,
    ErrorNotebookEntry,
    char_similarity,
    jaccard_similarity,
)
from harnesscad.eval.reliability.cot_grammar_gate import (
    gc_check,
    gc_filter,
    extract_final_answer,
    build_corrected_trajectory,
)
from harnesscad.agents.rag.rerank import (
    rerank_answer_sets,
    rerank_parts,
    select_top,
)
from harnesscad.eval.bench.retrieval.part_retrieval_eval import (
    relevance,
    exact_match,
    recall_at_k,
    reciprocal_rank,
    part_bucket,
    evaluate,
)


class TestErrorNotebookStore(unittest.TestCase):
    def _nb(self):
        nb = ErrorNotebook(scorer="char")
        nb.record_mistake(
            specification="The cylindrical pin must be inserted into the flat plate hole.",
            wrong_answer=["plate.png"],
            ground_truth=["pin.png", "plate.png"],
            part_descriptions={"pin.png": "a cylindrical pin",
                               "plate.png": "a flat plate with holes"},
            corrected_cot="Only pin.png is a cylindrical pin, but the plate holds it.",
            insight="include both mating parts",
        )
        nb.record_mistake(
            specification="The bracket must mount flush against the base plate.",
            wrong_answer=["base.png"],
            ground_truth=["bracket.png", "base.png"],
        )
        return nb

    def test_add_assigns_ids(self):
        nb = self._nb()
        self.assertEqual(len(nb), 2)
        ids = [e.entry_id for e in nb.entries]
        self.assertTrue(all(i is not None for i in ids))
        self.assertEqual(len(set(ids)), 2)

    def test_retrieve_ranks_by_similarity(self):
        nb = self._nb()
        hits = nb.retrieve("A cylindrical pin fits into a hole in the plate.", n=2)
        self.assertEqual(len(hits), 2)
        # the pin/plate entry should rank first
        self.assertIn("pin", hits[0][0].specification.lower())
        self.assertGreaterEqual(hits[0][1], hits[1][1])

    def test_retrieve_excludes_exact_spec_leak(self):
        nb = self._nb()
        spec = "The bracket must mount flush against the base plate."
        hits = nb.retrieve(spec, n=5)
        self.assertTrue(all(e.specification != spec for e, _ in hits))

    def test_retrieve_excludes_by_id(self):
        nb = self._nb()
        target = nb.entries[0]
        hits = nb.retrieve("something", n=5, exclude_id=target.entry_id)
        self.assertTrue(all(e.entry_id != target.entry_id for e, _ in hits))

    def test_few_shot_prompt_cot_toggle(self):
        nb = self._nb()
        with_cot = nb.few_shot_prompt("cylindrical pin into plate", n=1, include_cot=True)
        without = nb.few_shot_prompt("cylindrical pin into plate", n=1, include_cot=False)
        self.assertIn("Final Answer:", with_cot)
        self.assertIn("cylindrical pin", with_cot)  # cot text present
        self.assertNotIn("but the plate holds it", without)

    def test_known_wrong_for(self):
        nb = self._nb()
        kw = nb.known_wrong_for("A pin inserted into the plate hole", n=5)
        # the pin/plate entry recorded ("plate.png",) as wrong
        self.assertIn(("plate.png",), kw)
        self.assertGreater(kw[("plate.png",)], 0.0)

    def test_json_roundtrip(self):
        nb = self._nb()
        text = nb.to_json()
        nb2 = ErrorNotebook.from_json(text)
        self.assertEqual(len(nb2), 2)
        self.assertEqual(nb2.entries[0].ground_truth, nb.entries[0].ground_truth)
        self.assertEqual(nb2.scorer, "char")

    def test_save_load_file(self):
        nb = self._nb()
        d = tempfile.mkdtemp()
        path = os.path.join(d, "notebook.json")
        nb.save(path)
        nb2 = ErrorNotebook.load(path)
        self.assertEqual(len(nb2), 2)

    def test_similarity_helpers(self):
        self.assertEqual(char_similarity("", ""), 1.0)
        self.assertEqual(jaccard_similarity("a b c", "a b c"), 1.0)
        self.assertEqual(jaccard_similarity("a b", "c d"), 0.0)
        self.assertGreater(jaccard_similarity("pin plate", "pin bracket"), 0.0)

    def test_deterministic_retrieval(self):
        nb = self._nb()
        a = nb.retrieve("mount bracket to plate", n=2)
        b = nb.retrieve("mount bracket to plate", n=2)
        self.assertEqual([e.entry_id for e, _ in a], [e.entry_id for e, _ in b])


class TestGCVerifier(unittest.TestCase):
    ALLOWED = ["pin.png", "plate.png", "bracket.png", "base.png"]

    def test_extract_final_answer(self):
        traj = "Step 1: reason.\nFinal Answer: pin.png; plate.png"
        self.assertEqual(extract_final_answer(traj), ["pin.png", "plate.png"])

    def test_extract_uses_last_marker(self):
        traj = "Final Answer: wrong.png\nmore\nFinal Answer: pin.png"
        self.assertEqual(extract_final_answer(traj), ["pin.png"])

    def test_sgc_accepts_wellformed(self):
        traj = "Reason.\nFinal Answer: pin.png;plate.png"
        r = gc_check(traj, self.ALLOWED, variant="sGC")
        self.assertTrue(r.accepted)
        self.assertEqual(r.predicted, ("pin.png", "plate.png"))

    def test_sgc_rejects_missing_marker(self):
        traj = "Reason.\nThe answer is pin.png"
        r = gc_check(traj, self.ALLOWED, variant="sGC")
        self.assertFalse(r.accepted)
        self.assertIn("marker", r.reason)

    def test_sgc_rejects_out_of_vocab(self):
        traj = "Final Answer: gremlin.png"
        r = gc_check(traj, self.ALLOWED, variant="sGC")
        self.assertFalse(r.accepted)
        self.assertIn("not in allowed", r.reason)

    def test_sgc_rejects_empty_answer(self):
        traj = "Final Answer:   "
        r = gc_check(traj, self.ALLOWED, variant="sGC")
        self.assertFalse(r.accepted)

    def test_rgc_accepts_missing_marker(self):
        traj = "Reason step.\npin.png; plate.png"
        strict = gc_check(traj, self.ALLOWED, variant="sGC")
        relaxed = gc_check(traj, self.ALLOWED, variant="rGC")
        self.assertFalse(strict.accepted)
        self.assertTrue(relaxed.accepted)
        self.assertEqual(relaxed.predicted, ("pin.png", "plate.png"))

    def test_rgc_still_checks_vocab(self):
        traj = "Reason.\ngremlin.png"
        r = gc_check(traj, self.ALLOWED, variant="rGC")
        self.assertFalse(r.accepted)

    def test_gc_filter_indices(self):
        trajs = [
            "Final Answer: pin.png",          # ok
            "no marker here",                  # fail sGC
            "Final Answer: gremlin.png",       # out of vocab
        ]
        kept = gc_filter(trajs, self.ALLOWED, variant="sGC")
        self.assertEqual(kept, [0])

    def test_bad_variant_raises(self):
        with self.assertRaises(ValueError):
            gc_check("Final Answer: pin.png", self.ALLOWED, variant="xGC")

    def test_build_corrected_trajectory_passes_sgc(self):
        traj = build_corrected_trajectory(
            steps_up_to_first_error=["Step 1: check descriptions.",
                                     "Step 2: pin.png is a pin."],
            corrected_steps=["Reconsider: the plate holds the pin.",
                             "Both parts are relevant."],
            ground_truth=["pin.png", "plate.png"],
        )
        self.assertIn("Final Answer: pin.png;plate.png", traj)
        self.assertIn("But wait", traj)
        r = gc_check(traj, self.ALLOWED, variant="sGC")
        self.assertTrue(r.accepted)


class TestRerank(unittest.TestCase):
    def _nb(self):
        nb = ErrorNotebook(scorer="char")
        nb.record_mistake(
            specification="The cylindrical pin must be inserted into the plate hole.",
            wrong_answer=["plate.png"],
            ground_truth=["pin.png", "plate.png"],
        )
        return nb

    def test_rerank_downweights_known_wrong_set(self):
        nb = self._nb()
        spec = "A cylindrical pin inserted into the plate hole."
        cands = [
            (["plate.png"], 0.9),             # known wrong for similar spec
            (["pin.png", "plate.png"], 0.6),  # correct-ish, lower base
        ]
        ranked = rerank_answer_sets(spec, cands, nb, penalty_weight=1.0)
        # the known-wrong high-base candidate should be penalised below the other
        top = select_top(ranked, k=1)[0]
        self.assertEqual(top, ("pin.png", "plate.png"))
        # the penalised candidate carries a reason + positive penalty
        plate_only = next(c for c in ranked if c.answer == ("plate.png",))
        self.assertGreater(plate_only.penalty, 0.0)
        self.assertTrue(plate_only.reason)

    def test_rerank_no_penalty_when_no_similar_error(self):
        nb = self._nb()
        ranked = rerank_answer_sets("totally unrelated widget specification xyz",
                                    [(["a.png"], 0.5)], nb, min_similarity=0.9)
        self.assertEqual(ranked[0].penalty, 0.0)
        self.assertEqual(ranked[0].final_score, 0.5)

    def test_rerank_parts(self):
        nb = self._nb()
        spec = "A cylindrical pin inserted into the plate hole."
        cands = [("plate.png", 0.8), ("pin.png", 0.7)]
        ranked = rerank_parts(spec, cands, nb, penalty_weight=1.0)
        # plate.png flagged wrong -> penalised; pin.png should surface
        self.assertEqual(ranked[0].answer, ("pin.png",))

    def test_rerank_deterministic(self):
        nb = self._nb()
        spec = "pin into plate"
        cands = [(["plate.png"], 0.5), (["pin.png"], 0.5)]
        a = rerank_answer_sets(spec, cands, nb)
        b = rerank_answer_sets(spec, cands, nb)
        self.assertEqual([c.answer for c in a], [c.answer for c in b])


class TestEvalHarness(unittest.TestCase):
    def test_relevance(self):
        rel = relevance(["a", "b"], ["b", "c"])
        self.assertEqual(rel["tp"], 1)
        self.assertEqual(rel["fp"], 1)
        self.assertEqual(rel["fn"], 1)
        self.assertAlmostEqual(rel["recall"], 0.5)
        self.assertAlmostEqual(rel["precision"], 0.5)
        self.assertAlmostEqual(rel["f1"], 0.5)

    def test_exact_match_order_insensitive(self):
        self.assertTrue(exact_match(["b", "a"], ["a", "b"]))
        self.assertFalse(exact_match(["a"], ["a", "b"]))

    def test_recall_at_k(self):
        ranked = [["x.png"], ["a.png"], ["b.png"]]
        self.assertEqual(recall_at_k(ranked, ["a.png"], 1), 0.0)
        self.assertEqual(recall_at_k(ranked, ["a.png"], 2), 1.0)

    def test_reciprocal_rank(self):
        ranked = [["x.png"], ["a.png"]]
        self.assertAlmostEqual(reciprocal_rank(ranked, ["a.png"]), 0.5)
        self.assertEqual(reciprocal_rank(ranked, ["none.png"]), 0.0)

    def test_part_bucket(self):
        self.assertEqual(part_bucket(5), "<10")
        self.assertEqual(part_bucket(15), "10-20")
        self.assertEqual(part_bucket(30), "20-50")
        self.assertEqual(part_bucket(80), ">50")

    def test_evaluate_aggregate(self):
        queries = [
            {"pred": ["a", "b"], "gt": ["a", "b"],
             "ranked": [["a", "b"]], "n_parts": 5},
            {"pred": ["a"], "gt": ["a", "b"],
             "ranked": [["z"], ["a"]], "n_parts": 30},
        ]
        rep = evaluate(queries, ks=(1, 2))
        self.assertEqual(rep.n, 2)
        self.assertAlmostEqual(rep.accuracy, 0.5)   # first exact, second not
        self.assertAlmostEqual(rep.recall_at_k[1], 0.5)
        self.assertAlmostEqual(rep.recall_at_k[2], 1.0)
        self.assertAlmostEqual(rep.mrr, (1.0 + 0.5) / 2)
        self.assertIn("<10", rep.per_bucket)
        self.assertIn("20-50", rep.per_bucket)
        self.assertEqual(rep.per_bucket["<10"]["accuracy"], 1.0)

    def test_evaluate_empty(self):
        rep = evaluate([])
        self.assertEqual(rep.n, 0)
        self.assertEqual(rep.accuracy, 0.0)

    def test_report_to_dict(self):
        rep = evaluate([{"pred": ["a"], "gt": ["a"], "n_parts": 3}])
        d = rep.to_dict()
        self.assertEqual(d["accuracy"], 1.0)
        self.assertIn("per_bucket", d)


if __name__ == "__main__":
    unittest.main()
