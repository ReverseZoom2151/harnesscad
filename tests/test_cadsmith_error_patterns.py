import unittest

from harnesscad.agents.generation.cadsmith_error_patterns import (
    ErrorPattern, ErrorSolutionKB, default_kb, default_error_patterns,
)


class TestKBConstruction(unittest.TestCase):
    def test_default_kb_nonempty(self):
        self.assertGreaterEqual(len(default_kb()), 10)

    def test_duplicate_id_rejected(self):
        p = ErrorPattern("x", ("a",), "rc", "fix")
        with self.assertRaises(ValueError):
            ErrorSolutionKB([p, p])

    def test_unique_ids_in_default(self):
        ids = [p.id for p in default_error_patterns()]
        self.assertEqual(len(ids), len(set(ids)))


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.kb = default_kb()

    def test_fillet_traceback(self):
        tb = ("BRep_API: command not done, fillet radius too large for edge; "
              "ChFi2d error")
        hits = self.kb.retrieve(tb, top_k=1)
        self.assertEqual(hits[0].id, "fillet-radius-too-large")

    def test_wire_not_closed(self):
        tb = "ValueError: Wire is not closed, cannot extrude open wire"
        hits = self.kb.retrieve(tb, top_k=2)
        self.assertIn("wire-not-closed", [h.id for h in hits])

    def test_no_match_returns_empty(self):
        self.assertEqual(self.kb.retrieve("totally unrelated banana", top_k=3), ())

    def test_top_k_zero(self):
        self.assertEqual(self.kb.retrieve("fillet", top_k=0), ())

    def test_ranked_by_score(self):
        # A traceback hitting many fillet triggers should rank fillet first.
        tb = "fillet radius BRep_API_Error in ChFi2d"
        hits = self.kb.retrieve(tb, top_k=3)
        self.assertEqual(hits[0].id, "fillet-radius-too-large")

    def test_deterministic_tie_break(self):
        # Two runs identical.
        tb = "boolean cut produced empty null shape"
        self.assertEqual([h.id for h in self.kb.retrieve(tb)],
                         [h.id for h in self.kb.retrieve(tb)])

    def test_single_word_trigger_whole_token(self):
        # "arc" must not fire on "search".
        p = ErrorPattern("arc-only", ("arc",), "rc", "fix")
        kb = ErrorSolutionKB([p])
        self.assertEqual(kb.retrieve("please search the database"), ())
        self.assertEqual(len(kb.retrieve("threePointArc failed with arc error")), 1)

    def test_context_block(self):
        ctx = self.kb.context_for("fillet radius too large", top_k=1)
        self.assertIn("fillet-radius-too-large", ctx)
        self.assertIn("root cause", ctx)
        self.assertIn("fix", ctx)


if __name__ == "__main__":
    unittest.main()
