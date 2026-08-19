"""The LLM-backed solver, proven offline.

Every test here drives a ``ScriptedClient`` -- the same offline stand-in the
pressure suite uses -- so nothing needs ollama, a network, or a geometry kernel.
What is asserted is the part the hard corpus cares about: a good answer becomes
ops, and a BAD answer becomes a RECORDED INVALID ATTEMPT rather than an
exception or a silent skip.
"""

from __future__ import annotations

import unittest

from harnesscad.eval.hardcorpus import solver as solver_mod
from harnesscad.eval.pressure.model import ScriptedClient

PLATE = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 60, "h": 40},
    {"op": "extrude", "sketch": "sk1", "distance": 12},
]


class TestModelSolver(unittest.TestCase):

    def test_a_good_answer_becomes_ops(self):
        client = ScriptedClient([PLATE])
        s = solver_mod.ModelSolver(client)
        ops = s(" a 60 x 40 x 12 plate ")
        self.assertEqual(len(ops), 3)
        self.assertEqual(s.invalid, 0)
        self.assertEqual(s.model_calls, 1)
        self.assertFalse(s.records[0].invalid)

    def test_fenced_prose_is_still_parsed(self):
        client = ScriptedClient([
            'Sure! Here is the plan:\n```json\n[{"op":"new_sketch","plane":"XY"}]\n```'
        ])
        s = solver_mod.ModelSolver(client)
        self.assertEqual(len(s("plate")), 1)
        self.assertEqual(s.invalid, 0)

    def test_the_prompt_carries_the_brief_and_the_op_grammar(self):
        client = ScriptedClient([PLATE])
        solver_mod.ModelSolver(client)("a bracket with an M8 hole")
        _attempt, messages = client.calls[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("CISP", messages[0]["content"])
        self.assertIn("a bracket with an M8 hole", messages[1]["content"])

    def test_unparseable_output_is_recorded_not_raised(self):
        client = ScriptedClient(["I cannot help with that.",
                                 "Still no JSON here."])
        s = solver_mod.ModelSolver(client, max_attempts=2)
        ops = s("a plate")
        self.assertEqual(list(ops), [])
        self.assertEqual(s.invalid, 1)
        self.assertEqual(s.model_calls, 2)
        rec = s.records[0]
        self.assertTrue(rec.invalid)
        self.assertFalse(rec.errored)
        self.assertTrue(all(a.error for a in rec.attempts))

    def test_a_second_attempt_repairs_a_parse_failure(self):
        client = ScriptedClient(["nope, not JSON", PLATE])
        s = solver_mod.ModelSolver(client, max_attempts=2)
        ops = s("a plate")
        self.assertEqual(len(ops), 3)
        self.assertEqual(s.invalid, 0)
        # The repair turn tells the model it could not be parsed, and nothing
        # about the geometry (nothing has been built yet).
        _attempt, messages = client.calls[1]
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("could not be parsed", messages[-1]["content"])

    def test_a_single_attempt_budget_makes_exactly_one_call(self):
        client = ScriptedClient(["prose", PLATE])
        s = solver_mod.ModelSolver(client, max_attempts=1)
        self.assertEqual(list(s("a plate")), [])
        self.assertEqual(s.model_calls, 1)
        self.assertEqual(s.invalid, 1)

    def test_a_transport_failure_is_recorded_as_errored(self):
        class Dead:
            name = "dead"

            def complete(self, messages, attempt, seed=None, temperature=None):
                raise RuntimeError("connection refused")

        s = solver_mod.ModelSolver(Dead(), max_attempts=2)
        self.assertEqual(list(s("a plate")), [])
        self.assertEqual(s.invalid, 1)
        self.assertEqual(s.errored, 1)
        self.assertIn("connection refused", s.records[0].attempts[0].error)

    def test_records_never_carry_the_brief_text(self):
        secret = "a held-out brief nobody may republish"
        client = ScriptedClient([PLATE])
        s = solver_mod.ModelSolver(client)
        s(secret)
        blob = repr(s.to_dict())
        self.assertNotIn(secret, blob)
        self.assertIn(solver_mod.brief_digest(secret), blob)

    def test_the_digest_is_stable_and_short(self):
        d = solver_mod.brief_digest("a plate")
        self.assertEqual(d, solver_mod.brief_digest("a plate"))
        self.assertEqual(len(d), 12)
        self.assertNotEqual(d, solver_mod.brief_digest("a bracket"))

    def test_ordinals_number_distinct_briefs(self):
        client = ScriptedClient([PLATE, PLATE, PLATE])
        s = solver_mod.ModelSolver(client)
        s("one")
        s("two")
        s("one")
        self.assertEqual([r.ordinal for r in s.records], [0, 1, 0])

    def test_stats_report_the_whole_run(self):
        client = ScriptedClient([PLATE, "prose", "more prose"])
        s = solver_mod.ModelSolver(client, max_attempts=2)
        s("one")
        s("two")
        st = s.stats()
        self.assertEqual(st["briefs"], 2)
        self.assertEqual(st["invalid"], 1)
        self.assertEqual(st["model_calls"], 3)


if __name__ == "__main__":
    unittest.main()
