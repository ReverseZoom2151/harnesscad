"""The planner is no longer zero-shot, and it no longer hands the model an
unranked pile of diagnostics.

`system_prompt.py` was PURE ZERO-SHOT for a strict-format structured-output task
while `agents/rag/exemplar_select.py` sat orphaned. `agents/agents/roles.py:72
prioritize` existed and was never called: when nine diagnostics came back, all
nine went to the model in fleet order.
"""

from __future__ import annotations

import unittest

from harnesscad.agents.agent.planner import Planner, prioritize_diagnostics
from harnesscad.agents.agent.system_prompt import build_system_prompt
from harnesscad.agents.llm.base import CompletionResult
from harnesscad.eval.verifiers.verify import Diagnostic, Severity


class _NullLLM:
    def complete(self, messages, tools=None, **kwargs):
        return CompletionResult(text="[]")


def _planner(**kw):
    return Planner(_NullLLM(), **kw)


class TestFewShot(unittest.TestCase):
    def test_zero_shot_prompt_is_still_available_byte_for_byte(self):
        self.assertNotIn("WORKED EXAMPLES", build_system_prompt())
        self.assertNotIn("WORKED EXAMPLES", build_system_prompt(None))

    def test_a_brief_pins_verified_worked_examples_into_the_system_prompt(self):
        msgs = _planner(exemplars=3).build_messages(
            "A round flange, 80 mm diameter, 8 mm thick, with a 30 mm bore.")
        system = msgs[0].content
        self.assertIn("WORKED EXAMPLES", system)
        self.assertIn('"op": "extrude"', system)

    def test_exemplars_zero_restores_the_old_prompt(self):
        msgs = _planner(exemplars=0).build_messages("a plate with four holes")
        self.assertNotIn("WORKED EXAMPLES", msgs[0].content)
        self.assertEqual(msgs[0].content, build_system_prompt())

    def test_the_prompt_is_a_pure_function_of_the_brief(self):
        a = _planner().build_messages("a 50x30 plate")[0].content
        b = _planner().build_messages("a 50x30 plate")[0].content
        self.assertEqual(a, b)


class TestPrioritisation(unittest.TestCase):
    def test_errors_come_first_warnings_then_info(self):
        ds = [
            Diagnostic(Severity.INFO, "i", "info"),
            Diagnostic(Severity.WARNING, "w", "warn"),
            Diagnostic(Severity.ERROR, "e", "err"),
        ]
        self.assertEqual([d.code for d in prioritize_diagnostics(ds)],
                         ["e", "w", "i"])

    def test_order_is_stable_inside_a_severity_band(self):
        ds = [
            Diagnostic(Severity.ERROR, "e1", "a"),
            Diagnostic(Severity.ERROR, "e2", "b"),
            Diagnostic(Severity.ERROR, "e3", "c"),
        ]
        self.assertEqual([d.code for d in prioritize_diagnostics(ds)],
                         ["e1", "e2", "e3"])

    def test_top_k_caps_what_the_model_is_told(self):
        ds = [Diagnostic(Severity.WARNING, "w%d" % i, "m") for i in range(9)]
        ds.append(Diagnostic(Severity.ERROR, "boom", "the real one"))
        kept = prioritize_diagnostics(ds, 3)
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept[0].code, "boom")

    def test_dicts_work_too_and_the_original_objects_come_back(self):
        ds = [{"severity": "info", "code": "a"}, {"severity": "error", "code": "b"}]
        out = prioritize_diagnostics(ds)
        self.assertEqual([d["code"] for d in out], ["b", "a"])
        self.assertIs(out[0], ds[1])

    def test_the_planner_ranks_and_caps_the_retry_prompt(self):
        # 6 MEASURED diagnostics; only the top 2 may be spoken.
        ds = [Diagnostic(Severity.WARNING, "under-constrained", "sketch sk1 dof=4")
              for _ in range(5)]
        ds.append(Diagnostic(Severity.ERROR, "preflight-THICKNESS_TOO_LARGE",
                             "shell t=9 leaves no cavity (smallest extent 5)"))
        msgs = _planner(max_diagnostics=2).build_messages("a plate", diagnostics=ds)
        user = msgs[-1].content
        self.assertIn("preflight-THICKNESS_TOO_LARGE", user)
        self.assertEqual(user.count("under-constrained"), 1)

    def test_empty_in_empty_out(self):
        self.assertEqual(prioritize_diagnostics([]), [])


if __name__ == "__main__":
    unittest.main()
