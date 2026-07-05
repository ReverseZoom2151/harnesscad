"""Tests for the classify-then-route cost-control layer (routing.py).

No network, no keys: routes are `MockLLM`s (they satisfy the `LLM` protocol),
and a `FailingLLM` exercises the sequential fallback chain.
"""

import unittest
from typing import List, Optional

from llm.base import LLM, CompletionResult, Message, user, system
from routing import (
    TaskClass,
    Classifier,
    HeuristicClassifier,
    CostTable,
    ModelPrice,
    Usage,
    RoutingLLM,
    RouteDecision,
    AllRoutesFailed,
    count_tokens,
    messages_tokens,
    usage_from_result,
)


# --- test doubles ----------------------------------------------------------
class MockLLM(LLM):
    """A route that echoes which model answered. Records the calls it saw."""

    def __init__(self, name: str, usage: Optional[dict] = None) -> None:
        self.model = name
        self._usage = usage
        self.calls: List[List[Message]] = []

    def _result(self) -> CompletionResult:
        raw = {"usage": self._usage} if self._usage else None
        return CompletionResult(text=f"answered by {self.model}", raw=raw)

    def complete(self, messages, tools=None, response_schema=None, **opts) -> CompletionResult:
        self.calls.append(list(messages))
        return self._result()

    def stream(self, messages, tools=None, response_schema=None, **opts):
        self.calls.append(list(messages))
        yield self._result().text


class FailingLLM(LLM):
    """A route that always raises — to drive the fallback chain."""

    def __init__(self, name: str = "boom") -> None:
        self.model = name
        self.attempts = 0

    def complete(self, messages, tools=None, response_schema=None, **opts):
        self.attempts += 1
        raise RuntimeError(f"{self.model} is down")

    def stream(self, messages, tools=None, response_schema=None, **opts):
        self.attempts += 1
        raise RuntimeError(f"{self.model} is down")
        yield  # pragma: no cover - makes this a generator


def routes():
    return {
        TaskClass.CHEAP: MockLLM("cheap-model"),
        TaskClass.STANDARD: MockLLM("standard-model"),
        TaskClass.HARD: MockLLM("hard-model"),
    }


# --- classifier ------------------------------------------------------------
class TestHeuristicClassifier(unittest.TestCase):
    def setUp(self):
        self.clf = HeuristicClassifier()

    def _c(self, text, hints=None):
        return self.clf.classify([user(text)], hints)

    def test_is_a_classifier(self):
        self.assertIsInstance(self.clf, Classifier)

    def test_cheap_briefs(self):
        for brief in [
            "Change the parameter width from 20mm to 25mm",
            "Convert this dimension from mm to inches",
            "Rename the sketch and tweak the boilerplate",
            "Bump the fillet radius value",
        ]:
            self.assertEqual(self._c(brief), TaskClass.CHEAP, brief)

    def test_hard_briefs(self):
        for brief in [
            "Plan the spatial layout of the gearbox assembly",
            "Solve the constraint system so the parts mate without interference",
            "Assemble the bracket and check clearance against the housing",
            "Optimise the load path across the frame",
        ]:
            self.assertEqual(self._c(brief), TaskClass.HARD, brief)

    def test_standard_fallthrough(self):
        self.assertEqual(self._c("Make a bracket that holds a motor"), TaskClass.STANDARD)

    def test_hard_wins_over_cheap_when_ambiguous(self):
        # Mentions a cheap "parameter" edit but also constraint solving -> HARD.
        brief = "Change the parameter, then solve the assembly constraints"
        self.assertEqual(self._c(brief), TaskClass.HARD)

    def test_hint_override_enum_and_string(self):
        self.assertEqual(self._c("anything", {"task_class": TaskClass.HARD}), TaskClass.HARD)
        self.assertEqual(self._c("anything", {"task_class": "cheap"}), TaskClass.CHEAP)

    def test_hint_note_contributes_text(self):
        self.assertEqual(
            self.clf.classify([user("do it")], {"note": "constraint solving needed"}),
            TaskClass.HARD,
        )


# --- dispatch --------------------------------------------------------------
class TestRoutingDispatch(unittest.TestCase):
    def test_routes_to_mapped_model(self):
        r = routes()
        router = RoutingLLM(r)
        cheap = router.complete([user("convert mm to inches")])
        self.assertIn("cheap-model", cheap.text)
        hard = router.complete([user("solve the assembly constraints")])
        self.assertIn("hard-model", hard.text)
        std = router.complete([user("make a simple bracket")])
        self.assertIn("standard-model", std.text)
        # The right route actually received the messages.
        self.assertEqual(len(r[TaskClass.CHEAP].calls), 1)
        self.assertEqual(len(r[TaskClass.HARD].calls), 1)

    def test_router_is_an_llm(self):
        self.assertIsInstance(RoutingLLM(routes()), LLM)

    def test_hint_forces_route(self):
        r = routes()
        router = RoutingLLM(r)
        res = router.complete([user("convert mm to inches")], hints={"task_class": TaskClass.HARD})
        self.assertIn("hard-model", res.text)
        self.assertEqual(len(r[TaskClass.HARD].calls), 1)
        self.assertEqual(len(r[TaskClass.CHEAP].calls), 0)

    def test_hints_not_forwarded_to_backend(self):
        # A backend that rejects unknown kwargs must never see `hints`.
        class StrictLLM(LLM):
            model = "strict"

            def complete(self, messages, tools=None, response_schema=None):
                return CompletionResult(text="ok")

            def stream(self, messages, tools=None, response_schema=None):
                yield "ok"

        router = RoutingLLM({TaskClass.STANDARD: StrictLLM()})
        res = router.complete([user("plain request")], hints={"task_class": "standard"})
        self.assertEqual(res.text, "ok")

    def test_stream_dispatch(self):
        r = routes()
        router = RoutingLLM(r)
        out = "".join(router.stream([user("convert mm to inches")]))
        self.assertIn("cheap-model", out)

    def test_missing_route_uses_default_class(self):
        # No STANDARD route; default_class STANDARD missing -> falls to available.
        r = {TaskClass.HARD: MockLLM("only-hard")}
        router = RoutingLLM(r)
        res = router.complete([user("make a simple bracket")])  # classifies STANDARD
        self.assertIn("only-hard", res.text)


# --- fallback chain --------------------------------------------------------
class TestFallbackChain(unittest.TestCase):
    def test_failing_primary_falls_back_and_returns(self):
        primary = FailingLLM("primary")
        backup = MockLLM("backup")
        router = RoutingLLM(
            {TaskClass.STANDARD: primary},
            fallbacks=[backup],
        )
        res = router.complete([user("make a bracket")])
        self.assertIn("backup", res.text)
        self.assertEqual(primary.attempts, 1)
        self.assertEqual(router.stats()["fallbacks_taken"], 1)

    def test_sequential_multi_fallback(self):
        primary = FailingLLM("p")
        mid = FailingLLM("m")
        good = MockLLM("good")
        router = RoutingLLM({TaskClass.HARD: primary}, fallbacks=[mid, good])
        res = router.complete([user("solve the constraint assembly")])
        self.assertIn("good", res.text)
        self.assertEqual(router.stats()["fallbacks_taken"], 2)

    def test_all_fail_raises(self):
        router = RoutingLLM(
            {TaskClass.STANDARD: FailingLLM("a")}, fallbacks=[FailingLLM("b")]
        )
        with self.assertRaises(AllRoutesFailed):
            router.complete([user("make a bracket")])
        self.assertEqual(len(router.stats()["errors"]), 1)

    def test_stream_falls_back(self):
        router = RoutingLLM(
            {TaskClass.STANDARD: FailingLLM("p")}, fallbacks=[MockLLM("backup")]
        )
        out = "".join(router.stream([user("make a bracket")]))
        self.assertIn("backup", out)
        self.assertEqual(router.stats()["fallbacks_taken"], 1)


# --- stats -----------------------------------------------------------------
class TestStats(unittest.TestCase):
    def test_calls_per_class_and_totals(self):
        router = RoutingLLM(routes())
        router.complete([user("convert mm to inches")])       # CHEAP
        router.complete([user("rename the boilerplate")])     # CHEAP
        router.complete([user("make a bracket")])             # STANDARD
        router.complete([user("solve the assembly constraints")])  # HARD
        s = router.stats()
        self.assertEqual(s["total_calls"], 4)
        self.assertEqual(s["calls_per_class"]["cheap"], 2)
        self.assertEqual(s["calls_per_class"]["standard"], 1)
        self.assertEqual(s["calls_per_class"]["hard"], 1)
        self.assertEqual(s["fallbacks_taken"], 0)

    def test_spend_tally_uses_cost_table_and_usage(self):
        table = CostTable(
            {
                "cheap-model": ModelPrice(0.1, 0.2),
                "hard-model": (1.0, 2.0),  # tuple form accepted
            }
        )
        r = {
            TaskClass.CHEAP: MockLLM(
                "cheap-model", usage={"prompt_tokens": 1000, "completion_tokens": 1000}
            ),
            TaskClass.HARD: MockLLM(
                "hard-model", usage={"prompt_tokens": 1000, "completion_tokens": 1000}
            ),
        }
        router = RoutingLLM(r, cost_table=table)
        router.complete([user("convert mm to inches")])            # cheap: .1 + .2
        router.complete([user("solve the assembly constraints")])  # hard: 1.0 + 2.0
        self.assertAlmostEqual(router.stats()["estimated_spend"], 0.3 + 3.0)

    def test_unknown_model_costs_zero_with_note(self):
        router = RoutingLLM(
            {TaskClass.STANDARD: MockLLM("mystery", usage={"prompt_tokens": 5000, "completion_tokens": 5000})},
            cost_table=CostTable(),  # empty -> everything unpriced
        )
        router.complete([user("make a bracket")])
        s = router.stats()
        self.assertEqual(s["estimated_spend"], 0.0)
        self.assertTrue(s["notes"])
        self.assertIn("mystery", s["notes"][0])


# --- cost estimation -------------------------------------------------------
class TestCostEstimate(unittest.TestCase):
    def test_estimate_monotonic_in_message_size(self):
        table = CostTable({"standard-model": ModelPrice(1.0, 1.0)})
        router = RoutingLLM({TaskClass.STANDARD: MockLLM("standard-model")}, cost_table=table)
        small = [user("bracket")]
        medium = [user("bracket " * 20)]
        large = [system("context " * 100), user("bracket " * 100)]
        e_small = router.estimate(small)
        e_medium = router.estimate(medium)
        e_large = router.estimate(large)
        self.assertLess(e_small, e_medium)
        self.assertLess(e_medium, e_large)
        self.assertGreater(e_small, 0.0)

    def test_estimate_routes_by_class(self):
        table = CostTable(
            {"cheap-model": ModelPrice(0.01, 0.0), "hard-model": ModelPrice(5.0, 0.0)}
        )
        router = RoutingLLM(routes(), cost_table=table)
        msg = "the exact same length of text here...."
        cheap_est = router.estimate([user(msg)], hints={"task_class": TaskClass.CHEAP})
        hard_est = router.estimate([user(msg)], hints={"task_class": TaskClass.HARD})
        # Same message, but the hard route is priced far higher.
        self.assertLess(cheap_est, hard_est)

    def test_costtable_estimate_and_cost_of(self):
        table = CostTable({"m": ModelPrice(2.0, 4.0)})
        self.assertAlmostEqual(table.estimate("m", [user("a" * 4000)]), 2.0)  # ~1000 tok
        cost, note = table.cost_of("m", Usage(1000, 1000))
        self.assertAlmostEqual(cost, 6.0)
        self.assertIsNone(note)
        zero, note2 = table.cost_of("unknown", Usage(1000, 1000))
        self.assertEqual(zero, 0.0)
        self.assertIsNotNone(note2)


# --- helpers ---------------------------------------------------------------
class TestHelpers(unittest.TestCase):
    def test_count_tokens_monotonic(self):
        self.assertEqual(count_tokens(""), 0)
        self.assertLessEqual(count_tokens("ab"), count_tokens("abcdef"))
        self.assertLess(count_tokens("a" * 4), count_tokens("a" * 40))

    def test_messages_tokens_sums(self):
        n = messages_tokens([user("a" * 40), system("b" * 40)])
        self.assertEqual(n, count_tokens("a" * 40) + count_tokens("b" * 40))

    def test_usage_from_result_reads_raw(self):
        res = CompletionResult(text="hi", raw={"usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        u = usage_from_result(res)
        self.assertEqual(u.prompt_tokens, 7)
        self.assertEqual(u.completion_tokens, 3)
        self.assertEqual(u.total_tokens, 10)

    def test_usage_from_result_estimates_completion_when_absent(self):
        u = usage_from_result(CompletionResult(text="a" * 40))
        self.assertEqual(u.prompt_tokens, 0)
        self.assertGreater(u.completion_tokens, 0)

    def test_route_decision_defaults(self):
        d = RouteDecision(task_class=TaskClass.CHEAP, model="m")
        self.assertEqual(d.fallbacks_taken, 0)
        self.assertEqual(d.cost, 0.0)


if __name__ == "__main__":
    unittest.main()
