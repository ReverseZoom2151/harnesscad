"""Best-of-N is a selectable loop strategy on the SHIPPED path.

`eval/reliability/strategies/best_of_n.py` was written, tested and unreachable:
the harness spent its budget on typed feedback (which lost the A/B by 8.3 points)
while the mechanism that scales with an exact oracle sat orphaned.
"""

from __future__ import annotations

import json
import unittest

from harnesscad.agents.llm.base import CompletionResult
from harnesscad.core import cli, pipeline


GOOD = json.dumps([
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 50, "h": 30},
    {"op": "extrude", "sketch": "sk1", "distance": 6},
])
BAD = json.dumps([{"op": "extrude", "sketch": "sk9", "distance": 6}])


class _ScriptedLLM:
    """Returns BAD for the first `n_bad` calls, then GOOD. Records every brief."""

    def __init__(self, n_bad: int) -> None:
        self.n_bad = n_bad
        self.calls = 0
        self.briefs = []

    def complete(self, messages, tools=None, **kwargs):
        self.calls += 1
        self.briefs.append(messages[-1].content)
        text = BAD if self.calls <= self.n_bad else GOOD
        return CompletionResult(text=text)


class TestStrategySelection(unittest.TestCase):
    def test_the_cli_offers_exactly_the_pipeline_strategies(self):
        self.assertEqual(tuple(cli.BUILD_STRATEGIES), tuple(pipeline.STRATEGIES))

    def test_an_unknown_strategy_is_refused(self):
        with self.assertRaises(pipeline.BuildError):
            pipeline.build("a plate", llm=_ScriptedLLM(0), backend="frep",
                           strategy="mcts")

    def test_refine_is_still_the_default_and_is_labelled(self):
        out = pipeline.build("a 50x30 plate 6mm thick", llm=_ScriptedLLM(0),
                             backend="frep", max_iters=1)
        self.assertEqual(out["strategy"], "refine")
        self.assertTrue(out["ok"])


class TestBestOfN(unittest.TestCase):
    def test_n_bad_draws_then_a_good_one_still_wins(self):
        # The whole point: the verifier SELECTS. Two candidates are unusable and
        # the loop still ships a verified part, having spoken no diagnostic to the
        # model at all -- so there is no poisoning surface.
        llm = _ScriptedLLM(n_bad=2)
        out = pipeline.build("a 50x30 plate 6mm thick", llm=llm, backend="frep",
                             strategy="best-of-n", n=4)
        self.assertTrue(out["ok"])
        self.assertEqual(out["strategy"], "best-of-n")
        self.assertEqual(out["n"], 4)
        self.assertEqual(out["winner_index"], 2)
        self.assertEqual(llm.calls, 4)

    def test_all_candidates_bad_reports_not_ok_and_never_raises(self):
        out = pipeline.build("a plate", llm=_ScriptedLLM(n_bad=99), backend="frep",
                             strategy="best-of-n", n=2)
        self.assertFalse(out["ok"])
        self.assertEqual(len(out["trajectory"]), 2)

    def test_candidates_are_seeded_differently_so_the_draws_can_differ(self):
        llm = _ScriptedLLM(n_bad=0)
        pipeline.build("a plate", llm=llm, backend="frep", strategy="best-of-n", n=3)
        self.assertEqual(len(set(llm.briefs)), 3)

    def test_every_candidate_carries_its_own_per_op_credit(self):
        out = pipeline.build("a 50x30 plate 6mm thick", llm=_ScriptedLLM(n_bad=1),
                             backend="frep", strategy="best-of-n", n=2)
        loser, winner = out["trajectory"]
        self.assertEqual([r["reward"] for r in loser["step_rewards"]], [0.0])
        self.assertEqual([r["reward"] for r in winner["step_rewards"]],
                         [1.0, 1.0, 1.0])

    def test_n_must_be_at_least_one(self):
        with self.assertRaises(pipeline.BuildError):
            pipeline.build("a plate", llm=_ScriptedLLM(0), backend="frep",
                           strategy="best-of-n", n=0)


class TestRefineCarriesProcessSupervision(unittest.TestCase):
    def test_the_result_names_the_op_that_broke_the_trajectory(self):
        out = pipeline.build("a plate", llm=_ScriptedLLM(n_bad=99), backend="frep",
                             max_iters=1)
        self.assertFalse(out["ok"])
        self.assertEqual(out["first_divergence"], 0)
        self.assertEqual(out["step_accuracy"], 0.0)
        self.assertEqual(out["reward"]["outcome"], 0.0)

    def test_a_solved_run_scores_one_on_every_op(self):
        out = pipeline.build("a 50x30 plate 6mm thick", llm=_ScriptedLLM(0),
                             backend="frep", max_iters=1)
        self.assertIsNone(out["first_divergence"])
        self.assertEqual(out["step_accuracy"], 1.0)
        self.assertEqual(out["reward"]["step_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
