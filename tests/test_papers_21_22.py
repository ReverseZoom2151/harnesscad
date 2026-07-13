"""Tests for procedural CAD patterns, grammar compression, prior caching,
render distribution, camera pruning, lazy scene expansion, shape grammars,
multi-view consistency, and staged render assessment.

Rewritten from bare pytest-style module functions (never collected by
``python -m unittest``) into unittest.TestCase classes.
"""

import unittest

from harnesscad.eval.bench.generative.render_distribution import summarize
from harnesscad.data.dataengine.trace.prior_cache import PriorCache, cache_key
from harnesscad.domain.procedural.patterns import grid, linear, pipe, radial
from harnesscad.domain.procedural.lazy_scene import expand
from harnesscad.domain.procedural.shape_grammar import Production, derive
from harnesscad.eval.quality.perception.camera_pruning import CameraSample, prune
from harnesscad.eval.quality.sequence.grammar_compression import compression
from harnesscad.eval.quality.perception.multiview_consistency import consistency
from harnesscad.eval.quality.perception.render_stages import assess


class CADPatternsTest(unittest.TestCase):
    def test_linear_pattern_emits_one_instance_per_count(self):
        self.assertEqual(len(linear(3, 2)), 3)

    def test_grid_pattern_emits_the_full_cartesian_product(self):
        self.assertEqual(len(grid(2, 2, (1, 1))), 4)

    def test_radial_pattern_emits_one_instance_per_division(self):
        self.assertEqual(len(radial(4, 1)), 4)

    def test_pipe_over_a_two_point_path_emits_one_segment(self):
        self.assertEqual(len(pipe(((0, 0, 0), (1, 0, 0)))), 1)


class GrammarCompressionTest(unittest.TestCase):
    def test_repeated_symbol_reports_reuse(self):
        self.assertEqual(compression([1, 2], {1})["reuse"], 2)


class PriorCacheTest(unittest.TestCase):
    def test_put_then_get_returns_the_stored_value(self):
        cache = PriorCache()
        key = cache_key(prompt="x")
        cache.put(key, 1)
        self.assertEqual(cache.get(key)[0], 1)


class RenderDistributionTest(unittest.TestCase):
    def test_summarize_counts_every_record(self):
        records = [{"id": "a", "quality": 1, "prompt": 1, "feature": 0}]
        summary = summarize(records, lambda a, b: abs(a - b))
        self.assertEqual(summary["count"], 1)


class CameraPruningTest(unittest.TestCase):
    def test_single_sample_within_budget_is_kept_and_nothing_is_rejected(self):
        accepted, rejected = prune([CameraSample("a", 0, 0, 0)], 1, 1, 1)
        self.assertEqual(len(accepted), 1)
        self.assertFalse(rejected)


class LazySceneTest(unittest.TestCase):
    def test_expansion_stops_at_the_frontier_and_counts_visited_nodes(self):
        out, _, stats = expand([1], lambda x: True,
                               lambda x: () if x > 2 else (x + 1,),
                               lambda x: str(x))
        self.assertEqual(out, (3,))
        self.assertEqual(stats["visited"], 3)


class ShapeGrammarTest(unittest.TestCase):
    def test_derivation_reaches_a_terminal(self):
        self.assertTrue(derive("S", [Production("S", ("T",))], {"T"})[0])


class MultiviewConsistencyTest(unittest.TestCase):
    def test_mean_pairwise_distance_is_reported(self):
        result = consistency({"a": 0, "b": 1}, lambda a, b: abs(a - b))
        self.assertEqual(result["mean"], 1)


class RenderStagesTest(unittest.TestCase):
    def test_render_within_tolerance_is_accepted(self):
        result = assess(0, 1, lambda a, b: 1, lambda a, b: .1, .2)
        self.assertTrue(result["accepted"])


if __name__ == "__main__":
    unittest.main()
