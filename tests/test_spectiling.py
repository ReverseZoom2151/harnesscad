"""Tests for the Design-Specification Tiling (DST) modules (paper index 68).

Covers:
  * context/spectiling_components.py  -- n-gram extraction, weighting, ratio
  * rag/spectiling_greedy.py          -- greedy submodular selection (Alg. 1)
  * spec/spectiling_decompose.py      -- spec -> tiles + dependency ordering
  * context/spectiling_prompt.py      -- per-tile selection + ICL prompt
  * spec/spectiling_coverage.py       -- global + per-tile coverage metrics
  * dataengine/spectiling_complexity.py -- Easy/Middle/Hard stratification
"""

import unittest

from harnesscad.agents.context.spectiling_components import (
    ComponentSet,
    ngrams,
    tiling_ratio,
    tokenize,
    union_components,
    weighted_size,
)
from harnesscad.agents.rag.spectiling_greedy import (
    dst_select,
    marginal_gain,
    uncovered_components,
)
from harnesscad.domain.spec.spectiling_decompose import (
    build_dependencies,
    decompose_spec,
    ordered_tiles,
    resolve_order,
)
from harnesscad.agents.context.spectiling_prompt import (
    Exemplar,
    assemble_prompt,
    build_icl_prompt,
    select_for_query,
    select_per_tile,
)
from harnesscad.domain.spec.spectiling_coverage import coverage_report
from harnesscad.agents.generation.spectiling_compose import TileFragment, compose_fragments
from harnesscad.data.dataengine.curation.spectiling_complexity import (
    ComplexitySample,
    Tier,
    complexity_scores,
    partition,
    tier_counts,
)


class TestComponents(unittest.TestCase):
    def test_tokenize_lowercases_and_splits(self):
        self.assertEqual(
            tokenize("A Cylinder, with 2 Holes!"),
            ["a", "cylinder", "with", "2", "holes"],
        )

    def test_ngrams_sliding_window(self):
        toks = ["a", "b", "c", "d"]
        self.assertEqual(ngrams(toks, 2), [("a", "b"), ("b", "c"), ("c", "d")])
        self.assertEqual(ngrams(toks, 4), [("a", "b", "c", "d")])
        self.assertEqual(ngrams(toks, 5), [])  # shorter than window

    def test_ngram_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            ngrams(["a"], 0)

    def test_weighted_size_weights_by_granularity(self):
        # "a b c d" with granularities {2,4}: three 2-grams, one 4-gram.
        cs = ComponentSet.from_text("a b c d", granularities=(2, 4))
        self.assertEqual(len(cs.at(2)), 3)
        self.assertEqual(len(cs.at(4)), 1)
        # w = 2*3 + 4*1 = 10
        self.assertEqual(weighted_size(cs), 10)

    def test_tiling_ratio_bounds(self):
        q = ComponentSet.from_text("cylinder with a hole", granularities=(2,))
        # Identical exemplar tiles the query fully.
        self.assertAlmostEqual(tiling_ratio(q, q), 1.0)
        # Disjoint exemplar tiles nothing.
        other = ComponentSet.from_text("red green blue", granularities=(2,))
        self.assertEqual(tiling_ratio(other, q), 0.0)

    def test_empty_query_ratio_is_zero(self):
        empty = ComponentSet.empty(granularities=(2,))
        some = ComponentSet.from_text("a b", granularities=(2,))
        self.assertEqual(tiling_ratio(some, empty), 0.0)

    def test_union_algebra(self):
        a = ComponentSet.from_text("a b", granularities=(2,))
        b = ComponentSet.from_text("b c", granularities=(2,))
        u = union_components([a, b])
        self.assertEqual(len(u.at(2)), 2)  # (a,b) and (b,c)

    def test_deterministic(self):
        c1 = ComponentSet.from_text("a bracket with two holes")
        c2 = ComponentSet.from_text("a bracket with two holes")
        self.assertEqual(c1, c2)


class TestGreedySelection(unittest.TestCase):
    def setUp(self):
        # Query needs coverage of three disjoint features.
        self.query = ComponentSet.from_text(
            "cylinder hole bracket slot triangular base", granularities=(2,)
        )

    def test_marginal_gain_diminishes(self):
        exA = ComponentSet.from_text("cylinder hole", granularities=(2,))
        empty = ComponentSet.empty(granularities=(2,))
        g_first = marginal_gain(empty, exA, self.query)
        # After covering with exA, adding exA again yields zero gain.
        covered = empty.union(exA.intersection(self.query))
        g_again = marginal_gain(covered, exA, self.query)
        self.assertGreater(g_first, 0)
        self.assertEqual(g_again, 0)

    def test_greedy_prefers_complementary_over_redundant(self):
        # Two similar exemplars covering the same feature, one complementary.
        ex0 = ComponentSet.from_text("cylinder hole", granularities=(2,))
        ex1 = ComponentSet.from_text("cylinder hole", granularities=(2,))
        ex2 = ComponentSet.from_text("bracket slot", granularities=(2,))
        sel = dst_select(self.query, [ex0, ex1, ex2], k=2)
        # Should pick one of the redundant pair plus the complementary ex2.
        self.assertIn(2, sel.indices)
        self.assertEqual(len(sel.indices), 2)
        self.assertGreater(sel.tiling_ratio, 0.0)

    def test_early_termination_no_positive_gain(self):
        q = ComponentSet.from_text("alpha beta", granularities=(2,))
        useless = ComponentSet.from_text("x y", granularities=(2,))
        sel = dst_select(q, [useless, useless], k=2)
        self.assertEqual(sel.indices, [])  # nothing tiles -> stop immediately

    def test_k_clamped_to_pool(self):
        ex = [ComponentSet.from_text("cylinder hole", granularities=(2,))]
        sel = dst_select(self.query, ex, k=5)
        self.assertEqual(len(sel.indices), 1)

    def test_tie_break_lowest_index(self):
        ex0 = ComponentSet.from_text("cylinder hole", granularities=(2,))
        ex1 = ComponentSet.from_text("cylinder hole", granularities=(2,))
        sel = dst_select(self.query, [ex0, ex1], k=1)
        self.assertEqual(sel.indices, [0])

    def test_uncovered_reporting(self):
        ex = ComponentSet.from_text("cylinder hole", granularities=(2,))
        missing = uncovered_components(self.query, [ex])
        # "bracket slot" / "triangular base" remain uncovered at n=2.
        joined = missing.get(2, [])
        self.assertTrue(any("bracket" in m or "slot" in m for m in joined))

    def test_negative_k_raises(self):
        with self.assertRaises(ValueError):
            dst_select(self.query, [], k=-1)


class TestDecompose(unittest.TestCase):
    def test_splits_sentences_and_clauses(self):
        spec = "A base plate. A cylinder, and a hole through the top."
        tiles = decompose_spec(spec)
        texts = [t.text for t in tiles]
        self.assertIn("A base plate", texts)
        self.assertTrue(any("cylinder" in t for t in texts))
        self.assertTrue(any("hole" in t for t in texts))
        self.assertGreaterEqual(len(tiles), 3)

    def test_ids_are_sequential(self):
        tiles = decompose_spec("One. Two. Three.")
        self.assertEqual([t.id for t in tiles], [0, 1, 2])

    def test_empty_spec_yields_no_tiles(self):
        self.assertEqual(decompose_spec("   \n  "), [])

    def test_dependency_on_earlier_shared_noun(self):
        spec = "A base plate. A cylinder mounted on the base plate."
        tiles = decompose_spec(spec)
        deps = build_dependencies(tiles)
        # The mounted cylinder (id 1) shares 'base'/'plate' and has a cue.
        self.assertIn(0, deps[1])

    def test_resolve_order_is_topological_and_stable(self):
        spec = "A cylinder mounted on the base plate. A base plate."
        tiles = decompose_spec(spec)
        order = resolve_order(tiles)
        # Independent tiles: order must be a permutation of all ids.
        self.assertEqual(sorted(order), [t.id for t in tiles])

    def test_dependency_pushes_prerequisite_first(self):
        # Tile 0 references tile 1's feature; dependency should reorder.
        spec = "A gusset attached to the flange. A flange."
        tiles = decompose_spec(spec)
        ot = ordered_tiles(tiles)
        ids = [t.id for t in ot]
        # flange (id 1) must come before the gusset that depends on it (id 0).
        self.assertLess(ids.index(1), ids.index(0))


class TestPromptAssembly(unittest.TestCase):
    def setUp(self):
        self.db = [
            Exemplar("a cylinder with a hole", "cq.Workplane().circle(1)"),
            Exemplar("a rectangular bracket with a slot", "cq.Workplane().rect(2,1)"),
            Exemplar("a triangular base", "cq.Workplane().polygon(3,1)"),
        ]

    def test_select_for_query_returns_indices(self):
        idx = select_for_query(
            "cylinder with a hole and a bracket slot", self.db, k=2
        )
        self.assertTrue(set(idx) <= {0, 1, 2})
        self.assertLessEqual(len(idx), 2)

    def test_assemble_prompt_structure(self):
        prompt = assemble_prompt("a cylinder", self.db, [0])
        self.assertIn("System Prompt:", prompt)
        self.assertIn("Instruction:", prompt)
        self.assertIn("#Examples Begin:", prompt)
        self.assertIn("#Examples End", prompt)
        self.assertIn("User Input:", prompt)
        self.assertIn("a cylinder with a hole", prompt)  # exemplar spec
        self.assertIn("```python", prompt)
        # Query appears last under User Input.
        self.assertTrue(prompt.rstrip().endswith("Description: a cylinder"))

    def test_build_icl_prompt_whole_query(self):
        prompt, idx = build_icl_prompt(
            "cylinder with a hole", self.db, k=1
        )
        self.assertEqual(len(idx), 1)
        self.assertIn("#Examples Begin:", prompt)

    def test_per_tile_selection_covers_multiple_features(self):
        spec = "A cylinder with a hole. A rectangular bracket with a slot."
        idx = select_per_tile(spec, self.db, k_per_tile=1, total_budget=3)
        # Should pull exemplars for both distinct features.
        self.assertIn(0, idx)
        self.assertIn(1, idx)

    def test_per_tile_dedupes_and_respects_budget(self):
        spec = "A cylinder with a hole. Another cylinder with a hole."
        idx = select_per_tile(spec, self.db, k_per_tile=2, total_budget=2)
        self.assertLessEqual(len(idx), 2)
        self.assertEqual(len(idx), len(set(idx)))  # no dupes


class TestCoverage(unittest.TestCase):
    def test_global_and_per_tile(self):
        query = "A cylinder with a hole. A rectangular bracket with a slot."
        selected = ["a cylinder with a hole"]
        rep = coverage_report(query, selected)
        # First tile well covered, second tile poorly covered.
        self.assertGreater(rep.global_ratio, 0.0)
        self.assertLessEqual(rep.min_tile_ratio(), rep.mean_tile_ratio() + 1e-9)
        self.assertTrue(len(rep.tiles) >= 2)

    def test_full_coverage_when_exemplar_equals_query(self):
        query = "a cylinder with a hole"
        rep = coverage_report(query, [query])
        self.assertAlmostEqual(rep.global_ratio, 1.0)

    def test_uncovered_tiles_listed(self):
        query = "A cylinder. A completely unrelated widget."
        rep = coverage_report(query, ["a cylinder"])
        # The widget tile should be uncovered.
        self.assertTrue(len(rep.uncovered) >= 1)

    def test_no_exemplars_zero_coverage(self):
        rep = coverage_report("a cylinder with a hole", [])
        self.assertEqual(rep.global_ratio, 0.0)


class TestCompose(unittest.TestCase):
    def test_dedupes_imports_and_orders_bodies(self):
        frags = [
            TileFragment(0, "import cadquery as cq\nbox = cq.Workplane().box(1,1,1)"),
            TileFragment(1, "import cadquery as cq\ncyl = cq.Workplane().circle(1)"),
        ]
        merged = compose_fragments(frags)
        # Import appears exactly once.
        self.assertEqual(merged.count("import cadquery as cq"), 1)
        # Both bodies present with provenance banners in order.
        self.assertIn("# --- tile 0 ---", merged)
        self.assertIn("# --- tile 1 ---", merged)
        self.assertLess(merged.index("box ="), merged.index("cyl ="))

    def test_result_disambiguation_and_union(self):
        frags = [
            TileFragment(0, "result = cq.Workplane().box(1,1,1)"),
            TileFragment(1, "result = cq.Workplane().sphere(1)"),
        ]
        merged = compose_fragments(frags)
        self.assertIn("result_0", merged)
        self.assertIn("result_1", merged)
        self.assertIn("result = result_0 + result_1", merged)

    def test_single_result_not_renamed(self):
        frags = [TileFragment(0, "result = cq.Workplane().box(1,1,1)")]
        merged = compose_fragments(frags)
        self.assertIn("result =", merged)
        self.assertNotIn("result_0", merged)

    def test_deterministic(self):
        frags = [TileFragment(0, "import cq\nx = 1"), TileFragment(1, "y = 2")]
        self.assertEqual(compose_fragments(frags), compose_fragments(frags))


class TestComplexityPartition(unittest.TestCase):
    def _mk(self, n):
        return [
            ComplexitySample(
                sample_id=f"s{i}",
                nl_length=float(i),
                geom=float(i * 2),
                ops=float(i),
            )
            for i in range(n)
        ]

    def test_scores_monotone_with_inputs(self):
        samples = self._mk(6)
        scores = complexity_scores(samples)
        # All dims increase with i -> composite score is non-decreasing.
        for a, b in zip(scores, scores[1:]):
            self.assertLessEqual(a, b)

    def test_partition_three_equal_tiers(self):
        scored = partition(self._mk(9))
        counts = tier_counts(scored)
        self.assertEqual(counts[Tier.EASY], 3)
        self.assertEqual(counts[Tier.MIDDLE], 3)
        self.assertEqual(counts[Tier.HARD], 3)

    def test_lowest_complexity_is_easy_highest_is_hard(self):
        scored = partition(self._mk(9))
        by_id = {s.sample_id: s for s in scored}
        self.assertEqual(by_id["s0"].tier, Tier.EASY)
        self.assertEqual(by_id["s8"].tier, Tier.HARD)

    def test_remainder_never_enlarges_easy(self):
        # n=8 -> base=2, rem=2: easy=2, middle=3, hard=3
        scored = partition(self._mk(8))
        counts = tier_counts(scored)
        self.assertEqual(counts[Tier.EASY], 2)
        self.assertLessEqual(counts[Tier.EASY], counts[Tier.HARD])
        self.assertEqual(sum(counts.values()), 8)

    def test_empty_corpus(self):
        self.assertEqual(partition([]), [])
        self.assertEqual(complexity_scores([]), [])

    def test_degenerate_all_equal(self):
        samples = [
            ComplexitySample(f"s{i}", 5.0, 5.0, 5.0) for i in range(3)
        ]
        scores = complexity_scores(samples)
        self.assertEqual(scores, [0.0, 0.0, 0.0])
        scored = partition(samples)
        self.assertEqual(len(scored), 3)

    def test_determinism(self):
        s = self._mk(7)
        self.assertEqual(
            [x.tier for x in partition(s)], [x.tier for x in partition(s)]
        )


if __name__ == "__main__":
    unittest.main()
