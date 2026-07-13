"""Tests for the design-space exploration layer (docs/blueprint.md sec.12/sec.4).

Everything is deterministic and dependency-free. Variants are built with real CISP
``Op`` objects applied through a ``StubBackend``-backed ``HarnessSession`` (so
``result`` is a genuine ``ApplyOpsResult``), or with lightweight fakes where only the
verifier-visible surface (``ok`` / ``diagnostics`` / ``applied``) matters. No LLM, no
network, no geometry kernel.
"""

import unittest
from typing import List, Optional

from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.cisp.ops import Op, NewSketch, AddRectangle, AddCircle, Extrude
from harnesscad.core.cisp.protocol import ApplyOpsResult
from harnesscad.core.loop import HarnessSession
from harnesscad.eval.verifiers.verify import Diagnostic, Severity

from harnesscad.agents.exploration import (
    EloRating,
    Leaderboard,
    Variant,
    compare,
    debate,
    cluster_variants,
    cluster_representatives,
    op_signature,
    jaccard,
    EloTournament,
    evolve,
    explore,
    ExplorationResult,
)


# --- fixtures ---------------------------------------------------------------
def plate_ops(w: float = 20.0, h: float = 10.0, d: float = 5.0) -> List[Op]:
    """A plan the StubBackend accepts + verifies: sketch -> rectangle -> extrude."""
    return [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0, y=0, w=w, h=h),
        Extrude(sketch="sk1", distance=d),
    ]


def disc_ops(r: float = 8.0, d: float = 4.0) -> List[Op]:
    """A structurally distinct plan: sketch -> circle -> extrude."""
    return [
        NewSketch(plane="XY"),
        AddCircle(sketch="sk1", cx=0, cy=0, r=r),
        Extrude(sketch="sk1", distance=d),
    ]


def bad_ref_ops() -> List[Op]:
    """A plan the backend BLOCKS: extrude references a sketch that never existed."""
    return [Extrude(sketch="nope", distance=5.0)]


def build_variant(vid: str, ops: List[Op], **kw) -> Variant:
    """Apply ``ops`` through a fresh stub session and wrap as an evaluated Variant."""
    session = HarnessSession(StubBackend())
    result = session.apply_ops(ops)
    return Variant(id=vid, ops=ops, result=result, **kw)


def fake_variant(vid: str, ok: bool, n_diags: int = 0, applied: int = 0,
                 ops: Optional[List[Op]] = None) -> Variant:
    """A Variant with a synthetic ApplyOpsResult — for pure ranking tests."""
    diags = [Diagnostic(Severity.ERROR, "x", "m") for _ in range(n_diags)]
    res = ApplyOpsResult(ok=ok, applied=applied, digest="d", diagnostics=diags)
    return Variant(id=vid, ops=ops or [], result=res)


# --- Elo math ---------------------------------------------------------------
class TestElo(unittest.TestCase):
    def test_expected_symmetric_sums_to_one(self):
        elo = EloRating()
        self.assertAlmostEqual(elo.expected(1200, 1400) + elo.expected(1400, 1200), 1.0)

    def test_equal_ratings_expected_half(self):
        self.assertAlmostEqual(EloRating().expected(1500, 1500), 0.5)

    def test_winner_gains_loser_loses_conserved(self):
        elo = EloRating(k=32)
        nw, nl = elo.update(1200, 1200)
        self.assertGreater(nw, 1200)
        self.assertLess(nl, 1200)
        # A decisive result conserves total rating.
        self.assertAlmostEqual(nw + nl, 2400.0)
        # Even odds => each moves by exactly k/2.
        self.assertAlmostEqual(nw, 1216.0)
        self.assertAlmostEqual(nl, 1184.0)

    def test_upset_moves_more_than_expected_win(self):
        elo = EloRating(k=32)
        # Underdog (1200) beating favourite (1600) gains more than the reverse.
        under_up, _ = elo.update(1200, 1600)
        fav_up, _ = elo.update(1600, 1200)
        self.assertGreater(under_up - 1200, fav_up - 1600)

    def test_draw_moves_ratings_toward_each_other(self):
        elo = EloRating(k=32)
        na, nb = elo.update_draw(1400, 1200)
        self.assertLess(na, 1400)   # higher-rated slips on a draw
        self.assertGreater(nb, 1200)  # lower-rated climbs
        self.assertAlmostEqual(na + nb, 2600.0)  # conserved


class TestLeaderboard(unittest.TestCase):
    def test_lazy_base_rating(self):
        lb = Leaderboard(base=1200)
        self.assertEqual(lb.rating("unseen"), 1200)

    def test_record_and_rank(self):
        lb = Leaderboard()
        lb.record("a", "b")   # a beats b
        lb.record("a", "c")   # a beats c
        ranked = lb.rank()
        self.assertEqual(ranked[0][0], "a")
        self.assertGreater(lb.rating("a"), lb.rating("b"))

    def test_rank_tie_break_by_id(self):
        lb = Leaderboard()
        lb.add("b")
        lb.add("a")
        # Both at base; ties resolve to id ascending.
        self.assertEqual([cid for cid, _ in lb.rank()], ["a", "b"])


# --- deterministic debate comparator ----------------------------------------
class TestDebate(unittest.TestCase):
    def test_prefers_ok_over_failed(self):
        a = fake_variant("a", ok=True, applied=3)
        b = fake_variant("b", ok=False, n_diags=1)
        self.assertEqual(compare(a, b), 1)
        self.assertIs(debate(a, b), a)

    def test_prefers_fewer_diagnostics(self):
        a = fake_variant("a", ok=True, n_diags=0, applied=3)
        b = fake_variant("b", ok=True, n_diags=2, applied=3)
        self.assertIs(debate(a, b), a)

    def test_prefers_simpler_op_count_on_full_tie(self):
        a = fake_variant("a", ok=True, applied=3, ops=plate_ops())          # 3 ops
        b = fake_variant("b", ok=True, applied=3, ops=plate_ops() + plate_ops())  # 6 ops
        self.assertIs(debate(a, b), a)

    def test_true_tie_resolves_to_smaller_id(self):
        a = fake_variant("a", ok=True, applied=3, ops=plate_ops())
        b = fake_variant("b", ok=True, applied=3, ops=plate_ops())
        self.assertEqual(compare(a, b), 0)
        self.assertIs(debate(a, b), a)

    def test_injected_judge_overrides_comparator(self):
        # a is objectively worse (failed) but the judge always prefers a.
        a = fake_variant("a", ok=False, n_diags=1)
        b = fake_variant("b", ok=True, applied=3)

        def judge(x, y):
            return 1.0 if x.id == "a" else -1.0
        self.assertIs(debate(a, b, judge=judge), a)

    def test_swap_augmentation_cancels_position_bias(self):
        # A judge that always favours whichever variant is shown FIRST is pure
        # position bias; swap-augmentation must net it to a draw.
        a = fake_variant("a", ok=True, applied=3, ops=plate_ops())
        b = fake_variant("b", ok=True, applied=3, ops=plate_ops())

        def position_biased(x, y):
            return 1.0  # "the first argument is better", always
        self.assertEqual(compare(a, b, judge=position_biased), 0)

    def test_swap_augmentation_averages_a_biased_but_real_signal(self):
        # Judge genuinely prefers b by a wide margin one way, small the other; the
        # average still lands on b.
        a = fake_variant("a", ok=True, applied=3)
        b = fake_variant("b", ok=True, applied=3)

        def judge(x, y):
            # score = +2 when b is first arg, -1 when b is second arg => net favours b
            if x.id == "b":
                return 2.0
            return -1.0
        self.assertEqual(compare(a, b, judge=judge), -1)
        self.assertIs(debate(a, b, judge=judge), b)


# --- clustering -------------------------------------------------------------
class TestClustering(unittest.TestCase):
    def test_identical_ops_cluster_together(self):
        vs = [build_variant("a", plate_ops()), build_variant("b", plate_ops())]
        clusters = cluster_variants(vs, threshold=0.9)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0].members), 2)
        self.assertEqual(vs[0].cluster, vs[1].cluster)

    def test_distinct_ops_separate(self):
        vs = [build_variant("a", plate_ops()), build_variant("b", disc_ops())]
        clusters = cluster_variants(vs, threshold=0.9)
        self.assertEqual(len(clusters), 2)
        self.assertNotEqual(vs[0].cluster, vs[1].cluster)

    def test_near_duplicate_params_cluster(self):
        # Same op structure, slightly different dimensions => high Jaccard on tags,
        # partial on params. With a modest threshold they group.
        vs = [build_variant("a", plate_ops(w=20, h=10)),
              build_variant("b", plate_ops(w=21, h=10))]
        clusters = cluster_variants(vs, threshold=0.4)
        self.assertEqual(len(clusters), 1)

    def test_representative_is_highest_quality(self):
        good = build_variant("good", plate_ops())          # ok
        bad = Variant(id="bad", ops=plate_ops(),
                      result=ApplyOpsResult(False, 0, "d",
                                            [Diagnostic(Severity.ERROR, "e", "m")]))
        clusters = cluster_variants([bad, good], threshold=0.9)
        self.assertEqual(len(clusters), 1)
        self.assertIs(clusters[0].representative, good)

    def test_signature_and_jaccard_helpers(self):
        self.assertEqual(op_signature(plate_ops()), op_signature(plate_ops()))
        self.assertEqual(jaccard(op_signature(plate_ops()),
                                 op_signature(plate_ops())), 1.0)
        self.assertLess(jaccard(op_signature(plate_ops()),
                                op_signature(disc_ops())), 1.0)


# --- Elo tournament ---------------------------------------------------------
class TestEloTournament(unittest.TestCase):
    def test_clearly_better_variant_ranks_first(self):
        strong = build_variant("strong", plate_ops())       # ok
        weak1 = fake_variant("weak1", ok=False, n_diags=2)
        weak2 = fake_variant("weak2", ok=False, n_diags=3)
        tour = EloTournament([weak1, strong, weak2], seed=1)
        result = tour.run()
        self.assertEqual(result.winner.id, "strong")
        self.assertEqual(result.ranking[0][0], "strong")
        self.assertGreater(result.leaderboard.rating("strong"),
                           result.leaderboard.rating("weak1"))

    def test_scores_are_written_back_to_variants(self):
        strong = build_variant("strong", plate_ops())
        weak = fake_variant("weak", ok=False, n_diags=2)
        EloTournament([strong, weak], seed=0).run()
        self.assertIsNotNone(strong.score)
        self.assertGreater(strong.score, weak.score)

    def test_single_variant_pool_is_a_noop(self):
        only = build_variant("only", plate_ops())
        result = EloTournament([only]).run()
        self.assertEqual(result.winner.id, "only")
        self.assertEqual(result.pairings, [])

    def test_deterministic_across_runs(self):
        def run_ranking():
            vs = [build_variant("a", plate_ops()),
                  fake_variant("b", ok=False, n_diags=1),
                  build_variant("c", disc_ops())]
            return [cid for cid, _ in EloTournament(vs, seed=7).run().ranking]
        self.assertEqual(run_ranking(), run_ranking())

    def test_swiss_schedule_runs(self):
        vs = [build_variant("a", plate_ops()),
              fake_variant("b", ok=False, n_diags=1),
              build_variant("c", disc_ops()),
              fake_variant("d", ok=False, n_diags=2)]
        result = EloTournament(vs, seed=3, schedule="swiss", rounds=3).run()
        # The ok variants should out-rank the failed ones.
        ids = [cid for cid, _ in result.ranking]
        self.assertLess(ids.index("a"), ids.index("b"))


# --- evolve -----------------------------------------------------------------
class TestEvolve(unittest.TestCase):
    def test_produces_children_from_top_parents(self):
        parents_seen = {}

        def mutator(parents, rng):
            parents_seen["n"] = len(parents)
            # Recombine: child id derived from a deterministically chosen parent.
            p = parents[rng.randrange(len(parents))]
            return Variant(id=f"child-{p.id}", ops=list(p.ops),
                           result=p.result)

        ranked = [fake_variant(x, ok=True, applied=3) for x in ("a", "b", "c", "d")]
        children = evolve(ranked, mutator, top_k=2, n_children=3, seed=5)
        self.assertEqual(len(children), 3)
        self.assertEqual(parents_seen["n"], 2)  # only top_k parents offered

    def test_deterministic_children(self):
        def mutator(parents, rng):
            p = parents[rng.randrange(len(parents))]
            return Variant(id=f"c{p.id}", ops=list(p.ops), result=p.result)
        ranked = [fake_variant(x, ok=True) for x in ("a", "b", "c")]
        ids1 = [c.id for c in evolve(ranked, mutator, seed=9)]
        ids2 = [c.id for c in evolve(ranked, mutator, seed=9)]
        self.assertEqual(ids1, ids2)


# --- explore (the whole loop) ----------------------------------------------
class TestExplore(unittest.TestCase):
    def _generator(self):
        """A deterministic generator: variant 0 is the strong plate, rest are weak.

        Uses the seed only to label ids so successive generations get fresh ids.
        """
        def generate(n, seed):
            out = []
            for i in range(n):
                if i == 0:
                    out.append(build_variant(f"g{seed}-strong{i}", plate_ops()))
                else:
                    out.append(fake_variant(f"g{seed}-weak{i}", ok=False,
                                            n_diags=i, ops=bad_ref_ops()))
            return out
        return generate

    def test_runs_n_rounds_and_returns_winner(self):
        def mutator(parents, rng):
            best = parents[0]
            return build_variant(f"child-{rng.randrange(1000)}", plate_ops())

        res = explore(self._generator(), rounds=2, n=4, seed=0, mutator=mutator)
        self.assertIsInstance(res, ExplorationResult)
        self.assertEqual(len(res.generations), 2)
        self.assertIsNotNone(res.winner)
        self.assertTrue(res.winner.ok)  # the strong plate wins

    def test_clustering_shrinks_the_tournament_pool(self):
        # All-weak generation with identical bad_ref ops => they collapse to one
        # cluster, so the tournament fields a single representative.
        def generate(n, seed):
            return [fake_variant(f"w{i}", ok=False, n_diags=1, ops=bad_ref_ops())
                    for i in range(n)]
        res = explore(generate, rounds=1, n=5, seed=0)
        self.assertEqual(res.generations[0].n_variants, 5)
        self.assertEqual(res.generations[0].n_clusters, 1)

    def test_deterministic_with_fixed_seed(self):
        def mutator(parents, rng):
            return build_variant(f"child-{rng.randrange(1000)}", plate_ops())

        def run():
            return explore(self._generator(), rounds=2, n=4, seed=42,
                           mutator=mutator).winner.id
        self.assertEqual(run(), run())

    def test_single_round_without_mutator(self):
        res = explore(self._generator(), rounds=1, n=3, seed=0)
        self.assertEqual(len(res.generations), 1)
        self.assertTrue(res.winner.ok)

    def test_rejects_zero_rounds(self):
        with self.assertRaises(ValueError):
            explore(self._generator(), rounds=0, n=3, seed=0)


if __name__ == "__main__":
    unittest.main()
