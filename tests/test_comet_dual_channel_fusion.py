"""Tests for dual-channel (WHAT/WHEN) rank fusion + graph-aware link expansion."""

import unittest

from harnesscad.domain.library.comet_dual_channel_fusion import (
    ChannelHit,
    dual_channel_retrieve,
    expand_by_links,
    fuse_channels,
    rank_hits,
)


class TestRankHits(unittest.TestCase):
    def test_ranks_ascending_by_distance(self):
        hits = rank_hits({"b": 0.4, "a": 0.1, "c": 0.9})
        self.assertEqual([h.item_id for h in hits], ["a", "b", "c"])
        self.assertEqual([h.rank for h in hits], [0, 1, 2])

    def test_ties_broken_by_id_deterministically(self):
        first = rank_hits({"z": 0.5, "a": 0.5, "m": 0.5})
        second = rank_hits({"m": 0.5, "z": 0.5, "a": 0.5})
        self.assertEqual([h.item_id for h in first], ["a", "m", "z"])
        self.assertEqual([h.item_id for h in first], [h.item_id for h in second])

    def test_negative_rank_rejected(self):
        with self.assertRaises(ValueError):
            ChannelHit(item_id="x", distance=0.1, rank=-1)


class TestFuseChannels(unittest.TestCase):
    def test_trigger_only_match_still_surfaces(self):
        # "washer" is invisible to the summary channel but the trigger channel
        # (WHEN would I need this) ranks it first.
        summary = rank_hits({"bolt": 0.1, "nut": 0.3})
        trigger = rank_hits({"washer": 0.05, "bolt": 0.6})
        fused = fuse_channels(summary, trigger)
        ids = [h.item_id for h in fused]
        self.assertIn("washer", ids)
        self.assertLess(ids.index("washer"), ids.index("nut"))

    def test_alpha_one_ignores_trigger_ranks(self):
        summary = rank_hits({"bolt": 0.1})
        trigger = rank_hits({"washer": 0.0})
        fused = {h.item_id: h.score for h in fuse_channels(summary, trigger, alpha=1.0)}
        # washer contributes no rank mass, only its similarity term.
        self.assertGreater(fused["bolt"], 0.0)
        self.assertAlmostEqual(fused["washer"], 0.4 * 1.0, places=6)
        self.assertGreater(fused["bolt"], 0.0)

    def test_agreement_across_channels_wins(self):
        summary = rank_hits({"a": 0.2, "b": 0.21})
        trigger = rank_hits({"a": 0.2, "c": 0.21})
        fused = fuse_channels(summary, trigger, alpha=0.5)
        self.assertEqual(fused[0].item_id, "a")
        self.assertEqual(fused[0].rank, 0)

    def test_raw_channel_takes_its_weight(self):
        summary = rank_hits({"a": 0.1})
        trigger = rank_hits({"a": 0.1})
        raw = rank_hits({"r": 0.1})
        fused = {h.item_id: h.score for h in fuse_channels(summary, trigger, raw)}
        self.assertIn("r", fused)
        self.assertLess(fused["r"], fused["a"])

    def test_scores_are_monotone_in_rank(self):
        summary = rank_hits({"a": 0.1, "b": 0.2, "c": 0.3})
        trigger = rank_hits({"a": 0.1, "b": 0.2, "c": 0.3})
        fused = fuse_channels(summary, trigger)
        self.assertEqual([h.item_id for h in fused], ["a", "b", "c"])
        self.assertTrue(fused[0].score > fused[1].score > fused[2].score)

    def test_invalid_params(self):
        for kw in ({"alpha": 1.5}, {"raw_weight": 1.0}, {"rrf_blend": -0.1}, {"rrf_k": -1}):
            with self.assertRaises(ValueError):
                fuse_channels(rank_hits({"a": 0.1}), rank_hits({"a": 0.1}), **kw)


class TestExpandByLinks(unittest.TestCase):
    def setUp(self):
        self.links = {
            "bolt": ["washer", "nut"],
            "screw": ["washer"],
            "washer": ["shim"],
            "nut": ["bolt"],
        }
        self.seeds = fuse_channels(
            rank_hits({"bolt": 0.1, "screw": 0.2}),
            rank_hits({"bolt": 0.1, "screw": 0.2}),
        )

    def test_seeds_come_first_and_keep_scores(self):
        out = expand_by_links(self.seeds, self.links)
        self.assertEqual(out[0].item_id, "bolt")
        self.assertEqual(out[0].hop, 0)
        self.assertAlmostEqual(out[0].score, self.seeds[0].score, places=9)

    def test_multi_referenced_neighbour_outranks_single(self):
        out = {h.item_id: h for h in expand_by_links(self.seeds, self.links)}
        # washer is linked from both bolt and screw; nut only from bolt.
        self.assertEqual(out["washer"].hop, 1)
        self.assertEqual(out["nut"].hop, 1)
        self.assertGreater(out["washer"].score, out["nut"].score)

    def test_hop2_reached_and_decayed(self):
        out = {h.item_id: h for h in expand_by_links(self.seeds, self.links)}
        self.assertEqual(out["shim"].hop, 2)
        self.assertEqual(out["shim"].via[-1], "washer")
        self.assertAlmostEqual(out["shim"].score, out["washer"].score * 0.25, places=9)

    def test_cycles_do_not_reintroduce_seeds(self):
        out = expand_by_links(self.seeds, self.links)
        ids = [h.item_id for h in out]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids.count("bolt"), 1)

    def test_dangling_links_skipped_when_known_ids_given(self):
        links = {"bolt": ["ghost", "washer"], "screw": [], "washer": []}
        out = expand_by_links(self.seeds, links, known_ids={"bolt", "screw", "washer"})
        self.assertNotIn("ghost", [h.item_id for h in out])
        self.assertIn("washer", [h.item_id for h in out])

    def test_ranks_are_contiguous(self):
        out = expand_by_links(self.seeds, self.links)
        self.assertEqual([h.rank for h in out], list(range(len(out))))

    def test_bad_decay_rejected(self):
        with self.assertRaises(ValueError):
            expand_by_links(self.seeds, self.links, hop1_decay=1.5)


class TestDualChannelRetrieve(unittest.TestCase):
    def test_end_to_end_pulls_in_mating_part(self):
        out = dual_channel_retrieve(
            {"bolt_m6": 0.1, "gasket": 0.8},
            {"bolt_m6": 0.2, "gasket": 0.9},
            links={"bolt_m6": ["washer_m6"]},
            top_k=1,
        )
        ids = [h.item_id for h in out]
        self.assertEqual(ids[0], "bolt_m6")
        self.assertIn("washer_m6", ids)
        self.assertNotIn("gasket", ids)  # cut by top_k=1 and not linked

    def test_no_links_returns_plain_topk(self):
        out = dual_channel_retrieve({"a": 0.1, "b": 0.2}, {"a": 0.1, "b": 0.2}, top_k=2)
        self.assertEqual([h.item_id for h in out], ["a", "b"])
        self.assertTrue(all(h.hop == 0 for h in out))

    def test_deterministic_across_runs(self):
        kw = dict(links={"a": ["c"], "b": ["c"]}, top_k=2)
        first = [h.to_dict() for h in dual_channel_retrieve(
            {"a": 0.3, "b": 0.3}, {"a": 0.3, "b": 0.3}, **kw)]
        second = [h.to_dict() for h in dual_channel_retrieve(
            {"b": 0.3, "a": 0.3}, {"b": 0.3, "a": 0.3}, **kw)]
        self.assertEqual(first, second)

    def test_top_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            dual_channel_retrieve({"a": 0.1}, {"a": 0.1}, top_k=0)


if __name__ == "__main__":
    unittest.main()
