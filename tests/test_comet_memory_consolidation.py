"""Tests for the knowledge-store consolidation sweep."""

import unittest

from harnesscad.data.dataengine.comet_memory_consolidation import (
    MemoryNode,
    consolidate,
    find_clusters,
    merge_cluster,
    normalize_tags,
    prune_dangling_links,
    token_jaccard,
)


def _n(nid, summary, **kw):
    return MemoryNode(node_id=nid, summary=summary, **kw)


class TestTokenJaccard(unittest.TestCase):
    def test_identical_is_one(self):
        a = _n("a", "M6 clearance hole is 6.6 mm")
        self.assertAlmostEqual(token_jaccard(a, a), 1.0)

    def test_disjoint_is_zero(self):
        self.assertAlmostEqual(
            token_jaccard(_n("a", "bolt thread pitch"), _n("b", "fillet radius chamfer")), 0.0
        )

    def test_uses_trigger_too(self):
        a = MemoryNode("a", "clearance", trigger="drilling a through hole")
        b = MemoryNode("b", "clearance", trigger="drilling a through hole")
        self.assertAlmostEqual(token_jaccard(a, b), 1.0)

    def test_empty_pair(self):
        self.assertAlmostEqual(token_jaccard(_n("a", ""), _n("b", "")), 1.0)
        self.assertAlmostEqual(token_jaccard(_n("a", ""), _n("b", "x")), 0.0)


class TestFindClusters(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            _n("n1", "M6 clearance hole diameter is 6.6 mm"),
            _n("n2", "M6 clearance hole diameter 6.6 mm"),
            _n("n3", "fillet radius should not exceed wall thickness"),
        ]

    def test_near_duplicates_cluster(self):
        clusters = find_clusters(self.nodes, threshold=0.6)
        self.assertEqual(clusters, [["n1", "n2"]])

    def test_high_threshold_no_clusters(self):
        self.assertEqual(find_clusters(self.nodes, threshold=0.99), [])

    def test_single_link_transitivity(self):
        # a~b and b~c but a!~c: union-find still puts all three together.
        sim = lambda x, y: 1.0 if {x.node_id, y.node_id} in ({"a", "b"}, {"b", "c"}) else 0.0
        nodes = [_n("a", "x"), _n("b", "y"), _n("c", "z")]
        self.assertEqual(find_clusters(nodes, 0.5, similarity=sim), [["a", "b", "c"]])

    def test_order_independent(self):
        rev = list(reversed(self.nodes))
        self.assertEqual(find_clusters(self.nodes, 0.6), find_clusters(rev, 0.6))

    def test_bad_threshold(self):
        with self.assertRaises(ValueError):
            find_clusters(self.nodes, threshold=1.5)


class TestMergeCluster(unittest.TestCase):
    def test_keeper_is_most_important(self):
        a = MemoryNode("a", "s", importance=1, tags=("x",), recall_count=2)
        b = MemoryNode("b", "s", importance=5, tags=("y",), recall_count=3)
        keeper, absorbed = merge_cluster([a, b])
        self.assertEqual(keeper.node_id, "b")
        self.assertEqual(absorbed, ["a"])
        self.assertEqual(keeper.tags, ("x", "y"))
        self.assertEqual(keeper.recall_count, 5)
        self.assertEqual(keeper.importance, 5)

    def test_tiebreak_by_recall_then_length(self):
        a = MemoryNode("a", "short", recall_count=1)
        b = MemoryNode("b", "much longer summary", recall_count=1)
        self.assertEqual(merge_cluster([a, b])[0].node_id, "b")

    def test_links_to_absorbed_and_self_dropped(self):
        a = MemoryNode("a", "s", importance=9, links=("b", "c"))
        b = MemoryNode("b", "s", links=("a", "d"))
        keeper, _ = merge_cluster([a, b])
        self.assertEqual(keeper.node_id, "a")
        self.assertEqual(keeper.links, ("c", "d"))

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            merge_cluster([])


class TestNormalizeTags(unittest.TestCase):
    def test_case_and_substring_variants_collapse(self):
        nodes = [
            MemoryNode("a", "s", tags=("Fastener",)),
            MemoryNode("b", "s", tags=("fastener",)),
            MemoryNode("c", "s", tags=("fasteners",)),
        ]
        out, renames, changed = normalize_tags(nodes)
        canon = {t for n in out for t in n.tags}
        self.assertEqual(canon, {"fastener"})
        self.assertEqual(changed, 2)
        self.assertEqual(renames["fasteners"], "fastener")

    def test_meta_tags_untouched(self):
        nodes = [MemoryNode("a", "s", tags=("ORIGIN:session1", "ORIGIN:session12"))]
        out, renames, changed = normalize_tags(nodes)
        self.assertEqual(renames, {})
        self.assertEqual(out[0].tags, ("ORIGIN:session1", "ORIGIN:session12"))

    def test_unrelated_tags_survive(self):
        nodes = [MemoryNode("a", "s", tags=("bolt", "chamfer"))]
        out, renames, _ = normalize_tags(nodes)
        self.assertEqual(out[0].tags, ("bolt", "chamfer"))
        self.assertEqual(renames, {})


class TestPruneDanglingLinks(unittest.TestCase):
    def test_removes_missing_and_self_links(self):
        nodes = [MemoryNode("a", "s", links=("a", "b", "ghost")), MemoryNode("b", "s")]
        out, pruned = prune_dangling_links(nodes)
        self.assertEqual(pruned, 2)
        self.assertEqual(out[0].links, ("b",))

    def test_noop_leaves_nodes_alone(self):
        nodes = [MemoryNode("a", "s", links=("b",)), MemoryNode("b", "s")]
        out, pruned = prune_dangling_links(nodes)
        self.assertEqual(pruned, 0)
        self.assertEqual(out[0].links, ("b",))


class TestConsolidate(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            MemoryNode("n1", "M6 clearance hole diameter is 6.6 mm",
                       tags=("Fastener",), importance=3, links=("n3",)),
            MemoryNode("n2", "M6 clearance hole diameter 6.6 mm",
                       tags=("fasteners",), importance=1, recall_count=4),
            MemoryNode("n3", "fillet radius should not exceed wall thickness",
                       tags=("fillet",), links=("n2", "ghost")),
        ]

    def test_full_sweep(self):
        out, report = consolidate(self.nodes, threshold=0.6)
        ids = [n.node_id for n in out]
        self.assertEqual(ids, ["n1", "n3"])
        self.assertEqual(report.merged, 1)
        self.assertEqual(report.absorbed_into, {"n2": "n1"})
        keeper = out[0]
        # "fasteners" collapses into the shorter variant present in the store.
        self.assertEqual(keeper.tags, ("Fastener",))
        self.assertEqual(keeper.recall_count, 4)

    def test_links_rewired_to_keeper(self):
        out, _ = consolidate(self.nodes, threshold=0.6)
        n3 = [n for n in out if n.node_id == "n3"][0]
        self.assertEqual(n3.links, ("n1",))  # n2 -> keeper n1, ghost pruned

    def test_report_counts_pruned_links(self):
        _, report = consolidate(self.nodes, threshold=0.6)
        self.assertEqual(report.links_pruned, 1)
        self.assertIn("fasteners", report.tags_renamed)

    def test_deterministic(self):
        a, ra = consolidate(self.nodes, 0.6)
        b, rb = consolidate(list(reversed(self.nodes)), 0.6)
        self.assertEqual([n.to_dict() for n in a], [n.to_dict() for n in b])
        self.assertEqual(ra.to_dict(), rb.to_dict())

    def test_no_duplicates_is_identity(self):
        nodes = [MemoryNode("a", "bolt thread"), MemoryNode("b", "fillet radius")]
        out, report = consolidate(nodes, 0.6)
        self.assertEqual(report.merged, 0)
        self.assertEqual([n.node_id for n in out], ["a", "b"])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            consolidate([MemoryNode("a", "x"), MemoryNode("a", "y")])


if __name__ == "__main__":
    unittest.main()
