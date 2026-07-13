import unittest

from harnesscad.eval.quality.graph.tag_ontology import ModelTagGraph, Tag, TagOntology


def ontology():
    return TagOntology([
        Tag("feature"),
        Tag("subtractive", "feature", ("cut",)),
        Tag("hole", "subtractive", ("bore", "drilled hole")),
        Tag("pocket", "subtractive"),
        Tag("additive", "feature"),
        Tag("boss", "additive"),
    ])


class OntologyTests(unittest.TestCase):
    def test_aliases_hierarchy_and_children(self):
        tags = ontology()
        self.assertEqual("hole", tags.resolve("Drilled_Hole"))
        self.assertEqual(("subtractive", "feature"), tags.ancestors("bore"))
        self.assertEqual(("hole", "pocket"), tags.children("subtractive"))
        self.assertEqual(
            frozenset({"hole", "subtractive", "feature"}),
            tags.expand(["bore"]),
        )

    def test_missing_parent_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown parent"):
            TagOntology([Tag("hole", "missing")])

    def test_cycle_rejected(self):
        with self.assertRaisesRegex(ValueError, "cycle"):
            TagOntology([Tag("a", "b"), Tag("b", "a")])

    def test_alias_collision_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous alias"):
            TagOntology([Tag("a", aliases=("same",)), Tag("b", aliases=("Same",))])


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = ModelTagGraph(ontology())
        self.graph.add_model("plate", tags=["bore", "pocket"], categories=["machined"])
        self.graph.add_model("bracket", tags=["hole", "boss"], categories=["machined"])
        self.graph.add_model("pin", tags=["boss"], categories=["turned"])
        self.graph.categorize_tag("hole", "manufacturing-feature")

    def test_multi_label_and_heterogeneous_edges(self):
        self.graph.assign_tags("pin", ["bore"])
        edges = self.graph.edges
        self.assertIn(("hole", "pocket"), (
            tuple(sorted(edge.target for edge in edges
                         if edge.source == "plate" and edge.relation == "has-tag")),
        ))
        self.assertTrue(any(edge.relation == "is-a" for edge in edges))
        self.assertTrue(any(
            edge.source_type == "tag" and edge.target_type == "category"
            for edge in edges
        ))

    def test_hierarchy_aware_retrieval_is_deterministic_and_explainable(self):
        results = self.graph.retrieve(["subtractive"])
        self.assertEqual(("plate", "bracket", "pin"), tuple(item.model_id for item in results))
        self.assertIn("ancestor-level", results[0].explanation)
        exact = self.graph.retrieve(["drilled hole"], category="machined")
        self.assertEqual(("plate", "bracket"), tuple(item.model_id for item in exact))
        self.assertEqual(("hole",), exact[0].exact_tags)

    def test_retrieval_limit_and_missing_evidence(self):
        result = self.graph.retrieve(["pocket"], limit=1)[0]
        self.assertEqual("plate", result.model_id)
        self.assertEqual((), result.missing_query_tags)

    def test_frequent_motifs_integer_support(self):
        self.graph.assign_tags("pin", ["hole"])
        motifs = self.graph.frequent_motifs(min_support=2)
        hole_boss = next(item for item in motifs if item.tags == ("boss", "hole"))
        self.assertEqual(2, hole_boss.support)
        self.assertEqual(("bracket", "pin"), hole_boss.model_ids)
        self.assertIn("2/3", hole_boss.explanation)

    def test_fractional_support_rounds_up(self):
        self.graph.assign_tags("pin", ["hole"])
        motifs = self.graph.frequent_motifs(min_support=0.66)
        self.assertEqual((("boss", "hole"),), tuple(item.tags for item in motifs))

    def test_invalid_support_rejected(self):
        with self.assertRaises(ValueError):
            self.graph.frequent_motifs(min_support=0.0)


if __name__ == "__main__":
    unittest.main()
