import unittest

from harnesscad.eval.quality.taag import (
    EdgeSense,
    FeatureHypothesis,
    FeatureRecognizer,
    TopologyExtractor,
)


SAMPLE = {
    "vertices": [{"id": "v1", "point": (0, 0, 0)}, {"id": "v2"}],
    "edges": [
        {"id": "e1", "vertices": ["v1", "v2"], "faces": ["f1", "f2"],
         "sense": "concave", "curve": "line"},
        {"id": "e2", "vertices": ["v1"], "faces": ["f1", "f2"],
         "sense": "transitory"},
    ],
    "faces": [
        {"id": "f1", "edges": ["e1", "e2"], "surface": "plane"},
        {"id": "f2", "edges": ["e1", "e2"], "surface": "cylinder"},
    ],
}


class ExtractionTests(unittest.TestCase):
    def test_two_levels_and_attributes(self):
        graph = TopologyExtractor().extract(SAMPLE)
        self.assertEqual(6, len(graph.nodes))
        self.assertEqual("line", graph.node("e1").attributes["curve"])
        self.assertEqual("concave", graph.node("e1").attributes["sense"])
        self.assertEqual(2, len(graph.adjacent_faces("f1")))
        self.assertEqual(
            {EdgeSense.CONCAVE, EdgeSense.TRANSITORY},
            {item.sense for item in graph.face_adjacency},
        )

    def test_unknown_sense_is_preserved_as_unknown(self):
        source = {"edges": [{"id": "e", "sense": "ambiguous"}]}
        graph = TopologyExtractor().extract(source)
        self.assertEqual("unknown", graph.node("e").attributes["sense"])

    def test_rejects_dangling_topology(self):
        with self.assertRaisesRegex(ValueError, "unknown topology"):
            TopologyExtractor().extract({"faces": [{"id": "f", "edges": ["missing"]}]})

    def test_deterministic_order(self):
        shuffled = dict(SAMPLE)
        shuffled["faces"] = list(reversed(SAMPLE["faces"]))
        self.assertEqual(
            TopologyExtractor().extract(SAMPLE),
            TopologyExtractor().extract(shuffled),
        )


class RecognitionTests(unittest.TestCase):
    def test_set_valued_overlapping_hypotheses_with_provenance(self):
        graph = TopologyExtractor().extract(SAMPLE)

        def rule(_graph):
            yield FeatureHypothesis(
                "h-hole", "through_hole", frozenset({"f2", "e1"}), 0.92,
                "cylindrical-hole-v1", ("cylindrical face", "concave rim"),
                {"paper": "paper-3", "rule_version": 1},
            )
            yield FeatureHypothesis(
                "h-pocket", "pocket", frozenset({"f1", "e1"}), 0.71,
                "pocket-v2", ("planar floor", "concave boundary"),
                {"run_id": "r7"},
            )

        results = FeatureRecognizer([rule]).recognize(graph)
        self.assertEqual(2, len(results.hypotheses))
        self.assertEqual("h-pocket", results.overlapping("h-hole")[0].id)
        self.assertEqual(2, len(results.for_topology("e1")))
        self.assertEqual("paper-3", results.hypotheses[0].provenance["paper"])

    def test_boundary_rejects_recognizer_invented_topology(self):
        graph = TopologyExtractor().extract(SAMPLE)

        def bad_rule(_graph):
            return [FeatureHypothesis(
                "bad", "hole", frozenset({"invented"}), 0.5, "bad-rule"
            )]

        with self.assertRaisesRegex(ValueError, "unknown topology"):
            FeatureRecognizer([bad_rule]).recognize(graph)

    def test_confidence_is_validated(self):
        with self.assertRaisesRegex(ValueError, "confidence"):
            FeatureHypothesis("h", "hole", frozenset({"f1"}), 1.1, "rule")

    def test_duplicate_hypothesis_ids_rejected(self):
        graph = TopologyExtractor().extract(SAMPLE)
        hypothesis = FeatureHypothesis(
            "same", "hole", frozenset({"f1"}), 0.5, "rule"
        )
        with self.assertRaisesRegex(ValueError, "duplicate hypothesis"):
            FeatureRecognizer([lambda _graph: [hypothesis, hypothesis]]).recognize(graph)


if __name__ == "__main__":
    unittest.main()
