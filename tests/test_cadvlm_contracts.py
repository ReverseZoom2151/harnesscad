import unittest

from harnesscad.eval.bench.sketch.cadvlm_metrics import cadvlm_metrics, sliced_metrics
from harnesscad.eval.bench.harness.task_modality_ablation import ablation_matrix, promotion
from harnesscad.data.dataengine.schemas.sketch_constraint_ontology import (
    KINDS, resolve, validate_constraint,
)
from harnesscad.data.dataengine.schemas.sketch_modal_record import SketchModalRecord
from harnesscad.data.datagen.paired_sketch_prefix import paired_prefixes
from harnesscad.data.datagen.sketch_image_conditions import image_conditions
from harnesscad.io.ingest.cadvlm_codec import (
    CONSTRAINT_TOKENS, VERSION, decode_constraint, decode_entity,
    encode_constraint, encode_entity, fit_frame,
)
from harnesscad.eval.quality.sketch.constraint_label_stability import constraint_label_stability
from harnesscad.eval.quality.perception.sketch_crossmodal import crossmodal_consistency


class CadVLMContractsTests(unittest.TestCase):
    def test_versioned_entity_and_constraint_codec(self):
        self.assertEqual(VERSION, "cadvlm-sketch-v1")
        entities = (
            {"type": "line", "start": (-1, 0), "end": (1, 0)},
            {"type": "arc", "start": (-1, 0), "mid": (0, 1), "end": (1, 0)},
            {"type": "circle", "points": ((1, 0), (0, 1), (-1, 0), (0, -1))},
        )
        frame = fit_frame(tuple(p for entity in entities for p in (
            (entity["start"], entity["end"]) if entity["type"] == "line" else
            ((entity["start"], entity["mid"], entity["end"])
             if entity["type"] == "arc" else entity["points"]))))
        for entity in entities:
            encoded = encode_entity(entity, frame)
            self.assertTrue(all(1 <= value <= 64 for value in encoded[1:]))
            decoded = decode_entity(encoded, frame)
            self.assertEqual(decoded["type"], entity["type"])
        constraint = encode_constraint("parallel", (0, 1))
        self.assertEqual(constraint, (73, 0, 1))
        self.assertEqual(decode_constraint(constraint), ("parallel", (0, 1)))
        self.assertEqual(set(CONSTRAINT_TOKENS.values()), set(range(65, 78)))

    def test_modal_record_and_prefix_pairs(self):
        entities = ("line", "circle", "arc", "line2", "circle2")
        pairs = paired_prefixes(entities, renderer=lambda values: "|".join(values))
        self.assertEqual([item.ratio for item in pairs], [.2, .4, .6, .8])
        self.assertTrue(all(item.prefix + item.target == entities for item in pairs))
        record = SketchModalRecord(
            "x", "parent", entities, pairs[1].prefix, "full", "partial",
            .4, (), "mm", "origin", "precise", "train")
        self.assertEqual(record.partial_entities, entities[:2])
        with self.assertRaises(ValueError):
            SketchModalRecord("x", "p", entities, ("circle",), "f", "p",
                              .2, (), "mm", "o", "precise", "train")

    def test_exact_tolerance_and_sliced_metrics(self):
        expected = ((("line", 1.0, 2.0), ("circle", 3.0)),)
        actual = ((("line", 1.001, 2.0), ("arc", 3.0)),)
        exact = cadvlm_metrics(actual, expected)
        tolerant = cadvlm_metrics(actual, expected, tolerance=.01)
        self.assertEqual(exact["entity_accuracy"], 0)
        self.assertEqual(tolerant["entity_accuracy"], 1)
        self.assertEqual(tolerant["sketch_accuracy"], 0)
        rows = ({"ratio": .2, "condition": "precise",
                 "actual": actual[0], "expected": expected[0]},)
        self.assertIn((.2, "precise"), sliced_metrics(rows, tolerance=.01))

    def test_crossmodal_conditions_and_label_stability(self):
        report = crossmodal_consistency(
            ("line",), {(0, 0), (2, 2)},
            rasterizer=lambda primitives: {(0, 0), (1, 1)})
        self.assertEqual(report["iou"], 1/3)
        self.assertEqual(report["missing"], ((1, 1),))
        conditions = image_conditions(
            ("line",), render_precise=lambda p: ("precise", p),
            simulate_hand=lambda image, seed: ("hand", image, seed),
            affine_noise=lambda image, seed: ("noise", image, seed), seed=4)
        self.assertEqual([item.name for item in conditions],
                         ["precise", "hand_drawn", "noisy_hand_drawn"])
        stability = constraint_label_stability(
            (1, 2), {"quantized": lambda values: (1, 1),
                     "same": lambda values: values},
            classify=lambda values: ("equal",) if values[0] == values[1]
            else ("offset",))
        self.assertEqual(stability.flips, ("quantized",))

    def test_constraint_ontology_and_task_ablation(self):
        self.assertEqual(len(KINDS), 13)
        self.assertEqual(resolve("perp").token, 74)
        self.assertEqual(validate_constraint("parallel", (1,)),
                         ("insufficient-references",))
        rows = (
            {"task": "completion", "modalities": ("text",), "objectives": ("lm",),
             "quality": .5, "memory": 10, "latency": 1},
            {"task": "completion", "modalities": ("text", "image"),
             "objectives": ("lm", "contrastive"), "quality": .7,
             "memory": 12, "latency": 2},
        )
        matrix = ablation_matrix(rows)
        baseline = matrix[("completion", ("text",), ("lm",))]
        candidate = matrix[("completion", ("image", "text"),
                            ("contrastive", "lm"))]
        self.assertTrue(promotion(candidate, baseline,
                                  min_gain=.1, max_memory_increase=2)["promoted"])
        self.assertFalse(promotion(candidate, baseline,
                                   min_gain=.1, max_memory_increase=1)["promoted"])


if __name__ == "__main__":
    unittest.main()
