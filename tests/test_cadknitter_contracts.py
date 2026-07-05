import unittest

from bench.compositional_metrics import aggregate, evaluate_sample, slice_metrics
from bench.contact_heatmap import contact_heatmap
from dataengine.assembly_caption_workflow import caption_assembly
from dataengine.assembly_pair_record import (
    AssemblyPairRecord, audit_pairs, reverse_pair,
)
from dataengine.knitcad_filters import KnitLimits, filter_record, rejection_distribution
from exploration.guided_contact_search import guided_step, pareto_evidence
from ingest.assembly_normalization import fit_condition_transform
from ingest.contact_faces import contact_evidence
from quality.assembly_interaction import classify_interactions
from quality.contact_correspondence import assign
from quality.contact_objective import (
    bbox_geometry_cost, edge_shape_cost, position_cost, scheduled_weights,
)


class CADKnitterContractsTests(unittest.TestCase):
    def test_directed_pairs_reverse_duplicates_and_leakage(self):
        record = AssemblyPairRecord(
            "a", "source", "nut", "bolt", ("nf",), ("bf",), "add bolt",
            "train", {"scale": 1}, "dataset", "cc")
        reverse = reverse_pair(record, "b")
        self.assertEqual((reverse.condition_id, reverse.target_id), ("bolt", "nut"))
        duplicate = AssemblyPairRecord(
            "c", "source2", "nut", "bolt", ("nf",), ("bf",), "same",
            "test", {}, "dataset", "cc")
        audit = audit_pairs((record, reverse, duplicate))
        self.assertEqual(audit["duplicates"], (("a", "c"),))
        self.assertTrue(audit["split_leakage"])

    def test_bidirectional_contact_evidence(self):
        faces = {
            "left": (((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (1, 0, 0))),
            "right": (((0.05, 0, 0), (-1, 0, 0)),),
        }
        def projector(target, point):
            return target[0][0], target[0][1]
        result = contact_evidence(
            "l", faces["left"], "r", faces["right"],
            sampler=lambda face: face, projector=projector,
            tolerance=.1, min_support=1)
        self.assertTrue(result.contact)
        self.assertGreaterEqual(result.left_support, 1)
        self.assertGreaterEqual(result.right_support, 1)

    def test_stable_rectangular_assignment_and_ambiguity(self):
        result = assign(((1, 1), (1, 1)))
        self.assertEqual(result.pairs, ((0, 0, 1.0), (1, 1, 1.0)))
        self.assertFalse(result.ambiguous)
        rectangular = assign(((1,), (2,)))
        self.assertEqual(rectangular.pairs, ((0, 0, 1.0),))
        self.assertEqual(rectangular.unmatched_left, (1,))
        self.assertTrue(rectangular.ambiguous)

    def test_contact_objectives_and_schedule(self):
        self.assertEqual(position_cost(((1, 0, 0),), "face",
                                       lambda point, face: point[0]), 1)
        self.assertEqual(edge_shape_cost(((0, 0), (1, 0)),
                                         ((0, 0), (1, 0))), 0)
        candidate = ({"center": (0, 0, 0), "dimensions": (1, 2, 3)},)
        self.assertEqual(bbox_geometry_cost(candidate, candidate), 0)
        self.assertEqual(scheduled_weights(.8), (1, 1))
        self.assertEqual(scheduled_weights(.2), (0, 0))

    def test_guided_validity_first_search_and_pareto(self):
        selected, rows = guided_step(
            0, 5, guidance_steps=(5,), neighbors=lambda current, step: (1, 2, 3),
            is_valid=lambda value: value != 1,
            geometry=lambda value: {1: 0, 2: 1, 3: .5}[value],
            regularization=lambda value, current: {1: 0, 2: 0, 3: 1}[value],
            omega=1)
        self.assertEqual(selected, 2)
        self.assertEqual({row.value for row in pareto_evidence(rows)}, {2, 3})
        unchanged, evidence = guided_step(
            4, 1, guidance_steps=(5,), neighbors=lambda *_: (),
            is_valid=lambda _: True, geometry=lambda _: 0,
            regularization=lambda *_: 0)
        self.assertEqual((unchanged, evidence), (4, ()))

    def test_metrics_interactions_normalization_and_filters(self):
        valid = evaluate_sample(valid=True, cd=.2, intersection_volume=2,
                                condition_volume=10, proximity=.1)
        invalid = evaluate_sample(valid=False, cd=.1, intersection_volume=1,
                                  condition_volume=10, proximity=.1)
        report = aggregate((valid, invalid))
        self.assertEqual(report["vr"], .5)
        self.assertEqual(report["iv"], .2)
        slices = slice_metrics((
            {"contact_count": 1, "mapping_kind": "one_to_one", "metrics": valid},
            {"contact_count": 2, "mapping_kind": "ambiguous", "metrics": invalid},
        ))
        self.assertEqual(slices[(1, "one_to_one")]["pr"], .1)
        interactions = classify_interactions((
            {"faces": ("a", "b"), "distance": 0},
            {"faces": ("c", "d"), "distance": -1},
            {"faces": ("e", "f"), "distance": .2},
        ), allowed_contacts=(("a", "b"),), minimum_clearance=.5)
        self.assertEqual([x["classification"] for x in interactions],
                         ["allowed_contact", "forbidden_collision",
                          "clearance_violation"])
        transform = fit_condition_transform(((-2, -1, 0), (2, 1, 2)))
        point = (1, .5, 1)
        for left, right in zip(transform.invert(transform.apply(point)), point):
            self.assertAlmostEqual(left, right)
        bad = {"faces": 71, "contacts": 11, "max_edges_per_face": 41, "solids": 2}
        self.assertEqual(len(filter_record(bad)), 4)
        self.assertEqual(sum(rejection_distribution((bad,)).values()), 4)

    def test_heatmap_and_caption_workflow(self):
        heatmap = contact_heatmap(((0.1, 0.1), (0.2, 0.2), (1.1, 0)), bin_size=1)
        self.assertEqual(heatmap[(0, 0)], 2)
        caption = caption_assembly(
            "nut", "bolt", describe=lambda x: f"a {x}",
            fuse=lambda condition, target: f"generate {target} for {condition}")
        self.assertTrue(caption.consistent)
        self.assertIn("bolt", caption.prompt)
        self.assertIn("nut", caption.reverse_prompt)


if __name__ == "__main__":
    unittest.main()
