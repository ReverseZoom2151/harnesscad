import unittest

from harnesscad.eval.bench.judges.feasibility_novelty import (
    mann_whitney, pareto_items, rating_qc, spearman,
)
from harnesscad.eval.bench.geometry.mesh_topology import (
    dangling_edge_length, flux_enclosure_error, segment_error,
    self_intersection_ratio,
)
from harnesscad.eval.bench.data.modality_complementarity import (
    complementarity_delta, eliminate_points, gaussian_noise,
)
from harnesscad.eval.bench.data.modality_coverage_audit import audit_omnicad_splits
from harnesscad.eval.bench.judges.perceived_actual_gap import feasibility_gap
from harnesscad.eval.bench.generative.prompt_similarity import similarity_matrix
from harnesscad.data.dataengine.schemas.prompt_record import CADPromptRecord, audit_prompt_records
from harnesscad.data.dataengine.curation.modality_schedule import (
    combination_balance, modality_combinations, modality_curriculum,
)
from harnesscad.data.dataengine.schemas.multimodal_record import OmniCADRecord, PointNormal, ViewAsset
from harnesscad.data.datagen.command_prefixes import assert_split_before_expand, post_solid_prefixes
from harnesscad.data.datagen.modifier_ablation import ablate_modifiers
from harnesscad.data.datagen.multimodal_capture import DEFAULT_CAMERAS, capture_manifest, choose_views
from harnesscad.agents.exploration.image_prompt_sweep import sweep
from harnesscad.eval.quality.report.stage_policy import recommend_stage
from harnesscad.eval.quality.perception.modality_fusion import fusion_policy
from harnesscad.agents.rag.render_retrieval import retrieve_render


def omni(i="x", split="train", parent="p", text="part", views=True, points=True):
    return OmniCADRecord(
        i, "cmd", parent, text,
        (ViewAsset("front", (0, -1, 0), "img"),) if views else (),
        (PointNormal((0, 0, 0), (0, 0, 1)),) if points else (),
        "mm", "world", split, {"source": "fixture"})


class OmniCADTests(unittest.TestCase):
    def test_record_prefixes_and_split_leakage(self):
        record = omni()
        self.assertEqual(record.modalities, {"text", "image", "point"})
        prefixes = post_solid_prefixes(
            "p", "train", ("sketch", "extrude", "fillet"),
            is_checkpoint=lambda commands: commands[-1] == "extrude")
        self.assertEqual(prefixes[0].commands, ("sketch", "extrude"))
        assert_split_before_expand(prefixes)
        with self.assertRaises(ValueError):
            assert_split_before_expand(prefixes + (
                post_solid_prefixes("p", "test", ("extrude",),
                                    is_checkpoint=lambda _: True)[0],))
        report = audit_omnicad_splits(
            (record, omni("y", "test", "p", views=False)))
        self.assertIn("p", report["lineage_leakage"])
        self.assertEqual(report["missing_modalities"][0][1], ("image",))

    def test_modifier_ablation_and_capture(self):
        commands = ({"op": "box"}, {"op": "fillet"}, {"op": "chamfer"})
        result = ablate_modifiers(
            commands, rebuild=lambda ops: ops,
            topology_complete=lambda shape: not any(x["op"] == "chamfer" for x in shape))
        self.assertEqual([x.retained for x in result], [False, True])
        self.assertEqual(len(DEFAULT_CAMERAS), 8)
        self.assertEqual(choose_views(2, 3), choose_views(2, 3))
        manifest = capture_manifest(
            "shape", renderer=lambda shape, camera: camera.name.encode(),
            point_sampler=lambda shape, count, seed: [((0, 0, 0), (0, 0, 1))],
            count=1, seed=2)
        self.assertEqual(len(manifest["views"]), 8)

    def test_schedule_fusion_and_corruption(self):
        self.assertEqual(modality_curriculum()[-1][1], ("text", "point", "image"))
        combos = modality_combinations({"image", "text"})
        self.assertEqual(combos, (("text",), ("image",), ("text", "image")))
        self.assertEqual(combination_balance(combos)[("text",)], 1)
        decision = fusion_policy(
            {"text": "cube", "point": "cylinder"}, required=("image",),
            conflicts=(("text", "point"),))
        self.assertEqual(decision.route, "manual_review")
        points = ((0, 0, 0), (1, 1, 1), (2, 2, 2))
        self.assertEqual(gaussian_noise(points, 0, 1), points)
        self.assertEqual(len(eliminate_points(points, 2/3, 1)), 1)
        self.assertAlmostEqual(complementarity_delta(.4, .7), .3)

    def test_topology_metrics(self):
        vertices = ((0, 0), (1, 0), (1, 1), (0, 1))
        self.assertEqual(dangling_edge_length(vertices, ((0, 1, 2, 3),)), 4)
        self.assertEqual(segment_error(2, 3), .5)
        self.assertEqual(self_intersection_ratio(4, (1, 1, 2)), .5)
        self.assertEqual(flux_enclosure_error((((1, 0, 0), 2),
                                                ((-1, 0, 0), 2))), 0)


class CADPromptingTests(unittest.TestCase):
    def test_retrieval_sweep_and_similarity(self):
        records = (
            {"id": "b", "image": (0, 1), "verified_feasible": True},
            {"id": "a", "image": (1, 0), "verified_feasible": True},
            {"id": "z", "image": (1, 0), "verified_feasible": False},
        )
        found = retrieve_render(
            "query", records, embed_text=lambda _: (1, 0), embed_image=lambda x: x)
        self.assertEqual(found[0]["id"], "a")
        runs = sweep(
            lambda **kwargs: (kwargs["image_weight"], kwargs["seed"], kwargs["index"]),
            text="bike", image="cad", weights=(.5, 0), seeds=(2, 1),
            outputs_per_setting=2, config={"guidance": 7})
        self.assertEqual(len(runs), 8)
        matrix = similarity_matrix({"a": ((1, 0), (0, 1)), "b": ((1, 0),)},
                                   embed=lambda value: value)
        self.assertEqual(matrix[("b", "b")], None)
        self.assertEqual(matrix[("a", "a")], 0)

    def test_ratings_rank_tests_and_pareto(self):
        self.assertEqual(spearman((1, 2, 3), (3, 2, 1)), -1)
        self.assertEqual(mann_whitney((1, 2), (3, 4))["u"], 0)
        ratings = (
            {"item_id": "x", "rater_id": "a", "feasibility": 6, "novelty": 3},
            {"item_id": "x", "rater_id": "b", "feasibility": 4, "novelty": 5},
        )
        self.assertTrue(rating_qc(ratings)["x"]["accepted"])
        front = pareto_items((
            {"id": "f", "feasibility": 1, "novelty": .5},
            {"id": "n", "feasibility": .5, "novelty": 1},
            {"id": "bad", "feasibility": .2, "novelty": .2},
        ))
        self.assertEqual({x["id"] for x in front}, {"f", "n"})

    def test_calibration_records_and_claim_scope(self):
        evidence = {("concept", "provider", "bike"): (
            {"weight": .2, "objective": .8}, {"weight": .4, "objective": .9})}
        recommendation = recommend_stage("concept", "provider", "bike", evidence)
        self.assertEqual(recommendation.weight, .4)
        with self.assertRaises(LookupError):
            recommend_stage("detail", "provider", "bike", evidence)
        def record(i, split, license="cc"):
            return CADPromptRecord(i, "p", "render", "source", license, "provider",
                                   .5, 1, {}, i, split)
        audit = audit_prompt_records((record("a", "train"),
                                      record("b", "test", "")))
        self.assertEqual(audit["render_leakage"], ("render",))
        self.assertEqual(audit["missing_license"], ("b",))
        self.assertEqual(
            feasibility_gap(perceived=.8, actual_verified=None)["claim_scope"],
            "perceived_only")


if __name__ == "__main__":
    unittest.main()
