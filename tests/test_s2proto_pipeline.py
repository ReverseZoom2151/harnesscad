import unittest

from exploration.s2proto_pipeline import (
    SKETCH, TEXT, IMAGE, MESH,
    Stage, Pipeline, modality_rank,
    sketch2prototype_pipeline, direct_sketch_to_3d_pipeline, controlnet_pipeline,
)


class TestModalityRank(unittest.TestCase):
    def test_forward_order(self):
        self.assertLess(modality_rank(SKETCH), modality_rank(TEXT))
        self.assertLess(modality_rank(TEXT), modality_rank(IMAGE))
        self.assertLess(modality_rank(IMAGE), modality_rank(MESH))

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            modality_rank("hologram")


class TestValidation(unittest.TestCase):
    def test_canonical_is_valid(self):
        p = sketch2prototype_pipeline()
        self.assertEqual(p.validate(), [])
        self.assertTrue(p.is_valid())

    def test_empty_invalid(self):
        p = Pipeline([])
        self.assertIn("no stages", p.validate()[0])

    def test_must_start_at_sketch(self):
        p = Pipeline([Stage(TEXT), Stage(IMAGE)])
        self.assertTrue(any("start at the sketch" in m for m in p.validate()))

    def test_backward_modality_flagged(self):
        p = Pipeline([Stage(SKETCH), Stage(IMAGE), Stage(TEXT)])
        self.assertTrue(any("does not advance" in m for m in p.validate()))

    def test_repeated_modality_flagged(self):
        p = Pipeline([Stage(SKETCH), Stage(TEXT), Stage(TEXT)])
        self.assertTrue(any("does not advance" in m for m in p.validate()))

    def test_bad_fanout_flagged(self):
        p = Pipeline([Stage(SKETCH), Stage(TEXT, fanout=0)])
        self.assertTrue(any("fanout must be >= 1" in m for m in p.validate()))

    def test_unknown_modality_flagged(self):
        p = Pipeline([Stage(SKETCH), Stage("blob")])
        self.assertTrue(any("unknown modality" in m for m in p.validate()))


class TestClassify(unittest.TestCase):
    def test_sketch2prototype(self):
        self.assertEqual(sketch2prototype_pipeline().classify(), "sketch2prototype")

    def test_direct(self):
        self.assertEqual(direct_sketch_to_3d_pipeline().classify(), "direct_sketch_to_3d")

    def test_controlnet(self):
        self.assertEqual(controlnet_pipeline().classify(), "controlnet")

    def test_text_intermediary_present(self):
        self.assertTrue(sketch2prototype_pipeline().has_text_intermediary())
        self.assertFalse(direct_sketch_to_3d_pipeline().has_text_intermediary())

    def test_other(self):
        p = Pipeline([Stage(SKETCH), Stage(TEXT)])
        self.assertEqual(p.classify(), "other")


class TestArtifactCounts(unittest.TestCase):
    def test_total_artifacts_default(self):
        # 1 * 1 * 4 * 1 = 4 terminal meshes
        self.assertEqual(sketch2prototype_pipeline(n_images=4).total_artifacts(), 4)

    def test_total_artifacts_custom(self):
        self.assertEqual(sketch2prototype_pipeline(n_images=3).total_artifacts(), 3)

    def test_artifact_counts_progression(self):
        counts = sketch2prototype_pipeline(n_images=4).artifact_counts()
        self.assertEqual(counts[SKETCH], 1)
        self.assertEqual(counts[TEXT], 1)
        self.assertEqual(counts[IMAGE], 4)
        self.assertEqual(counts[MESH], 4)

    def test_selected_mesh_metadata(self):
        p = sketch2prototype_pipeline(n_images=4, n_meshes=2)
        self.assertEqual(p.stage_for(MESH).params["selected"], 2)

    def test_builders_validate_args(self):
        with self.assertRaises(ValueError):
            sketch2prototype_pipeline(n_images=0)
        with self.assertRaises(ValueError):
            controlnet_pipeline(n_images=0)


class TestFeedbackInjection(unittest.TestCase):
    def test_inject_returns_new_pipeline(self):
        p = sketch2prototype_pipeline()
        p2 = p.inject_feedback("made of wood and styled like an old saloon")
        self.assertIsNot(p, p2)
        self.assertEqual(p.feedback_history(), ())
        self.assertEqual(len(p2.feedback_history()), 1)

    def test_feedback_accumulates_in_order(self):
        p = sketch2prototype_pipeline()
        p = p.inject_feedback("make it wooden")
        p = p.inject_feedback("add a handle")
        self.assertEqual(p.feedback_history(), ("make it wooden", "add a handle"))

    def test_empty_feedback_raises(self):
        with self.assertRaises(ValueError):
            sketch2prototype_pipeline().inject_feedback("   ")

    def test_no_text_stage_raises(self):
        with self.assertRaises(ValueError):
            direct_sketch_to_3d_pipeline().inject_feedback("anything")


if __name__ == "__main__":
    unittest.main()
