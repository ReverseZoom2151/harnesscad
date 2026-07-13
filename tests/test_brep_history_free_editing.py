import unittest

from harnesscad.eval.bench.geometry.edit_metrics import (
    pass_at_k, relation_preservation, retention, symmetric_chamfer,
)
from harnesscad.data.dataengine.edits.brep_edit_annotations import (
    DirectionalAnnotation, validate_annotations,
)
from harnesscad.data.dataengine.edits.edit_complexity import (
    balanced_sample, complexity_bin, complexity_score,
)
from harnesscad.data.datagen.brep_edit_pairs import synthesize_delete_add_pairs
from harnesscad.domain.editing.brep import (
    EditCandidate, FaceDescriptor, canonicalize_faces, generate_candidates,
)
from harnesscad.io.ingest.brep_sequence import BrepEditSequence, BrepEditStep
from harnesscad.io.surfaces.edit_views import best_view, projected_bbox, select_edit_context


FACES = (
    FaceDescriptor((2, 0, 0), (1, 0, 0), 4, "plane", "b"),
    FaceDescriptor((0, 0, 0), (-1, 0, 0), 4, "plane", "a"),
    FaceDescriptor((1, 0, 0), (0, 1, 0), 2, "plane", "c"),
)


class ReverseProvider:
    def propose(self, shape, instruction, faces, k):
        # The provider observes canonical faces, regardless of input order.
        assert tuple(f.source_id for f in faces) == ("c", "a", "b")
        return [
            EditCandidate("bad", "move", {"d": 1}),
            EditCandidate("okay", "move", {"d": 2}),
            EditCandidate("best", "offset", {"d": 3}),
        ]


class HistoryFreeEditingTests(unittest.TestCase):
    def test_face_canonicalization_is_order_independent(self):
        self.assertEqual(canonicalize_faces(FACES), canonicalize_faces(reversed(FACES)))

    def test_candidate_pool_is_validity_first_then_score(self):
        ranked = generate_candidates(
            ReverseProvider(), "source", "edit it", reversed(FACES), 3,
            is_valid=lambda shape: shape != "bad",
            score=lambda shape, _: {"bad": 99, "okay": .4, "best": .9}[shape],
        )
        self.assertEqual([c.shape for c in ranked], ["best", "okay", "bad"])
        self.assertTrue(pass_at_k((c.verifier_score > .8 for c in ranked), 1))
        self.assertFalse(pass_at_k((False, True), 1))

    def test_delete_add_synthesis_keeps_only_valid_invertible_faces(self):
        def delete(shape, face):
            return None if face.source_id == "b" else shape - {face.source_id}

        pairs = synthesize_delete_add_pairs(
            {"a", "b", "c"}, FACES,
            delete_face=delete,
            add_face=lambda shape, face: shape | {face.source_id},
            is_valid=lambda shape: bool(shape),
            equivalent=lambda a, b: a == b,
        )
        self.assertEqual({p.face.source_id for p in pairs}, {"a", "c"})
        self.assertTrue(all(p.inverse["operation"] == "add_face" for p in pairs))

    def test_directional_annotations_invert_and_detect_leakage(self):
        item = DirectionalAnnotation(
            "face", "front", "flat", "recessed", "delete front", "add front")
        self.assertEqual(item.inverted().inverted(), item)
        self.assertEqual(validate_annotations([item]), ())
        self.assertEqual(
            validate_annotations([item], forbidden_tokens=("delete front",)),
            ("leakage:0",),
        )
        duplicate = DirectionalAnnotation(
            "face", "front", "a", "b", "forward", "reverse")
        self.assertIn("duplicate:face:front", validate_annotations([item, duplicate]))

    def test_sequence_digest_is_stable_for_parameter_order(self):
        a = BrepEditSequence.build(
            "abc", "remove face", [BrepEditStep("delete", {"z": 2, "a": 1})])
        b = BrepEditSequence.build(
            "abc", "remove face", [BrepEditStep("delete", {"a": 1, "z": 2})])
        self.assertEqual(a.digest, b.digest)

    def test_context_projection_and_best_view(self):
        context = select_edit_context(FACES, [0], {0: [1], 1: [2]}, rings=2)
        self.assertEqual(context, (0, 1, 2))
        points = ((0, 0, 0), (4, 2, 1))
        self.assertEqual(projected_bbox(points, "top"), (0, 0, 4, 2))
        self.assertEqual(best_view(points), "top")

    def test_geometry_and_relation_metrics(self):
        self.assertEqual(symmetric_chamfer([(0, 0, 0)], [(1, 0, 0)]), 1)
        self.assertAlmostEqual(retention(("a", "b"), ("b", "c")), .5)
        guard = relation_preservation(
            {"x": "parallel", "y": "symmetry", "note": "near"},
            {"x": "parallel", "y": "perpendicular"},
        )
        self.assertEqual(guard["broken"], ("y",))
        self.assertEqual(guard["fraction"], .5)

    def test_complexity_bins_and_balancing(self):
        self.assertEqual(
            complexity_score(affected_faces=1, context_faces=1,
                             relation_count=0, operation_count=1), 5)
        items = [
            {"id": 3, "bin": "easy"}, {"id": 1, "bin": "easy"},
            {"id": 4, "bin": "medium"}, {"id": 2, "bin": "hard"},
        ]
        sampled = balanced_sample(
            items, 1, bin_of=lambda x: x["bin"], key=lambda x: x["id"])
        self.assertEqual([x["id"] for x in sampled], [1, 4, 2])
        self.assertEqual([complexity_bin(x) for x in (5, 6, 15)],
                         ["easy", "medium", "hard"])


if __name__ == "__main__":
    unittest.main()
