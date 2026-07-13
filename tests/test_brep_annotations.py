import unittest

from harnesscad.io.ingest.brep_annotations import (
    EntityAnnotation,
    EntityRecord,
    ExternalTag,
    LocalFrame,
    assign_external_tags,
    persist_entity_ids,
)


def face(x=0.0, signature=(100.0, 1.0)):
    return EntityRecord("face", (x, 0, 0), signature, (4, 4, 0))


class PersistentEntityTests(unittest.TestCase):
    def test_derived_id_is_stable_for_equivalent_geometry(self):
        self.assertEqual(face().derived_id(), face(x=99).derived_id())

    def test_small_signature_noise_survives_revision(self):
        old = {"face-top": face(signature=(100.0, 1.0))}
        new = face(signature=(100.0000001, 1.0))
        self.assertEqual(list(persist_entity_ids(old, [new])), ["face-top"])

    def test_new_entity_gets_deterministic_id(self):
        one = persist_entity_ids({}, [face()])
        two = persist_entity_ids({}, [face()])
        self.assertEqual(one.keys(), two.keys())

    def test_duplicate_signatures_get_unique_ids(self):
        result = persist_entity_ids({}, [face(0), face(1)])
        self.assertEqual(len(result), 2)

    def test_invalid_tolerance_fails(self):
        with self.assertRaises(ValueError):
            face().derived_id(0)


class AnnotationTests(unittest.TestCase):
    def test_schema_supports_frame_labels_and_attributes(self):
        frame = LocalFrame((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        annotation = EntityAnnotation("f1", "face", ("mounting",), frame, {"finish": "ground"})
        self.assertEqual(annotation.frame.z_axis, (0, 0, 1))
        self.assertEqual(annotation.attributes["finish"], "ground")

    def test_zero_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            LocalFrame((0, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 1))

    def test_assigns_at_first_reaching_threshold(self):
        result = assign_external_tags(
            {"f1": face(0)}, [ExternalTag((0.006, 0, 0), ("datum",))],
            thresholds=(0.001, 0.01),
        )
        self.assertEqual(result.assignments[0].threshold, 0.01)
        self.assertEqual(result.annotations[0].labels, ("datum",))

    def test_samples_participate_in_proximity(self):
        entity = EntityRecord("edge", (100, 0, 0), (10,), (2,), ((0, 0, 0),))
        result = assign_external_tags({"e1": entity}, [ExternalTag((0, 0, 0), ("seam",))])
        self.assertEqual(result.assignments[0].entity_id, "e1")

    def test_reports_ambiguity_without_assignment(self):
        result = assign_external_tags(
            {"a": face(-1), "b": face(1)},
            [ExternalTag((0, 0, 0), ("center",))],
            thresholds=(2,),
        )
        self.assertEqual(result.assignments, ())
        self.assertEqual(result.issues[0].code, "ambiguous")
        self.assertEqual(result.issues[0].entity_ids, ("a", "b"))

    def test_reports_unassigned_tag(self):
        result = assign_external_tags(
            {"f": face(0)}, [ExternalTag((5, 0, 0), ("far",))], thresholds=(1,)
        )
        self.assertEqual(result.issues[0].code, "unassigned")

    def test_reports_conflicting_labels_and_preserves_both(self):
        existing = {"f": EntityAnnotation("f", "face", ("inlet",))}
        result = assign_external_tags(
            {"f": face()}, [ExternalTag((0, 0, 0), ("outlet",))], existing=existing
        )
        self.assertEqual(result.issues[0].code, "conflict")
        self.assertEqual(result.annotations[0].labels, ("inlet", "outlet"))

    def test_same_label_is_not_a_conflict(self):
        existing = {"f": EntityAnnotation("f", "face", ("datum",))}
        result = assign_external_tags(
            {"f": face()}, [ExternalTag((0, 0, 0), ("datum",))], existing=existing
        )
        self.assertEqual(result.issues, ())
        self.assertEqual(result.annotations[0].labels, ("datum",))

    def test_bad_thresholds_fail(self):
        with self.assertRaises(ValueError):
            assign_external_tags({}, [], thresholds=())


if __name__ == "__main__":
    unittest.main()
