import unittest

from harnesscad.domain.reconstruction.brep import entity_index as idx
from harnesscad.domain.reconstruction.brep.entity_index import EdgeRecord, FaceRecord


def _sample():
    faces = [
        FaceRecord(key="f_top"),
        FaceRecord(key="f_bottom"),
        FaceRecord(key="f_side"),
    ]
    edges = [
        EdgeRecord(key="e2", face_keys=("f_side", "f_top")),
        EdgeRecord(key="e1", face_keys=("f_top", "f_bottom")),
    ]
    return idx.build_index(faces, edges)


class OrderingTest(unittest.TestCase):
    def test_base_planes_first_and_fixed(self):
        ix = _sample()
        self.assertEqual([f.key for f in ix.faces[:3]],
                         ["base::Right", "base::Front", "base::Top"])
        self.assertEqual(ix.face_pointer("base::Right"), 0)
        self.assertEqual(ix.face_pointer("base::Top"), 2)

    def test_model_faces_sorted_after_base(self):
        ix = _sample()
        # keys sorted: f_bottom, f_side, f_top -> indices 3,4,5
        self.assertEqual(ix.face_pointer("f_bottom"), 3)
        self.assertEqual(ix.face_pointer("f_side"), 4)
        self.assertEqual(ix.face_pointer("f_top"), 5)

    def test_determinism_independent_of_input_order(self):
        a = _sample()
        faces = [FaceRecord(key="f_side"), FaceRecord(key="f_top"),
                 FaceRecord(key="f_bottom")]
        edges = [EdgeRecord(key="e1", face_keys=("f_bottom", "f_top")),
                 EdgeRecord(key="e2", face_keys=("f_top", "f_side"))]
        b = idx.build_index(faces, edges)
        self.assertEqual([f.key for f in a.faces], [f.key for f in b.faces])
        self.assertEqual([e.key for e in a.edges], [e.key for e in b.edges])

    def test_edges_ordered_by_face_indices(self):
        ix = _sample()
        # e1 spans (f_bottom=3, f_top=5); e2 spans (f_side=4, f_top=5)
        # sorted by (min,max): (3,5) then (4,5) -> e1, e2
        self.assertEqual([e.key for e in ix.edges], ["e1", "e2"])
        self.assertEqual(ix.edge_pointer("e1"), 0)
        self.assertEqual(ix.edge_pointer("e2"), 1)

    def test_edge_face_keys_stored_sorted(self):
        ix = _sample()
        self.assertEqual(ix.resolve_edge(1).face_keys, ("f_side", "f_top"))


class ResolutionTest(unittest.TestCase):
    def test_resolve_roundtrip(self):
        ix = _sample()
        self.assertEqual(ix.resolve_face(ix.face_pointer("f_top")).key, "f_top")
        self.assertEqual(ix.resolve_edge(ix.edge_pointer("e2")).key, "e2")

    def test_out_of_range(self):
        ix = _sample()
        self.assertFalse(ix.face_pointer_valid(999))
        self.assertFalse(ix.face_pointer_valid(-1))
        self.assertFalse(ix.edge_pointer_valid(ix.num_edges))
        with self.assertRaises(idx.PointerIndexError):
            ix.resolve_face(999)

    def test_empty_index_has_only_base_planes(self):
        ix = idx.build_index()
        self.assertEqual(ix.num_faces, 3)
        self.assertEqual(ix.num_edges, 0)

    def test_no_base_planes_option(self):
        ix = idx.build_index([FaceRecord(key="a")], include_base_planes=False)
        self.assertEqual(ix.num_faces, 1)
        self.assertEqual(ix.face_pointer("a"), 0)


class ValidationTest(unittest.TestCase):
    def test_edge_references_unknown_face(self):
        with self.assertRaises(idx.PointerIndexError):
            idx.build_index([FaceRecord(key="a")],
                            [EdgeRecord(key="e", face_keys=("a", "ghost"))])

    def test_duplicate_face_key(self):
        with self.assertRaises(idx.PointerIndexError):
            idx.build_index([FaceRecord(key="a"), FaceRecord(key="a")])

    def test_model_face_flagged_base_rejected(self):
        with self.assertRaises(idx.PointerIndexError):
            idx.build_index([FaceRecord(key="a", is_base=True)])


class IncidenceTest(unittest.TestCase):
    def test_incidence_counts(self):
        ix = _sample()
        inc = idx.face_incidence(ix)
        # f_top (index 5) bounds both edges
        self.assertEqual(sorted(inc[ix.face_pointer("f_top")]), [0, 1])
        # each edge shared by two faces -> total incidences = 2 * num_edges
        total = sum(len(v) for v in inc.values())
        self.assertEqual(total, 2 * ix.num_edges)


if __name__ == "__main__":
    unittest.main()
