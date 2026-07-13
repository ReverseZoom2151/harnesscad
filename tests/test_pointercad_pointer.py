import unittest

from harnesscad.domain.reconstruction import pointercad_indexing as idx
from harnesscad.domain.reconstruction import pointercad_pointer as pc
from harnesscad.domain.reconstruction.pointercad_indexing import EdgeRecord, FaceRecord


class CommandShapeTest(unittest.TestCase):
    def test_sketch_needs_one_face_pointer(self):
        cmd = pc.PointerCommand(kind=pc.SKETCH, face_pointers=(2,))
        self.assertEqual(cmd.target, pc.FACE)
        with self.assertRaises(pc.PointerError):
            pc.PointerCommand(kind=pc.SKETCH, face_pointers=(2, 3))
        with self.assertRaises(pc.PointerError):
            pc.PointerCommand(kind=pc.SKETCH, edge_pointers=(1,), face_pointers=(0,))

    def test_chamfer_fillet_need_edge_pointers(self):
        c = pc.PointerCommand(kind=pc.CHAMFER, edge_pointers=(0, 1), param=0.2)
        self.assertEqual(c.target, pc.EDGE)
        pc.PointerCommand(kind=pc.FILLET, edge_pointers=(3,), param=0.1)
        with self.assertRaises(pc.PointerError):
            pc.PointerCommand(kind=pc.CHAMFER, edge_pointers=())
        with self.assertRaises(pc.PointerError):
            pc.PointerCommand(kind=pc.FILLET, edge_pointers=(1,), face_pointers=(0,))

    def test_unknown_kind(self):
        with self.assertRaises(pc.PointerError):
            pc.PointerCommand(kind="drill", edge_pointers=(1,))


def _index():
    faces = [FaceRecord(key="a"), FaceRecord(key="b")]
    edges = [EdgeRecord(key="e", face_keys=("a", "b"))]
    return idx.build_index(faces, edges)  # 5 faces (3 base + a,b), 1 edge


class ResolutionTest(unittest.TestCase):
    def test_valid_command_resolves(self):
        ix = _index()
        cmd = pc.PointerCommand(kind=pc.SKETCH, face_pointers=(0,))
        res = pc.resolve_command(cmd, ix)
        self.assertTrue(res.is_valid)
        pc.validate_command(cmd, ix)  # does not raise

    def test_dangling_face_pointer(self):
        ix = _index()
        cmd = pc.PointerCommand(kind=pc.SKETCH, face_pointers=(99,))
        res = pc.resolve_command(cmd, ix)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.dangling_faces, (99,))
        with self.assertRaises(pc.PointerError):
            pc.validate_command(cmd, ix)

    def test_dangling_edge_pointer(self):
        ix = _index()
        cmd = pc.PointerCommand(kind=pc.FILLET, edge_pointers=(0, 7), param=0.1)
        res = pc.resolve_command(cmd, ix)
        self.assertEqual(res.dangling_edges, (7,))

    def test_dangling_report_over_sequence(self):
        ix = _index()
        cmds = [
            pc.PointerCommand(kind=pc.SKETCH, face_pointers=(1,)),          # ok
            pc.PointerCommand(kind=pc.CHAMFER, edge_pointers=(5,), param=0.1),  # bad
        ]
        bad = pc.dangling_pointers(cmds, ix)
        self.assertEqual([i for i, _ in bad], [1])


class CoplanarTest(unittest.TestCase):
    def test_coplanar_faces_grouped(self):
        faces = [
            FaceRecord(key="p1", plane=(0, 0, 1, 2.0)),
            FaceRecord(key="p2", plane=(0, 0, -1, -2.0)),   # same plane, flipped
            FaceRecord(key="p3", plane=(0, 0, 1, 5.0)),     # parallel, different d
        ]
        ix = idx.build_index(faces)
        groups = pc.coplanar_face_groups(ix)
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual({ix.faces[p].key for p in g}, {"p1", "p2"})

    def test_no_group_when_all_distinct(self):
        faces = [FaceRecord(key="p1", plane=(0, 0, 1, 1.0)),
                 FaceRecord(key="p2", plane=(1, 0, 0, 1.0))]
        ix = idx.build_index(faces)
        self.assertEqual(pc.coplanar_face_groups(ix), [])


class CollinearTest(unittest.TestCase):
    def test_collinear_edges_grouped(self):
        faces = [FaceRecord(key=k) for k in ("a", "b", "c", "d")]
        edges = [
            EdgeRecord(key="e1", face_keys=("a", "b"), line=(0, 0, 0, 1, 0, 0)),
            EdgeRecord(key="e2", face_keys=("c", "d"), line=(5, 0, 0, 2, 0, 0)),  # same x-axis line
            EdgeRecord(key="e3", face_keys=("a", "c"), line=(0, 1, 0, 1, 0, 0)),  # parallel, offset
        ]
        ix = idx.build_index(faces, edges)
        groups = pc.collinear_edge_groups(ix)
        self.assertEqual(len(groups), 1)
        keys = {ix.edges[p].key for p in groups[0]}
        self.assertEqual(keys, {"e1", "e2"})


class NonManifoldTest(unittest.TestCase):
    def test_edge_shared_by_three_faces_flagged(self):
        faces = [FaceRecord(key=k) for k in ("a", "b", "c")]
        # three coincident edge records on the same line -> 3 faces touch it
        edges = [
            EdgeRecord(key="e1", face_keys=("a", "b"), line=(0, 0, 0, 0, 0, 1)),
            EdgeRecord(key="e2", face_keys=("b", "c"), line=(0, 0, 3, 0, 0, 1)),
            EdgeRecord(key="e3", face_keys=("a", "c"), line=(0, 0, 9, 0, 0, 1)),
        ]
        ix = idx.build_index(faces, edges)
        nm = pc.non_manifold_edges(ix)
        self.assertEqual(len(nm), 3)

    def test_manifold_has_no_flags(self):
        faces = [FaceRecord(key=k) for k in ("a", "b")]
        edges = [EdgeRecord(key="e1", face_keys=("a", "b"), line=(0, 0, 0, 0, 0, 1))]
        ix = idx.build_index(faces, edges)
        self.assertEqual(pc.non_manifold_edges(ix), [])


if __name__ == "__main__":
    unittest.main()
