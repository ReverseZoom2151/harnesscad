import unittest

from harnesscad.domain.reconstruction.sequences import brep_linkage as lk
from harnesscad.domain.reconstruction.brep.entity_index import EdgeRecord, FaceRecord
from harnesscad.domain.reconstruction.sequences.pointer_commands import CHAMFER, FILLET, PointerCommand, PointerError


class SketchExtrudeTest(unittest.TestCase):
    def test_first_step_sketches_on_base_plane(self):
        state = lk.BrepState()
        # empty B-rep: only 3 base planes exist; sketch on Top (pointer 2)
        faces = [FaceRecord(key="prism_top"), FaceRecord(key="prism_bottom"),
                 FaceRecord(key="prism_side")]
        edges = [EdgeRecord(key="pe1", face_keys=("prism_top", "prism_side"))]
        new, res = lk.apply_sketch_extrude(state, 2, faces, edges)
        self.assertEqual(res.referenced_faces, ("base::Top",))
        self.assertEqual(set(res.added_faces), {"prism_top", "prism_bottom", "prism_side"})
        self.assertEqual(new.index().num_faces, 3 + 3)
        self.assertEqual(new.index().num_edges, 1)

    def test_dangling_plane_pointer_raises(self):
        state = lk.BrepState()
        with self.assertRaises(PointerError):
            lk.apply_sketch_extrude(state, 99, [], [])

    def test_new_edge_unknown_face_raises(self):
        state = lk.BrepState()
        with self.assertRaises(PointerError):
            lk.apply_sketch_extrude(
                state, 0, [FaceRecord(key="a")],
                [EdgeRecord(key="e", face_keys=("a", "ghost"))])

    def test_second_step_points_at_first_step_geometry(self):
        state = lk.BrepState()
        faces = [FaceRecord(key="f_a"), FaceRecord(key="f_b")]
        edges = [EdgeRecord(key="e_ab", face_keys=("f_a", "f_b"))]
        state, _ = lk.apply_sketch_extrude(state, 2, faces, edges)
        # now sketch on a face created by the first step
        ix = state.index()
        ptr = ix.face_pointer("f_a")
        state2, res = lk.apply_sketch_extrude(
            state, ptr, [FaceRecord(key="f_c")], [])
        self.assertEqual(res.referenced_faces, ("f_a",))


def _box_state():
    faces = [FaceRecord(key="top"), FaceRecord(key="bottom"),
             FaceRecord(key="side1"), FaceRecord(key="side2")]
    edges = [
        EdgeRecord(key="e_top_s1", face_keys=("top", "side1")),
        EdgeRecord(key="e_top_s2", face_keys=("top", "side2")),
        EdgeRecord(key="e_bot_s1", face_keys=("bottom", "side1")),
    ]
    state = lk.BrepState()
    state, _ = lk.apply_sketch_extrude(state, 2, faces, edges)
    return state


class ChamferFilletTest(unittest.TestCase):
    def test_chamfer_removes_edge_adds_face(self):
        state = _box_state()
        ix = state.index()
        ptr = ix.edge_pointer("e_top_s1")
        n_edges_before = ix.num_edges
        cmd = PointerCommand(kind=CHAMFER, edge_pointers=(ptr,), param=0.3)
        new, res = lk.apply_chamfer(state, cmd)
        self.assertEqual(res.referenced_edges, ("e_top_s1",))
        self.assertEqual(res.removed_edges, ("e_top_s1",))
        self.assertEqual(res.added_faces, ("chamfer_face::e_top_s1",))
        self.assertEqual(new.index().num_edges, n_edges_before - 1)

    def test_fillet_multiple_edges(self):
        state = _box_state()
        ix = state.index()
        ptrs = (ix.edge_pointer("e_top_s1"), ix.edge_pointer("e_bot_s1"))
        cmd = PointerCommand(kind=FILLET, edge_pointers=ptrs, param=0.1)
        new, res = lk.apply_fillet(state, cmd)
        self.assertEqual(set(res.referenced_edges), {"e_top_s1", "e_bot_s1"})
        self.assertEqual(len(res.added_faces), 2)

    def test_chamfer_dangling_edge_raises(self):
        state = _box_state()
        cmd = PointerCommand(kind=CHAMFER, edge_pointers=(999,), param=0.3)
        with self.assertRaises(PointerError):
            lk.apply_chamfer(state, cmd)

    def test_wrong_command_kind_rejected(self):
        state = _box_state()
        cmd = PointerCommand(kind=FILLET, edge_pointers=(0,), param=0.1)
        with self.assertRaises(PointerError):
            lk.apply_chamfer(state, cmd)


class ReplayTest(unittest.TestCase):
    def test_full_sequence_replay(self):
        state = lk.BrepState()
        faces = [FaceRecord(key="top"), FaceRecord(key="side")]
        edges = [EdgeRecord(key="e_ts", face_keys=("top", "side"))]
        # build sketch-extrude, then chamfer the edge it created
        steps = [
            ("sketch_extrude", (2, faces, edges)),
        ]
        mid, _ = lk.replay(state, steps)
        ptr = mid.index().edge_pointer("e_ts")
        final, results = lk.replay(state, steps + [
            ("chamfer", PointerCommand(kind=CHAMFER, edge_pointers=(ptr,), param=0.2)),
        ])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1].operation, "chamfer")
        # edge consumed, chamfer face added
        self.assertNotIn("e_ts", [e.key for e in final.edges])
        self.assertIn("chamfer_face::e_ts", [f.key for f in final.faces])

    def test_unknown_op_raises(self):
        with self.assertRaises(PointerError):
            lk.replay(lk.BrepState(), [("drill", None)])

    def test_replay_does_not_mutate_input(self):
        state = _box_state()
        n = len(state.edges)
        ix = state.index()
        cmd = PointerCommand(kind=CHAMFER, edge_pointers=(ix.edge_pointer("e_top_s1"),), param=0.1)
        lk.replay(state, [("chamfer", cmd)])
        self.assertEqual(len(state.edges), n)  # original unchanged


if __name__ == "__main__":
    unittest.main()
