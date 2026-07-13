import unittest

from harnesscad.domain.editing.s2cadsig_session import (
    ModelingSession,
    OpRecord,
    SessionError,
    deserialize_history,
    replay,
    serialize_history,
)
from harnesscad.domain.reconstruction.s2cadsig_op_router import spec_for
from harnesscad.domain.reconstruction.s2cadsig_param_decode import OrthoCamera, decode_operation

SEED = [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))]
CURVE = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0))


class TestFaces(unittest.TestCase):
    def test_seed(self):
        s = ModelingSession(SEED)
        self.assertEqual(len(s.faces), 1)
        self.assertEqual(s.faces[0].created_by, -1)
        self.assertTrue(s.is_active(0))

    def test_normal_normalized(self):
        s = ModelingSession([((0.0, 0.0, 0.0), (0.0, 0.0, 4.0))])
        self.assertEqual(s.face(0).normal, (0.0, 0.0, 1.0))

    def test_zero_normal_rejected(self):
        with self.assertRaises(SessionError):
            ModelingSession([((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))])

    def test_unknown_face(self):
        s = ModelingSession(SEED)
        with self.assertRaises(SessionError):
            s.face(7)

    def test_match_face(self):
        s = ModelingSession(SEED)
        self.assertEqual(s.match_face((3.0, 2.0, 0.0), (0.0, 0.0, 1.0)), 0)
        with self.assertRaises(SessionError):
            s.match_face((0.0, 0.0, 5.0), (0.0, 0.0, 1.0))

    def test_match_face_empty_shape(self):
        with self.assertRaises(SessionError):
            ModelingSession().match_face((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))


class TestApply(unittest.TestCase):
    def test_extrusion_consumes_and_caps(self):
        s = ModelingSession(SEED)
        step = s.apply(OpRecord("extrusion", 0, (0.0, 0.0, 2.0), CURVE))
        self.assertEqual(step.consumed, (0,))
        self.assertEqual(len(step.produced), 1)
        self.assertFalse(s.is_active(0))
        cap = s.face(step.produced[0])
        self.assertEqual(cap.point, (0.0, 0.0, 2.0))
        self.assertEqual(cap.normal, (0.0, 0.0, 1.0))
        self.assertEqual(cap.kind, "cap")
        self.assertEqual(s.step_count, 1)

    def test_subtractive_addsub_keeps_face_and_adds_floor(self):
        s = ModelingSession(SEED)
        step = s.apply(OpRecord("addSub", 0, (0.0, 0.0, -1.5), CURVE))
        self.assertEqual(step.consumed, ())
        self.assertTrue(s.is_active(0))
        floor = s.face(step.produced[0])
        self.assertEqual(floor.kind, "pocket_floor")
        self.assertEqual(floor.point, (0.0, 0.0, -1.5))
        self.assertEqual(floor.normal, (0.0, 0.0, -1.0))

    def test_additive_addsub_behaves_like_extrusion(self):
        s = ModelingSession(SEED)
        step = s.apply(OpRecord("addSub", 0, (0.0, 0.0, 1.0), CURVE))
        self.assertEqual(step.consumed, (0,))
        self.assertEqual(s.face(step.produced[0]).kind, "cap")

    def test_bevel_keeps_face(self):
        s = ModelingSession(SEED)
        step = s.apply(OpRecord("bevel", 0, None, CURVE))
        self.assertEqual(step.consumed, ())
        self.assertTrue(s.is_active(0))
        bevel = s.face(step.produced[0])
        self.assertEqual(bevel.kind, "bevel")
        # anchored at the base-curve centroid
        self.assertAlmostEqual(bevel.point[0], 2.0 / 3.0, places=6)

    def test_sweep_cap(self):
        s = ModelingSession(SEED)
        step = s.apply(OpRecord("sweep", 0, (0.0, 0.0, 3.0), CURVE))
        self.assertEqual(s.face(step.produced[0]).kind, "sweep_cap")

    def test_sequential_context(self):
        s = ModelingSession(SEED)
        s1 = s.apply(OpRecord("extrusion", 0, (0.0, 0.0, 2.0), CURVE))
        # the second op must stitch onto the face created by the first
        with self.assertRaises(SessionError):
            s.apply(OpRecord("extrusion", 0, (0.0, 0.0, 1.0), CURVE))
        s2 = s.apply(OpRecord("bevel", s1.produced[0], None, CURVE))
        self.assertEqual(s2.index, 1)
        self.assertEqual(len(s.summary()), 2)

    def test_validation_errors(self):
        s = ModelingSession(SEED)
        with self.assertRaises(SessionError):
            s.apply(OpRecord("extrusion", 0, None, CURVE))
        with self.assertRaises(SessionError):
            s.apply(OpRecord("extrusion", 0, (0.0, 0.0, 0.0), CURVE))
        with self.assertRaises(SessionError):
            s.apply(OpRecord("extrusion", 0, (0.0, 0.0, 1.0), ()))


class TestUndoRedoReplay(unittest.TestCase):
    def test_undo_restores_state(self):
        s = ModelingSession(SEED)
        sig0 = s.state_signature()
        s.apply(OpRecord("extrusion", 0, (0.0, 0.0, 2.0), CURVE))
        self.assertNotEqual(s.state_signature(), sig0)
        rec = s.undo()
        self.assertEqual(rec.op_name, "extrusion")
        self.assertEqual(s.state_signature(), sig0)
        self.assertTrue(s.is_active(0))
        self.assertEqual(s.step_count, 0)

    def test_redo(self):
        s = ModelingSession(SEED)
        s.apply(OpRecord("extrusion", 0, (0.0, 0.0, 2.0), CURVE))
        sig = s.state_signature()
        s.undo()
        step = s.redo()
        self.assertEqual(step.index, 0)
        self.assertEqual(s.state_signature(), sig)
        with self.assertRaises(SessionError):
            s.redo()

    def test_apply_clears_redo(self):
        s = ModelingSession(SEED)
        s.apply(OpRecord("bevel", 0, None, CURVE))
        s.undo()
        s.apply(OpRecord("sweep", 0, (0.0, 0.0, 1.0), CURVE))
        with self.assertRaises(SessionError):
            s.redo()

    def test_undo_empty(self):
        with self.assertRaises(SessionError):
            ModelingSession(SEED).undo()

    def test_replay_matches(self):
        s = ModelingSession(SEED)
        s.apply(OpRecord("extrusion", 0, (0.0, 0.0, 2.0), CURVE))
        s.apply(OpRecord("bevel", 1, None, CURVE))
        data = serialize_history(s)
        again = replay(SEED, deserialize_history(data))
        self.assertEqual(again.state_signature(), s.state_signature())
        self.assertEqual(serialize_history(again), data)


class TestApplyDecoded(unittest.TestCase):
    def test_decoded_operation_applies(self):
        h = w = 4
        n = h * w
        heat = [0.0] * n
        for i in (0, 1, 4, 5):
            heat[i] = 1.0
        normals = [(0.0, 0.0, 1.0)] * n
        depth = [0.0] * n
        curve = [0.0] * n
        curve[4] = 1.0
        curve[5] = 1.0
        maps = {
            "face_heatmap": heat,
            "context_normal": normals,
            "context_depth": depth,
            "offset_curve": curve,
            "offset_distance": [2.0] * n,
            "offset_direction": [(0.0, 0.0, 1.0)] * n,
            "offset_sign": [1.0] * n,
        }
        params = decode_operation(spec_for("extrusion"), maps, h, w, OrthoCamera())
        session = ModelingSession([((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))])
        step = session.apply_decoded(params, tolerance=1e-3)
        self.assertEqual(step.record.op_name, "extrusion")
        self.assertEqual(step.record.face_id, 0)
        self.assertEqual(session.face(step.produced[0]).point, (0.0, 0.0, 2.0))


if __name__ == "__main__":
    unittest.main()
