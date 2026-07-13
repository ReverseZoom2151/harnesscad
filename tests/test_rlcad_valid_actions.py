"""Tests for reconstruction.rlcad_valid_actions (RLCAD Algorithm 1)."""

import unittest

from harnesscad.domain.reconstruction.sequences import rlcad_valid_actions as va
from harnesscad.domain.reconstruction.tokens.rlcad_command_spec import (
    EXTRUDE, NEWBODY, REVOLVE, SUBTRACTION, UNION, ExtrudeCommand,
    RevolveCommand,
)


def _plane(fid, normal, offset):
    return va.FaceInfo(fid, va.PLANE, normal, offset)


class TestPartition(unittest.TestCase):
    def test_partition(self):
        faces = [
            _plane(0, (0, 0, 1), 0.0),
            va.FaceInfo(1, "cylinder"),
            _plane(2, (0, 0, 1), 5.0),
        ]
        planar, nonplanar = va.partition_faces(faces)
        self.assertEqual({f.face_id for f in planar}, {0, 2})
        self.assertEqual({f.face_id for f in nonplanar}, {1})


class TestGrouping(unittest.TestCase):
    def test_parallel_grouped_opposite_normals(self):
        faces = [
            _plane(0, (0, 0, 1), 0.0),
            _plane(1, (0, 0, -1), 5.0),   # antiparallel -> same group
            _plane(2, (1, 0, 0), 0.0),    # different direction
        ]
        groups = va.group_parallel_faces(faces)
        self.assertEqual(len(groups), 2)
        self.assertEqual({f.face_id for f in groups[0]}, {0, 1})
        self.assertEqual({f.face_id for f in groups[1]}, {2})


class TestValidExtrude(unittest.TestCase):
    def test_parallel_noncoplanar_valid(self):
        a = _plane(0, (0, 0, 1), 0.0)
        b = _plane(1, (0, 0, 1), 5.0)
        self.assertTrue(va.valid_extrude(a, b))

    def test_coplanar_invalid(self):
        a = _plane(0, (0, 0, 1), 3.0)
        b = _plane(1, (0, 0, 1), 3.0)
        self.assertFalse(va.valid_extrude(a, b))

    def test_antiparallel_offset_coplanar_invalid(self):
        # Same physical plane described with opposite normals: offsets negate.
        a = _plane(0, (0, 0, 1), 3.0)
        b = _plane(1, (0, 0, -1), -3.0)
        self.assertFalse(va.valid_extrude(a, b))

    def test_nonparallel_invalid(self):
        a = _plane(0, (0, 0, 1), 0.0)
        b = _plane(1, (1, 0, 0), 0.0)
        self.assertFalse(va.valid_extrude(a, b))


class TestValidRevolve(unittest.TestCase):
    def test_revolvable_types(self):
        for t in ("cylinder", "cone", "sphere", "torus"):
            self.assertTrue(va.valid_revolve(va.FaceInfo(0, t)))

    def test_plane_and_freeform_rejected(self):
        self.assertFalse(va.valid_revolve(_plane(0, (0, 0, 1), 0.0)))
        self.assertFalse(va.valid_revolve(va.FaceInfo(0, "bspline")))


class TestGenerateValidActions(unittest.TestCase):
    def setUp(self):
        # Two parallel planes + a cylinder + a freeform (non-revolvable) face.
        self.faces = [
            _plane(0, (0, 0, 1), 0.0),
            _plane(1, (0, 0, 1), 4.0),
            va.FaceInfo(2, "cylinder"),
            va.FaceInfo(3, "bspline"),
        ]

    def test_single_op_actions(self):
        actions = va.generate_valid_actions(self.faces)
        # Extrude both directions (0->1, 1->0) + revolve on cylinder only.
        self.assertIn((0, 1, NEWBODY, EXTRUDE), actions)
        self.assertIn((1, 0, NEWBODY, EXTRUDE), actions)
        self.assertIn((2, 2, NEWBODY, REVOLVE), actions)
        # bspline face never revolves.
        self.assertNotIn((3, 3, NEWBODY, REVOLVE), actions)
        self.assertEqual(len(actions), 3)

    def test_multi_op_expansion(self):
        actions = va.generate_valid_actions(
            self.faces, boolean_ops=(NEWBODY, UNION, SUBTRACTION))
        self.assertEqual(len(actions), 3 * 3)

    def test_unknown_op_raises(self):
        with self.assertRaises(ValueError):
            va.generate_valid_actions(self.faces, boolean_ops=("fillet",))


class TestValidActionSet(unittest.TestCase):
    def setUp(self):
        self.faces = [
            _plane(0, (0, 0, 1), 0.0),
            _plane(1, (0, 0, 1), 4.0),
            va.FaceInfo(2, "cone"),
        ]

    def test_commands_and_size(self):
        vas = va.ValidActionSet(self.faces)
        self.assertEqual(vas.action_space_size(), len(vas))
        self.assertEqual(len(vas.extrude_commands()), 2)
        self.assertEqual(len(vas.revolve_commands()), 1)
        self.assertIsInstance(vas.revolve_commands()[0], RevolveCommand)
        self.assertIsInstance(vas.extrude_commands()[0], ExtrudeCommand)


class TestSequenceValidity(unittest.TestCase):
    def setUp(self):
        self.faces = [
            _plane(0, (0, 0, 1), 0.0),
            _plane(1, (0, 0, 1), 4.0),
            va.FaceInfo(2, "sphere"),
        ]

    def test_valid_sequence(self):
        cmds = [ExtrudeCommand(0, 1, NEWBODY), RevolveCommand(2, UNION)]
        self.assertTrue(va.revolve_sequence_valid(cmds, self.faces))

    def test_revolve_on_plane_invalid(self):
        cmds = [RevolveCommand(0, UNION)]  # face 0 is a plane
        self.assertFalse(va.revolve_sequence_valid(cmds, self.faces))

    def test_missing_face_invalid(self):
        cmds = [RevolveCommand(99, UNION)]
        self.assertFalse(va.revolve_sequence_valid(cmds, self.faces))

    def test_extrude_coplanar_invalid(self):
        faces = self.faces + [va.FaceInfo(3, va.PLANE, (0, 0, 1), 0.0)]
        cmds = [ExtrudeCommand(0, 3, NEWBODY)]  # same offset as face 0
        self.assertFalse(va.revolve_sequence_valid(cmds, faces))


if __name__ == "__main__":
    unittest.main()
