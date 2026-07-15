"""The KCL codec: CISP ops -> deterministic, well-formed, field-live .kcl text.

These tests are offline and need no API key. They assert three things the build
promises: a box emits well-formed, deterministic KCL; the op->KCL mapping honours
every field (never silently drops one); and unemittable ops refuse with a typed
error rather than fabricating geometry.
"""

import unittest

from harnesscad.core.cisp.ops import (
    NewSketch, AddPoint, AddLine, AddCircle, AddRectangle, Constrain,
    Extrude, Revolve, Boolean, Hole, Shell, Fillet, Chamfer,
    LinearPattern, CircularPattern, Mirror, AddInstance, Mate, SetParam,
)
from harnesscad.io.formats import kcl as kcl_codec
from harnesscad.io.formats.kcl import emit_kcl, KclEmitError


def _box_ops():
    return [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0),
        Extrude(sketch="sk1", distance=5.0),
    ]


class TestBoxEmission(unittest.TestCase):
    def test_box_is_wellformed(self):
        text = emit_kcl(_box_ops(), name="box")
        # Header + settings.
        self.assertIn("@settings(defaultLengthUnit = mm, kclVersion = 1.0)", text)
        # A real sketch->profile->extrude spine.
        self.assertIn("startSketchOn(XY)", text)
        self.assertIn("startProfile(at = [0, 0])", text)
        self.assertIn("close()", text)
        self.assertRegex(text, r"extrude\(profile\d+, length = 5\)")
        # POSIX-clean: exactly one trailing newline, no CRLF.
        self.assertTrue(text.endswith("\n"))
        self.assertNotIn("\r", text)
        # No annotation comments for a fully-expressible model.
        self.assertNotIn("not expressible", text)

    def test_deterministic_and_idempotent(self):
        a = emit_kcl(_box_ops(), name="box")
        b = emit_kcl(_box_ops(), name="box")
        self.assertEqual(a, b)

    def test_rectangle_fields_all_live(self):
        # Every AddRectangle field must appear in the emitted text.
        text = emit_kcl([NewSketch(plane="XZ"),
                         AddRectangle(sketch="sk1", x=3.0, y=7.0, w=21.0, h=13.0)])
        self.assertIn("startSketchOn(XZ)", text)      # plane
        self.assertIn("[3, 7]", text)                 # x, y
        self.assertIn("xLine(length = 21)", text)     # w
        self.assertIn("yLine(length = 13)", text)     # h
        self.assertIn("xLine(length = -21)", text)    # closing side from w

    def test_number_formatting(self):
        text = emit_kcl([NewSketch(plane="XY"),
                         AddCircle(sketch="sk1", cx=-2.5, cy=0.0, r=4.0)])
        # Integers render without a trailing .0; floats keep precision.
        self.assertIn("center = [-2.5, 0], radius = 4", text)


class TestFieldLiveness(unittest.TestCase):
    def test_circle_to_cylinder(self):
        text = emit_kcl([NewSketch(plane="XY"),
                         AddCircle(sketch="sk1", cx=1.0, cy=2.0, r=3.0),
                         Extrude(sketch="sk1", distance=8.0)])
        self.assertIn("circle(center = [1, 2], radius = 3)", text)
        self.assertRegex(text, r"extrude\(profile\d+, length = 8\)")

    def test_negative_extrude_sign_preserved(self):
        text = emit_kcl([NewSketch(plane="XY"),
                         AddRectangle(sketch="sk1", x=0, y=0, w=2, h=2),
                         Extrude(sketch="sk1", distance=-4.0)])
        self.assertIn("length = -4", text)

    def test_line_and_point_coordinates_preserved(self):
        text = emit_kcl([
            NewSketch(plane="XY"),
            AddLine(sketch="sk1", x1=1.0, y1=2.0, x2=3.0, y2=4.0),
            AddPoint(sketch="sk1", x=9.0, y=8.0),
        ])
        self.assertIn("startProfile(at = [1, 2])", text)
        self.assertIn("line(endAbsolute = [3, 4])", text)
        self.assertIn("startProfile(at = [9, 8])", text)

    def test_boolean_kinds(self):
        base = [NewSketch(plane="XY"),
                AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10),
                Extrude(sketch="sk1", distance=10),
                NewSketch(plane="XY"),
                AddCircle(sketch="sk2", cx=5, cy=5, r=2),
                Extrude(sketch="sk2", distance=10)]
        cut = emit_kcl(base + [Boolean(kind="cut", target="f1", tool="f2")])
        self.assertIn("subtract(solid1, tools = [solid2])", cut)
        uni = emit_kcl(base + [Boolean(kind="union", target="f1", tool="f2")])
        self.assertIn("union([solid1, solid2])", uni)
        isec = emit_kcl(base + [Boolean(kind="intersect", target="f1", tool="f2")])
        self.assertIn("intersect([solid1, solid2])", isec)

    def test_shell_maps_cap_and_keeps_kind(self):
        ops = _box_ops() + [Shell(faces=(">Z",), thickness=2.0, kind="intersection")]
        text = emit_kcl(ops)
        self.assertIn("faces = [END]", text)      # >Z cap -> END
        self.assertIn("thickness = 2", text)
        self.assertIn("join=intersection", text)  # kind preserved

    def test_hole_fields_preserved(self):
        ops = _box_ops() + [Hole(face_or_sketch="solid", x=4.0, y=6.0,
                                  diameter=3.0, through=True, kind="counterbore",
                                  cbore_diameter=6.0, cbore_depth=2.0)]
        text = emit_kcl(ops)
        self.assertIn("radius = 1.5", text)          # diameter/2
        self.assertIn("[4, 6]", text)                # x, y
        self.assertIn("counterbore", text)           # kind
        self.assertIn("cbore_diameter=6", text)      # stepped fields
        self.assertIn("cbore_depth=2", text)
        self.assertIn("subtract(", text)

    def test_patterns(self):
        base = _box_ops()
        lin = emit_kcl(base + [LinearPattern(feature="f1", direction=(1, 0, 0),
                                             count=3, spacing=12.0)])
        self.assertIn("patternLinear3d(solid1, instances = 3, distance = 12, "
                      "axis = [1, 0, 0])", lin)
        circ = emit_kcl(base + [CircularPattern(feature="f1",
                                                axis=(0, 0, 0, 0, 0, 1),
                                                count=6, angle=360.0)])
        self.assertIn("patternCircular3d(solid1, instances = 6", circ)
        self.assertIn("arcDegrees = 360", circ)

    def test_revolve_axis_and_full_vector_kept(self):
        text = emit_kcl([NewSketch(plane="XY"),
                         AddRectangle(sketch="sk1", x=1, y=0, w=2, h=5),
                         Revolve(sketch="sk1", axis=(0, 0, 0, 0, 1, 0),
                                 angle=270.0)])
        self.assertIn("revolve(", text)
        self.assertIn("axis = Y", text)            # dominant direction
        self.assertIn("angle = 270", text)
        self.assertIn("axis6=[0, 0, 0, 0, 1, 0]", text)  # full 6-tuple retained

    def test_annotated_ops_keep_all_fields(self):
        # Fillet / Chamfer / Constrain / Mirror / AddInstance / Mate are annotated,
        # and every field must still appear in the text.
        text = emit_kcl(_box_ops() + [
            Constrain(kind="distance", a="e1", b="e2", value=5.0),
            Fillet(edges=("|Z", ">Y"), radius=1.25),
            Chamfer(edges=("|Z",), distance=0.8, distance2=0.4),
            Mirror(feature_or_body="f1", plane="XZ"),
            AddInstance(part="f1", x=1, y=2, z=3, rx=10, ry=20, rz=30),
            Mate(kind="revolute", a="i1", b="i2", value=45.0),
        ])
        for token in ("distance", "value = 5", "radius = 1.25",
                      "distance2 = 0.4", "plane = XZ",
                      "rotate_deg = [10, 20, 30]", "revolute", "value = 45"):
            self.assertIn(token, text)


class TestRefusals(unittest.TestCase):
    def test_unknown_plane_refused(self):
        with self.assertRaises(KclEmitError):
            emit_kcl([NewSketch(plane="ABC")])

    def test_extrude_without_closed_profile_refused(self):
        # A lone line is an open profile: there is nothing to extrude.
        with self.assertRaises(KclEmitError):
            emit_kcl([NewSketch(plane="XY"),
                      AddLine(sketch="sk1", x1=0, y1=0, x2=1, y2=1),
                      Extrude(sketch="sk1", distance=5)])

    def test_shell_side_wall_refused(self):
        with self.assertRaises(KclEmitError):
            emit_kcl(_box_ops() + [Shell(faces=(">X",), thickness=1.0)])

    def test_setparam_is_not_a_statement(self):
        # SetParam is an edit meta-op; it must never appear as KCL.
        text = emit_kcl(_box_ops() + [SetParam(target=1, param="w", value=99)])
        self.assertNotIn("SetParam", text)
        self.assertNotIn("set_param", text)


class TestPublicApi(unittest.TestCase):
    def test_ops_of_accepts_op_list_and_backend(self):
        ops = _box_ops()
        self.assertEqual(kcl_codec.ops_of(ops), ops)

        class _FakeBackend:
            _oplog = ops

        self.assertEqual(kcl_codec.ops_of(_FakeBackend()), ops)

    def test_serialize_and_write_roundtrip_determinism(self):
        import os
        import tempfile

        class _FakeBackend:
            _oplog = _box_ops()

        be = _FakeBackend()
        text = kcl_codec.serialize_kcl(be, name="box")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "box.kcl")
            kcl_codec.write_kcl(be, path)
            with open(path, "r", encoding="utf-8") as fh:
                on_disk = fh.read()
        # The file name seeds the model name, so compare on the same name.
        self.assertEqual(on_disk, kcl_codec.serialize_kcl(be, name="box"))
        self.assertIn("startSketchOn(XY)", text)


if __name__ == "__main__":
    unittest.main()
