import math
import unittest

from harnesscad.io.formats.freecad_document import (
    DocumentContext,
    DocumentInfo,
    FreeCADObject,
    Placement,
    Rotation,
    ShapeInfo,
    ViewState,
    axis_angle_to_quaternion,
    decode_document_context,
    encode_document_context,
    normalize_axis,
    parse_type_id,
    quaternion_to_axis_angle,
    validate_context,
)


def _sample_context():
    box = FreeCADObject(
        name="Box",
        label="MyBox",
        type_id="Part::Box",
        visibility=True,
        placement=Placement(
            position=(10.0, 0.0, 5.0),
            rotation=Rotation(axis=(0.0, 0.0, 1.0), angle=math.pi / 2),
        ),
        shape=ShapeInfo(type="Solid", volume=8000.0, area=2400.0),
    )
    sketch = FreeCADObject(
        name="Sketch",
        label="Sketch",
        type_id="Sketcher::SketchObject",
        visibility=False,
    )
    view = ViewState(
        camera_type="PerspectiveCamera",
        camera_position=(0.0, 0.0, 100.0),
        camera_orientation=(0.0, 0.0, 0.0, 1.0),
    )
    doc = DocumentInfo(name="Unnamed", filename=None, object_count=2)
    return DocumentContext(document=doc, objects=[box, sketch], view=view)


class TypeIdTests(unittest.TestCase):
    def test_split(self):
        self.assertEqual(parse_type_id("Part::Box"), ("Part", "Box"))
        self.assertEqual(
            parse_type_id("Sketcher::SketchObject"), ("Sketcher", "SketchObject")
        )

    def test_no_namespace(self):
        self.assertEqual(parse_type_id("Box"), ("", "Box"))


class AxisAngleTests(unittest.TestCase):
    def test_normalize(self):
        ax = normalize_axis((0.0, 0.0, 5.0))
        self.assertAlmostEqual(ax[2], 1.0)
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in ax)), 1.0)

    def test_zero_axis_defaults(self):
        self.assertEqual(normalize_axis((0.0, 0.0, 0.0)), (0.0, 0.0, 1.0))

    def test_quaternion_roundtrip(self):
        axis = normalize_axis((1.0, 2.0, 3.0))
        angle = 1.2345
        q = axis_angle_to_quaternion(axis, angle)
        # quaternion is unit length
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in q)), 1.0, places=12)
        axis2, angle2 = quaternion_to_axis_angle(q)
        self.assertAlmostEqual(angle, angle2, places=10)
        for a, b in zip(axis, axis2):
            self.assertAlmostEqual(a, b, places=10)

    def test_identity_quaternion(self):
        axis, angle = quaternion_to_axis_angle((0.0, 0.0, 0.0, 1.0))
        self.assertEqual(axis, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(angle, 0.0)

    def test_rotation_quaternion_helpers(self):
        r = Rotation(axis=(0.0, 1.0, 0.0), angle=math.pi / 3)
        r2 = Rotation.from_quaternion(r.to_quaternion())
        self.assertAlmostEqual(r.angle, r2.angle, places=10)
        for a, b in zip(r.axis, r2.axis):
            self.assertAlmostEqual(a, b, places=10)

    def test_identity_rotation_flag(self):
        self.assertTrue(Rotation().is_identity)
        self.assertFalse(Rotation(axis=(0.0, 0.0, 1.0), angle=0.5).is_identity)


class EncodeDecodeTests(unittest.TestCase):
    def test_roundtrip(self):
        ctx = _sample_context()
        encoded = encode_document_context(ctx)
        decoded = decode_document_context(encoded)
        self.assertEqual(encode_document_context(decoded), encoded)

    def test_wire_shape(self):
        ctx = _sample_context()
        enc = encode_document_context(ctx)
        self.assertEqual(set(enc), {"document", "objects", "view"})
        self.assertEqual(enc["document"]["object_count"], 2)
        box = enc["objects"][0]
        self.assertEqual(box["type"], "Part::Box")
        self.assertEqual(box["name"], "Box")
        self.assertEqual(box["label"], "MyBox")
        self.assertEqual(box["placement"]["position"], [10.0, 0.0, 5.0])
        self.assertEqual(len(box["placement"]["rotation"]), 4)
        self.assertEqual(box["shape"]["type"], "Solid")

    def test_object_without_placement_or_shape(self):
        enc = encode_document_context(_sample_context())
        sketch = enc["objects"][1]
        self.assertNotIn("placement", sketch)
        self.assertNotIn("shape", sketch)
        self.assertIsNone(sketch["visibility"] or None)

    def test_empty_document(self):
        ctx = DocumentContext(document=None, objects=[], view=None)
        enc = encode_document_context(ctx)
        self.assertEqual(enc, {"document": None, "objects": [], "view": None})
        dec = decode_document_context(enc)
        self.assertIsNone(dec.document)
        self.assertEqual(dec.objects, [])
        self.assertIsNone(dec.view)

    def test_object_lookup(self):
        ctx = decode_document_context(encode_document_context(_sample_context()))
        self.assertIsNotNone(ctx.object_by_name("Box"))
        self.assertIsNone(ctx.object_by_name("Nope"))

    def test_view_orientation_decode(self):
        ctx = _sample_context()
        axis, angle = ctx.view.orientation_axis_angle()
        self.assertAlmostEqual(angle, 0.0)


class ValidationTests(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(validate_context(_sample_context()), [])

    def test_count_mismatch(self):
        ctx = _sample_context()
        bad = DocumentContext(
            document=DocumentInfo(name="x", object_count=99),
            objects=ctx.objects,
            view=None,
        )
        issues = validate_context(bad)
        self.assertTrue(any("object_count" in i for i in issues))

    def test_malformed_type_id(self):
        ctx = DocumentContext(
            document=DocumentInfo(name="x", object_count=1),
            objects=[FreeCADObject(name="A", label="A", type_id="Box")],
        )
        issues = validate_context(ctx)
        self.assertTrue(any("TypeId" in i for i in issues))

    def test_duplicate_names(self):
        ctx = DocumentContext(
            document=DocumentInfo(name="x", object_count=2),
            objects=[
                FreeCADObject(name="A", label="A", type_id="Part::Box"),
                FreeCADObject(name="A", label="B", type_id="Part::Box"),
            ],
        )
        issues = validate_context(ctx)
        self.assertTrue(any("duplicate" in i for i in issues))

    def test_negative_volume(self):
        ctx = DocumentContext(
            document=DocumentInfo(name="x", object_count=1),
            objects=[
                FreeCADObject(
                    name="A", label="A", type_id="Part::Box",
                    shape=ShapeInfo(type="Solid", volume=-1.0),
                )
            ],
        )
        issues = validate_context(ctx)
        self.assertTrue(any("negative volume" in i for i in issues))


if __name__ == "__main__":
    unittest.main()
