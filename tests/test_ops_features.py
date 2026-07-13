"""Tests for the extended mechanical-op vocabulary (revolve, chamfer, hole,
shell, draft, loft, sweep, patterns, mirror) plus the query('metrics') and
STEP/STL/IGES export additions.

Every new op is exercised three ways:
  1. structural  — parse_op(to_dict) round-trips and canonical_json is stable;
  2. stub        — applies on StubBackend (ok, correct ids, block-and-correct on
                   a bad reference), with zero geometry dependency;
  3. real (opt.) — guarded by cadquery availability, Revolve/Chamfer/Hole/Shell
                   yield a valid OCCT solid with sensible query('metrics') volume
                   and a non-empty STL export.
"""

import unittest

from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.cisp.ops import (
    NewSketch, AddRectangle,
    Revolve, Chamfer, Hole, Shell, Draft,
    Loft, Sweep, LinearPattern, CircularPattern, Mirror,
    parse_op, canonical_json, _REGISTRY,
)


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_CQ = _cadquery_available()

NEW_OPS = [
    Revolve(sketch="sk1", axis=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0), angle=270.0),
    Chamfer(edges=("e1", "e2"), distance=0.5),
    Hole(face_or_sketch="f1", x=1.0, y=2.0, diameter=3.0, depth=None, through=True),
    Hole(face_or_sketch="f1", x=0.0, y=0.0, diameter=2.0, depth=4.0, through=False),
    Shell(faces=("f2",), thickness=1.5),
    Draft(faces=("f3",), angle=5.0, neutral_plane="XY"),
    Loft(sketches=("sk1", "sk2"), ruled=True),
    Sweep(sketch="sk1", path="sk2"),
    LinearPattern(feature="f1", direction=(1.0, 0.0, 0.0), count=3, spacing=10.0),
    CircularPattern(feature="f1", axis=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0), count=6, angle=360.0),
    Mirror(feature_or_body="f1", plane="XZ"),
]


class TestRegistryAndRoundTrip(unittest.TestCase):
    def test_all_new_ops_registered(self):
        for tag in ("revolve", "chamfer", "hole", "shell", "draft",
                    "loft", "sweep", "linear_pattern", "circular_pattern", "mirror"):
            self.assertIn(tag, _REGISTRY)

    def test_round_trip_via_parse_op(self):
        for op in NEW_OPS:
            d = op.to_dict()
            back = parse_op(d)
            self.assertEqual(back, op, f"round-trip failed for {op}")

    def test_tuple_fields_survive_json(self):
        # JSON turns tuples into lists; parse_op must re-tuple so ops stay equal
        # and hashable (frozen dataclasses).
        op = Revolve(sketch="sk1", axis=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        back = parse_op(op.to_dict())
        self.assertIsInstance(back.axis, tuple)
        self.assertEqual(back, op)
        # canonical_json is deterministic (sorted keys).
        self.assertEqual(canonical_json(op), canonical_json(parse_op(op.to_dict())))


def _plate_stub(w=20.0, h=10.0) -> StubBackend:
    b = StubBackend()
    assert b.apply(NewSketch(plane="XY")).ok
    assert b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h)).ok
    return b


class TestStubApplies(unittest.TestCase):
    def test_revolve_creates_feature(self):
        b = _plate_stub()
        res = b.apply(Revolve(sketch="sk1", angle=360.0))
        self.assertTrue(res.ok)
        self.assertEqual(res.created, ["f1"])
        self.assertTrue(b.query("summary")["solid_present"])

    def test_revolve_bad_ref_blocks(self):
        b = _plate_stub()
        before = b.state_digest()
        res = b.apply(Revolve(sketch="nope"))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-ref")
        self.assertEqual(b.state_digest(), before)

    def test_chamfer_requires_solid(self):
        b = _plate_stub()
        self.assertFalse(b.apply(Chamfer(edges=(), distance=1.0)).ok)
        b.apply(Revolve(sketch="sk1"))  # gives the stub a solid
        res = b.apply(Chamfer(edges=(), distance=1.0))
        self.assertTrue(res.ok)
        self.assertFalse(b.apply(Chamfer(edges=(), distance=0.0)).ok)

    def test_hole_semantics_and_block(self):
        b = _plate_stub()
        b.apply(Revolve(sketch="sk1"))  # gives us a solid
        self.assertTrue(b.apply(Hole(face_or_sketch="", x=1, y=1, diameter=2.0)).ok)
        # bad-value: non-positive diameter
        self.assertFalse(b.apply(Hole(diameter=0.0)).ok)
        # bad-ref: sketch-looking ref that does not exist
        res = b.apply(Hole(face_or_sketch="sk999", diameter=1.0))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-ref")

    def test_shell_and_draft(self):
        b = _plate_stub()
        b.apply(Revolve(sketch="sk1"))
        self.assertTrue(b.apply(Shell(faces=(), thickness=1.0)).ok)
        self.assertFalse(b.apply(Shell(thickness=-1.0)).ok)
        self.assertTrue(b.apply(Draft(angle=3.0, neutral_plane="XY")).ok)
        self.assertFalse(b.apply(Draft(angle=3.0, neutral_plane="")).ok)

    def test_loft_and_sweep_refs(self):
        b = StubBackend()
        b.apply(NewSketch())
        b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=5, h=5))
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk2", x=0, y=0, w=3, h=3))
        self.assertTrue(b.apply(Loft(sketches=("sk1", "sk2"), ruled=False)).ok)
        self.assertTrue(b.apply(Sweep(sketch="sk1", path="sk2")).ok)
        # bad-ref propagates
        self.assertFalse(b.apply(Loft(sketches=("sk1", "nope"))).ok)
        self.assertFalse(b.apply(Sweep(sketch="sk1", path="nope")).ok)

    def test_patterns_and_mirror(self):
        b = _plate_stub()
        r = b.apply(Revolve(sketch="sk1"))
        fid = r.created[0]
        self.assertTrue(b.apply(LinearPattern(feature=fid, count=3, spacing=5.0)).ok)
        self.assertTrue(b.apply(CircularPattern(feature=fid, count=4)).ok)
        self.assertTrue(b.apply(Mirror(feature_or_body=fid, plane="XZ")).ok)
        # bad feature ref
        res = b.apply(LinearPattern(feature="f999", count=2, spacing=1.0))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-ref")
        # count < 2
        self.assertFalse(b.apply(CircularPattern(feature=fid, count=1)).ok)

    def test_stub_metrics_is_empty(self):
        b = _plate_stub()
        b.apply(Revolve(sketch="sk1"))
        self.assertEqual(b.query("metrics"), {})  # callers INFO-skip


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestRealGeometry(unittest.TestCase):
    def _cq_plate(self, w=20.0, h=10.0, t=5.0):
        from harnesscad.io.backends.cadquery_backend import CadQueryBackend
        from harnesscad.core.cisp.ops import Extrude
        b = CadQueryBackend()
        self.assertTrue(b.apply(NewSketch(plane="XY")).ok)
        self.assertTrue(b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h)).ok)
        self.assertTrue(b.apply(Extrude(sketch="sk1", distance=t)).ok)
        return b

    def test_revolve_makes_valid_solid_with_metrics(self):
        from harnesscad.io.backends.cadquery_backend import CadQueryBackend
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        # rectangle offset from the Y axis -> a valid annular solid of revolution
        b.apply(AddRectangle(sketch="sk1", x=5.0, y=0.0, w=2.0, h=4.0))
        res = b.apply(Revolve(sketch="sk1", axis=(0, 0, 0, 0, 1, 0), angle=360.0))
        self.assertTrue(res.ok, res.diagnostics and res.diagnostics[0].message)
        self.assertTrue(b.query("validity")["is_valid"])
        m = b.query("metrics")
        self.assertGreater(m["volume"], 0.0)
        self.assertAlmostEqual(m["mass"], m["volume"], places=6)  # density 1.0
        self.assertGreater(m["surface_area"], 0.0)
        self.assertEqual(len(m["bbox"]), 3)
        self.assertEqual(len(m["center_of_mass"]), 3)

    def test_chamfer_keeps_valid_solid(self):
        b = self._cq_plate()
        vol0 = b.query("metrics")["volume"]
        res = b.apply(Chamfer(edges=(), distance=1.0))
        self.assertTrue(res.ok, res.diagnostics and res.diagnostics[0].message)
        self.assertTrue(b.query("validity")["is_valid"])
        self.assertLess(b.query("metrics")["volume"], vol0)  # material removed

    def test_hole_cuts_material(self):
        b = self._cq_plate()
        vol0 = b.query("metrics")["volume"]
        res = b.apply(Hole(face_or_sketch="", x=10.0, y=5.0, diameter=3.0, through=True))
        self.assertTrue(res.ok, res.diagnostics and res.diagnostics[0].message)
        self.assertTrue(b.query("validity")["is_valid"])
        self.assertLess(b.query("metrics")["volume"], vol0)

    def test_shell_hollows_solid(self):
        b = self._cq_plate(t=8.0)
        vol0 = b.query("metrics")["volume"]
        res = b.apply(Shell(faces=(), thickness=1.0))
        self.assertTrue(res.ok, res.diagnostics and res.diagnostics[0].message)
        self.assertTrue(b.query("validity")["is_valid"])
        self.assertLess(b.query("metrics")["volume"], vol0)

    def test_export_stl_and_iges_nonempty(self):
        b = self._cq_plate()
        stl = b.export("stl")
        self.assertTrue(stl)
        iges = b.export("iges")
        self.assertTrue(iges)
        self.assertIn("S", iges)  # IGES sections are line-tagged (S/G/D/P/T)

    def test_linear_pattern_grows_bbox(self):
        b = self._cq_plate(w=5.0, h=5.0, t=5.0)
        r = b.apply(Revolve(sketch="sk1"))  # noqa: F841 - just to have a feature id
        # use the extrude feature id (f1) as the pattern reference
        res = b.apply(LinearPattern(feature="f1", direction=(1, 0, 0), count=3, spacing=20.0))
        self.assertTrue(res.ok, res.diagnostics and res.diagnostics[0].message)
        self.assertGreater(b.query("metrics")["bbox"][0], 5.0)

    def test_mirror_produces_valid_solid(self):
        b = self._cq_plate()
        res = b.apply(Mirror(feature_or_body="f1", plane="YZ"))
        self.assertTrue(res.ok, res.diagnostics and res.diagnostics[0].message)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_draft_loft_sweep_are_typed_unsupported(self):
        from harnesscad.io.backends.cadquery_backend import CadQueryBackend
        b = self._cq_plate()
        d = b.apply(Draft(angle=5.0, neutral_plane="XY"))
        self.assertFalse(d.ok)
        self.assertEqual(d.diagnostics[0].code, "not-yet-supported")
        # loft: build two profile sketches, expect typed unsupported (not bad-ref)
        b2 = CadQueryBackend()
        b2.apply(NewSketch())
        b2.apply(AddRectangle(sketch="sk1", x=0, y=0, w=5, h=5))
        b2.apply(NewSketch())
        b2.apply(AddRectangle(sketch="sk2", x=0, y=0, w=3, h=3))
        lo = b2.apply(Loft(sketches=("sk1", "sk2")))
        self.assertFalse(lo.ok)
        self.assertEqual(lo.diagnostics[0].code, "not-yet-supported")
        sw = b2.apply(Sweep(sketch="sk1", path="sk2"))
        self.assertFalse(sw.ok)
        self.assertEqual(sw.diagnostics[0].code, "not-yet-supported")


if __name__ == "__main__":
    unittest.main()
