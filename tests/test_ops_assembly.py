"""Tests for the assembly ops (AddInstance / Mate), the SetParam editability
primitive, and the ``query('assembly')`` surface the backends now produce.

Covered:
  * parse_op / to_dict / canonical_json round-trip for the three new ops
    (including SetParam values of int / float / str);
  * apply on the dependency-free StubBackend: AddInstance + Mate succeed, and
    block-and-correct (ok=False, no mutation) on unknown part / mate refs and
    bad mate kinds;
  * query('assembly') returns the {parts, mates, transforms} shape and drives
    checks_assembly.AssemblyCheck to a sensible residual-DOF result;
  * SetParam edits a prior op's param and the state digest changes
    deterministically (and block-and-corrects on a bad target / param);
  * guarded by cadquery: a two-instance overlapping assembly yields a
    query('assembly') that InterferenceCheck flags as a hard clash.
"""

import unittest

from harnesscad.core.cisp.ops import (
    AddInstance, Mate, SetParam, NewSketch, AddRectangle, Extrude,
    parse_op, canonical_json,
)
from harnesscad.io.backends.stub import StubBackend
from harnesscad.eval.verifiers.verify import Severity
from harnesscad.eval.verifiers.assembly import AssemblyCheck, AssemblyModel
from harnesscad.eval.verifiers.interference import InterferenceCheck


def _codes(report):
    return {d.code for d in report.diagnostics}


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_CQ = _cadquery_available()


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #
class TestNewOpRoundTrip(unittest.TestCase):
    def test_add_instance_round_trip(self):
        op = AddInstance(part="f1", x=1.0, y=2.0, z=3.0, rx=0.0, ry=90.0, rz=0.0)
        restored = parse_op(op.to_dict())
        self.assertEqual(op, restored)
        self.assertIs(type(op), type(restored))

    def test_mate_round_trip(self):
        op = Mate(kind="revolute", a="i1", b="i2", value=None)
        restored = parse_op(op.to_dict())
        self.assertEqual(op, restored)
        op2 = Mate(kind="cylindrical", a="i1", b="i2", value=12.5)
        self.assertEqual(op2, parse_op(op2.to_dict()))

    def test_set_param_round_trip_all_scalar_types(self):
        for value in (5, 2.5, "counterbore"):
            op = SetParam(target=1, param="distance", value=value)
            with self.subTest(value=value):
                restored = parse_op(op.to_dict())
                self.assertEqual(op, restored)
                self.assertEqual(restored.value, value)

    def test_canonical_json_is_sorted_and_stable(self):
        op = AddInstance(part="f1", x=1.0, y=2.0, z=3.0)
        self.assertEqual(canonical_json(op), canonical_json(op))
        d = op.to_dict()
        keys = sorted(d.keys())
        positions = [canonical_json(op).index('"%s"' % k) for k in keys]
        self.assertEqual(positions, sorted(positions))


# --------------------------------------------------------------------------- #
# StubBackend apply + block-and-correct
# --------------------------------------------------------------------------- #
def _plate_stub() -> StubBackend:
    b = StubBackend()
    assert b.apply(NewSketch(plane="XY")).ok
    assert b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0)).ok
    assert b.apply(Extrude(sketch="sk1", distance=5.0)).ok  # -> feature f1
    return b


class TestStubAssemblyApply(unittest.TestCase):
    def test_add_instance_ok_and_tracked(self):
        b = _plate_stub()
        res = b.apply(AddInstance(part="f1", x=1.0, y=0.0, z=0.0))
        self.assertTrue(res.ok)
        self.assertEqual(res.created, ["i1"])
        self.assertEqual(len(b.instances), 1)

    def test_add_instance_unknown_part_blocks(self):
        b = _plate_stub()
        before = b.state_digest()
        res = b.apply(AddInstance(part="ghost"))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-ref")
        self.assertEqual(b.state_digest(), before)  # no mutation

    def test_mate_ok(self):
        b = _plate_stub()
        b.apply(AddInstance(part="f1"))
        b.apply(AddInstance(part="f1", x=5.0))
        res = b.apply(Mate(kind="revolute", a="i1", b="i2"))
        self.assertTrue(res.ok)
        self.assertEqual(len(b.mates), 1)

    def test_mate_unknown_ref_blocks(self):
        b = _plate_stub()
        b.apply(AddInstance(part="f1"))
        before = b.state_digest()
        res = b.apply(Mate(kind="revolute", a="i1", b="nope"))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-ref")
        self.assertEqual(b.state_digest(), before)

    def test_mate_bad_kind_blocks(self):
        b = _plate_stub()
        b.apply(AddInstance(part="f1"))
        b.apply(AddInstance(part="f1", x=5.0))
        res = b.apply(Mate(kind="wobble", a="i1", b="i2"))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-value")


# --------------------------------------------------------------------------- #
# query('assembly') shape + AssemblyCheck integration
# --------------------------------------------------------------------------- #
class TestAssemblyQuery(unittest.TestCase):
    def test_empty_when_no_instances(self):
        self.assertEqual(_plate_stub().query("assembly"), {})

    def test_shape_of_assembly_query(self):
        b = _plate_stub()
        b.apply(AddInstance(part="f1", x=0.0))
        b.apply(AddInstance(part="f1", x=30.0))
        b.apply(Mate(kind="revolute", a="i1", b="i2"))
        asm = b.query("assembly")
        self.assertEqual({p["id"] for p in asm["parts"]}, {"i1", "i2"})
        self.assertEqual(asm["parts"][1]["transform"]["translate"], [30.0, 0.0, 0.0])
        self.assertEqual(asm["mates"][0]["kind"], "revolute")
        self.assertIn("i1", asm["transforms"])

    def test_assembly_check_reports_residual_dof(self):
        # Two free parts + one revolute mate -> 6*2 - 5 = 7 DOF (under).
        b = _plate_stub()
        b.apply(AddInstance(part="f1"))
        b.apply(AddInstance(part="f1", x=30.0))
        b.apply(Mate(kind="revolute", a="i1", b="i2"))
        model = AssemblyModel.from_dict(b.query("assembly"))
        self.assertEqual(model.residual_dof(), 7)
        report = AssemblyCheck().check(b, None)
        self.assertIn("assembly-dof", _codes(report))
        self.assertIn("under-constrained", _codes(report))
        self.assertTrue(report.ok)  # under-constrained is only a WARNING

    def test_stub_still_info_skips_before_any_instance(self):
        report = AssemblyCheck().check(StubBackend(), None)
        self.assertIn("assembly-skipped", _codes(report))
        self.assertTrue(report.ok)


# --------------------------------------------------------------------------- #
# SetParam editability
# --------------------------------------------------------------------------- #
class TestSetParam(unittest.TestCase):
    def test_set_param_changes_digest_deterministically(self):
        b = _plate_stub()
        b.apply(AddInstance(part="f1", x=1.0))          # oplog index 3
        before = b.state_digest()
        res = b.apply(SetParam(target=3, param="x", value=9.0))
        self.assertTrue(res.ok)
        after = b.state_digest()
        self.assertNotEqual(before, after)
        self.assertEqual(b.instances[0]["transform"]["translate"][0], 9.0)

        # Deterministic: an identical build + edit yields the same digest.
        b2 = _plate_stub()
        b2.apply(AddInstance(part="f1", x=1.0))
        b2.apply(SetParam(target=3, param="x", value=9.0))
        self.assertEqual(after, b2.state_digest())

        # And equals building it directly with the edited value.
        b3 = _plate_stub()
        b3.apply(AddInstance(part="f1", x=9.0))
        self.assertEqual(after, b3.state_digest())

    def test_set_param_edits_upstream_op_and_replays(self):
        # Edit the extrude distance (oplog index 2); replay regenerates.
        b = _plate_stub()
        before = b.state_digest()
        res = b.apply(SetParam(target=2, param="distance", value=12.0))
        self.assertTrue(res.ok)
        self.assertNotEqual(before, b.state_digest())
        self.assertTrue(b.query("summary")["solid_present"])

    def test_set_param_bad_target_blocks(self):
        b = _plate_stub()
        before = b.state_digest()
        res = b.apply(SetParam(target=99, param="x", value=1.0))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-ref")
        self.assertEqual(b.state_digest(), before)

    def test_set_param_bad_param_blocks(self):
        b = _plate_stub()
        before = b.state_digest()
        res = b.apply(SetParam(target=2, param="not_a_field", value=1.0))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-param")
        self.assertEqual(b.state_digest(), before)


# --------------------------------------------------------------------------- #
# Guarded by cadquery: exact interference on placed instances
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestCadQueryAssemblyInterference(unittest.TestCase):
    def _plate_cq(self):
        from harnesscad.io.backends.cadquery_backend import CadQueryBackend
        b = CadQueryBackend()
        assert b.apply(NewSketch(plane="XY")).ok
        assert b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=10.0)).ok
        assert b.apply(Extrude(sketch="sk1", distance=10.0)).ok
        return b

    def test_parts_carry_real_bbox_and_shape(self):
        b = self._plate_cq()
        b.apply(AddInstance(part="f1"))
        asm = b.query("assembly")
        part = asm["parts"][0]
        self.assertIn("bbox", part)
        self.assertEqual(len(part["bbox"]), 6)
        self.assertIsNotNone(part.get("shape"))

    def test_overlapping_instances_flagged(self):
        b = self._plate_cq()
        b.apply(AddInstance(part="f1", x=0.0))    # box at origin
        b.apply(AddInstance(part="f1", x=5.0))    # overlaps by 5mm in x
        report = InterferenceCheck().check(b, None)
        self.assertIn("interference", _codes(report))  # exact OCCT clash
        self.assertFalse(report.ok)

    def test_disjoint_instances_clear(self):
        b = self._plate_cq()
        b.apply(AddInstance(part="f1", x=0.0))
        b.apply(AddInstance(part="f1", x=50.0))
        report = InterferenceCheck().check(b, None)
        self.assertEqual([d for d in report.diagnostics
                          if d.severity is Severity.ERROR], [])
        self.assertTrue(report.ok)


if __name__ == "__main__":
    unittest.main()
