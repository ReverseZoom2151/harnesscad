"""Tests for the semantic diff layer (diff.py).

The op-level diff is pure stdlib and always runs. The geometric diff degrades
to a metrics delta on the dependency-free stub (asserted here to never crash)
and, when cadquery/OCCT is installed, reports a real volume delta between two
different solids.
"""

import unittest

from backends.stub import StubBackend
from cisp.ops import (
    NewSketch, AddRectangle, Extrude, Fillet, Chamfer, Hole,
)
from state.opdag import OpDAG
from diff import op_diff, geom_diff, diff_checkpoints, OpDiff, GeomDiff


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_CQ = _cadquery_available()


# ---------------------------------------------------------------------------
# op-level diff
# ---------------------------------------------------------------------------
class TestOpDiff(unittest.TestCase):
    def _base(self):
        return [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", w=20.0, h=10.0),
            Extrude(sketch="sk1", distance=5.0),
        ]

    def test_identical_lists_yield_empty_diff(self):
        ops = self._base()
        d = op_diff(ops, list(ops))
        self.assertIsInstance(d, OpDiff)
        self.assertTrue(d.is_empty())
        self.assertEqual(d.added, [])
        self.assertEqual(d.removed, [])
        self.assertEqual(d.modified, [])
        self.assertEqual(d.unchanged_count, len(ops))
        self.assertEqual(d.render(), "no changes")

    def test_detects_added_removed_and_modified(self):
        a = self._base() + [Chamfer(edges=(), distance=1.0), Hole(
            face_or_sketch="", x=0.0, y=0.0, diameter=6.0)]
        b = self._base() + [Hole(
            face_or_sketch="", x=0.0, y=0.0, diameter=5.4),
            Fillet(edges=(), radius=2.0)]

        d = op_diff(a, b)

        # unchanged prefix (the three base ops).
        self.assertEqual(d.unchanged_count, 3)

        # added: the fillet.
        self.assertEqual(len(d.added), 1)
        self.assertEqual(d.added[0]["tag"], "fillet")

        # removed: the chamfer.
        self.assertEqual(len(d.removed), 1)
        self.assertEqual(d.removed[0]["tag"], "chamfer")

        # modified: the hole, with the diameter before/after values.
        self.assertEqual(len(d.modified), 1)
        m = d.modified[0]
        self.assertEqual(m["tag"], "hole")
        diam = [p for p in m["params"] if p["field"] == "diameter"]
        self.assertEqual(len(diam), 1)
        self.assertEqual(diam[0]["before"], 6.0)
        self.assertEqual(diam[0]["after"], 5.4)

    def test_render_is_readable(self):
        a = self._base() + [Hole(face_or_sketch="", diameter=6.0)]
        b = self._base() + [
            Hole(face_or_sketch="", diameter=5.4),
            Fillet(edges=(), radius=2.0),
        ]
        summary = op_diff(a, b).render()
        self.assertIn("+1 fillet", summary)
        self.assertIn("hole Ø6->5.4", summary)

    def test_accepts_opdags(self):
        a, b = OpDAG(), OpDAG()
        for op in self._base():
            a.append(op)
            b.append(op)
        b.append(Fillet(edges=(), radius=1.0))
        d = op_diff(a, b)
        self.assertEqual(len(d.added), 1)
        self.assertEqual(d.added[0]["tag"], "fillet")
        self.assertTrue(d.removed == [] and d.modified == [])

    def test_to_dict_is_json_serialisable(self):
        import json
        a = self._base()
        b = self._base() + [Fillet(edges=(), radius=1.0)]
        blob = json.dumps(op_diff(a, b).to_dict(), sort_keys=True)
        self.assertIn("fillet", blob)

    def test_diff_checkpoints(self):
        dag = OpDAG()
        for op in self._base():
            dag.append(op)
        dag.checkpoint("v1")
        dag.append(Fillet(edges=(), radius=1.0))
        dag.checkpoint("v2")

        d = diff_checkpoints(dag, "v1", "v2")
        self.assertEqual(len(d.added), 1)
        self.assertEqual(d.added[0]["tag"], "fillet")
        self.assertEqual(d.unchanged_count, 3)

    def test_deterministic(self):
        a = self._base() + [Hole(face_or_sketch="", diameter=6.0)]
        b = self._base() + [Hole(face_or_sketch="", diameter=5.4)]
        self.assertEqual(op_diff(a, b).render(), op_diff(a, b).render())


# ---------------------------------------------------------------------------
# geometric diff
# ---------------------------------------------------------------------------
class TestGeomDiffDegrades(unittest.TestCase):
    def test_stub_degrades_to_metrics_without_crash(self):
        # The stub exposes no OCCT combined shape, so geom_diff must degrade to
        # a metrics delta and never raise.
        a, b = StubBackend(), StubBackend()
        for op in (NewSketch(plane="XY"),
                   AddRectangle(sketch="sk1", w=10.0, h=10.0),
                   Extrude(sketch="sk1", distance=5.0)):
            a.apply(op)
            b.apply(op)

        d = geom_diff(a, b)
        self.assertIsInstance(d, GeomDiff)
        self.assertEqual(d.mode, "metrics")
        self.assertFalse(d.available)
        self.assertIsNotNone(d.reason)
        # stub reports no measurable volume -> zero delta, still JSON-safe.
        self.assertEqual(d.volume_delta, d.volume_b - d.volume_a)
        import json
        json.dumps(d.to_dict())
        self.assertIsInstance(d.render(), str)


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestGeomDiffReal(unittest.TestCase):
    def _plate(self, w, h, t):
        from backends.cadquery_backend import CadQueryBackend
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h))
        b.apply(Extrude(sketch="sk1", distance=t))
        return b

    def test_reports_volume_delta_between_two_solids(self):
        a = self._plate(20.0, 10.0, 5.0)   # volume 1000
        b = self._plate(20.0, 10.0, 6.0)   # volume 1200
        d = geom_diff(a, b)

        self.assertEqual(d.mode, "boolean")
        self.assertTrue(d.available)
        self.assertAlmostEqual(d.volume_a, 1000.0, places=3)
        self.assertAlmostEqual(d.volume_b, 1200.0, places=3)
        self.assertAlmostEqual(d.volume_delta, 200.0, places=3)
        # the taller plate adds material, removes none.
        self.assertGreater(d.added_volume, 0.0)
        self.assertAlmostEqual(d.removed_volume, 0.0, places=3)
        self.assertIsNotNone(d.face_count_delta)


if __name__ == "__main__":
    unittest.main()
