"""Tests for the Rhino ``.3dm`` (openNURBS / rhino3dm) container codec.

These require the optional ``rhino3dm`` wheel; when it is not installed the whole
suite is skipped cleanly (the codec module still imports -- rhino3dm is guarded).

The invariant under test is a GEOMETRY + UNIT round trip: a box written to a
``.3dm`` and read back must return the same triangle mesh, the same bounding box
(measured by rhino3dm's own kernel), the same volume, and -- critically -- the
same declared unit. Rhino has its own unit system, so the unit is exercised
explicitly (a metre-unit file must not silently come back as millimetres).
"""

import math
import os
import tempfile
import unittest

from harnesscad.io.formats import threedm
from harnesscad.io.formats import registry as fmt
from harnesscad.io import gate


HAVE_R3 = threedm.RHINO3DM_AVAILABLE

# A 10 x 20 x 30 box as 8 vertices / 6 quad faces.
_BOX_V = [(0, 0, 0), (10, 0, 0), (10, 20, 0), (0, 20, 0),
          (0, 0, 30), (10, 0, 30), (10, 20, 30), (0, 20, 30)]
_BOX_F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
          (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
_BOX_VOLUME = 10.0 * 20.0 * 30.0
_BOX_BBOX = [10.0, 20.0, 30.0]


@unittest.skipUnless(HAVE_R3, "rhino3dm is not installed")
class ThreeDmCodecTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()

    def _path(self, name: str = "part.3dm") -> str:
        return os.path.join(self._tmp, name)

    def test_write_read_roundtrip_geometry(self) -> None:
        path = self._path()
        threedm.write_3dm(path, _BOX_V, _BOX_F, unit="millimeter", name="box")
        self.assertTrue(os.path.getsize(path) > 0)
        verts, tris, unit = threedm.read_3dm(path)
        self.assertEqual(unit, "millimeter")
        # 6 quads -> 12 triangles.
        self.assertEqual(len(tris), 12)
        # The volume and bounding box survive the round trip.
        m = gate.measure(verts, tris)
        self.assertAlmostEqual(m["volume"], _BOX_VOLUME, places=6)
        for got, want in zip(m["bbox"], _BOX_BBOX):
            self.assertAlmostEqual(got, want, places=6)
        self.assertTrue(m["watertight"])
        self.assertTrue(m["manifold"])

    def test_unit_is_explicit_and_survives(self) -> None:
        # A non-default unit must round-trip exactly, not silently become mm.
        for unit in ("meter", "inch", "foot", "centimeter", "micron"):
            path = self._path("u_%s.3dm" % unit)
            threedm.write_3dm(path, _BOX_V, _BOX_F, unit=unit)
            _v, _t, got = threedm.read_3dm(path)
            self.assertEqual(got, unit)
            measured = threedm.measure_3dm(path)
            self.assertEqual(measured["unit"], unit)
            # The unit is a label; the coordinate magnitudes are unchanged by it.
            for g, w in zip(measured["bbox"], _BOX_BBOX):
                self.assertAlmostEqual(g, w, places=6)

    def test_rhino_bounding_box_matches_analytic(self) -> None:
        # measure_3dm reads rhino3dm's OWN GetBoundingBox -- an independent path.
        path = self._path()
        threedm.write_3dm(path, _BOX_V, _BOX_F, unit="millimeter")
        m = threedm.measure_3dm(path)
        for got, want in zip(m["bbox"], _BOX_BBOX):
            self.assertAlmostEqual(got, want, places=6)
        self.assertEqual(m["vertex_count"], 8)

    def test_serialize_bytes_roundtrip(self) -> None:
        data = threedm.serialize_3dm(_BOX_V, _BOX_F, unit="millimeter")
        self.assertIsInstance(data, bytes)
        path = self._path("from_bytes.3dm")
        with open(path, "wb") as fh:
            fh.write(data)
        _v, tris, unit = threedm.read_3dm(path)
        self.assertEqual(unit, "millimeter")
        self.assertEqual(len(tris), 12)

    def test_unknown_unit_refused(self) -> None:
        with self.assertRaises(threedm.ThreeDmError):
            threedm.write_3dm(self._path(), _BOX_V, _BOX_F, unit="furlong")

    def test_registry_roundtrip_through_gate(self) -> None:
        # The .3dm codec must route through the format registry and its gate,
        # exactly like stl/obj/ply/3mf.
        spec = fmt.spec_for_extension(".3dm")
        self.assertEqual(spec.name, "threedm")
        self.assertTrue(spec.can_read)
        self.assertTrue(spec.can_write)
        self.assertTrue(spec.round_trip)
        self.assertEqual(spec.kind, "mesh")

        mesh = fmt.Mesh.from_vertices_faces(_BOX_V, _BOX_F, name="box", unit="meter")
        path = self._path("via_registry.3dm")
        fmt.write(mesh, path)
        back = fmt.read(path)
        self.assertEqual(back.unit, "meter")
        v, f = back.indexed()
        m = gate.measure(v, f)
        self.assertAlmostEqual(m["volume"], _BOX_VOLUME, places=6)


class ThreeDmAvailabilityTest(unittest.TestCase):
    """These must hold whether or not rhino3dm is installed."""

    def test_module_imports_without_rhino3dm(self) -> None:
        # Importing the codec never requires the wheel.
        self.assertIn("write_3dm", threedm.__all__)
        self.assertIn("read_3dm", threedm.__all__)

    def test_units_vocabulary(self) -> None:
        self.assertEqual(
            set(threedm.UNITS),
            {"micron", "millimeter", "centimeter", "inch", "foot", "meter"})

    @unittest.skipIf(HAVE_R3, "rhino3dm IS installed here")
    def test_clean_error_when_unavailable(self) -> None:  # pragma: no cover
        with self.assertRaises(threedm.ThreeDmError):
            threedm.write_3dm("x.3dm", _BOX_V, _BOX_F)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
