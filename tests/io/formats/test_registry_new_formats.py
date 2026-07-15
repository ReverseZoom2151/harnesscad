"""3MF and PLY are discovered by the format registry and route through the gate.

The point of registering a codec is that ``FormatSpec.write`` sends it through
``io/gate.py`` automatically -- one ungated codec is a hole in the hull. These
tests exercise the full ``fmt.write -> gate -> codec`` and ``fmt.read`` path and
assert the axis and unit conventions survive the round trip.
"""

import os
import tempfile
import unittest

from harnesscad.io.formats import registry as fmt
from harnesscad.io.formats.registry import Mesh
from harnesscad.domain.geometry.mesh.polyhedron import Polyhedron, unit_cube


def _box_mesh(sx, sy, sz, unit="millimeter"):
    c = unit_cube(1.0)
    verts = [(x * sx, y * sy, z * sz) for (x, y, z) in c.vertices]
    return Mesh.from_vertices_faces(verts, c.faces, unit=unit)


class TestDiscovery(unittest.TestCase):
    def test_new_formats_are_discovered(self):
        names = {s.name for s in fmt.specs(refresh=True)}
        self.assertIn("ply", names)
        self.assertIn("threemf", names)

    def test_new_formats_advertise_round_trip(self):
        by_name = {s.name: s for s in fmt.specs()}
        self.assertTrue(by_name["ply"].round_trip)
        self.assertTrue(by_name["threemf"].round_trip)

    def test_extensions_route(self):
        self.assertEqual(fmt.spec_for_extension(".ply").name, "ply")
        self.assertEqual(fmt.spec_for_extension(".3mf").name, "threemf")


class TestGatedRoundTrip(unittest.TestCase):
    def _round_trip(self, ext, unit):
        mesh = _box_mesh(1.0, 2.0, 4.0, unit=unit)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m" + ext)
            fmt.write(mesh, path)               # through the gate
            back = fmt.read(path)
        p = Polyhedron(*back.indexed())
        lo, hi = p.bounds()
        # Volume survives.
        self.assertAlmostEqual(p.volume(), 8.0)
        # Axis convention: no rotation / permutation.
        self.assertEqual(lo, (0.0, 0.0, 0.0))
        self.assertEqual(hi, (1.0, 2.0, 4.0))
        return back

    def test_ply_gated_round_trip(self):
        back = self._round_trip(".ply", "inch")
        self.assertEqual(back.unit, "inch")

    def test_threemf_gated_round_trip(self):
        back = self._round_trip(".3mf", "inch")
        self.assertEqual(back.unit, "inch")

    def test_gate_refuses_a_broken_mesh(self):
        # A single triangle is not a closed watertight solid; the gate must refuse
        # it rather than write a bogus artifact -- proving the codec is gated.
        from harnesscad.io.formats.stl import Triangle
        from harnesscad.io import gate
        bad = Mesh((Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0)),))
        with tempfile.TemporaryDirectory() as d:
            for ext in (".ply", ".3mf"):
                path = os.path.join(d, "bad" + ext)
                with self.assertRaises(gate.InvalidArtifact):
                    fmt.write(bad, path)
                self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
