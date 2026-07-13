"""Tests for adapters.ccc_interop_matrix (deterministic, stdlib-only)."""

from __future__ import annotations

import unittest

from harnesscad.io.adapters import ecosystem_catalog as eco
from harnesscad.io.adapters import interop_matrix as ix


class TestExplicitBridges(unittest.TestCase):
    def test_endpoints_are_real_catalogue_systems(self):
        for b in ix.explicit_bridges():
            self.assertTrue(eco.has(b.src), b.src)
            self.assertTrue(eco.has(b.dst), b.dst)
            self.assertIn(b.kind, ix.BRIDGE_KINDS)

    def test_transpilers_emit_openscad(self):
        # The list states SolidPython/scad-clj/scad-hs emit OpenSCAD source.
        for name in ("solidpython", "scad-clj", "scad-hs"):
            self.assertEqual(ix.transpile_targets(name), ["openscad"])

    def test_freecad_embeds_openscad_and_cadquery(self):
        self.assertEqual(ix.embed_targets("freecad"), ["cadquery", "openscad"])

    def test_angelcad_runs_openscad(self):
        self.assertEqual(ix.embed_targets("angelcad"), ["openscad"])

    def test_explicit_bridges_sorted_and_deterministic(self):
        self.assertEqual(ix.explicit_bridges(), ix.explicit_bridges())
        rows = [(b.src, b.dst, b.via) for b in ix.explicit_bridges()]
        self.assertEqual(rows, sorted(rows))


class TestSharedFormats(unittest.TestCase):
    def test_shared_formats_symmetry_of_data(self):
        # cadquery exports step; freecad imports step -> step is shared.
        self.assertIn("step", ix.shared_formats("cadquery", "freecad"))

    def test_interchange_prefers_exact_brep(self):
        # cadquery -> freecad can go via step, stl, dxf, svg; step wins.
        self.assertEqual(ix.interchange_format("cadquery", "freecad"), "step")

    def test_interchange_none_when_no_common_format(self):
        # scad-clj outputs nothing importable by curv.
        self.assertIsNone(ix.interchange_format("scad-clj", "curv"))

    def test_shared_formats_sorted_by_fidelity(self):
        shared = ix.shared_formats("cadquery", "freecad")
        keys = [ix._fidelity_key(f) for f in shared]
        self.assertEqual(keys, sorted(keys))


class TestBridges(unittest.TestCase):
    def test_format_bridges_have_no_self_edges(self):
        for b in ix.format_bridges():
            self.assertNotEqual(b.src, b.dst)
            self.assertEqual(b.kind, ix.BRIDGE_FORMAT)

    def test_format_bridges_deterministic(self):
        self.assertEqual(ix.format_bridges(), ix.format_bridges())

    def test_all_bridges_includes_both_channels(self):
        kinds = {b.kind for b in ix.all_bridges()}
        self.assertIn(ix.BRIDGE_FORMAT, kinds)
        self.assertIn(ix.BRIDGE_TRANSPILE, kinds)
        self.assertIn(ix.BRIDGE_EMBED, kinds)

    def test_can_handoff_format_channel(self):
        # openscad and freecad share mesh/2D formats both ways.
        self.assertTrue(ix.can_handoff("openscad", "freecad"))

    def test_can_handoff_unknown_system_raises(self):
        with self.assertRaises(eco.UnknownSystem):
            ix.can_handoff("nope", "openscad")


class TestQueries(unittest.TestCase):
    def test_consumers_and_producers_are_inverse_relations(self):
        for consumer in ix.consumers_of("cadquery"):
            self.assertIn("cadquery", ix.producers_for(consumer))

    def test_openscad_is_a_common_transpile_target(self):
        producers = ix.producers_for("openscad")
        for t in ("solidpython", "scad-clj", "scad-hs"):
            self.assertIn(t, producers)

    def test_interop_hubs_sorted_desc_by_degree(self):
        hubs = ix.interop_hubs()
        degrees = [d for _, d in hubs]
        self.assertEqual(degrees, sorted(degrees, reverse=True))
        # Every catalogue system appears exactly once.
        self.assertEqual(sorted(n for n, _ in hubs), eco.system_names())


class TestPaths(unittest.TestCase):
    def test_path_to_self_is_empty(self):
        self.assertEqual(ix.handoff_path("openscad", "openscad"), [])

    def test_transpiler_reaches_freecad_via_openscad(self):
        # solidpython -> openscad (transpile) -> freecad (shared format).
        path = ix.handoff_path("solidpython", "freecad")
        self.assertIsNotNone(path)
        self.assertEqual(path[0].src, "solidpython")
        self.assertEqual(path[-1].dst, "freecad")
        # Chain is contiguous.
        for a, b in zip(path, path[1:]):
            self.assertEqual(a.dst, b.src)

    def test_path_is_shortest(self):
        # Direct format handoff should be a single hop.
        path = ix.handoff_path("cadquery", "freecad")
        self.assertEqual(len(path), 1)

    def test_reachable_from_excludes_self(self):
        reach = ix.reachable_from("solidpython")
        self.assertNotIn("solidpython", reach)
        self.assertIn("openscad", reach)

    def test_kinds_filter_restricts_graph(self):
        # Restricting to format-only removes the transpile hop out of solidpython.
        self.assertIsNone(
            ix.handoff_path("solidpython", "freecad", kinds=(ix.BRIDGE_FORMAT,))
        )


class TestMatrix(unittest.TestCase):
    def test_matrix_is_square_over_catalogue(self):
        m = ix.interop_matrix()
        n = len(eco.system_names())
        self.assertEqual(len(m), n)
        for row in m:
            self.assertEqual(len(row), n)
            for cell in row:
                self.assertIn(cell, (0, 1))

    def test_matrix_diagonal_is_zero(self):
        m = ix.interop_matrix()
        for i in range(len(m)):
            self.assertEqual(m[i][i], 0)

    def test_matrix_agrees_with_can_handoff(self):
        names = eco.system_names()
        m = ix.interop_matrix()
        # Spot-check a couple of known relations.
        i = names.index("cadquery")
        j = names.index("freecad")
        self.assertEqual(m[i][j], 1 if ix.can_handoff("cadquery", "freecad") else 0)


if __name__ == "__main__":
    unittest.main()
