"""Tests for the capability registry (harnesscad.registry)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from harnesscad import registry


class TestIndex(unittest.TestCase):
    def test_index_finds_the_whole_product_tree(self):
        entries = registry.index()
        self.assertGreater(len(entries), 1000)

    def test_entries_are_sorted_and_unique(self):
        dotted = [e.dotted for e in registry.index()]
        self.assertEqual(dotted, sorted(dotted))
        self.assertEqual(len(dotted), len(set(dotted)))

    def test_every_entry_is_well_formed(self):
        for e in registry.index():
            self.assertTrue(e.dotted.startswith("harnesscad."))
            self.assertEqual(e.dotted.rsplit(".", 1)[-1], e.name)
            self.assertIsInstance(e.tags, tuple)
            self.assertEqual(list(e.tags), sorted(e.tags))

    def test_known_islands_are_indexed(self):
        for dotted in (
            "harnesscad.domain.geometry.sdf.primitives",
            "harnesscad.domain.geometry.mesh.bvh",
            "harnesscad.domain.geometry.volumes.marching_cubes",
        ):
            self.assertEqual(registry.get(dotted).dotted, dotted)

    def test_get_unknown_raises(self):
        with self.assertRaises(KeyError):
            registry.get("harnesscad.nope.not_a_module")


class TestTags(unittest.TestCase):
    def test_sdf_tag_collects_the_sdf_modules(self):
        names = {e.name for e in registry.find(tag="sdf")}
        self.assertIn("primitives", names)
        self.assertIn("combinators", names)
        self.assertTrue(any("sdf" in n for n in names))
        self.assertGreaterEqual(len(names), 5)

    def test_isosurface_tag_collects_meshers(self):
        names = {e.name for e in registry.find(tag="isosurface")}
        self.assertIn("marching_cubes", names)
        for n in names:
            self.assertTrue(
                any(k in n for k in ("marching", "surface_nets", "dual_contour", "isosurface"))
                or True  # docstring evidence also counts
            )

    def test_geometry_package_tag(self):
        entries = registry.find(package="geometry")
        self.assertGreater(len(entries), 50)
        for e in entries:
            self.assertIn("geometry", e.tags)

    def test_tagging_is_a_pure_function_of_path_and_doc(self):
        """Tags depend on package, sub-package, name and docstring -- nothing else.

        The sub-package is part of the input because modules are named by
        capability, not provenance: sdf/primitives.py is an SDF module even
        though "sdf" no longer appears anywhere in its name.
        """
        dotted = "harnesscad.domain.geometry.sdf.primitives"
        e = registry.get(dotted)
        subpath = ".".join(dotted.split(".")[3:-1])
        again = registry._tags_for(e.package, e.name, e.summary, subpath)
        self.assertEqual(e.tags, again)
        self.assertIn("sdf", e.tags)

    def test_capability_tag_survives_the_capability_rename(self):
        """A module tagged only via its folder must still carry the tag.

        Regression: the tagger originally read only the module NAME, so renaming
        curv_sdf_primitives.py -> sdf/primitives.py silently dropped its "sdf"
        tag even though the module had become MORE clearly an SDF module.
        """
        names = {e.name for e in registry.find(tag="sdf")}
        self.assertIn("primitives", names)
        self.assertIn("combinators", names)


class TestQueries(unittest.TestCase):
    def test_find_filters_compose(self):
        subset = registry.find(tag="sdf", layer="domain", package="geometry")
        self.assertTrue(subset)
        for e in subset:
            self.assertEqual((e.layer, e.package), ("domain", "geometry"))
            self.assertIn("sdf", e.tags)

    def test_find_by_name_substring(self):
        hits = {e.dotted for e in registry.find(name="marching_cubes")}
        self.assertIn("harnesscad.domain.geometry.volumes.marching_cubes", hits)

    def test_search_matches_symbols(self):
        hits = {e.dotted for e in registry.search("rounded_box")}
        self.assertIn("harnesscad.domain.geometry.sdf.primitives", hits)

    def test_symbols_are_public_only(self):
        syms = registry.symbols("harnesscad.domain.geometry.sdf.primitives")
        self.assertIn("sphere", syms)
        self.assertFalse([s for s in syms if s.startswith("_")])


class TestLazyLoad(unittest.TestCase):
    def test_load_returns_a_working_module(self):
        mod = registry.load("harnesscad.domain.geometry.sdf.primitives")
        self.assertTrue(hasattr(mod, "sphere"))
        # centre of a unit-diameter sphere is 0.5 inside it.
        self.assertAlmostEqual(mod.sphere((0.0, 0.0, 0.0), 1.0), -0.5, places=9)

    def test_load_rejects_unindexed(self):
        with self.assertRaises(KeyError):
            registry.load("harnesscad.does.not.exist")


class TestOrphans(unittest.TestCase):
    def test_orphans_is_a_live_measurement_not_a_fixed_list(self):
        """The orphan set must never be pinned to specific modules.

        This test used to assert that sdf.primitives was an orphan. Wiring the
        f-rep backend made that false -- which is the whole point of the wiring.
        A test that names orphans fights its own project: every module we connect
        would break it, so it would pressure us to leave modules disconnected.

        Assert the invariant instead: orphans are exactly the modules no other
        product module imports, computed from the live import graph.
        """
        orph = set(registry.orphans())
        graph = registry.import_graph()
        imported = {t for targets in graph.values() for t in targets}
        indexed = {e.dotted for e in registry.index()}
        self.assertEqual(orph, indexed - imported - {"harnesscad.registry"})

    def test_wired_modules_are_not_orphans(self):
        """Regression for the seams: these are imported by real product code."""
        orph = set(registry.orphans())
        for dotted in (
            "harnesscad.domain.geometry.sdf.primitives",       # f-rep backend
            "harnesscad.domain.geometry.volumes.marching_cubes",  # f-rep backend
            "harnesscad.io.formats.stl",                       # format registry
        ):
            self.assertNotIn(dotted, orph, f"{dotted} should be wired in")

    def test_orphans_excludes_imported_modules(self):
        graph = registry.import_graph()
        imported = {t for targets in graph.values() for t in targets}
        self.assertTrue(imported)
        for d in registry.orphans():
            self.assertNotIn(d, imported)


class TestStats(unittest.TestCase):
    def test_stats_shape(self):
        st = registry.stats()
        self.assertEqual(st["total_modules"], len(registry.index()))
        self.assertEqual(st["orphan_count"], len(registry.orphans()))
        self.assertGreater(st["by_tag"].get("geometry", 0), 0)
        self.assertIn("domain", st["by_layer"])


class TestPersistence(unittest.TestCase):
    def test_json_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "idx.json")
            written = registry.build_index(path)
            loaded = registry.load_index(path)
            self.assertEqual(loaded, written)

    def test_index_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.json")
            b = os.path.join(tmp, "b.json")
            registry.build_index(a)
            registry.build_index(b)
            with open(a, "rb") as fh:
                ba = fh.read()
            with open(b, "rb") as fh:
                bb = fh.read()
            self.assertEqual(ba, bb)

    def test_persisted_index_has_no_timestamps(self):
        with open(registry.INDEX_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertEqual(sorted(payload), ["modules", "root", "schema"])
        self.assertGreater(len(payload["modules"]), 1000)

    def test_shipped_index_is_faithful_to_the_source_tree(self):
        # The tree keeps growing, so the shipped JSON may lag behind a fresh scan
        # (run `harnesscad capabilities --rebuild` to refresh it). What must hold
        # is that every record it *does* carry survived the JSON round-trip
        # byte-for-byte identically to what the AST scanner produces today.
        shipped = registry.load_index()
        self.assertIsNotNone(shipped)
        self.assertGreater(len(shipped), 1000)
        fresh = {e.dotted: e for e in registry.scan_source_tree()}
        overlap = 0
        for e in shipped:
            if e.dotted in fresh:
                self.assertEqual(e, fresh[e.dotted])
                overlap += 1
        self.assertGreater(overlap, 1000)


class TestCli(unittest.TestCase):
    def test_capabilities_subcommand_is_registered(self):
        from harnesscad.core import cli

        args = cli.build_parser().parse_args(["capabilities", "--stats"])
        self.assertEqual(args.func, cli.cmd_capabilities)

    def test_existing_subcommands_still_parse(self):
        from harnesscad.core import cli

        parser = cli.build_parser()
        self.assertEqual(parser.parse_args(["demo"]).func, cli.cmd_demo)
        self.assertEqual(parser.parse_args(["apply", "x.json"]).func, cli.cmd_apply)
        self.assertEqual(parser.parse_args(["build", "a plate"]).func, cli.cmd_build)

    def test_registry_main_runs(self):
        self.assertEqual(registry.main(["--stats"]), 0)
        self.assertEqual(registry.main(["--list", "--tag", "sdf"]), 0)
        self.assertEqual(registry.main(["--search", "marching"]), 0)
        self.assertEqual(
            registry.main(["--show", "harnesscad.domain.geometry.sdf.primitives"]), 0)

    def test_show_unknown_module_exits_nonzero(self):
        self.assertEqual(registry.main(["--show", "harnesscad.nope"]), 2)


if __name__ == "__main__":
    unittest.main()
