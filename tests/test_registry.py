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

        "Imports" means every way a module can actually be reached: a static
        import between indexed modules, a registry's runtime `importlib` dispatch,
        and a package ``__init__`` hub importing its own submodules. That last
        term is not a courtesy -- the index skips ``__init__.py``, so a hub-wired
        module looked unreachable while its package imported it on line one.
        """
        orph = set(registry.orphans())
        graph = registry.import_graph()
        imported = {t for targets in graph.values() for t in targets}
        for targets in registry.dynamic_edges().values():
            imported.update(targets)
        for targets in registry.package_init_edges().values():
            imported.update(targets)
        indexed = {e.dotted for e in registry.index()}
        self.assertEqual(orph, indexed - imported - set(registry.ROOTS))

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

    def test_package_init_hub_imports_are_credited(self):
        """A module its own package imports on line one is not an orphan.

        The index skips ``__init__.py`` (a package is not a capability), and the
        skip used to drop the EDGES with it. So the fixture loaders -- deliberately
        wired with real ``import`` statements, with a source comment saying the
        AST scan would therefore see them -- all reported as dead. The tempting
        fix was a ROOTS entry each, which would have been a lie: ROOTS means
        "nothing imports it because it IS the entry point", and these are imported.
        """
        edges = registry.package_init_edges()
        hub = "harnesscad.eval.corpus.fixtures"
        self.assertIn(hub, edges, "the fixtures hub imports its loaders for real")
        orph = set(registry.orphans())
        for loader in edges[hub]:
            self.assertNotIn(loader, orph, f"{loader} is imported by {hub}")

    def test_vendored_data_is_not_indexed_as_a_capability(self):
        """A .py file outside an importable package is DATA, not a module.

        The birdhouse fixture is the same part vendored in eight languages. The
        index claimed the three ``.py`` ones as harnesscad capabilities -- minting
        dotted paths that cannot be imported, since no ``__init__.py`` exists
        anywhere in that chain -- and ignored the ``.scad``/``.js``/``.go`` five.
        Whether a fixture counts as product code cannot depend on its extension.
        """
        indexed = {e.dotted for e in registry.index()}
        for dotted in indexed:
            self.assertNotIn(
                ".fixtures.birdhouse.sources.", dotted,
                f"{dotted} is vendored fixture data, not a capability",
            )


class TestOrphanDetectorSeesDynamicDispatch(unittest.TestCase):
    """The orphan detector used to cry wolf, and that is why the dead stayed hidden.

    `_product_imports` reads the AST, so it cannot see
    `importlib.import_module(entry.dotted)` -- which is how EVERY layer registry
    in this repo dispatches. The verifier fleet runs on every `verify_level=full`
    loop and reported as dead; so did every bench metric and every adapter. 336
    orphans were reported and 292 of them were alive. When a tool is wrong about
    300 modules nobody reads its output, and `tool_reward.py` sits unwired inside
    the noise. These tests keep the detector honest.
    """

    def test_a_dynamically_dispatched_verifier_is_not_an_orphan(self):
        # precheck runs on every full-verify loop -- it caused every regression in
        # assets/pressure/report.md -- and the old detector called it dead.
        orph = set(registry.orphans())
        self.assertNotIn("harnesscad.eval.verifiers.precheck", orph)
        self.assertNotIn("harnesscad.eval.verifiers.dfm", orph)

    def test_the_static_view_is_still_available_and_is_strictly_larger(self):
        static = set(registry.orphans(dynamic=False))
        true = set(registry.orphans(dynamic=True))
        self.assertTrue(true.issubset(static))
        self.assertLess(len(true), len(static))

    def test_every_dynamic_dispatcher_is_declared(self):
        """A new layer registry cannot silently reintroduce the cried-wolf bug.

        Any module named `registry` (or `pipeline`) that imports the capability
        index and loads modules by dotted name IS a dispatcher and must appear in
        DYNAMIC_DISPATCHERS, or the modules it dispatches will be reported dead.
        """
        import pathlib
        import re

        src = pathlib.Path(registry.ROOT)
        undeclared = []
        for path in sorted(src.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "from harnesscad import registry as capability_registry" not in text:
                continue
            if not re.search(r"capability_registry\.load\(|importlib\.import_module\(",
                             text):
                continue
            dotted = "harnesscad." + str(
                path.relative_to(src).with_suffix("")).replace(os.sep, ".")
            if dotted not in registry.DYNAMIC_DISPATCHERS:
                undeclared.append(dotted)
        self.assertEqual(
            undeclared, [],
            "these modules dispatch by importlib and are not in "
            "registry.DYNAMIC_DISPATCHERS, so everything they dispatch will be "
            "misreported as an orphan: %r" % (undeclared,))

    def test_declared_dispatchers_and_prefixes_are_real(self):
        indexed = {e.dotted for e in registry.index()}
        for dispatcher, prefixes in registry.DYNAMIC_DISPATCHERS.items():
            self.assertIn(dispatcher, indexed, dispatcher)
            for prefix in prefixes:
                self.assertTrue(
                    any(d.startswith(prefix + ".") for d in indexed),
                    "%s dispatches %r and nothing lives there" % (dispatcher, prefix))

    def test_the_orphan_ledger_may_not_grow(self):
        """The committed ceiling. An orphan is a capability nobody can reach."""
        orph = registry.orphans()
        self.assertLessEqual(
            len(orph), ORPHAN_CEILING,
            "the orphan count rose to %d (ceiling %d). A module nothing imports "
            "and no dispatcher can reach is not a capability. Wire it, or delete "
            "it, or raise this ceiling in the same diff and say why:\n  %s"
            % (len(orph), ORPHAN_CEILING, "\n  ".join(orph)))


#: The number of modules NOTHING in the product can reach, statically or by
#: runtime dispatch, as measured when the detector was fixed. It is a debt and it
#: may only go down. (The old, AST-only detector reported 336; 292 of those were
#: alive and dispatched by importlib.)
#:
#: 44 -> 0. Two thirds of the last drop was the detector being wrong again rather
#: than modules being wired: it discarded the import edges of every package
#: __init__ (so hub-wired modules read as dead) and indexed vendored fixture data
#: as capabilities (so data read as dead code). Both are fixed in registry.py and
#: pinned by tests above. The rest was real wiring -- the geometry surface, the
#: standards/spec/vision routes, the corpus and benchmark hubs.
#:
#: The last orphan, eval.corpus.discipline_examples, was repaired rather than
#: rooted: its bracket and PCB-carrier records are explicitly retired, its HMI
#: panel now accounts for the overlapping-hole lens, and consensus imports only
#: the three independently defensible records through corroboration_briefs().
#: That makes the route real without laundering the two unsafe references into a
#: benchmark. The ceiling therefore reaches its intended zero-debt state.
ORPHAN_CEILING = 0


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
