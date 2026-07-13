"""Tests for adapters.cadhub_language_registry."""

import unittest

from harnesscad.io.adapters.language_registry import (
    ARTIFACT_IMAGE,
    ARTIFACT_MESH,
    ARTIFACT_PRIMITIVES,
    EXEC_CLI,
    EXEC_WORKER,
    PARAMS_NONE,
    UnknownLanguage,
    capability_matrix,
    entry_file,
    for_extension,
    get,
    has,
    language_names,
    languages_with_artifact,
    mesh_exporters,
    missing_capabilities,
    parametric_languages,
    select,
    working_files,
)


class TestLookup(unittest.TestCase):
    def test_names(self):
        self.assertEqual(
            language_names(), ["cadquery", "curv", "jscad", "openscad"]
        )

    def test_alias_and_case(self):
        self.assertEqual(get("CQ").name, "cadquery")
        self.assertEqual(get("OpenSCAD").name, "openscad")
        self.assertTrue(has("openjscad"))
        self.assertFalse(has("solidworks"))

    def test_unknown_raises(self):
        with self.assertRaises(UnknownLanguage):
            get("solidworks")

    def test_entry_files(self):
        self.assertEqual(entry_file("openscad"), "main.scad")
        self.assertEqual(entry_file("cadquery"), "main.py")
        self.assertEqual(entry_file("curv"), "main.curv")


class TestExtensions(unittest.TestCase):
    def test_path_resolution(self):
        self.assertEqual(for_extension("/tmp/x/main.scad").name, "openscad")
        self.assertEqual(for_extension("part.curv").name, "curv")
        self.assertEqual(for_extension("model.py").name, "cadquery")

    def test_bare_extension(self):
        self.assertEqual(for_extension(".jscad").name, "jscad")
        self.assertEqual(for_extension("scad").name, "openscad")

    def test_unknown_extension(self):
        with self.assertRaises(UnknownLanguage):
            for_extension("model.step")


class TestCapabilities(unittest.TestCase):
    def test_mesh_exporters_exclude_jscad(self):
        self.assertEqual(mesh_exporters(), ["cadquery", "curv", "openscad"])

    def test_artifact_queries(self):
        self.assertEqual(
            languages_with_artifact(ARTIFACT_IMAGE), ["curv", "openscad"]
        )
        self.assertEqual(languages_with_artifact(ARTIFACT_PRIMITIVES), ["jscad"])
        self.assertIn("cadquery", languages_with_artifact(ARTIFACT_MESH))

    def test_bad_artifact(self):
        with self.assertRaises(ValueError):
            languages_with_artifact("hologram")

    def test_parametric(self):
        self.assertEqual(parametric_languages(), ["cadquery", "jscad", "openscad"])
        self.assertEqual(get("curv").params, PARAMS_NONE)

    def test_execution_model(self):
        self.assertEqual(get("jscad").execution, EXEC_WORKER)
        self.assertEqual(get("openscad").execution, EXEC_CLI)

    def test_missing_capabilities(self):
        gaps = missing_capabilities("jscad")
        self.assertIn("mesh_export", gaps)
        self.assertIn("artifact:image", gaps)
        self.assertEqual(gaps, sorted(gaps))
        self.assertEqual(missing_capabilities("openscad"), ["artifact:primitives"])

    def test_curv_missing_params(self):
        self.assertIn("params", missing_capabilities("curv"))


class TestSelectAndMatrix(unittest.TestCase):
    def test_select_combined(self):
        self.assertEqual(
            select(mesh_export=True, execution=EXEC_CLI, params=True),
            ["cadquery", "openscad"],
        )
        self.assertEqual(select(execution=EXEC_WORKER), ["jscad"])
        self.assertEqual(select(artifact=ARTIFACT_IMAGE, params=True), ["openscad"])

    def test_matrix_shape_and_determinism(self):
        rows = capability_matrix()
        self.assertEqual(len(rows), 4)
        self.assertEqual([r["name"] for r in rows], language_names())
        self.assertEqual(rows, capability_matrix())
        self.assertEqual(rows[3]["entry_file"], "main.scad")

    def test_working_files(self):
        self.assertEqual(working_files("openscad"), ("main.scad", "params.json"))
        self.assertEqual(working_files("curv"), ("main.curv",))
        self.assertEqual(working_files("jscad"), ("main.jscad.js",))


if __name__ == "__main__":
    unittest.main()
