"""Tests for the code-CAD ecosystem knowledge base."""

from __future__ import annotations

import unittest

from harnesscad.io.adapters import ccc_codecad_ecosystem as eco


class TestCatalogueIntegrity(unittest.TestCase):
    def test_names_sorted_and_unique(self) -> None:
        names = eco.system_names()
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(names), 25)

    def test_every_spec_uses_known_vocabulary(self) -> None:
        for spec in eco.all_systems():
            self.assertIn(spec.category, eco.CATEGORIES, spec.name)
            self.assertIn(spec.kernel, eco.KERNELS, spec.name)
            for paradigm in spec.paradigms:
                self.assertIn(paradigm, eco.PARADIGMS, spec.name)
            self.assertTrue(spec.license, spec.name)
            self.assertTrue(spec.name.islower(), spec.name)

    def test_lookup_by_alias_is_case_insensitive(self) -> None:
        self.assertEqual(eco.get("CQ").name, "cadquery")
        self.assertEqual(eco.get("OCCT").name, "opencascade")
        self.assertEqual(eco.get("openjscad").name, "jscad")
        self.assertTrue(eco.has("scad"))
        self.assertFalse(eco.has("solidworks"))
        with self.assertRaises(eco.UnknownSystem):
            eco.get("solidworks")

    def test_recommended_and_birdhouse_entries_exist(self) -> None:
        for name in eco.RECOMMENDED_BREP_SYSTEMS + eco.BIRDHOUSE_IMPLEMENTATIONS:
            self.assertTrue(eco.has(name), name)


class TestGroundedAttributes(unittest.TestCase):
    def test_openscad_is_a_mesh_csg_language(self) -> None:
        spec = eco.get("openscad")
        self.assertTrue(spec.is_language)
        self.assertEqual(spec.host_language, "custom-dsl")
        self.assertEqual(spec.representation, "mesh")
        self.assertIn(eco.PARA_CSG, spec.paradigms)
        self.assertEqual(spec.license, "GPL-2")

    def test_cadquery_is_occt_brep_python(self) -> None:
        spec = eco.get("cadquery")
        self.assertEqual(spec.kernel, eco.K_OCCT)
        self.assertEqual(spec.representation, "brep")
        self.assertEqual(spec.host_language, "python")
        self.assertTrue(spec.exports("STEP"))
        self.assertTrue(spec.exports(".stl"))

    def test_transpilers_have_no_kernel(self) -> None:
        for name in ("solidpython", "scad-clj", "scad-hs"):
            spec = eco.get(name)
            self.assertTrue(spec.kernel_free, name)
            self.assertIn(eco.PARA_TRANSPILE, spec.paradigms)
            self.assertEqual(spec.formats_out, ("scad",))

    def test_unknowns_are_declared_not_guessed(self) -> None:
        self.assertIn("kernel", eco.unknown_attributes("bitbybit"))
        self.assertIn("license", eco.unknown_attributes("declaracad"))
        self.assertEqual(eco.unknown_attributes("cadquery"), [])


class TestQueries(unittest.TestCase):
    def test_sdf_systems(self) -> None:
        sdf = eco.sdf_systems()
        for name in ("curv", "libfive", "sdfx", "sdf-csg", "implicitcad", "tovero", "manifold"):
            self.assertIn(name, sdf, name)
        self.assertNotIn("cadquery", sdf)

    def test_step_exporters_are_all_brep(self) -> None:
        for name in eco.exporters_of("step"):
            self.assertIn(eco.PARA_BREP, eco.get(name).paradigms, name)
        self.assertIn("cadquery", eco.exporters_of("step"))
        self.assertIn("freecad", eco.exporters_of("STEP"))
        self.assertNotIn("openscad", eco.exporters_of("step"))

    def test_occt_systems(self) -> None:
        occt = eco.occt_based_systems()
        for name in ("cadquery", "cascadestudio", "declaracad", "pythonocc", "freecad", "replicad"):
            self.assertIn(name, occt, name)

    def test_kernel_free_systems(self) -> None:
        free = eco.kernel_free_systems()
        self.assertIn("solidpython", free)
        self.assertIn("cadhub", free)
        self.assertNotIn("libfive", free)

    def test_by_host_language(self) -> None:
        python = eco.by_host_language("Python")
        self.assertEqual(python, ["build123d", "cadquery", "freecad", "pythonocc", "solidpython"])
        self.assertEqual(eco.by_host_language("go"), ["sdfx"])
        self.assertEqual(eco.by_host_language("clojure"), ["scad-clj"])

    def test_select_composes_constraints(self) -> None:
        got = eco.select(paradigm=eco.PARA_BREP, host_language="python", exports="step")
        self.assertEqual(got, ["build123d", "cadquery", "freecad", "pythonocc"])
        self.assertEqual(eco.select(category=eco.CAT_KERNEL), ["manifold", "opencascade"])
        self.assertEqual(
            eco.select(paradigm=eco.PARA_SDF, host_language="javascript"), ["sdf-csg"]
        )

    def test_select_rejects_unknown_vocabulary(self) -> None:
        with self.assertRaises(ValueError):
            eco.by_paradigm("wireframe")
        with self.assertRaises(ValueError):
            eco.by_kernel("parasolid")
        with self.assertRaises(ValueError):
            eco.by_category("plugin")

    def test_matrix_is_deterministic(self) -> None:
        first = eco.catalogue_matrix()
        second = eco.catalogue_matrix()
        self.assertEqual(first, second)
        self.assertEqual([r["name"] for r in first], eco.system_names())

    def test_representation_notes_taxonomy(self) -> None:
        self.assertIn("brep", eco.REPRESENTATION_NOTES)
        self.assertTrue(eco.REPRESENTATION_NOTES["mesh"].limitations)
        self.assertIn("fillets", eco.CSG_CAVEAT)
        for key, note in eco.REPRESENTATION_NOTES.items():
            self.assertEqual(key, note.representation)


if __name__ == "__main__":
    unittest.main()
