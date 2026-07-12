"""Tests for deterministic code-CAD backend selection and the coverage report."""

from __future__ import annotations

import unittest

from adapters import ccc_backend_selector as sel
from adapters import ccc_codecad_ecosystem as eco


class TestSupportLevels(unittest.TestCase):
    def test_executable_systems_come_from_the_language_registry(self) -> None:
        self.assertEqual(sel.executable_systems(), ["cadquery", "curv", "jscad", "openscad"])

    def test_support_level(self) -> None:
        self.assertEqual(sel.support_level("cadquery"), sel.SUPPORT_EXECUTE)
        self.assertEqual(sel.support_level("solidpython"), sel.SUPPORT_EMIT)
        self.assertEqual(sel.support_level("fornjot"), sel.SUPPORT_NONE)


class TestRanking(unittest.TestCase):
    def test_step_plus_internal_fillets_picks_a_brep_system(self) -> None:
        req = sel.Requirement(exact_exchange=True, internal_fillets=True, export_format="step")
        top = sel.best(req)
        self.assertIsNotNone(top)
        self.assertIn(eco.PARA_BREP, eco.get(top.name).paradigms)
        # cadquery is executable AND a listed recommendation -> it must win.
        self.assertEqual(top.name, "cadquery")
        self.assertIn("exports STEP for exact downstream exchange", top.reasons)

    def test_organic_blends_picks_an_implicit_system(self) -> None:
        req = sel.Requirement(organic_blends=True)
        top = sel.best(req)
        self.assertEqual(eco.get(top.name).representation, "implicit")

    def test_hard_constraints_filter(self) -> None:
        names = [c.name for c in sel.rank(sel.Requirement(host_language="go"))]
        self.assertEqual(names, ["sdfx"])
        names = [c.name for c in sel.rank(sel.Requirement(must_be_executable=True))]
        self.assertEqual(sorted(names), sel.executable_systems())

    def test_node_editors_excluded_unless_scripting_relaxed(self) -> None:
        scripted = [c.name for c in sel.rank(sel.Requirement())]
        self.assertNotIn("sverchok", scripted)
        relaxed = [c.name for c in sel.rank(sel.Requirement(scripting_only=False))]
        self.assertIn("sverchok", relaxed)

    def test_impossible_requirement_yields_nothing(self) -> None:
        req = sel.Requirement(host_language="go", export_format="step")
        self.assertEqual(sel.rank(req), [])
        self.assertIsNone(sel.best(req))

    def test_ranking_is_deterministic_and_sorted(self) -> None:
        req = sel.Requirement(paradigm=eco.PARA_CSG)
        first = sel.rank(req)
        second = sel.rank(req)
        self.assertEqual([(c.name, c.score) for c in first], [(c.name, c.score) for c in second])
        keys = [(-c.score, c.name) for c in first]
        self.assertEqual(keys, sorted(keys))

    def test_explain_reports_exclusion(self) -> None:
        cand = sel.explain("openscad", sel.Requirement(export_format="step"))
        self.assertEqual(cand.score, 0)
        self.assertIn("excluded: fails a hard constraint", cand.reasons)
        cand = sel.explain("openscad", sel.Requirement(export_format="stl"))
        self.assertGreater(cand.score, 0)
        self.assertIn("harness can execute it", cand.reasons)

    def test_limit(self) -> None:
        self.assertEqual(len(sel.rank(sel.Requirement(), limit=3)), 3)


class TestCoverage(unittest.TestCase):
    def test_report_partitions_the_catalogue(self) -> None:
        rep = sel.coverage_report()
        self.assertEqual(rep.total_systems, len(eco.system_names()))
        buckets = set(rep.executable) | set(rep.emit_only) | set(rep.unsupported)
        self.assertEqual(buckets, set(eco.system_names()))
        self.assertEqual(
            len(rep.executable) + len(rep.emit_only) + len(rep.unsupported), rep.total_systems
        )

    def test_covered_and_missing_are_disjoint(self) -> None:
        rep = sel.coverage_report()
        self.assertFalse(set(rep.covered_paradigms) & set(rep.missing_paradigms))
        self.assertFalse(set(rep.covered_kernels) & set(rep.missing_kernels))
        self.assertFalse(set(rep.covered_formats_out) & set(rep.missing_formats_out))

    def test_known_harness_gaps(self) -> None:
        rep = sel.coverage_report()
        # openscad(cgal) + cadquery(occt) + jscad(custom) + curv(custom) reach these:
        self.assertIn(eco.PARA_BREP, rep.covered_paradigms)
        self.assertIn(eco.PARA_SDF, rep.covered_paradigms)
        # ... but nothing supported is a node/visual editor or a manifold/libfive kernel.
        self.assertIn(eco.PARA_VISUAL, rep.missing_paradigms)
        self.assertIn(eco.K_MANIFOLD, rep.missing_kernels)
        self.assertIn(eco.K_LIBFIVE, rep.missing_kernels)
        self.assertIn("gltf", rep.missing_formats_out)

    def test_hypothetical_support_closes_gaps(self) -> None:
        rep = sel.coverage_report(supported=sel.executable_systems() + ["manifold"])
        self.assertNotIn(eco.K_MANIFOLD, rep.missing_kernels)
        self.assertNotIn("gltf", rep.missing_formats_out)

    def test_gap_recommendations_are_unsupported_and_unique(self) -> None:
        rep = sel.coverage_report()
        recs = sel.gap_recommendations()
        self.assertTrue(recs)
        names = [n for n, _ in recs]
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), len(set(names)))
        for name, gap in recs:
            self.assertIn(name, rep.unsupported)
            self.assertTrue(gap.split(":")[0] in ("paradigm", "kernel", "format"))
        self.assertEqual(sel.gap_recommendations(), recs)

    def test_report_row_is_serialisable(self) -> None:
        row = sel.coverage_report().as_row()
        self.assertEqual(row["total_systems"], len(eco.system_names()))
        self.assertIsInstance(row["covered_paradigms"], list)


if __name__ == "__main__":
    unittest.main()
