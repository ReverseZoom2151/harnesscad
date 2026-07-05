"""Tests for the fabrication-workflow taxonomy, feasibility and comparison layer.

Covers fabworkflow_taxonomy, fabworkflow_feasibility and fabworkflow_compare
(distilled from Feng et al., "Comparing Fabrication Workflows in CAD to Support
Design Reasoning" / CAMeleon). Stdlib unittest, deterministic.
"""

import unittest

import fabrication.fabworkflow_taxonomy as tax
import fabrication.fabworkflow_feasibility as feas
import fabrication.fabworkflow_compare as comp


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class TaxonomyTests(unittest.TestCase):
    def test_categories_and_workflows_present(self):
        self.assertIn("wire_forming", tax.CATEGORIES)
        self.assertGreaterEqual(len(tax.WORKFLOWS), 16)

    def test_every_workflow_category_is_valid(self):
        for wf in tax.WORKFLOWS.values():
            self.assertIn(wf.category, tax.CATEGORIES, wf.id)

    def test_every_machine_and_material_reference_resolves(self):
        for wf in tax.WORKFLOWS.values():
            self.assertTrue(wf.machines, wf.id)
            for mid in wf.machines:
                self.assertIn(mid, tax.MACHINES, f"{wf.id}->{mid}")
            for mat in wf.materials:
                self.assertIn(mat, tax.MATERIALS, f"{wf.id}->{mat}")

    def test_levels_in_range(self):
        for wf in tax.WORKFLOWS.values():
            for lvl in (wf.cost, wf.time, wf.precision, wf.skill):
                self.assertTrue(1 <= lvl <= 5, wf.id)

    def test_workflows_in_category(self):
        molds = tax.workflows_in_category("mold_casting")
        ids = {w.id for w in molds}
        self.assertIn("silicone_molding", ids)
        self.assertIn("mold_making", ids)

    def test_workflows_in_category_rejects_unknown(self):
        with self.assertRaises(KeyError):
            tax.workflows_in_category("nope")

    def test_available_workflows_any_of_machines(self):
        got = tax.available_workflows(["laser_generic"])
        ids = {w.id for w in got}
        self.assertIn("laser_cut_interlocking", ids)
        # a printer-only workflow should not appear
        self.assertNotIn("fdm_3d_printing", ids)

    def test_available_workflows_deterministic_order(self):
        a = [w.id for w in tax.available_workflows(["manual", "laser_generic"])]
        b = [w.id for w in tax.available_workflows(["laser_generic", "manual"])]
        self.assertEqual(a, b)
        self.assertEqual(a, sorted(a))

    def test_workflows_for_machine_unknown(self):
        with self.assertRaises(KeyError):
            tax.workflows_for_machine("no_such_machine")

    def test_get_workflow_unknown(self):
        with self.assertRaises(KeyError):
            tax.get_workflow("no_such_wf")


# --------------------------------------------------------------------------- #
# Feasibility
# --------------------------------------------------------------------------- #
class MachineFitTests(unittest.TestCase):
    def test_small_part_fits(self):
        part = feas.PartSpec(bbox=(100.0, 80.0, 50.0))
        f = feas.check_machine_fit("fdm_3d_printing", part, "prusa_mk3")
        self.assertTrue(f.ok)
        self.assertEqual(f.data["splits"], 1)

    def test_oversized_part_splits(self):
        # A stool-sized part far exceeds a Prusa bed -> must split.
        part = feas.PartSpec(bbox=(500.0, 450.0, 420.0))
        f = feas.check_machine_fit("fdm_3d_printing", part, "prusa_mk3")
        self.assertEqual(f.severity, "warning")
        self.assertGreater(f.data["splits"], 1)

    def test_orientation_is_free(self):
        # Long-thin part fits if rotated: 240 long vs 250 max axis.
        part = feas.PartSpec(bbox=(240.0, 30.0, 30.0))
        f = feas.check_machine_fit("fdm_3d_printing", part, "prusa_mk3")
        self.assertTrue(f.ok)

    def test_manual_workflow_has_no_envelope(self):
        part = feas.PartSpec(bbox=(9999.0, 9999.0, 9999.0))
        f = feas.check_machine_fit("paper_mache", part)
        self.assertTrue(f.ok)

    def test_invalid_machine_for_workflow(self):
        with self.assertRaises(ValueError):
            feas.check_machine_fit("fdm_3d_printing",
                                   feas.PartSpec(bbox=(1, 1, 1)), "laser_generic")


class PrintTimeTests(unittest.TestCase):
    def test_large_print_warns(self):
        # ~ a stool volume; should be many hours.
        part = feas.PartSpec(volume_mm3=3_000_000.0)  # 3000 cm^3
        f = feas.estimate_print_time(part, infill=0.20, material="pla")
        self.assertEqual(f.severity, "warning")
        self.assertGreater(f.data["minutes"], 24 * 60)

    def test_small_print_ok(self):
        part = feas.PartSpec(volume_mm3=20_000.0)
        f = feas.estimate_print_time(part, infill=0.20, material="pla")
        self.assertTrue(f.ok)

    def test_higher_infill_takes_longer(self):
        part = feas.PartSpec(volume_mm3=500_000.0)
        low = feas.estimate_print_time(part, infill=0.10).data["minutes"]
        high = feas.estimate_print_time(part, infill=0.80).data["minutes"]
        self.assertGreater(high, low)

    def test_deterministic(self):
        part = feas.PartSpec(volume_mm3=123_456.0)
        a = feas.estimate_print_time(part, infill=0.2, material="petg").data["minutes"]
        b = feas.estimate_print_time(part, infill=0.2, material="petg").data["minutes"]
        self.assertEqual(a, b)

    def test_no_volume_skips(self):
        f = feas.estimate_print_time(feas.PartSpec())
        self.assertTrue(f.ok)


class MaterialStockTests(unittest.TestCase):
    def test_exact_stock_ok(self):
        part = feas.PartSpec(sheet_thickness=3.0)
        f = feas.check_material_stock("laser_cut_interlocking", part, "plywood_3mm")
        self.assertTrue(f.ok)

    def test_snaps_to_nearest(self):
        part = feas.PartSpec(sheet_thickness=4.0)
        f = feas.check_material_stock("laser_cut_interlocking", part, "plywood_3mm")
        self.assertEqual(f.severity, "warning")
        self.assertEqual(f.data["snapped"], 3.0)

    def test_snaps_up_when_closer(self):
        part = feas.PartSpec(sheet_thickness=5.5)
        f = feas.check_material_stock("laser_cut_interlocking", part, "mdf_6mm")
        self.assertEqual(f.data["snapped"], 6.0)

    def test_no_thickness_skips(self):
        f = feas.check_material_stock("laser_cut_interlocking", feas.PartSpec())
        self.assertTrue(f.ok)


class WireFormTests(unittest.TestCase):
    def test_feasible_polyline(self):
        part = feas.PartSpec(wire_segments=[(50.0, 90.0), (40.0, 45.0), (60.0, 30.0)])
        f = feas.check_wire_form(part)
        self.assertTrue(f.ok)

    def test_short_segment_warns(self):
        part = feas.PartSpec(wire_segments=[(50.0, 90.0), (5.0, 45.0)])
        f = feas.check_wire_form(part)
        self.assertEqual(f.severity, "warning")
        self.assertIn(1, f.data["short_segments"])

    def test_impossible_bend_errors(self):
        part = feas.PartSpec(wire_segments=[(50.0, 170.0)])
        f = feas.check_wire_form(part)
        self.assertEqual(f.severity, "error")
        self.assertIn(0, f.data["impossible_bends"])

    def test_empty_skips(self):
        self.assertTrue(feas.check_wire_form(feas.PartSpec()).ok)


class FoamLoadTests(unittest.TestCase):
    def test_structural_foam_warns(self):
        part = feas.PartSpec(load_bearing=True)
        f = feas.check_foam_load("hot_wire_foam_cutting", part)
        self.assertEqual(f.severity, "warning")

    def test_decorative_foam_ok(self):
        part = feas.PartSpec(load_bearing=False)
        f = feas.check_foam_load("hot_wire_foam_cutting", part)
        self.assertTrue(f.ok)

    def test_load_bearing_material_ok(self):
        part = feas.PartSpec(load_bearing=True)
        f = feas.check_foam_load("laser_cut_interlocking", part)
        self.assertTrue(f.ok)


class DraftAngleTests(unittest.TestCase):
    def test_sufficient_draft_ok(self):
        f = feas.check_draft_angle(feas.PartSpec(min_draft_deg=3.0))
        self.assertTrue(f.ok)

    def test_low_draft_warns(self):
        f = feas.check_draft_angle(feas.PartSpec(min_draft_deg=1.0))
        self.assertEqual(f.severity, "warning")

    def test_undercut_errors(self):
        f = feas.check_draft_angle(feas.PartSpec(min_draft_deg=-2.0))
        self.assertEqual(f.severity, "error")

    def test_no_data_skips(self):
        self.assertTrue(feas.check_draft_angle(feas.PartSpec()).ok)


class DispatcherTests(unittest.TestCase):
    def test_dispatch_runs_declared_checks(self):
        part = feas.PartSpec(bbox=(100, 80, 50), volume_mm3=50000.0,
                             sheet_thickness=None)
        findings = feas.analyze_workflow("fdm_3d_printing", part)
        checks = {f.check for f in findings}
        self.assertEqual(checks, {"machine_fit", "print_time", "material_stock"})

    def test_worst_severity_rollup(self):
        part = feas.PartSpec(wire_segments=[(50.0, 170.0)])
        findings = feas.analyze_workflow("wire_forming", part)
        self.assertEqual(feas.worst_severity(findings), "error")

    def test_wire_forming_dispatch(self):
        part = feas.PartSpec(bbox=(100, 100, 100),
                             wire_segments=[(50.0, 90.0), (50.0, 45.0)])
        findings = feas.analyze_workflow("wire_forming", part)
        checks = [f.check for f in findings]
        self.assertEqual(checks, ["machine_fit", "wire_form"])


# --------------------------------------------------------------------------- #
# Comparison / ranking / reflection
# --------------------------------------------------------------------------- #
class CompareTests(unittest.TestCase):
    def test_side_by_side_table(self):
        table = comp.compare_workflows(
            ["fdm_3d_printing", "wire_forming", "epoxy_laminating"])
        self.assertEqual(set(table.keys()),
                         {"fdm_3d_printing", "wire_forming", "epoxy_laminating"})
        for row in table.values():
            for c in comp.COMPARISON_CRITERIA:
                self.assertIn(c, row)

    def test_max_four_workflows(self):
        with self.assertRaises(ValueError):
            comp.compare_workflows(
                ["fdm_3d_printing", "wire_forming", "epoxy_laminating",
                 "mold_making", "silicone_molding"])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            comp.compare_workflows([])

    def test_deltas_flag_differences(self):
        d = comp.comparison_deltas(["fdm_3d_printing", "wire_forming"])
        # time and cost differ between these two
        self.assertTrue(d["time"])
        self.assertTrue(d["cost"])

    def test_deltas_identical_workflow(self):
        d = comp.comparison_deltas(["fdm_3d_printing", "fdm_3d_printing"])
        self.assertFalse(any(d.values()))


class IntentRankTests(unittest.TestCase):
    def test_transparent_food_safe_prefers_matching_workflow(self):
        reqs = {"transparent": True, "food_safe": True}
        ranked = comp.rank_by_intent(reqs, top_k=3)
        self.assertLessEqual(len(ranked), 3)
        top = ranked[0]
        # silicone (food safe) or epoxy/petg-based transparent should rank; the
        # top result must satisfy at least one requirement.
        self.assertGreater(top.score, 0)
        self.assertTrue(top.matched)

    def test_lightweight_prefers_wire_or_foam(self):
        ranked = comp.rank_by_intent({"lightweight": True}, top_k=5)
        top_ids = {m.workflow_id for m in ranked}
        self.assertTrue(
            top_ids & {"wire_forming", "hot_wire_foam_cutting", "paper_folding"})

    def test_max_time_penalizes_slow(self):
        # Without cap, 3D printing (time=5) can rank; with a tight time cap it
        # should be penalized below a fast workflow.
        reqs = {"lightweight": True, "max_time": 2}
        ranked = comp.rank_by_intent(reqs, top_k=8)
        by_id = {m.workflow_id: m.score for m in ranked}
        if "fdm_3d_printing" in by_id and "paper_folding" in by_id:
            self.assertGreater(by_id["paper_folding"], by_id["fdm_3d_printing"])

    def test_reasons_present_for_each_requirement(self):
        ranked = comp.rank_by_intent({"durable": True, "max_cost": 3}, top_k=1)
        self.assertTrue(ranked[0].reasons)

    def test_deterministic_order(self):
        reqs = {"durable": True, "lightweight": True}
        a = [m.workflow_id for m in comp.rank_by_intent(reqs, top_k=5)]
        b = [m.workflow_id for m in comp.rank_by_intent(reqs, top_k=5)]
        self.assertEqual(a, b)

    def test_candidate_restriction(self):
        ranked = comp.rank_by_intent(
            {"durable": True},
            candidate_ids=["wire_forming", "mold_making"], top_k=5)
        self.assertEqual({m.workflow_id for m in ranked},
                         {"wire_forming", "mold_making"})

    def test_machine_restriction(self):
        ranked = comp.rank_by_intent(
            {"durable": True}, machine_ids=["laser_generic"], top_k=20)
        for m in ranked:
            wf = tax.get_workflow(m.workflow_id)
            self.assertIn("laser_generic", wf.machines)


class ReflectionTests(unittest.TestCase):
    def test_general_and_specific(self):
        cl = comp.reflection_checklist("fdm_3d_printing")
        self.assertTrue(cl.general)
        self.assertTrue(cl.specific)
        joined = " ".join(cl.specific).lower()
        self.assertIn("bed", joined)

    def test_laser_specific_questions(self):
        cl = comp.reflection_checklist("laser_cut_interlocking")
        joined = " ".join(cl.specific).lower()
        self.assertIn("kerf", joined)

    def test_workflow_without_specific_questions(self):
        cl = comp.reflection_checklist("escape_loom")
        self.assertTrue(cl.general)
        self.assertEqual(cl.specific, ())

    def test_unknown_workflow(self):
        with self.assertRaises(KeyError):
            comp.reflection_checklist("no_such")


class ExplorationTraceTests(unittest.TestCase):
    def test_breadth_gain_and_selection_change(self):
        t = comp.ExplorationTrace()
        t.consider("fdm_3d_printing", after=False)
        t.select("fdm_3d_printing", after=False)
        for w in ("fdm_3d_printing", "wire_forming", "silicone_molding"):
            t.consider(w, after=True)
        t.select("wire_forming", after=True)
        t.cite("durability")
        self.assertEqual(t.breadth_gain(), 2)
        self.assertTrue(t.changed_selection())
        s = t.summary()
        self.assertEqual(s["selected_after"], "wire_forming")
        self.assertIn("durability", s["criteria_cited"])

    def test_no_change_when_same_selection(self):
        t = comp.ExplorationTrace()
        t.select("fdm_3d_printing", after=False)
        t.select("fdm_3d_printing", after=True)
        self.assertFalse(t.changed_selection())

    def test_consider_dedup(self):
        t = comp.ExplorationTrace()
        t.consider("wire_forming")
        t.consider("wire_forming")
        self.assertEqual(len(t.considered_after), 1)

    def test_consider_validates(self):
        t = comp.ExplorationTrace()
        with self.assertRaises(KeyError):
            t.consider("nope")


if __name__ == "__main__":
    unittest.main()
