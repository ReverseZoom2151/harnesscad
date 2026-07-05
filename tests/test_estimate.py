"""Tests for the quantitative measurement + cost/BOM layer (estimate.py)."""

from __future__ import annotations

import unittest

from estimate import (
    BOM, BOMEstimator, BudgetCheck, BudgetSpec, Material, MaterialTable,
    PartEstimate, estimate_part, resolve_metrics,
)
from verify import Severity


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class MetricsBackend:
    """A fake backend answering query('metrics') (and 'measure')."""

    def __init__(self, volume=None, bbox=None, surface_area=None,
                 center_of_mass=None, only_measure=False):
        self._volume = volume
        self._bbox = bbox
        self._sa = surface_area
        self._com = center_of_mass
        self._only_measure = only_measure

    def query(self, q):
        if q == "measure":
            return {"volume": self._volume, "bbox": self._bbox}
        if q == "metrics" and not self._only_measure:
            return {
                "volume": self._volume,
                "bbox": self._bbox,
                "surface_area": self._sa,
                "center_of_mass": self._com,
            }
        return {}


class BboxOnlyBackend:
    """A fake backend that only knows a bbox (no volume) via 'measure'."""

    def __init__(self, bbox):
        self._bbox = bbox

    def query(self, q):
        if q == "measure":
            return {"volume": None, "bbox": self._bbox}
        return {}


class BlindBackend:
    """A stub-like backend that answers no measurement query."""

    def query(self, q):
        return {}


class AssemblyBackend:
    """A fake backend exposing query('assembly') with two part instances."""

    def __init__(self, parts):
        self._parts = parts

    def query(self, q):
        if q == "assembly":
            return {"parts": self._parts}
        return {}


# --------------------------------------------------------------------------- #
# MaterialTable
# --------------------------------------------------------------------------- #
class TestMaterialTable(unittest.TestCase):
    def test_defaults_present(self):
        t = MaterialTable()
        self.assertIn("aluminium", t.names())
        self.assertIn("steel", t.names())
        self.assertIn("abs", t.names())

    def test_alias_and_fallback(self):
        t = MaterialTable()
        name, mat = t.resolve("aluminum")   # US spelling alias
        self.assertEqual(name, "aluminium")
        # unknown -> default
        name2, _ = t.resolve("unobtainium")
        self.assertEqual(name2, "aluminium")

    def test_round_trip(self):
        t = MaterialTable()
        d = t.to_dict()
        t2 = MaterialTable.from_dict(d)
        self.assertEqual(t.to_dict(), t2.to_dict())
        # A specific density survives the round trip.
        self.assertAlmostEqual(
            t2.get("aluminium").density, t.get("aluminium").density)

    def test_material_from_dict_defaults(self):
        m = Material.from_dict({"density": 3.0})
        self.assertEqual(m.density, 3.0)
        self.assertEqual(m.cost_per_kg, 0.0)


# --------------------------------------------------------------------------- #
# estimate_part
# --------------------------------------------------------------------------- #
class TestEstimatePart(unittest.TestCase):
    def test_known_part_mass_from_volume_x_density(self):
        # A 10x10x10 mm cube = 1000 mm^3 = 1 cm^3.
        # Aluminium density 2.70 g/cm^3 -> mass 2.70 g.
        b = MetricsBackend(volume=1000.0, bbox=[10.0, 10.0, 10.0])
        est = estimate_part(b, material="aluminium")
        self.assertTrue(est.measured)
        self.assertFalse(est.volume_estimated)
        self.assertAlmostEqual(est.mass, 2.70, places=6)
        self.assertAlmostEqual(est.mass_kg, 0.0027, places=8)
        # Stock is bbox + 2mm/face = 14x14x14.
        self.assertEqual(est.stock_size, [14.0, 14.0, 14.0])
        self.assertGreater(est.material_cost, 0.0)
        self.assertGreater(est.rough_machining_cost, 0.0)
        self.assertAlmostEqual(
            est.total_cost, est.material_cost + est.rough_machining_cost)

    def test_material_changes_mass(self):
        b = MetricsBackend(volume=1000.0, bbox=[10.0, 10.0, 10.0])
        alu = estimate_part(b, material="aluminium")
        steel = estimate_part(b, material="steel")
        self.assertGreater(steel.mass, alu.mass)  # steel is denser

    def test_degrade_bbox_only(self):
        # Only a bbox: volume is approximated by the bounding box.
        b = BboxOnlyBackend(bbox=[20.0, 10.0, 5.0])
        est = estimate_part(b, material="aluminium")
        self.assertTrue(est.measured)
        self.assertTrue(est.volume_estimated)
        self.assertAlmostEqual(est.volume, 1000.0)  # 20*10*5
        self.assertIsNotNone(est.mass)
        self.assertIsNotNone(est.material_cost)

    def test_degrade_no_metrics(self):
        est = estimate_part(BlindBackend(), material="aluminium")
        self.assertFalse(est.measured)
        self.assertIsNone(est.mass)
        self.assertIsNone(est.material_cost)
        self.assertIsNone(est.total_cost)

    def test_accepts_raw_metrics_dict(self):
        est = estimate_part({"volume": 2000.0, "bbox": [10, 10, 20]},
                            material="steel")
        self.assertAlmostEqual(est.mass, 2.0 * 7.85, places=6)

    def test_passthrough_partestimate(self):
        original = PartEstimate(material="steel", mass=5.0)
        self.assertIs(estimate_part(original), original)

    def test_resolve_metrics_prefers_metrics_over_measure(self):
        b = MetricsBackend(volume=1000.0, bbox=[10, 10, 10],
                           surface_area=600.0)
        m = resolve_metrics(b)
        self.assertIn("surface_area", m)  # only 'metrics' has this


# --------------------------------------------------------------------------- #
# BOMEstimator
# --------------------------------------------------------------------------- #
class TestBOM(unittest.TestCase):
    def test_single_part_no_assembly(self):
        b = MetricsBackend(volume=1000.0, bbox=[10, 10, 10])
        bom = BOMEstimator().estimate(b)
        self.assertEqual(bom.line_count, 1)
        self.assertEqual(bom.part_count, 1)
        self.assertAlmostEqual(bom.total_mass, 2.70, places=6)

    def test_two_part_assembly_totals(self):
        parts = [
            {"name": "bracket", "material": "aluminium", "qty": 2,
             "metrics": {"volume": 1000.0, "bbox": [10, 10, 10]}},
            {"name": "pin", "material": "steel", "qty": 4,
             "metrics": {"volume": 500.0, "bbox": [5, 5, 20]}},
        ]
        bom = BOMEstimator().estimate(AssemblyBackend(parts))
        self.assertEqual(bom.line_count, 2)
        self.assertEqual(bom.part_count, 6)  # 2 + 4

        # bracket: 1 cm^3 * 2.70 = 2.70 g each, x2 = 5.40 g
        # pin:     0.5 cm^3 * 7.85 = 3.925 g each, x4 = 15.70 g
        expected_mass = 2 * 2.70 + 4 * (0.5 * 7.85)
        self.assertAlmostEqual(bom.total_mass, expected_mass, places=6)

        # Totals equal the sum of the per-line extended costs.
        line_cost = sum(l.total_cost for l in bom.lines)
        self.assertAlmostEqual(bom.total_cost, line_cost)
        self.assertTrue(bom.fully_measured)

    def test_render_csv_and_markdown(self):
        parts = [
            {"name": "bracket", "material": "aluminium", "qty": 2,
             "metrics": {"volume": 1000.0, "bbox": [10, 10, 10]}},
            {"name": "pin", "material": "steel", "qty": 4,
             "metrics": {"volume": 500.0, "bbox": [5, 5, 20]}},
        ]
        bom = BOMEstimator().estimate(AssemblyBackend(parts))

        csv_text = bom.to_csv()
        self.assertIn("part,qty,material", csv_text)
        self.assertIn("bracket", csv_text)
        self.assertIn("pin", csv_text)
        self.assertIn("TOTAL", csv_text)
        # header + 2 lines + total = 4 rows
        self.assertEqual(len([r for r in csv_text.strip().splitlines()]), 4)

        md = bom.to_markdown()
        self.assertIn("| Part |", md)
        self.assertIn("bracket", md)
        self.assertIn("**Total**", md)

    def test_bom_to_dict_totals(self):
        b = MetricsBackend(volume=1000.0, bbox=[10, 10, 10])
        d = BOMEstimator().estimate(b).to_dict()
        self.assertIn("totals", d)
        self.assertEqual(d["totals"]["part_count"], 1)
        self.assertTrue(d["totals"]["fully_measured"])

    def test_assembly_as_bare_list(self):
        class BareListAssembly:
            def query(self, q):
                if q == "assembly":
                    return [{"name": "a",
                             "metrics": {"volume": 1000.0, "bbox": [10, 10, 10]}}]
                return {}
        bom = BOMEstimator().estimate(BareListAssembly())
        self.assertEqual(bom.line_count, 1)


# --------------------------------------------------------------------------- #
# BudgetCheck
# --------------------------------------------------------------------------- #
class TestBudgetCheck(unittest.TestCase):
    def test_within_budget_no_error(self):
        b = MetricsBackend(volume=1000.0, bbox=[10, 10, 10])
        rep = BudgetCheck(BudgetSpec(max_mass=10.0)).check(b, None)
        self.assertTrue(rep.ok)  # advisory: never an ERROR
        codes = {d.code for d in rep.diagnostics}
        self.assertIn("mass-within-budget", codes)

    def test_over_budget_warns_not_errors(self):
        b = MetricsBackend(volume=1000.0, bbox=[10, 10, 10])
        rep = BudgetCheck(BudgetSpec(max_mass=1.0)).check(b, None)
        self.assertTrue(rep.ok)  # still ok: only a WARNING
        sevs = {d.severity for d in rep.diagnostics}
        self.assertIn(Severity.WARNING, sevs)
        self.assertIn("over-mass-budget", {d.code for d in rep.diagnostics})

    def test_unmeasurable_skips(self):
        rep = BudgetCheck(BudgetSpec(max_mass=1.0)).check(BlindBackend(), None)
        self.assertTrue(rep.ok)
        self.assertIn("budget-skipped", {d.code for d in rep.diagnostics})

    def test_spec_round_trip(self):
        spec = BudgetSpec(max_mass=5.0, max_cost=20.0, material="steel")
        spec2 = BudgetSpec.from_dict(spec.to_dict())
        self.assertEqual(spec.to_dict(), spec2.to_dict())


if __name__ == "__main__":
    unittest.main()
