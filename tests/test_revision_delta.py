import unittest

from quality.estimate import BOM, BOMLine, PartEstimate
from quality.revision_delta import compare_revisions


class Backend:
    def __init__(self, volume):
        self.volume = volume

    def query(self, name):
        return {"volume": self.volume, "bbox": [10, 10, 10]} if name == "metrics" else {}


class TestRevisionDelta(unittest.TestCase):
    def test_part_estimate_scalar_deltas(self):
        before = PartEstimate(
            "steel", volume=100, mass=10, material_cost=2,
            rough_machining_cost=3, embodied_carbon=1, embodied_energy=4,
        )
        after = PartEstimate(
            "steel", volume=110, mass=12, material_cost=3,
            rough_machining_cost=3, embodied_carbon=0.8, embodied_energy=5,
        )
        report = compare_revisions(before, after)
        self.assertTrue(report.available)
        self.assertEqual(report.metric("volume").absolute, 10)
        self.assertEqual(report.metric("cost").absolute, 1)
        self.assertEqual(report.metric("carbon").direction, "decreased")
        self.assertAlmostEqual(report.metric("mass").percent, 20)

    def test_backend_is_estimated_through_existing_layer(self):
        report = compare_revisions(Backend(1000), Backend(2000), material="steel")
        self.assertEqual(report.metric("volume").absolute, 1000)
        self.assertGreater(report.metric("mass").absolute, 0)
        self.assertTrue(report.metric("carbon").available)

    def test_bom_lines_are_added_removed_and_changed(self):
        estimate = PartEstimate("steel", embodied_energy=2)
        before = BOM([
            BOMLine("bolt", 2, "steel", 10, 1, 0.1, estimate),
            BOMLine("washer", 1, "steel", 2, 0.2, 0.01, estimate),
        ])
        after = BOM([
            BOMLine("bolt", 3, "steel", 10, 1, 0.1, estimate),
            BOMLine("nut", 1, "steel", 5, 0.5, 0.03, estimate),
        ])
        report = compare_revisions(before, after)
        lines = {(line.part, line.change): line for line in report.bom_lines}
        self.assertEqual(lines["bolt", "changed"].quantity_delta, 1)
        self.assertEqual(lines["bolt", "changed"].energy_delta, 2)
        self.assertEqual(lines["nut", "added"].cost_delta, 0.5)
        self.assertEqual(lines["washer", "removed"].mass_delta, -2)

    def test_bom_aggregate_deltas_include_energy(self):
        e1 = PartEstimate("steel", embodied_energy=2)
        e2 = PartEstimate("steel", embodied_energy=3)
        before = BOM([BOMLine("p", 2, "steel", 10, 1, 0.1, e1)])
        after = BOM([BOMLine("p", 2, "steel", 12, 2, 0.2, e2)])
        report = compare_revisions(before, after)
        self.assertEqual(report.metric("mass").absolute, 4)
        self.assertEqual(report.metric("energy").absolute, 2)

    def test_mapping_revision_metrics_and_lines(self):
        before = {
            "totals": {"volume": 10, "cost": 2, "carbon": 1},
            "lines": [{"part": "x", "qty": 1, "material": "pla", "unit_cost": 2}],
        }
        after = {
            "totals": {"volume": 15, "cost": 3, "carbon": 1},
            "lines": [{"part": "x", "qty": 2, "material": "pla", "unit_cost": 2}],
        }
        report = compare_revisions(before, after)
        self.assertEqual(report.metric("volume").absolute, 5)
        self.assertEqual(report.bom_lines[0].cost_delta, 2)
        self.assertFalse(report.metric("mass").available)

    def test_zero_baseline_has_absolute_but_no_percent(self):
        report = compare_revisions({"volume": 0}, {"volume": 4})
        delta = report.metric("volume")
        self.assertEqual(delta.absolute, 4)
        self.assertIsNone(delta.percent)

    def test_unavailable_degrades_cleanly(self):
        report = compare_revisions(None, object())
        self.assertFalse(report.available)
        self.assertIn("unavailable", report.note)
        self.assertTrue(all(not metric.available for metric in report.metrics))
        self.assertEqual(report.bom_lines, ())

    def test_serialization_is_deterministic(self):
        before = BOM([
            BOMLine("z", 1, "a", unit_cost=1),
            BOMLine("a", 1, "a", unit_cost=1),
        ])
        report = compare_revisions(before, BOM([]))
        data = report.to_dict()
        self.assertEqual([line["part"] for line in data["bom_lines"]], ["a", "z"])
        self.assertEqual(list(data["metrics"]), ["volume", "mass", "cost", "carbon", "energy"])


if __name__ == "__main__":
    unittest.main()
