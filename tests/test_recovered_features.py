"""Integration coverage for modules recovered from the interrupted build wave."""

import unittest

from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch
from harnesscad.data.dataengine import EditPairStore, capture_edit_pair, to_preference
from harnesscad.io.ingest import import_fidelity, roundtrip_fidelity
from harnesscad.eval.quality.ask import ask
from harnesscad.eval.quality.pareto import Objective, pareto_front, pareto_rank
from harnesscad.eval.reliability import RetrievalFallback


class MetricsBackend:
    def __init__(self, metrics):
        self.metrics = metrics

    def query(self, kind):
        if kind in ("metrics", "measure"):
            return dict(self.metrics)
        if kind == "summary":
            return {"feature_count": 0, "solid_present": False}
        return {}


class RecoveredFeaturesTests(unittest.TestCase):
    def test_retrieval_fallback_always_returns_buildable_ops(self):
        result = RetrievalFallback().fallback("small mounting plate", "timeout")
        self.assertTrue(result.approximate)
        self.assertEqual(result.source, "default")
        self.assertEqual([op.OP for op in result.ops_or_part],
                         ["new_sketch", "add_rectangle", "extrude"])

    def test_roundtrip_fidelity_compares_measured_geometry(self):
        source = MetricsBackend({
            "volume": 1000.0,
            "bbox": [10.0, 20.0, 5.0],
            "faces": 6,
        })
        rebuilt = MetricsBackend({
            "volume": 1000.5,
            "bbox": [10.0, 20.0, 5.0],
            "faces": 6,
        })
        report = roundtrip_fidelity(source, rebuilt, rel_tol=0.001)
        self.assertTrue(report.available)
        self.assertTrue(report.matched)

    def test_import_fidelity_rejects_degenerate_geometry(self):
        report = import_fidelity(MetricsBackend({
            "volume": 0.0,
            "bbox": [10.0, 0.0, 5.0],
            "faces": 0,
        }))
        self.assertFalse(report.matched)

    def test_human_edit_pair_exports_preference_signal(self):
        proposed = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10),
            Extrude(sketch="sk1", distance=2),
        ]
        final = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10),
            Extrude(sketch="sk1", distance=2),
        ]
        pair = capture_edit_pair(proposed, final, brief="wider plate")
        preference = to_preference(pair)
        self.assertGreater(pair.n_changes, 0)
        self.assertTrue(preference["has_signal"])
        store = EditPairStore()
        store.add(pair)
        self.assertEqual(len(store.to_preferences()), 1)

    def test_pareto_helpers_preserve_tradeoffs(self):
        items = [
            {"name": "light", "mass": 1.0, "cost": 5.0},
            {"name": "cheap", "mass": 2.0, "cost": 2.0},
            {"name": "dominated", "mass": 3.0, "cost": 6.0},
        ]
        objectives = [
            Objective("mass", "min", key="mass"),
            Objective("cost", "min", key="cost"),
        ]
        front = pareto_front(items, objectives)
        self.assertEqual([item["name"] for item in front], ["light", "cheap"])
        self.assertEqual(len(pareto_rank(items, objectives)), 2)

    def test_ask_answers_grounded_mass_query(self):
        backend = MetricsBackend({
            "volume": 1000.0,
            "bbox": [10.0, 10.0, 10.0],
        })
        answer = ask("What is the mass?", backend=backend)
        self.assertIn("mass", answer.lower())


if __name__ == "__main__":
    unittest.main()
