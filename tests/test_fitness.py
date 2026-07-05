"""Tests for the quantitative objective layer (fitness.py)."""

from __future__ import annotations

import unittest

from estimate import PartEstimate
from fitness import (
    Objective, PENALTY, Term, cost_objective, dominates, mass_objective,
    multi_objective, target_dims_objective,
)
from verify import Diagnostic, Severity, VerifyReport


class MetricsBackend:
    """Fake backend answering query('metrics')."""

    def __init__(self, volume, bbox):
        self._volume = volume
        self._bbox = bbox

    def query(self, q):
        if q in ("metrics", "measure"):
            return {"volume": self._volume, "bbox": self._bbox}
        return {}


class FakeVariant:
    """A tournament-style competitor exposing a backend + ok flag."""

    def __init__(self, backend, ok=True):
        self.backend = backend
        self.ok = ok


# --------------------------------------------------------------------------- #
# Monotonicity
# --------------------------------------------------------------------------- #
class TestMonotonic(unittest.TestCase):
    def test_lower_mass_scores_higher(self):
        obj = mass_objective()
        light = MetricsBackend(volume=500.0, bbox=[10, 10, 5])
        heavy = MetricsBackend(volume=2000.0, bbox=[10, 10, 20])
        self.assertGreater(obj.score(light), obj.score(heavy))

    def test_score_strictly_monotonic_across_volumes(self):
        obj = mass_objective()
        scores = [obj.score(MetricsBackend(volume=v, bbox=[10, 10, v / 100.0]))
                  for v in (100.0, 500.0, 1000.0, 5000.0)]
        # As volume (mass) increases, score must strictly decrease.
        for a, b in zip(scores, scores[1:]):
            self.assertGreater(a, b)

    def test_lower_cost_scores_higher(self):
        obj = cost_objective()
        cheap = MetricsBackend(volume=500.0, bbox=[10, 10, 5])
        pricey = MetricsBackend(volume=5000.0, bbox=[20, 20, 12.5])
        self.assertGreater(obj.score(cheap), obj.score(pricey))

    def test_accepts_partestimate_directly(self):
        obj = mass_objective()
        light = PartEstimate(material="aluminium", mass=1.0)
        heavy = PartEstimate(material="aluminium", mass=9.0)
        self.assertGreater(obj.score(light), obj.score(heavy))


# --------------------------------------------------------------------------- #
# Multi-objective weighting
# --------------------------------------------------------------------------- #
class TestMultiObjective(unittest.TestCase):
    def test_violations_dominate_when_weighted(self):
        obj = multi_objective(mass_weight=1.0, cost_weight=1.0,
                              violation_weight=1000.0)
        clean = MetricsBackend(volume=2000.0, bbox=[10, 10, 20])   # heavier
        broken = MetricsBackend(volume=500.0, bbox=[10, 10, 5])    # lighter
        rep_ok = VerifyReport([])
        rep_bad = VerifyReport([
            Diagnostic(Severity.ERROR, "over-constrained", "bad")])
        # The lighter part has an ERROR; with a big violation weight the
        # heavier-but-valid part must win.
        self.assertGreater(obj.score(clean, rep_ok),
                           obj.score(broken, rep_bad))

    def test_weight_shifts_preference(self):
        # Two parts: A lighter but (say) pricier material, B heavier cheaper.
        a = PartEstimate(material="x", mass=1.0, material_cost=10.0,
                         rough_machining_cost=0.0)
        b = PartEstimate(material="x", mass=5.0, material_cost=1.0,
                         rough_machining_cost=0.0)
        mass_heavy = multi_objective(mass_weight=100.0, cost_weight=1.0,
                                     violation_weight=0.0)
        cost_heavy = multi_objective(mass_weight=1.0, cost_weight=100.0,
                                     violation_weight=0.0)
        # Mass-weighted prefers the lighter A.
        self.assertGreater(mass_heavy.score(a), mass_heavy.score(b))
        # Cost-weighted prefers the cheaper B.
        self.assertGreater(cost_heavy.score(b), cost_heavy.score(a))

    def test_unmeasurable_gets_penalised(self):
        obj = mass_objective()
        good = MetricsBackend(volume=1000.0, bbox=[10, 10, 10])
        blind = PartEstimate(material="aluminium", mass=None)  # unmeasured
        self.assertGreater(obj.score(good), obj.score(blind))
        self.assertLessEqual(obj.score(blind), -PENALTY + 1)


# --------------------------------------------------------------------------- #
# Target dims + Pareto vector
# --------------------------------------------------------------------------- #
class TestTargetAndPareto(unittest.TestCase):
    def test_target_dims_prefers_closer(self):
        obj = target_dims_objective((10.0, 10.0, 10.0))
        close = MetricsBackend(volume=1000.0, bbox=[10, 10, 11])
        far = MetricsBackend(volume=1000.0, bbox=[10, 10, 30])
        self.assertGreater(obj.score(close), obj.score(far))

    def test_vector_length_and_dominance(self):
        obj = multi_objective(violation_weight=1.0)
        light = MetricsBackend(volume=500.0, bbox=[10, 10, 5])
        heavy = MetricsBackend(volume=5000.0, bbox=[20, 20, 12.5])
        v_light = obj.vector(light, VerifyReport([]))
        v_heavy = obj.vector(heavy, VerifyReport([]))
        self.assertEqual(len(v_light), 3)
        # Lighter+cheaper+equal-violations dominates (min-orientation).
        self.assertTrue(dominates(v_light, v_heavy))
        self.assertFalse(dominates(v_heavy, v_light))

    def test_dominates_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            dominates((1.0,), (1.0, 2.0))


# --------------------------------------------------------------------------- #
# Integration as a scorer
# --------------------------------------------------------------------------- #
class TestScorerIntegration(unittest.TestCase):
    def test_scores_two_variants_better_wins(self):
        obj = mass_objective()
        light = FakeVariant(MetricsBackend(volume=500.0, bbox=[10, 10, 5]))
        heavy = FakeVariant(MetricsBackend(volume=3000.0, bbox=[10, 10, 30]))
        winner = max([light, heavy], key=obj.score)
        self.assertIs(winner, light)

    def test_not_ok_variant_loses(self):
        obj = mass_objective()
        # A lighter but not-ok variant must lose to a heavier valid one.
        broken = FakeVariant(
            MetricsBackend(volume=100.0, bbox=[5, 5, 4]), ok=False)
        valid = FakeVariant(
            MetricsBackend(volume=3000.0, bbox=[10, 10, 30]), ok=True)
        self.assertGreater(obj.score(valid), obj.score(broken))

    def test_callable_alias(self):
        obj = mass_objective()
        b = MetricsBackend(volume=1000.0, bbox=[10, 10, 10])
        self.assertEqual(obj(b), obj.score(b))

    def test_as_scorer_closure(self):
        obj = multi_objective(violation_weight=1000.0)
        rep = VerifyReport([Diagnostic(Severity.ERROR, "x", "bad")])
        scorer = obj.as_scorer(rep)
        b = MetricsBackend(volume=1000.0, bbox=[10, 10, 10])
        # Same as calling score with the report.
        self.assertEqual(scorer(b), obj.score(b, rep))


if __name__ == "__main__":
    unittest.main()
