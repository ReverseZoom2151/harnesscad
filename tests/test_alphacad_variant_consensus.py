import unittest

from harnesscad.agents.exploration.alphacad_variant_consensus import (
    analyze_variants,
    base_positions,
    consensus_base,
    ensemble_brick_confidence,
    position_frequency,
)


def _b(bid, x, y, z):
    return {"id": bid, "x": x, "y": y, "z": z}


def _m(*bricks):
    return {"bricks": list(bricks)}


class TestVariantConsensus(unittest.TestCase):
    def test_base_positions(self):
        m = _m(_b(0, 0, 0, 0), _b(1, 1, 0, 0), _b(2, 0, 0, 1))
        self.assertEqual(base_positions(m), {(0, 0, 0), (1, 0, 0)})

    def test_consensus_intersection(self):
        a = _m(_b(0, 0, 0, 0), _b(1, 1, 0, 0))
        b = _m(_b(0, 0, 0, 0), _b(1, 2, 0, 0))
        c = _m(_b(0, 0, 0, 0), _b(1, 3, 0, 0))
        self.assertEqual(consensus_base([a, b, c]), {(0, 0, 0)})

    def test_consensus_empty_ensemble(self):
        self.assertEqual(consensus_base([]), set())

    def test_position_frequency(self):
        a = _m(_b(0, 0, 0, 0))
        b = _m(_b(0, 0, 0, 0), _b(1, 5, 5, 0))
        freq = position_frequency([a, b])
        self.assertEqual(freq[(0, 0, 0)], 2)
        self.assertEqual(freq[(5, 5, 0)], 1)

    def test_ensemble_confidence_tiers(self):
        models = [_m(_b(0, 0, 0, 0)), _m(_b(0, 0, 0, 0)), _m(_b(0, 0, 0, 0))]
        freq = position_frequency(models)
        confs = ensemble_brick_confidence(models[0], freq, 3)
        self.assertEqual(confs[0].confidence, 100)

    def test_ensemble_unique_low(self):
        model = _m(_b(0, 9, 9, 0))
        freq = position_frequency([model])
        confs = ensemble_brick_confidence(model, freq, 3)
        self.assertEqual(confs[0].confidence, 33)

    def test_violation_penalty(self):
        model = _m(_b(0, 0, 0, 0))
        freq = position_frequency([model, model, model])
        confs = ensemble_brick_confidence(model, freq, 3, violations={0})
        self.assertEqual(confs[0].confidence, 60)

    def test_analyze_variants_shape(self):
        a = _m(_b(0, 0, 0, 0), _b(1, 1, 0, 0))
        b = _m(_b(0, 0, 0, 0), _b(1, 2, 0, 0))
        rep = analyze_variants([a, b])
        self.assertEqual(rep["consensus"], [(0, 0, 0)])
        self.assertEqual(len(rep["per_model"]), 2)
        self.assertIn("brick_confidence", rep["per_model"][0])

    def test_deterministic(self):
        a = _m(_b(0, 0, 0, 0), _b(1, 1, 0, 0))
        b = _m(_b(0, 0, 0, 0), _b(1, 2, 0, 0))
        self.assertEqual(analyze_variants([a, b]), analyze_variants([a, b]))


if __name__ == "__main__":
    unittest.main()
