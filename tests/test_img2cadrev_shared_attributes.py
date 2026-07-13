import unittest

from harnesscad.domain.reconstruction.sequences.shared_attributes import SharedAttributePrior
from harnesscad.domain.reconstruction.sequences.factorization import factorize, assemble


def chair(leg_radius, seat_ext):
    return [
        {"label": "seat", "commands": [
            {"type": "L", "attrs": [1.0, 0.0]},
            {"type": "Ej", "attrs": [0, 0, 0, 0, 0, 0, seat_ext]},
        ]},
        {"label": "leg", "commands": [
            {"type": "R", "attrs": [0.0, 0.0, leg_radius]},
            {"type": "Ej", "attrs": [0, 0, 0, 0, 0, 0, 1.0]},
        ]},
    ]


class SharedAttributePriorTest(unittest.TestCase):
    def test_mean_aggregation(self):
        prior = SharedAttributePrior().fit([chair(0.1, 0.2), chair(0.3, 0.4)])
        # leg circle radius key mean of 0.1 and 0.3 -> 0.2
        key = ("leg", "R", 0)
        self.assertEqual(prior.count(key), 2)
        self.assertAlmostEqual(prior.mean(key)[2], 0.2)

    def test_predict_from_prior(self):
        prior = SharedAttributePrior().fit([chair(0.1, 0.2), chair(0.3, 0.4)])
        structure, _ = factorize(chair(0.0, 0.0))
        pred = prior.predict(structure)
        # Predicted leg radius (command index 2, attr 2) is the shared mean 0.2.
        self.assertAlmostEqual(pred[2][2], 0.2)
        # Predictions assemble into a valid model.
        model = assemble(structure, pred)
        self.assertEqual(model[1]["commands"][0]["attrs"][2], 0.2)

    def test_predict_unseen_key_zero(self):
        prior = SharedAttributePrior().fit([chair(0.1, 0.2)])
        structure = [{"label": "armrest", "command_types": ["L"]}]
        pred = prior.predict(structure)
        self.assertEqual(pred, [[0.0, 0.0]])

    def test_coverage(self):
        prior = SharedAttributePrior().fit([chair(0.1, 0.2)])
        seen, _ = factorize(chair(0.5, 0.5))
        self.assertEqual(prior.coverage(seen), 1.0)
        mixed = [{"label": "unknown", "command_types": ["L"]}]
        self.assertEqual(prior.coverage(mixed), 0.0)

    def test_regularize_blends(self):
        prior = SharedAttributePrior().fit([chair(0.2, 0.2), chair(0.2, 0.2)])
        structure, attrs = factorize(chair(0.0, 0.0))
        blended = prior.regularize(structure, attrs, weight=0.5)
        # leg radius: 0.5*0.0 + 0.5*0.2 = 0.1
        self.assertAlmostEqual(blended[2][2], 0.1)

    def test_regularize_weight_bounds(self):
        prior = SharedAttributePrior().fit([chair(0.2, 0.2)])
        structure, attrs = factorize(chair(0.0, 0.0))
        with self.assertRaises(ValueError):
            prior.regularize(structure, attrs, weight=2.0)

    def test_regularize_zero_weight_identity(self):
        prior = SharedAttributePrior().fit([chair(0.2, 0.2)])
        structure, attrs = factorize(chair(0.5, 0.5))
        self.assertEqual(prior.regularize(structure, attrs, weight=0.0), attrs)


if __name__ == "__main__":
    unittest.main()
