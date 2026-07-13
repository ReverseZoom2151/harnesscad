import unittest

from harnesscad.domain.reconstruction.img2cadrev_factorization import (
    normalize_model, factorize, assemble, round_trip,
    structure_command_count, structure_attribute_dim, validate_structure,
    structure_signature, part_signature,
)


def sample_chair():
    return [
        {"label": "seat", "commands": [
            {"type": "L", "attrs": [1.0, 0.0]},
            {"type": "L", "attrs": [1.0, 1.0]},
            {"type": "L", "attrs": [0.0, 1.0]},
            {"type": "L", "attrs": [0.0, 0.0]},
            {"type": "Ej", "attrs": [0, 0, 0, 0, 0, 0, 0.2]},
        ]},
        {"label": "leg", "commands": [
            {"type": "R", "attrs": [0.0, 0.0, 0.1]},
            {"type": "Ej", "attrs": [0, 0, 0, 0.1, 0.1, 0, 1.0]},
        ]},
    ]


class FactorizeTest(unittest.TestCase):
    def test_factorize_shapes(self):
        structure, attributes = factorize(sample_chair())
        self.assertEqual(len(structure), 2)
        self.assertEqual(structure[0]["label"], "seat")
        self.assertEqual(structure[0]["command_types"],
                         ["L", "L", "L", "L", "Ej"])
        self.assertEqual(len(attributes), 7)  # 5 + 2 commands
        self.assertEqual(attributes[0], [1.0, 0.0])

    def test_round_trip_lossless(self):
        model = sample_chair()
        self.assertEqual(round_trip(model), normalize_model(model))

    def test_assemble_inverse(self):
        model = sample_chair()
        structure, attributes = factorize(model)
        rebuilt = assemble(structure, attributes)
        self.assertEqual(rebuilt, normalize_model(model))

    def test_command_count_and_dim(self):
        structure, _ = factorize(sample_chair())
        self.assertEqual(structure_command_count(structure), 7)
        # 4*2 (L) + 7 (Ej) + 3 (R) + 7 (Ej) = 8+7+3+7 = 25
        self.assertEqual(structure_attribute_dim(structure), 25)

    def test_assemble_wrong_count(self):
        structure, attributes = factorize(sample_chair())
        with self.assertRaises(ValueError):
            assemble(structure, attributes[:-1])

    def test_assemble_wrong_arity(self):
        structure, attributes = factorize(sample_chair())
        attributes[0] = [1.0]  # L needs 2
        with self.assertRaises(ValueError):
            assemble(structure, attributes)

    def test_normalize_rejects_bad_type(self):
        bad = [{"label": "x", "commands": [{"type": "Z", "attrs": []}]}]
        with self.assertRaises(ValueError):
            normalize_model(bad)

    def test_validate_structure(self):
        validate_structure([{"label": "a", "command_types": ["L"]}])
        with self.assertRaises(ValueError):
            validate_structure([{"label": "a", "command_types": ["Q"]}])
        with self.assertRaises(ValueError):
            validate_structure([{"label": "a"}])


class SignatureTest(unittest.TestCase):
    def test_signature_shared(self):
        s1, _ = factorize(sample_chair())
        s2, _ = factorize(sample_chair())
        self.assertEqual(structure_signature(s1), structure_signature(s2))

    def test_signature_differs(self):
        s1, _ = factorize(sample_chair())
        other = sample_chair()
        other[0]["label"] = "backrest"
        s2, _ = factorize(other)
        self.assertNotEqual(structure_signature(s1), structure_signature(s2))

    def test_part_signature(self):
        s, _ = factorize(sample_chair())
        self.assertEqual(part_signature(s[1]), "leg:R,Ej")


if __name__ == "__main__":
    unittest.main()
