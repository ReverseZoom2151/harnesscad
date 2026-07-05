import json
import unittest

from library.family import (
    FamilySpec,
    ParameterAxis,
    Validation,
    generate_family,
    parameter_grid,
    range_axis,
)


class PartFamilyTests(unittest.TestCase):
    def test_parameter_product_is_deterministic(self):
        spec = FamilySpec(
            "bolt",
            (
                ParameterAxis("diameter", (6, 8), "mm"),
                ParameterAxis("length", (10, 20), "mm"),
            ),
        )
        self.assertEqual(
            parameter_grid(spec),
            (
                {"diameter": 6, "length": 10},
                {"diameter": 6, "length": 20},
                {"diameter": 8, "length": 10},
                {"diameter": 8, "length": 20},
            ),
        )

    def test_validated_family_manifest(self):
        spec = FamilySpec(
            "M12-screw",
            (range_axis("length", 10, 30, 10, unit="mm"),),
            filename_template="{family}_L{length}",
            metadata={"standard": "fixture"},
        )
        manifest = generate_family(
            spec,
            lambda p: {"radius": 6, **p},
            lambda artifact, p: Validation(
                p["length"] <= 20,
                {"diameter": artifact["radius"] * 2 == 12, "length": p["length"] <= 20},
                "test limit exceeded",
            ),
            serializer=lambda artifact: json.dumps(
                artifact, sort_keys=True
            ).encode(),
        )
        self.assertEqual([entry.name for entry in manifest.entries], [
            "M12-screw_L10", "M12-screw_L20", "M12-screw_L30"
        ])
        self.assertEqual(len(manifest.accepted), 2)
        self.assertEqual(manifest.entries[0].units, {"length": "mm"})
        self.assertEqual(len(manifest.entries[0].digest), 64)
        self.assertEqual(json.loads(manifest.to_json())["metadata"]["standard"], "fixture")

    def test_failed_member_does_not_abort_family(self):
        def build(parameters):
            if parameters["length"] == 20:
                raise RuntimeError("kernel failure")
            return parameters

        manifest = generate_family(
            FamilySpec("pin", (ParameterAxis("length", (10, 20, 30)),)),
            build,
            lambda artifact, parameters: Validation(True, {"solid": True}),
        )
        self.assertEqual(len(manifest.entries), 3)
        self.assertIn("kernel failure", manifest.entries[1].error)
        self.assertEqual(len(manifest.accepted), 2)

    def test_variant_limit_prevents_accidental_explosion(self):
        spec = FamilySpec(
            "large",
            (ParameterAxis("a", tuple(range(11))), ParameterAxis("b", tuple(range(10)))),
            maximum_variants=100,
        )
        with self.assertRaisesRegex(ValueError, "110 variants"):
            parameter_grid(spec)

    def test_duplicate_axis_and_values_are_rejected(self):
        with self.assertRaises(ValueError):
            ParameterAxis("x", (1, 1))
        axis = ParameterAxis("x", (1,))
        with self.assertRaises(ValueError):
            FamilySpec("bad", (axis, axis))

    def test_strict_mode_propagates_builder_error(self):
        with self.assertRaisesRegex(RuntimeError, "stop"):
            generate_family(
                FamilySpec("x", (ParameterAxis("n", (1,)),)),
                lambda parameters: (_ for _ in ()).throw(RuntimeError("stop")),
                lambda artifact, parameters: Validation(True, {}),
                continue_on_error=False,
            )


if __name__ == "__main__":
    unittest.main()
