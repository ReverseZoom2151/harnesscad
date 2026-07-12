import math
import unittest

from bench.blenderllm_complexity import (
    complexity,
    parameter_density,
    safe_complexity,
    shannon_entropy,
    type_distribution,
    unit_number,
)

MIXED = (
    "import bpy\n"
    "bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))\n"
    "bpy.ops.mesh.primitive_uv_sphere_add(radius=1)\n"
    "bpy.ops.mesh.primitive_cone_add(radius1=1, depth=2)\n"
)

UNIFORM = (
    "import bpy\n"
    "bpy.ops.mesh.primitive_cube_add()\n"
    "bpy.ops.mesh.primitive_cube_add()\n"
)


class UnitNumberTests(unittest.TestCase):
    def test_counts_primitives(self):
        self.assertEqual(unit_number(MIXED), 3)

    def test_ignores_non_primitive_ops(self):
        script = "import bpy\nbpy.ops.transform.resize(value=(1, 2, 3))\n"
        self.assertEqual(unit_number(script), 0)


class TypeDistributionTests(unittest.TestCase):
    def test_distribution(self):
        self.assertEqual(
            type_distribution(MIXED),
            {"primitive_cube_add": 1, "primitive_uv_sphere_add": 1, "primitive_cone_add": 1},
        )

    def test_uniform(self):
        self.assertEqual(type_distribution(UNIFORM), {"primitive_cube_add": 2})


class EntropyTests(unittest.TestCase):
    def test_single_type_zero(self):
        self.assertEqual(shannon_entropy({"a": 5}), 0.0)

    def test_empty_zero(self):
        self.assertEqual(shannon_entropy({}), 0.0)

    def test_uniform_three_types(self):
        self.assertAlmostEqual(shannon_entropy({"a": 1, "b": 1, "c": 1}), math.log2(3))

    def test_two_equal_types_one_bit(self):
        self.assertAlmostEqual(shannon_entropy({"a": 3, "b": 3}), 1.0)

    def test_never_negative(self):
        self.assertGreaterEqual(shannon_entropy({"a": 2, "b": 2, "c": 4}), 0.0)


class ParameterDensityTests(unittest.TestCase):
    def test_counts_numeric_params_per_unit(self):
        # cube: size=2, location tuple 0,0,0 -> 4 numerics
        # sphere: radius=1 -> 1
        # cone: radius1=1, depth=2 -> 2 ; total 7 over 3 units
        self.assertAlmostEqual(parameter_density(MIXED), 7 / 3)

    def test_zero_units_zero_density(self):
        self.assertEqual(parameter_density("import bpy\nx = 5\n"), 0.0)

    def test_booleans_not_counted(self):
        script = "import bpy\nbpy.ops.mesh.primitive_cube_add(enter_editmode=True, size=1)\n"
        # Only size=1 counts, not the boolean.
        self.assertAlmostEqual(parameter_density(script), 1.0)


class ComplexityTests(unittest.TestCase):
    def test_bundle(self):
        c = complexity(MIXED)
        self.assertEqual(c.unit_number, 3)
        self.assertAlmostEqual(c.parameter_density, 7 / 3)
        self.assertAlmostEqual(c.entropy, math.log2(3))
        self.assertEqual(len(c.type_distribution), 3)

    def test_uniform_zero_entropy(self):
        c = complexity(UNIFORM)
        self.assertEqual(c.unit_number, 2)
        self.assertEqual(c.entropy, 0.0)

    def test_safe_complexity_on_bad_script(self):
        self.assertIsNone(safe_complexity("bpy.ops.mesh.primitive_cube_add(\n"))

    def test_safe_complexity_on_good_script(self):
        self.assertIsNotNone(safe_complexity(UNIFORM))


if __name__ == "__main__":
    unittest.main()
