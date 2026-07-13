import unittest

from harnesscad.io.formats.stepllm_parser import Entity, Enum, Ref, StepFile, parse
from harnesscad.io.formats.stepllm_graph import (
    build_graph, dangling_references, is_acyclic, reachable, roots,
    topological_order, unreachable, validate,
)


def _brep_file():
    return "\n".join([
        "ISO-10303-21;", "HEADER;", "ENDSEC;", "DATA;",
        "#1=CARTESIAN_POINT('',(0.,0.,0.));",
        "#2=DIRECTION('',(0.,0.,1.));",
        "#3=DIRECTION('',(1.,0.,0.));",
        "#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);",
        "#5=PLANE('',#4);",
        "#6=CLOSED_SHELL('',(#5));",
        "#7=MANIFOLD_SOLID_BREP('',#6);",
        "ENDSEC;", "END-ISO-10303-21;", "",
    ])


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.step = parse(_brep_file())

    def test_out_edges(self):
        g = build_graph(self.step)
        self.assertEqual(g.successors(4), [1, 2, 3])
        self.assertEqual(g.successors(7), [6])

    def test_in_degree(self):
        g = build_graph(self.step)
        self.assertEqual(g.in_degree[1], 1)   # referenced by #4
        self.assertEqual(g.in_degree[7], 0)   # root

    def test_no_dangling(self):
        self.assertEqual(dangling_references(self.step), [])

    def test_roots(self):
        self.assertEqual(roots(self.step), [7])

    def test_topological_order(self):
        order = topological_order(self.step)
        # A referrer must precede its referents.
        self.assertLess(order.index(7), order.index(6))
        self.assertLess(order.index(4), order.index(1))

    def test_acyclic(self):
        self.assertTrue(is_acyclic(self.step))

    def test_reachable_from_root(self):
        self.assertEqual(reachable(self.step), {1, 2, 3, 4, 5, 6, 7})

    def test_no_unreachable(self):
        self.assertEqual(unreachable(self.step), [])

    def test_validate_ok(self):
        report = validate(self.step)
        self.assertTrue(report.valid, report.summary())


class TestDangling(unittest.TestCase):
    def test_dangling_detected(self):
        step = StepFile()
        step.add(Entity(1, "PLANE", ["", Ref(99)]))
        self.assertEqual(dangling_references(step), [(1, 99)])
        self.assertFalse(validate(step).valid)


class TestCycle(unittest.TestCase):
    def test_cycle_rejected(self):
        step = StepFile()
        step.add(Entity(1, "MANIFOLD_SOLID_BREP", ["", Ref(2)]))
        step.add(Entity(2, "CLOSED_SHELL", ["", [Ref(1)]]))
        self.assertFalse(is_acyclic(step))
        with self.assertRaises(ValueError):
            topological_order(step)


class TestReachability(unittest.TestCase):
    def test_dead_instance_reported(self):
        text = _brep_file().replace(
            "#7=MANIFOLD_SOLID_BREP('',#6);",
            "#7=MANIFOLD_SOLID_BREP('',#6);\n"
            "#8=CARTESIAN_POINT('',(9.,9.,9.));")
        step = parse(text)
        self.assertEqual(unreachable(step), [8])
        # Unreachable instances are reported but do not invalidate the file.
        self.assertTrue(validate(step).valid)


class TestMissingRoot(unittest.TestCase):
    def test_no_root_is_invalid(self):
        text = "\n".join([
            "ISO-10303-21;", "HEADER;", "ENDSEC;", "DATA;",
            "#1=CARTESIAN_POINT('',(0.,0.,0.));",
            "ENDSEC;", "END-ISO-10303-21;", "",
        ])
        step = parse(text)
        report = validate(step)
        self.assertEqual(report.roots, [])
        self.assertFalse(report.valid)


class TestSchemaProblem(unittest.TestCase):
    def test_schema_violation_makes_invalid(self):
        text = _brep_file().replace(
            "#5=PLANE('',#4);", "#5=PLANE('');")  # wrong arity
        step = parse(text)
        report = validate(step)
        self.assertTrue(report.schema_problems)
        self.assertFalse(report.valid)


if __name__ == "__main__":
    unittest.main()
