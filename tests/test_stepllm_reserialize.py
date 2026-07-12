import unittest

from formats.stepllm_parser import Entity, Real, Ref, StepFile, parse, serialize
from formats.stepllm_graph import validate
from ingest.stepllm_reserialize import (
    annotate, branch_stats, dfs_order, format_real, normalize_reals,
    renumber, reserialize,
)


def _gapped_file():
    # Intentionally gapped/scrambled ids; root #40 references a shell etc.
    return "\n".join([
        "ISO-10303-21;", "HEADER;", "ENDSEC;", "DATA;",
        "#10=CARTESIAN_POINT('',(0.0000000,0.,0.));",
        "#20=DIRECTION('',(0.,0.,1.));",
        "#25=DIRECTION('',(1.,0.,0.));",
        "#30=AXIS2_PLACEMENT_3D('',#10,#20,#25);",
        "#35=PLANE('',#30);",
        "#38=CLOSED_SHELL('',(#35));",
        "#40=MANIFOLD_SOLID_BREP('',#38);",
        "ENDSEC;", "END-ISO-10303-21;", "",
    ])


class TestDfsOrder(unittest.TestCase):
    def test_root_first(self):
        step = parse(_gapped_file())
        order = dfs_order(step)
        self.assertEqual(order[0], 40)  # the root
        # depth-first: the shell then placement expanded before siblings
        self.assertLess(order.index(38), order.index(35))
        self.assertLess(order.index(30), order.index(10))

    def test_each_instance_once(self):
        step = parse(_gapped_file())
        order = dfs_order(step)
        self.assertEqual(sorted(order), sorted(step.order))
        self.assertEqual(len(order), len(set(order)))

    def test_shared_reference_visited_once(self):
        # #4 and #5 both reference #1 (shared); DFS emits #1 exactly once.
        step = StepFile()
        step.add(Entity(9, "MANIFOLD_SOLID_BREP", ["", Ref(8)]))
        step.add(Entity(8, "CLOSED_SHELL", ["", [Ref(4), Ref(5)]]))
        step.add(Entity(4, "PLANE", ["", Ref(1)]))
        step.add(Entity(5, "PLANE", ["", Ref(1)]))
        step.add(Entity(1, "CARTESIAN_POINT", ["", [Real("0."), Real("0."), Real("0.")]]))
        order = dfs_order(step)
        self.assertEqual(order.count(1), 1)


class TestRenumber(unittest.TestCase):
    def test_sequential_ids(self):
        step = parse(_gapped_file())
        out = renumber(step)
        self.assertEqual(out.order, [1, 2, 3, 4, 5, 6, 7])

    def test_references_remapped_consistently(self):
        step = parse(_gapped_file())
        out = renumber(step)
        # root becomes #1 and references the shell which becomes #2
        root = out.entities[1]
        self.assertEqual(root.keyword, "MANIFOLD_SOLID_BREP")
        self.assertEqual(root.params[1], Ref(2))
        self.assertEqual(out.entities[2].keyword, "CLOSED_SHELL")

    def test_renumber_preserves_validity(self):
        step = parse(_gapped_file())
        out = renumber(step)
        self.assertTrue(validate(out).valid)

    def test_no_information_lost(self):
        step = parse(_gapped_file())
        out = renumber(step)
        self.assertEqual(len(out.entities), len(step.entities))


class TestFormatReal(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(format_real(0.0), "0.")

    def test_integer_valued(self):
        self.assertEqual(format_real(2.0), "2.")

    def test_strip_trailing_zeros(self):
        self.assertEqual(format_real(1.5000000), "1.5")

    def test_rounds_to_precision(self):
        self.assertEqual(format_real(0.123456789, 4), "0.1235")

    def test_negative(self):
        self.assertEqual(format_real(-3.0), "-3.")


class TestNormalizeReals(unittest.TestCase):
    def test_long_decimals_shortened(self):
        step = parse("\n".join([
            "ISO-10303-21;", "HEADER;", "ENDSEC;", "DATA;",
            "#1=CARTESIAN_POINT('',(0.3333333333,1.9999999,0.));",
            "ENDSEC;", "END-ISO-10303-21;", "",
        ]))
        out = normalize_reals(step, precision=3)
        coords = out.entities[1].params[1]
        self.assertEqual([c.text for c in coords], ["0.333", "2.", "0."])

    def test_topology_preserved(self):
        step = parse(_gapped_file())
        out = normalize_reals(step, 6)
        self.assertEqual(out.order, step.order)
        self.assertTrue(validate(out).valid)


class TestBranchStats(unittest.TestCase):
    def setUp(self):
        self.step = parse(_gapped_file())
        self.stats = branch_stats(self.step)

    def test_leaf_stats(self):
        leaf = self.stats[10]  # CARTESIAN_POINT
        self.assertEqual(leaf.child_count, 0)
        self.assertEqual(leaf.depth, 0)
        self.assertEqual(leaf.subtree_size, 1)

    def test_placement_children(self):
        placement = self.stats[30]  # references #10,#20,#25
        self.assertEqual(placement.child_count, 3)
        self.assertEqual(placement.depth, 1)
        self.assertEqual(placement.subtree_size, 4)

    def test_root_depth(self):
        root = self.stats[40]
        # 40 -> 38 -> 35 -> 30 -> 10  == depth 4
        self.assertEqual(root.depth, 4)
        self.assertEqual(root.subtree_size, 7)


class TestAnnotate(unittest.TestCase):
    def test_annotations_present_and_parseable(self):
        step = reserialize(parse(_gapped_file()))
        text = annotate(step)
        self.assertIn("/* c=", text)
        # annotations are block comments -> round-trip parse must still work
        reparsed = parse(text)
        self.assertEqual(reparsed.order, step.order)


class TestReserialize(unittest.TestCase):
    def test_full_pipeline(self):
        step = parse(_gapped_file())
        out = reserialize(step, precision=6)
        self.assertEqual(out.order, [1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(validate(out).valid)

    def test_deterministic(self):
        step = parse(_gapped_file())
        a = serialize(reserialize(step))
        b = serialize(reserialize(parse(_gapped_file())))
        self.assertEqual(a, b)

    def test_idempotent_ordering(self):
        step = parse(_gapped_file())
        once = reserialize(step)
        twice = reserialize(once)
        self.assertEqual(serialize(once), serialize(twice))


if __name__ == "__main__":
    unittest.main()
