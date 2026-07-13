import unittest

from harnesscad.domain.reconstruction.scene.graphcad_knowledge_graph import (
    GraphNode,
    KnowledgeGraph,
    Material,
    build_waves,
    iter_subtree,
    parse_document,
    parse_knowledge_graph,
    parse_material_library,
    serialize_knowledge_graph,
    serialize_material_library,
    validate_graph,
)


DOC = """
-- MATERIAL LIBRARY --
wood_oak | diffuse_color=(0.6,0.4,0.2,1.0)
metal_steel | diffuse_color=(0.7,0.7,0.7,1.0)
#END_MATERIALS

# ----------  BEGIN_GRAPH  ----------
L0: id=table | parent=- | type=assembly | create_method=composite
    | assembly_order=[top], [leg_a, leg_b]
L1: id=top | parent=table | type=part | size=(1.0,0.6,0.03)
    | mat=wood_oak | create_method=primitive
L1: id=leg_a | parent=table | type=part | mat=metal_steel
    | create_method=primitive | pos=offset(-0.45,-0.25,0)
L1: id=leg_b | parent=table | type=part | mat=metal_steel
    | create_method=primitive | after=leg_a
# ----------  END_GRAPH  ----------
"""


def _graph(text=DOC):
    return parse_document(text)


class MaterialLibraryTests(unittest.TestCase):
    def test_parses_entries(self):
        materials = parse_material_library(DOC)
        self.assertEqual(
            [material.name for material in materials], ["wood_oak", "metal_steel"]
        )
        self.assertEqual(materials[0].diffuse_color, (0.6, 0.4, 0.2, 1.0))

    def test_alpha_defaults_to_one(self):
        text = "-- MATERIAL LIBRARY --\nred | diffuse_color=(1,0,0)\n#END_MATERIALS"
        self.assertEqual(parse_material_library(text)[0].diffuse_color, (1.0, 0.0, 0.0, 1.0))

    def test_serialisation_is_alphabetical(self):
        block = serialize_material_library(parse_material_library(DOC))
        lines = block.splitlines()
        self.assertEqual(lines[0], "-- MATERIAL LIBRARY --")
        self.assertEqual(lines[1], "metal_steel | diffuse_color=(0.7,0.7,0.7,1)")
        self.assertEqual(lines[2], "wood_oak | diffuse_color=(0.6,0.4,0.2,1)")
        self.assertEqual(lines[3], "#END_MATERIALS")

    def test_channel_range_is_checked(self):
        with self.assertRaises(ValueError):
            Material("bad", (1.5, 0.0, 0.0, 1.0))


class ParseGraphTests(unittest.TestCase):
    def test_node_ids_and_layers(self):
        nodes = parse_knowledge_graph(DOC)
        self.assertEqual([node.node_id for node in nodes], ["table", "top", "leg_a", "leg_b"])
        self.assertEqual([node.layer for node in nodes], [0, 1, 1, 1])

    def test_continuation_lines_are_merged(self):
        node = _graph().by_id()["top"]
        self.assertEqual(node.mat, "wood_oak")
        self.assertEqual(node.size, "(1.0,0.6,0.03)")
        self.assertEqual(node.create_method, "primitive")

    def test_dash_placeholder_becomes_none(self):
        self.assertIsNone(_graph().by_id()["table"].parent)

    def test_pipe_inside_brackets_is_not_a_separator(self):
        text = "L1: id=a | parent=- | constraint=f(x|y) | create_method=primitive"
        node = parse_knowledge_graph(text)[0]
        self.assertEqual(node.constraint, "f(x|y)")

    def test_after_list(self):
        self.assertEqual(_graph().by_id()["leg_b"].after, ("leg_a",))

    def test_assembly_groups(self):
        self.assertEqual(
            _graph().by_id()["table"].assembly_groups(),
            (("top",), ("leg_a", "leg_b")),
        )

    def test_roots_children_leaves(self):
        graph = _graph()
        self.assertEqual([node.node_id for node in graph.roots()], ["table"])
        self.assertEqual(
            [node.node_id for node in graph.children_of("table")],
            ["top", "leg_a", "leg_b"],
        )
        self.assertEqual(
            [node.node_id for node in graph.leaves()], ["top", "leg_a", "leg_b"]
        )

    def test_unknown_field_lands_in_extra(self):
        node = parse_knowledge_graph("L2: id=a | parent=- | weirdo=7")[0]
        self.assertEqual(node.extra, {"weirdo": "7"})

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            KnowledgeGraph((GraphNode("a", 0), GraphNode("a", 1)))


class SerializeGraphTests(unittest.TestCase):
    def test_round_trip_is_stable(self):
        graph = _graph()
        text = serialize_knowledge_graph(graph.nodes)
        reparsed = parse_knowledge_graph(text)
        self.assertEqual(reparsed, graph.nodes)

    def test_markers_present(self):
        text = serialize_knowledge_graph(_graph().nodes)
        self.assertIn("BEGIN_GRAPH", text.splitlines()[0])
        self.assertIn("END_GRAPH", text.splitlines()[-1])


class ValidateTests(unittest.TestCase):
    def test_valid_graph_has_no_errors(self):
        self.assertEqual(validate_graph(_graph()), ())

    def test_empty_graph(self):
        self.assertEqual(validate_graph(KnowledgeGraph(())), ("graph has no nodes",))

    def test_unknown_parent(self):
        graph = KnowledgeGraph((GraphNode("a", 0, create_method="primitive"),
                                GraphNode("b", 1, parent="ghost", create_method="primitive")))
        errors = validate_graph(graph)
        self.assertIn("b: unknown parent 'ghost'", errors)

    def test_parent_cycle(self):
        graph = KnowledgeGraph((
            GraphNode("a", 0, parent="b", create_method="primitive"),
            GraphNode("b", 0, parent="a", create_method="primitive"),
        ))
        errors = validate_graph(graph)
        self.assertIn("a: parent chain forms a cycle", errors)

    def test_dependency_cycle(self):
        graph = KnowledgeGraph((
            GraphNode("r", 0, create_method="composite"),
            GraphNode("a", 1, parent="r", create_method="primitive", depends_on=("b",)),
            GraphNode("b", 1, parent="r", create_method="primitive", depends_on=("a",)),
        ))
        self.assertIn(
            "dependency edges (after/depends_on) form a cycle", validate_graph(graph)
        )

    def test_leaf_without_create_method(self):
        graph = KnowledgeGraph((
            GraphNode("r", 0, create_method="composite"),
            GraphNode("a", 1, parent="r"),
        ))
        self.assertIn("a: leaf node has no create_method", validate_graph(graph))

    def test_boolean_needs_tool_and_target(self):
        graph = KnowledgeGraph((
            GraphNode("r", 0, create_method="boolean_subtract"),
        ))
        errors = validate_graph(graph)
        self.assertIn("r: boolean_subtract needs a tool_id", errors)
        self.assertIn("r: boolean_subtract needs a target_id", errors)

    def test_unknown_material(self):
        graph = KnowledgeGraph(
            (GraphNode("r", 0, create_method="primitive", mat="pink"),),
            (Material("wood_oak", (0.6, 0.4, 0.2, 1.0)),),
        )
        self.assertIn("r: material 'pink' is not in the library", validate_graph(graph))

    def test_assembly_order_must_partition_children(self):
        graph = KnowledgeGraph((
            GraphNode("r", 0, create_method="composite", assembly_order="[a], [a, ghost]"),
            GraphNode("a", 1, parent="r", create_method="primitive"),
            GraphNode("b", 1, parent="r", create_method="primitive"),
        ))
        errors = validate_graph(graph)
        self.assertIn("r: assembly_order repeats 'a'", errors)
        self.assertIn("r: assembly_order lists non-child 'ghost'", errors)
        self.assertIn("r: assembly_order omits child 'b'", errors)

    def test_after_must_be_sibling(self):
        graph = KnowledgeGraph((
            GraphNode("r", 0, create_method="composite"),
            GraphNode("a", 1, parent="r", create_method="primitive"),
            GraphNode("b", 1, parent="a", create_method="primitive", after=("a",)),
        ))
        self.assertIn("b: after 'a' is not a sibling", validate_graph(graph))


class BuildWaveTests(unittest.TestCase):
    def test_children_precede_parents_and_order_groups(self):
        waves = build_waves(_graph())
        self.assertEqual(waves, (("top",), ("leg_a",), ("leg_b",), ("table",)))

    def test_parallel_siblings_share_a_wave(self):
        graph = KnowledgeGraph((
            GraphNode("r", 0, create_method="composite"),
            GraphNode("a", 1, parent="r", create_method="primitive"),
            GraphNode("b", 1, parent="r", create_method="primitive"),
        ))
        self.assertEqual(build_waves(graph), (("a", "b"), ("r",)))

    def test_boolean_operands_precede_the_boolean(self):
        graph = KnowledgeGraph((
            GraphNode("cut", 0, create_method="boolean_subtract", tool_id="drill",
                      target_id="block"),
            GraphNode("block", 1, parent="cut", create_method="primitive"),
            GraphNode("drill", 1, parent="cut", create_method="primitive"),
        ))
        self.assertEqual(build_waves(graph), (("block", "drill"), ("cut",)))

    def test_cycle_raises(self):
        graph = KnowledgeGraph((
            GraphNode("r", 0, create_method="composite"),
            GraphNode("a", 1, parent="r", create_method="primitive", depends_on=("b",)),
            GraphNode("b", 1, parent="r", create_method="primitive", depends_on=("a",)),
        ))
        with self.assertRaises(ValueError):
            build_waves(graph)

    def test_subtree_preorder(self):
        graph = _graph()
        self.assertEqual(
            [node.node_id for node in iter_subtree(graph, "table")],
            ["table", "top", "leg_a", "leg_b"],
        )
        with self.assertRaises(KeyError):
            list(iter_subtree(graph, "ghost"))


if __name__ == "__main__":
    unittest.main()
