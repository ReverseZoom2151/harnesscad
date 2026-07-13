import unittest

from harnesscad.domain.reconstruction.graphcad_action_plan import (
    Action,
    action_histogram,
    plan_actions,
    render_plan,
    validate_plan,
)
from harnesscad.domain.reconstruction.graphcad_knowledge_graph import (
    GraphNode,
    KnowledgeGraph,
    parse_document,
)


TABLE = """
# ----------  BEGIN_GRAPH  ----------
L0: id=table | parent=- | type=assembly | create_method=composite
    | assembly_order=[top], [leg_a, leg_b]
L1: id=top | parent=table | create_method=primitive | size=(1.0,0.6,0.03)
L1: id=leg_a | parent=table | create_method=primitive
    | align=Align(Z) leg_a.top_face to top.bottom_face
L1: id=leg_b | parent=table | create_method=primitive | after=leg_a
# ----------  END_GRAPH  ----------
"""


class PlanOrderTests(unittest.TestCase):
    def setUp(self):
        self.actions = plan_actions(parse_document(TABLE))

    def test_children_created_before_the_parent_assembles(self):
        ops = [(action.op, action.node_id) for action in self.actions]
        self.assertEqual(ops[0], ("create", "top"))
        self.assertEqual(ops[-1], ("assemble", "table"))

    def test_assembly_order_groups_are_respected(self):
        ids = [action.node_id for action in self.actions if action.op == "create"]
        self.assertEqual(ids, ["top", "leg_a", "leg_b"])

    def test_assemble_lists_children_in_group_order(self):
        assemble = self.actions[-1]
        self.assertEqual(assemble.operands, ("top", "leg_a", "leg_b"))
        self.assertEqual(assemble.method, "composite")

    def test_placement_action_follows_creation(self):
        place = [action for action in self.actions if action.op == "place"]
        self.assertEqual(len(place), 1)
        self.assertEqual(place[0].node_id, "leg_a")
        self.assertIn("align=Align(Z) leg_a.top_face to top.bottom_face", place[0].detail)

    def test_plan_is_valid(self):
        self.assertEqual(validate_plan(self.actions), ())

    def test_plan_is_deterministic(self):
        self.assertEqual(plan_actions(parse_document(TABLE)), self.actions)


class BooleanAndBevelTests(unittest.TestCase):
    def test_boolean_operands_follow_target_then_tool(self):
        graph = KnowledgeGraph((
            GraphNode("cut", 0, create_method="boolean_subtract", tool_id="drill",
                      target_id="block"),
            GraphNode("block", 1, parent="cut", create_method="primitive"),
            GraphNode("drill", 1, parent="cut", create_method="primitive",
                      orientation="normal:block"),
        ))
        actions = plan_actions(graph)
        boolean = actions[-1]
        self.assertEqual(boolean.op, "boolean")
        self.assertEqual(boolean.operands, ("block", "drill"))
        self.assertEqual(validate_plan(actions), ())

    def test_bevel_targets_its_tool_id(self):
        graph = KnowledgeGraph((
            GraphNode("root", 0, create_method="composite"),
            GraphNode("body", 1, parent="root", create_method="primitive"),
            GraphNode("edge", 1, parent="root", create_method="bevel", tool_id="body",
                      constraint="radius 0.002 segments 3"),
        ))
        actions = plan_actions(graph)
        bevel = [action for action in actions if action.op == "bevel"][0]
        self.assertEqual(bevel.operands, ("body",))
        self.assertEqual(bevel.detail, "radius 0.002 segments 3")
        self.assertEqual(validate_plan(actions), ())


class PatternTests(unittest.TestCase):
    def test_template_node_expands_to_instances(self):
        graph = KnowledgeGraph((
            GraphNode("root", 0, create_method="composite"),
            GraphNode("leg", 1, parent="root", create_method="primitive",
                      pattern="polar(count:3, radius:0.4, start_angle:0, angle_step:120)"),
        ))
        actions = plan_actions(graph)
        created = [action.node_id for action in actions if action.op == "create"]
        self.assertEqual(created, ["leg_0", "leg_1", "leg_2"])
        assemble = [action for action in actions if action.node_id == "leg"][0]
        self.assertEqual(assemble.op, "assemble")
        self.assertEqual(assemble.operands, ("leg_0", "leg_1", "leg_2"))
        self.assertEqual(validate_plan(actions), ())

    def test_grid_pattern_instances(self):
        graph = KnowledgeGraph((
            GraphNode("tile", 0, create_method="primitive",
                      pattern="grid(rows:2, cols:2, x_spacing:0.1, y_spacing:0.1)"),
        ))
        created = [action.node_id for action in plan_actions(graph) if action.op == "create"]
        self.assertEqual(created, ["tile_0_0", "tile_0_1", "tile_1_0", "tile_1_1"])


class ValidatePlanTests(unittest.TestCase):
    def test_undefined_operand_is_reported(self):
        actions = (Action("boolean", "cut", "boolean_subtract", ("block", "drill")),)
        errors = validate_plan(actions)
        self.assertIn("step 1: boolean 'cut' uses undefined 'block'", errors)

    def test_double_definition_is_reported(self):
        actions = (
            Action("create", "a", "primitive"),
            Action("create", "a", "primitive"),
        )
        self.assertIn("step 2: 'a' was already defined at step 1", validate_plan(actions))

    def test_place_before_create_is_reported(self):
        actions = (Action("place", "a", detail="pos=(0,0,0)"),)
        self.assertIn("step 1: place on undefined 'a'", validate_plan(actions))

    def test_reordered_plan_becomes_invalid(self):
        actions = plan_actions(parse_document(TABLE))
        self.assertNotEqual(validate_plan(tuple(reversed(actions))), ())


class RenderTests(unittest.TestCase):
    def test_render_is_numbered(self):
        text = render_plan(plan_actions(parse_document(TABLE)))
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("1. create(top"))
        self.assertIn("assemble(table", lines[-1])

    def test_action_histogram(self):
        histogram = action_histogram(plan_actions(parse_document(TABLE)))
        self.assertEqual(histogram, {"create": 3, "place": 1, "assemble": 1})


if __name__ == "__main__":
    unittest.main()
