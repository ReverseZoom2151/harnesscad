import unittest

from harnesscad.core.state.constraint_hierarchy import (
    ConstraintScope,
    LocalFrame,
    PrunedBranch,
    local_editability,
    solve_hierarchy,
    solve_pruned_branches,
)


class ConstraintHierarchyTests(unittest.TestCase):
    def test_children_solve_before_parent_with_qualified_values(self):
        root = ConstraintScope("assembly", constraints=("align children",))
        root.add_child(ConstraintScope(
            "left", LocalFrame(origin=(-10, 0, 0)), {"width": 4}
        ))
        root.add_child(ConstraintScope(
            "right", LocalFrame(origin=(10, 0, 0)), {"width": 6}
        ))
        order = []

        def solver(scope, values):
            order.append(scope.name)
            if scope.name == "assembly":
                return {"span": values["left.width"] + values["right.width"]}
            return scope.parameters

        result = solve_hierarchy(root, solver)
        self.assertEqual(order, ["left", "right", "assembly"])
        self.assertEqual(result.values_for("assembly")["span"], 10)

    def test_local_editability_penalizes_sibling_changes(self):
        before = {"left": {"w": 4}, "right": {"w": 6, "h": 3}}
        local = local_editability(
            before, {"left": {"w": 5}, "right": {"w": 6, "h": 3}}, "left"
        )
        self.assertEqual(local.changed_in_scope, 1)
        self.assertEqual(local.locality, 1.0)
        leaked = local_editability(
            before, {"left": {"w": 5}, "right": {"w": 7, "h": 3}}, "left"
        )
        self.assertLess(leaked.locality, 1.0)

    def test_pruned_solver_revalidates_original_expression(self):
        branches = [
            PrunedBranch("min-is-x", ("x=target",)),
            PrunedBranch("min-is-y", ("y=target",)),
        ]

        def solve(constraints, seed):
            if constraints == ("x=target",):
                return {"x": 5, "y": 2}  # min is 2: invalid original
            return {"x": 7, "y": 5}

        result = solve_pruned_branches(
            branches, solve, lambda values: min(values["x"], values["y"]) == 5
        )
        self.assertEqual(result.branch, "min-is-y")
        self.assertEqual(result.attempts, 2)

    def test_pruned_solver_fails_when_no_original_solution(self):
        with self.assertRaises(ValueError):
            solve_pruned_branches(
                [PrunedBranch("a", ("x=1",))],
                lambda constraints, seed: {"x": 1},
                lambda values: False,
            )


if __name__ == "__main__":
    unittest.main()
