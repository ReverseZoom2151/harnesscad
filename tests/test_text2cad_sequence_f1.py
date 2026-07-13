import unittest

from harnesscad.eval.bench.text2cad_sequence_f1 import (
    SequenceF1Error,
    aggregate_f1,
    evaluate_sequence,
    hungarian_assignment,
    invalidity_ratio,
    loop_primitive_counts,
)
from harnesscad.domain.reconstruction.deepcad_command_spec import command


def _circle_seq():
    return [
        command("SOL"), command("Circle", x=0.5, y=0.5, r=0.4),
        command("Ext", theta=0, phi=0, gamma=0, px=0, py=0, pz=0,
                s=1, e1=0.2, e2=0, b=0, u=0),
    ]


def _rect_seq():
    return [
        command("SOL"),
        command("Line", x=0, y=0), command("Line", x=1, y=0),
        command("Line", x=1, y=1), command("Line", x=0, y=1),
        command("Ext", theta=0, phi=0, gamma=0, px=0, py=0, pz=0,
                s=1, e1=0.3, e2=0, b=0, u=0),
    ]


class LoopExtractionTests(unittest.TestCase):
    def test_single_circle_loop(self):
        loops = loop_primitive_counts(_circle_seq())
        self.assertEqual(len(loops), 1)
        self.assertEqual(loops[0]["Circle"], 1)

    def test_rectangle_loop(self):
        loops = loop_primitive_counts(_rect_seq())
        self.assertEqual(len(loops), 1)
        self.assertEqual(loops[0]["Line"], 4)

    def test_two_loops(self):
        cmds = [
            command("SOL"), command("Circle", x=0.5, y=0.5, r=0.4),
            command("SOL"), command("Circle", x=0.5, y=0.5, r=0.2),
            command("Ext", theta=0, phi=0, gamma=0, px=0, py=0, pz=0,
                    s=1, e1=0.2, e2=0, b=0, u=0),
        ]
        self.assertEqual(len(loop_primitive_counts(cmds)), 2)


class HungarianTests(unittest.TestCase):
    def test_identity(self):
        cost = [[0, 1], [1, 0]]
        self.assertEqual(hungarian_assignment(cost), [0, 1])

    def test_swap(self):
        cost = [[5, 0], [0, 5]]
        self.assertEqual(hungarian_assignment(cost), [1, 0])

    def test_empty(self):
        self.assertEqual(hungarian_assignment([]), [])

    def test_optimal_three(self):
        # Classic assignment; optimum picks the zero-diagonal-ish assignment.
        cost = [[9, 2, 7], [6, 4, 3], [5, 8, 1]]
        assign = hungarian_assignment(cost)
        total = sum(cost[i][assign[i]] for i in range(3))
        self.assertEqual(total, 2 + 6 + 1)  # cols {1,0,2}
        self.assertEqual(sorted(assign), [0, 1, 2])

    def test_non_square_raises(self):
        with self.assertRaises(SequenceF1Error):
            hungarian_assignment([[1, 2, 3], [4, 5, 6]])


class EvaluateSequenceTests(unittest.TestCase):
    def test_perfect_match(self):
        e = evaluate_sequence(_rect_seq(), _rect_seq())
        self.assertEqual(e.line.f1, 1.0)
        self.assertEqual(e.extrusion.f1, 1.0)
        self.assertEqual(e.line.fp, 0)
        self.assertEqual(e.line.fn, 0)

    def test_loop_order_invariant(self):
        gt = [
            command("SOL"), command("Circle", x=0.5, y=0.5, r=0.4),
            command("SOL"),
            command("Line", x=0, y=0), command("Line", x=1, y=0),
            command("Line", x=1, y=1), command("Line", x=0, y=1),
            command("Ext", theta=0, phi=0, gamma=0, px=0, py=0, pz=0,
                    s=1, e1=0.2, e2=0, b=0, u=0),
        ]
        # Predicted with the two loops in swapped order.
        pred = [
            command("SOL"),
            command("Line", x=0, y=0), command("Line", x=1, y=0),
            command("Line", x=1, y=1), command("Line", x=0, y=1),
            command("SOL"), command("Circle", x=0.5, y=0.5, r=0.4),
            command("Ext", theta=0, phi=0, gamma=0, px=0, py=0, pz=0,
                    s=1, e1=0.2, e2=0, b=0, u=0),
        ]
        e = evaluate_sequence(pred, gt)
        self.assertEqual(e.line.f1, 1.0)
        self.assertEqual(e.circle.f1, 1.0)

    def test_missing_extrusion_counts_fn(self):
        pred = [command("SOL"), command("Circle", x=0.5, y=0.5, r=0.4)]
        e = evaluate_sequence(pred, _circle_seq())
        self.assertEqual(e.extrusion.fn, 1)
        self.assertEqual(e.extrusion.f1, 0.0)
        self.assertEqual(e.circle.f1, 1.0)

    def test_extra_lines_are_false_positives(self):
        pred = _rect_seq()  # 4 lines
        gt = [
            command("SOL"), command("Line", x=0, y=0), command("Line", x=1, y=0),
            command("Ext", theta=0, phi=0, gamma=0, px=0, py=0, pz=0,
                    s=1, e1=0.3, e2=0, b=0, u=0),
        ]  # 2 lines
        e = evaluate_sequence(pred, gt)
        self.assertEqual(e.line.tp, 2)
        self.assertEqual(e.line.fp, 2)


class AggregateAndIRTests(unittest.TestCase):
    def test_aggregate_perfect(self):
        evals = [evaluate_sequence(_rect_seq(), _rect_seq()) for _ in range(3)]
        f1 = aggregate_f1(evals)
        self.assertEqual(f1["line"], 1.0)
        self.assertEqual(f1["extrusion"], 1.0)
        self.assertEqual(f1["arc"], 0.0)  # no arcs anywhere

    def test_invalidity_ratio(self):
        self.assertEqual(invalidity_ratio([True, True, False, True]), 0.25)
        self.assertEqual(invalidity_ratio([True] * 5), 0.0)

    def test_invalidity_ratio_empty_raises(self):
        with self.assertRaises(SequenceF1Error):
            invalidity_ratio([])


if __name__ == "__main__":
    unittest.main()
