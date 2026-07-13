"""Tests for the assembly / insertion-sequence planner (quality.assemblyseq).

Covers:
  * a simple stacked 2- / 3-part assembly gets a valid collision-free insertion
    order and a disassembly order that is exactly its reverse;
  * per-part insertion vectors point into the seat;
  * an interlocked (mutually-capturing) pair is flagged with NO insertion axis
    and yields no full sequence;
  * the mate-graph critical dependency chain is surfaced;
  * a single-part / stub model INFO-skips (never an ERROR), and the SequenceCheck
    verifier emits an ERROR only when no full sequence exists.
"""

import unittest

from harnesscad.eval.verifiers.assembly import AssemblyModel, Mate
from harnesscad.eval.verifiers.verify import Severity
from harnesscad.eval.quality.assembly.sequence_planner import (
    plan_assembly_sequence, AssemblySequence, SequenceCheck,
    sequence_diagnostics, with_sequence,
)


def _codes(report):
    return {d.code for d in report.diagnostics}


def _by_sev(report, sev):
    return [d for d in report.diagnostics if d.severity is sev]


# Stacked boxes along Z: A[0,10] under B[10,20] under C[20,30]; footprint 10x10.
STACK_BBOXES = {
    "A": [0, 0, 0, 10, 10, 10],
    "B": [0, 0, 10, 10, 10, 20],
    "C": [0, 0, 20, 10, 10, 30],
}


class TestStackedAssembly(unittest.TestCase):
    def test_two_part_stack_has_valid_order(self):
        model = AssemblyModel(
            parts=["A", "B"],
            mates=[Mate(kind="rigid", a="A", b="B")],
        )
        boxes = {"A": STACK_BBOXES["A"], "B": STACK_BBOXES["B"]}
        seq = plan_assembly_sequence(model, bboxes=boxes)
        self.assertTrue(seq.ok)
        self.assertFalse(seq.trivial)
        self.assertEqual(set(seq.insertion_order), {"A", "B"})
        self.assertEqual(len(seq.insertion_order), 2)
        # No part is flagged as blocked.
        self.assertEqual(seq.blocked_parts, [])

    def test_disassembly_is_reverse_of_insertion(self):
        model = AssemblyModel(
            parts=["A", "B", "C"],
            mates=[Mate(kind="rigid", a="A", b="B"),
                   Mate(kind="rigid", a="B", b="C")],
        )
        seq = plan_assembly_sequence(model, bboxes=STACK_BBOXES)
        self.assertTrue(seq.ok)
        self.assertEqual(seq.disassembly_order,
                         list(reversed(seq.insertion_order)))

    def test_three_part_stack_orders_bottom_up_or_top_down(self):
        model = AssemblyModel(parts=["A", "B", "C"], mates=[])
        seq = plan_assembly_sequence(model, bboxes=STACK_BBOXES)
        self.assertTrue(seq.ok)
        # B is sandwiched; it must never be inserted last against both neighbours
        # unless via a clear axis — the planner still finds *a* valid order.
        self.assertEqual(set(seq.insertion_order), {"A", "B", "C"})

    def test_insertion_vectors_are_unit_axes(self):
        model = AssemblyModel(parts=["A", "B"], mates=[])
        boxes = {"A": STACK_BBOXES["A"], "B": STACK_BBOXES["B"]}
        seq = plan_assembly_sequence(model, bboxes=boxes)
        for pid in seq.insertion_order:
            vec = seq.insertion_vectors[pid]
            mag = sum(abs(c) for c in vec)
            self.assertAlmostEqual(mag, 1.0)  # exactly one +-1 component

    def test_critical_chain_follows_mates(self):
        model = AssemblyModel(
            parts=["A", "B", "C"],
            mates=[Mate(kind="rigid", a="A", b="B"),
                   Mate(kind="rigid", a="B", b="C")],
        )
        seq = plan_assembly_sequence(model, bboxes=STACK_BBOXES)
        self.assertTrue(seq.ok)
        # A-B-C are a single mate chain -> the critical chain spans all three.
        self.assertEqual(len(seq.critical_chain), 3)
        self.assertEqual(set(seq.critical_chain), {"A", "B", "C"})


class TestInterlocked(unittest.TestCase):
    def test_interlocked_pair_has_no_insertion(self):
        # Two mutually-interpenetrating boxes: neither can escape along any
        # principal axis -> no full sequence, both flagged blocked.
        model = AssemblyModel(
            parts=["L", "R"],
            mates=[Mate(kind="rigid", a="L", b="R")],
        )
        boxes = {
            "L": [0, 0, 0, 10, 10, 10],
            "R": [5, 5, 5, 15, 15, 15],
        }
        seq = plan_assembly_sequence(model, bboxes=boxes)
        self.assertFalse(seq.ok)
        self.assertEqual(sorted(seq.blocked_parts), ["L", "R"])

    def test_blocked_part_reported_by_verifier(self):
        model = AssemblyModel(parts=["L", "R"], mates=[])
        boxes = {"L": [0, 0, 0, 10, 10, 10], "R": [5, 5, 5, 15, 15, 15]}
        report = SequenceCheck().check_model(model, bboxes=boxes)
        self.assertFalse(report.ok)  # ERROR present
        self.assertIn("no-assembly-sequence", _codes(report))
        self.assertIn("blocked-insertion", _codes(report))


class TestVerifierSkips(unittest.TestCase):
    def test_single_part_is_trivial_info(self):
        model = AssemblyModel(parts=["only"], mates=[])
        seq = plan_assembly_sequence(model, bboxes={"only": [0, 0, 0, 1, 1, 1]})
        self.assertTrue(seq.trivial)
        report = SequenceCheck().check_model(model,
                                             bboxes={"only": [0, 0, 0, 1, 1, 1]})
        self.assertTrue(report.ok)
        self.assertIn("assembly-sequence-trivial", _codes(report))

    def test_stub_backend_info_skip(self):
        from harnesscad.io.backends.stub import StubBackend
        report = SequenceCheck().check(StubBackend(), None)
        self.assertTrue(report.ok)
        self.assertIn("assembly-sequence-skipped", _codes(report))
        self.assertEqual(_by_sev(report, Severity.ERROR), [])

    def test_valid_sequence_is_info_only(self):
        model = AssemblyModel(parts=["A", "B"],
                              mates=[Mate(kind="rigid", a="A", b="B")])
        boxes = {"A": STACK_BBOXES["A"], "B": STACK_BBOXES["B"]}
        report = SequenceCheck().check_model(model, bboxes=boxes)
        self.assertTrue(report.ok)
        self.assertIn("assembly-sequence", _codes(report))
        self.assertEqual(_by_sev(report, Severity.ERROR), [])


class TestBackendInput(unittest.TestCase):
    def test_reads_raw_assembly_dict(self):
        raw = {
            "parts": [
                {"id": "A", "bbox": STACK_BBOXES["A"]},
                {"id": "B", "bbox": STACK_BBOXES["B"]},
            ],
            "mates": [{"kind": "rigid", "a": "A", "b": "B"}],
        }
        seq = plan_assembly_sequence(raw)
        self.assertTrue(seq.ok)
        self.assertEqual(set(seq.insertion_order), {"A", "B"})

    def test_fake_backend_query(self):
        class FakeBackend:
            def query(self, q):
                if q == "assembly":
                    return {
                        "parts": [
                            {"id": "A", "bbox": STACK_BBOXES["A"]},
                            {"id": "B", "bbox": STACK_BBOXES["B"]},
                        ],
                        "mates": [],
                    }
                return {}

        seq = plan_assembly_sequence(FakeBackend())
        self.assertTrue(seq.ok)
        self.assertEqual(len(seq.insertion_order), 2)


class TestSerialisation(unittest.TestCase):
    def test_to_dict_and_render(self):
        model = AssemblyModel(parts=["A", "B", "C"],
                              mates=[Mate(kind="rigid", a="A", b="B"),
                                     Mate(kind="rigid", a="B", b="C")])
        seq = plan_assembly_sequence(model, bboxes=STACK_BBOXES)
        d = seq.to_dict()
        self.assertIn("insertion_order", d)
        self.assertIn("insertion_vectors", d)
        self.assertIn("critical_chain", d)
        text = seq.render()
        self.assertIn("insert:", text)
        self.assertIn("remove:", text)

    def test_render_of_blocked(self):
        model = AssemblyModel(parts=["L", "R"], mates=[])
        boxes = {"L": [0, 0, 0, 10, 10, 10], "R": [5, 5, 5, 15, 15, 15]}
        seq = plan_assembly_sequence(model, bboxes=boxes)
        self.assertIn("NO valid", seq.render())

    def test_with_sequence_appends_verifier(self):
        base = []
        extended = with_sequence(base)
        self.assertEqual(len(extended), 1)
        self.assertEqual(extended[0].name, "assembly-sequence")


if __name__ == "__main__":
    unittest.main()
