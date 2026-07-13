import unittest

from harnesscad.agents.agent.cad_plan import parse_envelope
from harnesscad.eval.bench.cad_geometry_protocol import (
    GeometryProtocol, evaluate_geometry, normalize, squared_chamfer,
)
from harnesscad.eval.bench.edit_alignment import (
    aggregate_edits, directional_cosine, occupancy_jsd, slice_by_se,
)
from harnesscad.eval.bench.edit_splits import balance_se, split_leakage
from harnesscad.data.dataengine.cot_records import CoTRecord, cot_leakage
from harnesscad.data.dataengine.edit_caption import caption_edit
from harnesscad.data.dataengine.edit_filters import filter_edit
from harnesscad.data.dataengine.selective_edits import (
    EditCandidateRecord, create_selection,
)
from harnesscad.data.datagen.edit_triplets import enumerate_pairs
from harnesscad.data.datagen.geometry_triplets import quality_tier, select_triplet
from harnesscad.domain.editing.iterative_session import IterativeEditSession
from harnesscad.domain.editing.locate_infill import MASK, context_preserved, infill, locate_mask
from harnesscad.eval.quality.cad_reward import format_reward, geometric_reward, score_candidate
from harnesscad.eval.quality.sampling_guard import sampling_diagnostics


class CADRewardTests(unittest.TestCase):
    def envelope(self):
        return """<think>
description: two blocks
coordinates: global XY
sketch: rectangles
extrusion: extrude then union
implementation: construct r
</think>
```python
r = build()
```"""

    def test_plan_envelope_and_format(self):
        envelope = parse_envelope(self.envelope())
        self.assertEqual(envelope.plan.sketch, "rectangles")
        self.assertEqual(format_reward(self.envelope()), 1)
        with self.assertRaises(ValueError):
            parse_envelope("```python\nx=1\n```")

    def test_piecewise_execution_first_reward(self):
        self.assertEqual(geometric_reward(0), 1)
        self.assertAlmostEqual(geometric_reward(.5), .01)
        self.assertEqual(geometric_reward(.6), 0)
        result = score_candidate(
            self.envelope(), execute=lambda _: "shape",
            sample=lambda _: ((0, 0),), target_points=((0, 0),),
            distance=lambda a, b: 0,
        )
        self.assertEqual((result.geometric, result.format, result.total), (1, 1, 2))
        failed = score_candidate(
            self.envelope(), execute=lambda _: (_ for _ in ()).throw(RuntimeError()),
            sample=lambda x: x, target_points=(), distance=lambda a, b: 0)
        self.assertFalse(failed.executable)
        self.assertEqual(failed.geometric, 0)

    def test_geometry_protocol_and_triplet_selection(self):
        protocol = GeometryProtocol(sample_count=4)
        points = normalize(((1, 0), (3, 0)), protocol)
        self.assertEqual(points, ((-1.0, 0.0), (1.0, 0.0)))
        self.assertEqual(squared_chamfer(((0, 0),), ((1, 0),)), 2)
        stats = evaluate_geometry((.2, None, .4))
        self.assertAlmostEqual(stats["invalidity_ratio"], 1/3)
        self.assertAlmostEqual(stats["median_cd"], .3)
        triplet = select_triplet(
            "box", ("bad", "near", "best"), "target",
            execute=lambda c: (_ for _ in ()).throw(ValueError()) if c == "bad" else c,
            distance=lambda shape, target: {"near": 1e-3, "best": 1e-5}[shape],
        )
        self.assertEqual(triplet.candidate_index, 2)
        self.assertEqual(triplet.tier, "high")
        self.assertEqual(quality_tier(.01), "hard")

    def test_sampling_and_cot_lineage_guards(self):
        issues = sampling_diagnostics(
            ((0, 0),), thin_regions=(lambda p: p[0] > 5,),
            interior_regions=(lambda p: p[1] == 0,),
            coarse_cd=.1, fine_cd=.2, relative_tolerance=.2)
        self.assertEqual(issues, ("thin-region-uncovered", "multires-disagreement"))
        def record(i, split):
            return CoTRecord(i, "p", (("description", "x"),), "code",
                             "geometry", 0, True, True, split, "source")
        self.assertTrue(cot_leakage((record("1", "train"), record("2", "test"))))


class LocateInfillTests(unittest.TestCase):
    def test_lcs_masks_insert_delete_replace_and_infill(self):
        self.assertEqual(locate_mask("abc", "axbc"), ("a", MASK, "b", "c"))
        self.assertEqual(locate_mask("abc", "ac"), ("a", MASK, "c"))
        masked = locate_mask(("box", "10", "end"), ("box", "20", "end"))
        self.assertEqual(infill(masked, (("20",),)), ("box", "20", "end"))
        self.assertTrue(context_preserved(masked, ("box", "20", "end")))
        self.assertFalse(context_preserved(masked, ("sphere", "20", "end")))

    def test_all_direction_pair_synthesis_and_caption(self):
        pairs = enumerate_pairs("base", ("v1", "v2"))
        self.assertEqual(len(pairs), 6)
        self.assertEqual({p.direction for p in pairs},
                         {"base-forward", "base-reverse", "cross"})
        caption = caption_edit(
            "small", "large",
            visual_describer=lambda x: f"visual {x}",
            sequence_describer=lambda x: f"sequence {x}",
            differ=lambda a, b: f"{a} -> {b}",
            compress=lambda change: "increase size",
        )
        self.assertEqual(caption.instruction, "increase size")

    def test_filters_selection_and_alignment_metrics(self):
        self.assertEqual(
            filter_edit(instruction="no transformation is needed", mask_count=6,
                        changed_spans=4, se_count=4),
            ("too-many-masks", "too-many-changes", "sequence-too-complex", "no-op"))
        candidates = (
            EditCandidateRecord("b", (), "r2", True, .8),
            EditCandidateRecord("a", (), "r1", True, .7),
        )
        selected = create_selection("source", "edit", candidates, "b", ("worker",))
        self.assertEqual([x.id for x in selected.candidates], ["a", "b"])
        self.assertEqual(occupancy_jsd({"a": 1}, {"a": 1}), 0)
        cosine = directional_cosine(
            "old", "new", "neutral", "edit",
            image_embed=lambda x: {"old": (0, 0), "new": (1, 0)}[x],
            text_embed=lambda x: {"neutral": (0, 0), "edit": (2, 0)}[x])
        self.assertEqual(cosine, 1)
        rows = ({"valid": True, "cd": .2, "se_count": 1},
                {"valid": False, "cd": None, "se_count": 1})
        self.assertEqual(aggregate_edits(rows)["valid_ratio"], .5)
        self.assertEqual(slice_by_se(rows)[1]["mean_cd"], .2)

    def test_split_balance_and_iterative_rollback(self):
        records = [{"id": f"{n}-{i}", "se_count": n, "source_id": f"s{n}-{i}",
                    "lineage": f"l{n}-{i}", "split": "train"}
                   for n in range(1, 6) for i in range(2)]
        self.assertEqual(len(balance_se(records, 1)), 5)
        leaked = records + [{**records[0], "id": "other", "split": "test"}]
        self.assertTrue(split_leakage(leaked))
        session = IterativeEditSession(1)
        session.apply("double", lambda value, _: value * 2)
        session.apply("add", lambda value, _: value + 3)
        self.assertEqual(session.current, 5)
        self.assertEqual(session.rollback(1), 2)
        self.assertEqual(len(session.revisions), 1)


if __name__ == "__main__":
    unittest.main()
