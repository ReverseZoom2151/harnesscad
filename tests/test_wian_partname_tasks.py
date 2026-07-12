import unittest

from bench.wian_partname_tasks import (
    FEATURE_KEYS,
    SPLIT_NAMES,
    build_corpus,
    build_tasks,
    corpus_line,
    document_name_task,
    feature_subset_splits,
    missing_part_task,
    parts_only_line,
    rank_of_target,
    retrieval_metrics,
    split_corpus,
    stratified_split,
)
from library.wian_name_normalizer import CleanDocument, clean_corpus

DOC = CleanDocument(
    document_id="d1",
    document_name="Skateboard truck",
    document_description="a truck",
    body_names=("baseplate", "hanger", "kingpin"),
    feature_names=("chamfer top edge",),
)

DOC_NO_DESC = CleanDocument(
    document_id="d2",
    document_name="Drone frame",
    document_description="",
    body_names=("arm", "hub"),
    feature_names=(),
)


class TestCorpus(unittest.TestCase):
    def test_line_with_description(self):
        self.assertEqual(
            corpus_line(DOC),
            'An assembly with name "Skateboard truck" and description "a truck",'
            " contains the following parts: baseplate, hanger, kingpin, chamfer top edge.",
        )

    def test_line_without_description(self):
        self.assertEqual(
            corpus_line(DOC_NO_DESC),
            'An assembly with name "Drone frame" contains the following parts: arm, hub.',
        )

    def test_lowercase(self):
        self.assertEqual(corpus_line(DOC_NO_DESC, lower=True), corpus_line(DOC_NO_DESC).lower())

    def test_parts_only(self):
        self.assertEqual(parts_only_line(DOC), "baseplate, hanger, kingpin.")

    def test_parts_only_dedup(self):
        doc = CleanDocument("d", "n", "", ("a", "a", "b"))
        self.assertEqual(parts_only_line(doc, remove_duplicates=True), "a, b.")

    def test_build_corpus(self):
        lines = build_corpus([DOC, DOC_NO_DESC])
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("An assembly"))
        self.assertEqual(build_corpus([DOC], parts_only=True), ["baseplate, hanger, kingpin."])


class TestTasks(unittest.TestCase):
    def test_missing_part_holds_out_exactly_one(self):
        task = missing_part_task(DOC, seed=3)
        self.assertEqual(task.kind, "missing_part")
        self.assertEqual(len(task.inputs), 2)
        self.assertNotIn(task.target, task.inputs)
        self.assertEqual(set(task.inputs) | {task.target}, set(DOC.body_names))

    def test_missing_part_deterministic(self):
        self.assertEqual(missing_part_task(DOC, seed=3), missing_part_task(DOC, seed=3))

    def test_missing_part_requires_two_parts(self):
        with self.assertRaises(ValueError):
            missing_part_task(CleanDocument("d", "n", "", ("only",)))

    def test_document_name_task(self):
        task = document_name_task(DOC)
        self.assertEqual(task.target, "Skateboard truck")
        self.assertEqual(task.inputs, DOC.body_names)

    def test_document_name_requires_name(self):
        with self.assertRaises(ValueError):
            document_name_task(CleanDocument("d", "", "", ("a",)))

    def test_build_tasks_skips_infeasible(self):
        docs = [DOC, CleanDocument("d3", "x", "", ("solo",))]
        self.assertEqual(len(build_tasks(docs, kind="missing_part")), 1)
        self.assertEqual(len(build_tasks(docs, kind="document_name")), 2)

    def test_build_tasks_unknown_kind(self):
        with self.assertRaises(ValueError):
            build_tasks([DOC], kind="nope")


class TestSplits(unittest.TestCase):
    def setUp(self):
        self.raw = {
            f"doc{i:03d}": {
                "body_names": ["bracket", "wheel hub"] if i % 2 == 0 else ["bracket"],
                "feature_names": ["chamfer edge"] if i % 3 == 0 else [],
                "document_name": f"assembly {i}",
                "document_description": "desc" if i % 5 == 0 else "",
            }
            for i in range(100)
        }
        self.docs = clean_corpus(self.raw)

    def test_partition_is_disjoint_and_complete(self):
        split = stratified_split(self.docs, seed=1)
        ids = [i for name in SPLIT_NAMES for i in split[name]]
        self.assertEqual(len(ids), 100)
        self.assertEqual(len(set(ids)), 100)

    def test_fractions_respected(self):
        split = stratified_split(self.docs, validation_fraction=0.2, test_fraction=0.2, seed=1)
        self.assertGreaterEqual(len(split["train"]), 55)
        self.assertGreater(len(split["validation"]), 10)
        self.assertGreater(len(split["test"]), 10)

    def test_deterministic(self):
        self.assertEqual(stratified_split(self.docs, seed=4), stratified_split(self.docs, seed=4))

    def test_bad_fractions(self):
        with self.assertRaises(ValueError):
            stratified_split(self.docs, validation_fraction=0.6, test_fraction=0.6)

    def test_feature_subsets(self):
        split = stratified_split(self.docs, seed=2)
        subsets = feature_subset_splits(self.docs, split)
        self.assertEqual(sorted(subsets), sorted(FEATURE_KEYS))
        two_or_more = subsets["two_or_more_partnames"]
        for name in SPLIT_NAMES:
            for doc_id in two_or_more[name]:
                self.assertGreaterEqual(len(self.docs[doc_id].body_names), 2)
        total = sum(len(two_or_more[n]) for n in SPLIT_NAMES)
        self.assertEqual(total, 50)

    def test_split_corpus(self):
        cleaned, split = split_corpus(self.raw, seed=8)
        self.assertEqual(len(cleaned), 100)
        self.assertEqual(sum(len(split[n]) for n in SPLIT_NAMES), 100)


class TestRetrievalMetrics(unittest.TestCase):
    def test_rank_of_target(self):
        self.assertEqual(rank_of_target(["a", "b", "c"], "b"), 2)
        self.assertEqual(rank_of_target(["a"], "z"), 0)

    def test_metrics(self):
        ranked = [["a", "b", "c"], ["x", "y", "z"], ["p", "q", "r"]]
        targets = ["a", "z", "missing"]
        m = retrieval_metrics(ranked, targets, ks=(1, 3))
        self.assertAlmostEqual(m["acc@1"], 1 / 3)
        self.assertAlmostEqual(m["acc@3"], 2 / 3)
        self.assertAlmostEqual(m["mrr"], (1.0 + 1 / 3 + 0.0) / 3)

    def test_metrics_empty(self):
        m = retrieval_metrics([], [], ks=(1,))
        self.assertEqual(m["n"], 0.0)
        self.assertEqual(m["acc@1"], 0.0)

    def test_metrics_mismatch(self):
        with self.assertRaises(ValueError):
            retrieval_metrics([["a"]], [])


if __name__ == "__main__":
    unittest.main()
