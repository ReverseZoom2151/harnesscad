"""The real-photo held-out loader's manifest-only + closed-axis invariants.

Kernel-free and instant: these assert the committed FACTS (400 rows, the five
closed axis vocabularies, the 50-object id map) are intact and internally
consistent, and that the raw pixels degrade cleanly when resources/ is absent.
"""

from __future__ import annotations

import unittest

from harnesscad.eval.corpus.fixtures import loader
from harnesscad.eval.corpus.fixtures import real_photo_heldout as rph


class TestManifest(unittest.TestCase):

    def test_manifest_only_nothing_vendored(self):
        m = rph.manifest()
        self.assertEqual(m.license, "NOLICENSE-DEEPCAD-DERIVED")
        self.assertEqual(m.verify_vendored(), [])
        for e in m.entries:
            self.assertIsNone(e.vendored, e.name)
            self.assertTrue(e.resource, e.name)
            self.assertEqual(len(e.sha256), 64, e.name)

    def test_role_census(self):
        m = rph.manifest()
        self.assertEqual(len(m.by_role("photo")), rph.EXPECTED_PHOTOS)
        self.assertEqual(len(m.by_role("metadata_source")), 1)
        self.assertEqual(len(m.by_role("idmap_source")), 1)

    def test_reachable_through_hub(self):
        self.assertIs(loader("real_photo_heldout"), rph)


class TestFacts(unittest.TestCase):

    def test_id_map_covers_all_fifty_objects(self):
        imap = rph.id_map()
        self.assertEqual(len(imap), rph.EXPECTED_OBJECTS)
        self.assertEqual(set(imap), {str(i) for i in range(1, 51)})
        for did in imap.values():
            self.assertTrue(did)

    def test_four_hundred_records_all_resolve_to_a_deepcad_id(self):
        recs = rph.photo_records()
        self.assertEqual(len(recs), rph.EXPECTED_PHOTOS)
        imap = rph.id_map()
        for r in recs:
            self.assertEqual(r.deepcad_id, imap[str(r.object_id)], r.file)
            self.assertEqual(rph.resolve_deepcad_id(r), r.deepcad_id)

    def test_fifty_objects_each_with_eight_photos(self):
        counts = {}
        for r in rph.photo_records():
            counts[r.object_id] = counts.get(r.object_id, 0) + 1
        self.assertEqual(len(counts), rph.EXPECTED_OBJECTS)
        self.assertEqual(set(counts.values()), {rph.EXPECTED_PHOTOS_PER_OBJECT})

    def test_every_axis_value_is_in_its_closed_vocabulary(self):
        for r in rph.photo_records():
            self.assertIn(r.color, rph.AXES["color"], r.file)
            self.assertIn(r.orientation, rph.AXES["orientation"], r.file)
            self.assertIn(r.proximity, rph.AXES["proximity"], r.file)
            self.assertIn(r.background, rph.AXES["background"], r.file)
            self.assertIn(r.lighting, rph.AXES["lighting"], r.file)

    def test_embedded_axes_match_the_manifest(self):
        raw_axes = rph.axes()
        self.assertEqual(set(raw_axes), set(rph.AXES))
        for name, vocab in rph.AXES.items():
            self.assertEqual(raw_axes[name], tuple(vocab))

    def test_slicing_partitions_a_multivalue_axis(self):
        p1 = rph.records_by_axis("orientation", "Position1")
        p2 = rph.records_by_axis("orientation", "Position2")
        self.assertEqual(len(p1) + len(p2), rph.EXPECTED_PHOTOS)

    def test_slicing_rejects_unknown_axis_and_out_of_vocab_value(self):
        with self.assertRaises(KeyError):
            rph.records_by_axis("nope", "x")
        with self.assertRaises(ValueError):
            rph.records_by_axis("background", "marble")


class TestDegradeAndSelfcheck(unittest.TestCase):

    def test_available_is_a_subset_and_facts_survive_absence(self):
        # The facts (rows) are always present; only pixels may be absent.
        self.assertEqual(len(rph.photo_records()), rph.EXPECTED_PHOTOS)
        avail = rph.available_photos()
        self.assertLessEqual(len(avail), rph.EXPECTED_PHOTOS)
        for r in avail:
            self.assertIsNotNone(r.path)

    def test_selfcheck_exits_zero(self):
        self.assertEqual(rph.main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
