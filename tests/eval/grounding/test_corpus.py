"""The self-labelling corpus: determinism, honesty about discards, and one live sweep."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from harnesscad.eval.grounding import corpus
from harnesscad.io.cua import viewport as vp


class TestParts(unittest.TestCase):
    def test_seeded_and_reproducible(self):
        a = corpus.sample_parts(6, seed=3)
        b = corpus.sample_parts(6, seed=3)
        self.assertEqual([(s, brief) for s, brief, _o, _p in a],
                         [(s, brief) for s, brief, _o, _p in b])

    def test_a_different_seed_is_a_different_part(self):
        a = corpus.sample_parts(1, seed=1)[0]
        b = corpus.sample_parts(1, seed=2)[0]
        self.assertNotEqual(a[1], b[1])

    def test_the_family_spans_the_topology_that_makes_grounding_hard(self):
        """Boxes alone would report a lovely discard rate and teach nothing.

        The corpus needs bores (cylindrical faces), blends (slivers) and shells
        (interior faces that occlude each other) or it is not exercising the thing
        that has no accessibility tree.
        """
        names = {p[3]["generator"] for p in corpus.sample_parts(7, seed=0)}
        for needed in ("bored_plate", "filleted_block", "shelled_box", "boss"):
            self.assertIn(needed, names)


class TestStats(unittest.TestCase):
    def _pair(self, verified: bool, kind: str = "face", reason: str = "") -> corpus.GroundingPair:
        return corpus.GroundingPair(
            sample="s", view="isometric", screenshot="a.png", entity="Face1",
            kind=kind, description="d", x=1, y=2, point=(0.0, 0.0, 0.0),
            verified=verified, reason=reason)

    def test_discard_rate_is_reported_not_hidden(self):
        stats = corpus.CorpusStats()
        for _ in range(3):
            stats.record(self._pair(True))
        for _ in range(7):
            stats.record(self._pair(False, reason="occluded by Face3"))
        self.assertEqual(stats.verified, 3)
        self.assertEqual(stats.discarded, 7)
        self.assertAlmostEqual(stats.discard_rate, 0.7)
        # The reason is bucketed, so "how much of a viewport is un-clickable, and
        # why" is answerable from the stats alone.
        self.assertEqual(stats.by_reason["occluded"], 7)

    def test_empty_stats_do_not_divide_by_zero(self):
        stats = corpus.CorpusStats()
        self.assertEqual(stats.discard_rate, 0.0)
        self.assertEqual(stats.pairs_per_minute, 0.0)

    def test_pairs_per_minute(self):
        stats = corpus.CorpusStats()
        for _ in range(10):
            stats.record(self._pair(True))
        stats.elapsed = 30.0
        self.assertAlmostEqual(stats.pairs_per_minute, 20.0)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="hc_corpus_test_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_discards_are_kept_in_their_own_file(self):
        """A corpus that silently drops its discards cannot report its own coverage."""
        pairs = [
            corpus.GroundingPair(sample="s", view="top", screenshot="a.png",
                                 entity="Face1", kind="face", description="the top face",
                                 x=10, y=20, point=(1.0, 2.0, 3.0), verified=True,
                                 selected="Face1", width=800, height=600),
            corpus.GroundingPair(sample="s", view="top", screenshot="a.png",
                                 entity="Face2", kind="face", description="the bottom face",
                                 x=-1, y=-1, point=(1.0, 2.0, 0.0), verified=False,
                                 selected="Face1", reason="occluded by Face1"),
        ]
        stats = corpus.CorpusStats()
        for p in pairs:
            stats.record(p)
        corpus.write_corpus(self.dir, pairs, stats)

        kept = corpus.read_pairs(os.path.join(self.dir, "pairs.jsonl"))
        self.assertEqual([p.entity for p in kept], ["Face1"])
        self.assertEqual(kept[0].width, 800)      # the frame travels WITH the label
        dropped = corpus.read_pairs(os.path.join(self.dir, "discards.jsonl"))
        self.assertEqual([p.entity for p in dropped], ["Face2"])
        self.assertEqual(dropped[0].reason, "occluded by Face1")

        with open(os.path.join(self.dir, "stats.json"), encoding="utf-8") as fh:
            written = json.load(fh)
        self.assertAlmostEqual(written["discard_rate"], 0.5)


@unittest.skipUnless(vp.gui_available(), "the FreeCAD GUI is not installed")
class TestLiveCorpus(unittest.TestCase):
    """One real sweep. Skipped, never failed, when FreeCAD is absent."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="hc_corpus_live_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_sweep_yields_verified_pairs_and_an_honest_discard_rate(self):
        with corpus.CorpusGenerator(self.dir, views=("isometric", "top")) as gen:
            pairs, stats = gen.run(count=2, seed=0)
        self.assertGreater(stats.verified, 0)
        self.assertGreater(stats.discarded, 0,
                           "a discard rate of zero would mean nothing occludes "
                           "anything, which is not a CAD viewport")
        self.assertLess(stats.discard_rate, 1.0)

        for pair in pairs:
            if pair.verified:
                # The label is the app's own verdict, never our projection's.
                self.assertEqual(pair.selected, pair.entity)
                self.assertTrue(0 <= pair.x < pair.width)
                self.assertTrue(0 <= pair.y < pair.height)
                self.assertTrue(os.path.isfile(
                    os.path.join(self.dir, pair.screenshot)))
            else:
                self.assertTrue(pair.reason)

    def test_the_same_seed_gives_the_same_corpus(self):
        """Byte-for-byte, twice. A corpus that cannot be regenerated is not one."""
        def sweep(path):
            with corpus.CorpusGenerator(path, views=("isometric",)) as gen:
                pairs, _s = gen.run(count=1, seed=5)
            return {(p.sample, p.view, p.entity): (p.x, p.y)
                    for p in pairs if p.verified}

        first = sweep(os.path.join(self.dir, "a"))
        second = sweep(os.path.join(self.dir, "b"))
        self.assertTrue(first)
        self.assertEqual(first, second)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
