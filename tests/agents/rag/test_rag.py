"""Tests for the hybrid RAG grounding layer (blueprint sec.2, sec.7, sec.19 P2).

Covers:
  - structure-aware chunking keeps heading breadcrumbs and splits sections;
    fenced code/API blocks stay atomic;
  - BM25 ranks the on-topic chunk first for a keyword query;
  - hybrid fusion beats either index alone on a mixed query;
  - retrieve(k) returns exactly k ranked results, most-relevant first;
  - filtering by source / heading;
  - empty-corpus and single-doc edge cases.
"""

import unittest

from harnesscad.agents.rag.chunk import Chunk, chunk_document
from harnesscad.agents.rag.index import BM25Index, HashedVectorIndex, HashedEmbedder
from harnesscad.agents.rag.retriever import HybridRetriever, build_from_docs


MARKDOWN = """\
# Fasteners

Intro prose about fasteners in general.

## M6 bolts

### Torque

The recommended tightening torque for an M6 socket head cap screw is 10 Nm.

```python
def torque_m6(mu=0.2):
    return 10.0  # Nm, class 8.8
```

## Washers

Use a flat washer under the head to spread the load.
"""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
class TestChunking(unittest.TestCase):
    def test_splits_sections_and_keeps_breadcrumbs(self):
        chunks = chunk_document(MARKDOWN, source="fasteners.md")
        self.assertGreaterEqual(len(chunks), 4)
        # every chunk knows its source
        self.assertTrue(all(c.source == "fasteners.md" for c in chunks))

        # the torque prose chunk carries the full heading breadcrumb
        torque = next(c for c in chunks if "tightening torque" in c.text)
        self.assertEqual(torque.heading_path, ["Fasteners", "M6 bolts", "Torque"])
        self.assertEqual(torque.breadcrumb, "Fasteners > M6 bolts > Torque")

        # a sibling section resets the breadcrumb to the right depth
        washer = next(c for c in chunks if "flat washer" in c.text)
        self.assertEqual(washer.heading_path, ["Fasteners", "Washers"])

    def test_code_block_is_atomic(self):
        chunks = chunk_document(MARKDOWN, source="fasteners.md")
        code = [c for c in chunks if c.kind == "code"]
        self.assertEqual(len(code), 1)
        # the whole fenced block is kept intact, fences included
        self.assertIn("def torque_m6", code[0].text)
        self.assertIn("return 10.0", code[0].text)
        self.assertTrue(code[0].text.strip().startswith("```"))
        # and it inherits the section breadcrumb it lives under
        self.assertEqual(code[0].heading_path, ["Fasteners", "M6 bolts", "Torque"])

    def test_ids_are_deterministic_and_unique(self):
        a = chunk_document(MARKDOWN, source="fasteners.md")
        b = chunk_document(MARKDOWN, source="fasteners.md")
        self.assertEqual([c.id for c in a], [c.id for c in b])
        self.assertEqual(len({c.id for c in a}), len(a))

    def test_plaintext_no_headings(self):
        chunks = chunk_document("just a flat note with no headings", "note")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].heading_path, [])

    def test_empty_document(self):
        self.assertEqual(chunk_document("", "empty"), [])
        self.assertEqual(chunk_document("   \n\n  ", "blank"), [])


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
class TestBM25(unittest.TestCase):
    def _corpus(self):
        return [
            Chunk("a", "The tightening torque for an M6 bolt is ten newton metres.", "d"),
            Chunk("b", "Aluminium 6061 has a density of about 2.7 grams per cc.", "d"),
            Chunk("c", "A fillet radius must be smaller than the adjacent edge length.", "d"),
        ]

    def test_ranks_on_topic_chunk_first(self):
        idx = BM25Index()
        for c in self._corpus():
            idx.add(c)
        hits = idx.search("M6 bolt tightening torque", k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0].id, "a")

    def test_empty_index(self):
        self.assertEqual(BM25Index().search("anything", k=3), [])


# ---------------------------------------------------------------------------
# Hashed vector index
# ---------------------------------------------------------------------------
class TestHashedVector(unittest.TestCase):
    def test_cosine_ranks_and_is_deterministic(self):
        chunks = [
            Chunk("a", "socket head cap screw torque specification", "d"),
            Chunk("b", "surface finish and anodising of aluminium", "d"),
        ]
        idx1 = HashedVectorIndex()
        idx2 = HashedVectorIndex(embedder=HashedEmbedder())
        for c in chunks:
            idx1.add(c)
            idx2.add(c)
        h1 = idx1.search("cap screw torque", k=2)
        h2 = idx2.search("cap screw torque", k=2)
        self.assertEqual(h1[0][0].id, "a")
        # deterministic across instances (stable hashing, no salted hash())
        self.assertEqual([c.id for c, _ in h1], [c.id for c, _ in h2])

    def test_empty_index(self):
        self.assertEqual(HashedVectorIndex().search("anything", k=3), [])


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------
class TestHybridRetriever(unittest.TestCase):
    def test_retrieve_returns_k_ranked(self):
        docs = [
            ("M6 bolt torque is 10 Nm for class 8.8 fasteners.", "s1"),
            ("Aluminium 6061 density is 2.7 g/cc.", "s2"),
            ("Fillet radius rule of thumb for sheet metal bends.", "s3"),
            ("STL export must be watertight and manifold.", "s4"),
        ]
        r = build_from_docs(docs)
        # query touches two docs (bolt/torque -> s1, aluminium/density -> s2)
        hits = r.retrieve("bolt torque and aluminium density", k=2)
        self.assertEqual(len(hits), 2)
        # results are sorted by fused score descending
        self.assertGreaterEqual(hits[0].score, hits[1].score)
        # on-topic chunk surfaces at the top
        self.assertIn("torque", hits[0].text.lower())

    def test_fusion_beats_either_alone(self):
        # Mixed query: 'flangebolt'/'sprocketpin' are rare (BM25 signal),
        # 'torque aluminium bracket mounting' are common (dense/cosine signal).
        #  - the BM decoys spam the rare terms   -> top BM25, weak cosine;
        #  - the VEC decoys spam the common phrase-> top cosine, weak BM25;
        #  - the target has both, so it is only ~2nd in EACH index alone,
        #    but rank fusion (RRF over each retriever's top-N list) lifts it to #1.
        query = "flangebolt sprocketpin torque aluminium bracket mounting"
        bmd1 = "flangebolt sprocketpin " * 4
        bmd2 = "flangebolt sprocketpin flangebolt sprocketpin flangebolt sprocketpin"
        vcd1 = "torque aluminium bracket mounting " * 4
        vcd2 = ("torque aluminium bracket mounting torque aluminium bracket "
                "mounting torque aluminium")
        target = ("flangebolt sprocketpin torque aluminium bracket mounting reference "
                  "clause with some assorted explanatory padding words appended here")
        pad = "lorem ipsum dolor sit amet consectetur adipiscing elit sed eiusmod"
        docs = [(target, "target"), (bmd1, "bmd1"), (bmd2, "bmd2"),
                (vcd1, "vcd1"), (vcd2, "vcd2")]
        docs += [(f"torque aluminium bracket mounting {pad} v{n}", f"f{n}")
                 for n in range(6)]

        r = build_from_docs(docs)

        bm_first = r.bm25.search(query, k=3)[0][0].source
        vc_first = r.vector.search(query, k=3)[0][0].source
        # RRF fuses each retriever's top-N candidate list (standard practice).
        hybrid_first = r.retrieve(query, k=3, candidate_pool=3)[0].source

        # neither index alone puts the target on top ...
        self.assertNotEqual(bm_first, "target")
        self.assertNotEqual(vc_first, "target")
        # ... but the fusion does.
        self.assertEqual(hybrid_first, "target")

    def test_filter_by_source_and_heading(self):
        md = ("# Standards\n\nGeneral note.\n\n"
              "## Torque\n\nM6 torque is 10 Nm.\n")
        r = HybridRetriever()
        r.add_document(md, "iso.md")
        r.add_document("M6 torque hints in loose notes.", "notes.txt")

        only_iso = r.retrieve("M6 torque", k=5, source="iso")
        self.assertTrue(only_iso)
        self.assertTrue(all(h.source == "iso.md" for h in only_iso))

        under_torque = r.retrieve("M6 torque", k=5, heading="Torque")
        self.assertTrue(under_torque)
        self.assertTrue(all("Torque" in h.heading_path for h in under_torque))

    def test_weighted_fusion_mode(self):
        docs = [("M6 bolt torque is 10 Nm.", "a"), ("Anodising aluminium.", "b")]
        r = build_from_docs(docs, fusion="weighted")
        hits = r.retrieve("bolt torque", k=2)
        self.assertEqual(hits[0].source, "a")

    def test_empty_corpus(self):
        r = HybridRetriever()
        self.assertEqual(r.retrieve("anything", k=5), [])
        self.assertEqual(r.retrieve_chunks("anything", k=5), [])

    def test_single_doc(self):
        r = build_from_docs([("The only document, about M6 bolts.", "solo")])
        hits = r.retrieve("M6 bolts", k=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source, "solo")

    def test_retrieve_chunks_returns_chunks(self):
        r = build_from_docs([("M6 bolt torque note.", "s1")])
        cs = r.retrieve_chunks("torque", k=1)
        self.assertEqual(len(cs), 1)
        self.assertIsInstance(cs[0], Chunk)


if __name__ == "__main__":
    unittest.main()
