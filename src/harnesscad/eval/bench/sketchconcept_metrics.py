"""Evaluation metrics for CAD-sketch **concept libraries** (Yang & Pan,
"Discovering Design Concepts for CAD Sketches", NeurIPS 2022).

The paper judges a discovered library on two axes, and a good library must win on
both at once (either is trivial alone -- one concept per sketch reconstructs
everything; an empty library is maximally compact):

* **compactness / modularity** -- how few, how small, how *reused* the concepts
  are, and how much the hierarchy itself compresses the library (a concept built
  from sub-concepts stores calls, not geometry);
* **reconstruction coverage** -- how much of a sketch corpus the library actually
  explains: the fraction of primitives and constraints absorbed into concept
  instances, and whether the decomposition is *lossless*.

Both are combined by the description-length **compression ratio**: the cost of
spelling the corpus out primitive-by-primitive, divided by the cost of storing the
library once plus emitting one call (name + bindings + input wiring) per instance
and spelling out only the residual. A ratio above 1 means the library pays for
itself.

Token model (shared with :mod:`library.sketchconcept_induction`):
``primitive = 1 + |params|``, ``constraint = 1 + arity``,
``concept call = 1 + |bindings| + |inputs|``, plus a small header per stored
concept.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.reconstruction.sketchconcept_decompose import Decomposition, decompose, is_exact
from harnesscad.domain.reconstruction.sketchconcept_template import Concept, Sketch

# ---------------------------------------------------------------------------
# Token costs
# ---------------------------------------------------------------------------


def sketch_cost(sketch: Sketch) -> int:
    """Raw description length of a sketch: primitives + constraints, spelled out."""
    cost = 0
    for p in sketch.primitives:
        cost += 1 + len(p.params)
    for c in sketch.constraints:
        cost += 1 + len(c.refs)
    return cost


def concept_cost(concept: Concept) -> int:
    """Description length of a *stored* concept (sub-instances counted as calls)."""
    cost = 1 + len(concept.slots) + concept.in_arity + len(concept.out_refs)  # header
    for m in concept.members:
        cost += 1 + len(m.params)
    for c in concept.constraints:
        cost += 1 + len(c.refs)
    for s in concept.subs:
        cost += 1 + len(s.bindings) + len(s.inputs)
    return cost


def library_cost(library) -> int:
    """Description length of the whole library, in its stored (hierarchical) form."""
    return sum(concept_cost(library.get(n)) for n in library.names())


def instance_call_cost(placement) -> int:
    inst = placement.instance
    return 1 + len(inst.bindings) + len(inst.inputs)


def decomposition_cost(decomp: Decomposition) -> int:
    """Cost of a decomposed sketch: concept calls + spelled-out residual."""
    cost = sum(instance_call_cost(p) for p in decomp.placements)
    by_id = decomp.sketch.by_id()
    for pid in decomp.residual_primitives:
        cost += 1 + len(by_id[pid].params)
    for i in decomp.residual_constraints:
        cost += 1 + len(decomp.sketch.constraints[i].refs)
    return cost


# ---------------------------------------------------------------------------
# Library compactness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibraryStats:
    n_concepts: int
    n_aliases: int
    total_members: int          # flattened members across the library
    avg_members: float
    max_members: int
    max_depth: int              # deepest abstraction level
    total_constraints: int
    cost: int                   # description length of the stored library
    flat_cost: int              # cost if every concept were stored fully flattened
    hierarchy_saving: int       # flat_cost - cost (what the hierarchy buys)


def library_compactness(library) -> LibraryStats:
    names = list(library.names())
    if not names:
        return LibraryStats(0, len(library.aliases()), 0, 0.0, 0, 0, 0, 0, 0, 0)
    flats = [library.flatten(n) for n in names]
    members = [len(f.members) for f in flats]
    cons = sum(len(f.constraints) for f in flats)
    flat_cost = sum(concept_cost(f) for f in flats)
    cost = library_cost(library)
    return LibraryStats(
        n_concepts=len(names),
        n_aliases=len(library.aliases()),
        total_members=sum(members),
        avg_members=sum(members) / len(names),
        max_members=max(members),
        max_depth=max(library.depth(n) for n in names),
        total_constraints=cons,
        cost=cost,
        flat_cost=flat_cost,
        hierarchy_saving=flat_cost - cost,
    )


# ---------------------------------------------------------------------------
# Coverage of one decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageStats:
    n_primitives: int
    n_constraints: int
    n_instances: int
    covered_primitives: int
    covered_constraints: int
    primitive_coverage: float
    constraint_coverage: float
    raw_cost: int
    encoded_cost: int
    compression_ratio: float
    lossless: bool


def coverage(decomp: Decomposition, library, check_lossless: bool = True) -> CoverageStats:
    raw = sketch_cost(decomp.sketch)
    enc = decomposition_cost(decomp)
    n_p = len(decomp.sketch.primitives)
    n_c = len(decomp.sketch.constraints)
    return CoverageStats(
        n_primitives=n_p,
        n_constraints=n_c,
        n_instances=len(decomp.placements),
        covered_primitives=len(decomp.covered_primitives()),
        covered_constraints=n_c - len(decomp.residual_constraints),
        primitive_coverage=decomp.primitive_coverage(),
        constraint_coverage=decomp.constraint_coverage(),
        raw_cost=raw,
        encoded_cost=enc,
        compression_ratio=(raw / enc) if enc else float("inf"),
        lossless=is_exact(decomp, library) if check_lossless else True,
    )


# ---------------------------------------------------------------------------
# Corpus-level evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusStats:
    n_sketches: int
    library: LibraryStats
    primitive_coverage: float
    constraint_coverage: float
    n_instances: int
    raw_cost: int
    encoded_cost: int          # library cost + per-sketch encoded cost
    compression_ratio: float
    lossless: bool
    usage: Tuple[Tuple[str, int], ...]   # concept -> instance count, sorted
    unused_concepts: Tuple[str, ...]
    reuse_rate: float          # fraction of concepts instantiated at least twice


def evaluate_corpus(corpus: Sequence[Sketch], library,
                    decompositions: Optional[Sequence[Decomposition]] = None,
                    check_lossless: bool = True) -> CorpusStats:
    """Decompose a corpus with ``library`` and report compactness + coverage + DL."""
    decomps = list(decompositions) if decompositions is not None else \
        [decompose(sk, library) for sk in corpus]
    if len(decomps) != len(corpus):
        raise ValueError("decompositions do not match the corpus")

    raw = sum(sketch_cost(sk) for sk in corpus)
    enc = library_cost(library) + sum(decomposition_cost(d) for d in decomps)

    tot_p = sum(len(d.sketch.primitives) for d in decomps)
    cov_p = sum(len(d.covered_primitives()) for d in decomps)
    tot_c = sum(len(d.sketch.constraints) for d in decomps)
    cov_c = sum(len(d.sketch.constraints) - len(d.residual_constraints) for d in decomps)

    usage: Dict[str, int] = {n: 0 for n in library.names()}
    n_inst = 0
    for d in decomps:
        for name, k in d.concept_counts().items():
            usage[library.resolve(name)] = usage.get(library.resolve(name), 0) + k
            n_inst += k

    n_concepts = len(library)
    reused = sum(1 for v in usage.values() if v >= 2)
    lossless = all(is_exact(d, library) for d in decomps) if check_lossless else True

    return CorpusStats(
        n_sketches=len(corpus),
        library=library_compactness(library),
        primitive_coverage=(cov_p / tot_p) if tot_p else 1.0,
        constraint_coverage=(cov_c / tot_c) if tot_c else 1.0,
        n_instances=n_inst,
        raw_cost=raw,
        encoded_cost=enc,
        compression_ratio=(raw / enc) if enc else float("inf"),
        lossless=lossless,
        usage=tuple(sorted(usage.items(), key=lambda kv: (-kv[1], kv[0]))),
        unused_concepts=tuple(sorted(n for n, v in usage.items() if v == 0)),
        reuse_rate=(reused / n_concepts) if n_concepts else 0.0,
    )
