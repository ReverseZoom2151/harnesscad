"""Graph-level contrastive pretext objective for GC-CAD.

Quan et al., *Self-supervised GNN for Mechanical CAD Retrieval* (GC-CAD),
Section 3.4.2. For a mini-batch of ``N`` CAD parts, each part is augmented twice
(:mod:`reconstruction.ssgnn_graph_augment`) into two graphs whose GNN embeddings
are ``z'_n`` and ``z''_n``. The NT-Xent objective pulls the two views of the same
part together and pushes views of *different* parts apart (paper Eq. in Section
3.4.2)::

    l_n = -log( exp(sim(z'_n, z''_n) / tau)
                / sum_{n'} exp(sim(z'_n, z''_{n'}) / tau) )

with cosine similarity ``sim`` and temperature ``tau``. This is the *graph-level*
contrastive loss (one embedding per graph, ``N`` pairs), which is distinct from
ContrastCAD's node/sequence-latent 2m-view NT-Xent
(:mod:`bench.contrastcad_contrastive`): there the negatives are all other views
in a ``2m`` list; here the anchor is a first-view graph embedding and the
candidates are the *second-view* embeddings of the whole batch.

The learned GNN encoder is external; callers supply either precomputed embeddings
or a deterministic ``embed_fn`` mapping a :class:`CADGraph` to a vector (e.g. the
structural descriptor in :mod:`reconstruction.ssgnn_graph_descriptors`). Given
fixed embeddings and a fixed seed the loss is byte-reproducible. Stdlib-only.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

from harnesscad.eval.bench.retrieval.nt_xent_loss import cosine_similarity
from harnesscad.domain.reconstruction.recognize.graph_augment import CADGraph, positive_pair

Vector = Sequence[float]


def graph_nt_xent_anchor(first: Sequence[Vector], second: Sequence[Vector],
                         n: int, temperature: float = 0.5,
                         include_positive: bool = True) -> float:
    """NT-Xent loss for one anchor part ``n`` (paper Section 3.4.2).

    ``first[n]`` (``z'_n``) is the anchor, ``second[n]`` (``z''_n``) its positive.
    The denominator sums ``exp(sim(z'_n, z''_{n'}) / tau)`` over the batch. With
    ``include_positive`` (the standard NT-Xent / GraphCL form) the positive term
    ``n' = n`` is kept in the denominator; set it False to reproduce the paper's
    literal ``n' != n`` indexing (negatives only).
    """
    if len(first) != len(second):
        raise ValueError("first and second view lists must have equal length")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0 <= n < len(first):
        raise ValueError("anchor index out of range")
    num = math.exp(cosine_similarity(first[n], second[n]) / temperature)
    denom = 0.0
    for np_ in range(len(second)):
        if np_ == n and not include_positive:
            continue
        denom += math.exp(cosine_similarity(first[n], second[np_]) / temperature)
    if denom == 0.0:
        raise ValueError("degenerate denominator in NT-Xent loss")
    return -math.log(num / denom)


def graph_nt_xent(first: Sequence[Vector], second: Sequence[Vector],
                  temperature: float = 0.5, include_positive: bool = True,
                  symmetric: bool = False) -> float:
    """Mean NT-Xent loss over a batch of ``N`` augmented-graph embedding pairs.

    Averages :func:`graph_nt_xent_anchor` over every part. With ``symmetric`` the
    loss is also computed with the roles of the two views swapped and the two
    directions are averaged (the standard symmetric contrastive objective).
    """
    n = len(first)
    if n == 0:
        raise ValueError("empty batch")
    forward = sum(graph_nt_xent_anchor(first, second, i, temperature,
                                       include_positive) for i in range(n)) / n
    if not symmetric:
        return forward
    backward = sum(graph_nt_xent_anchor(second, first, i, temperature,
                                        include_positive) for i in range(n)) / n
    return (forward + backward) / 2.0


def similarity_matrix(first: Sequence[Vector],
                      second: Sequence[Vector]) -> List[List[float]]:
    """Cross-view cosine-similarity matrix ``M[i][j] = sim(z'_i, z''_j)``.

    The diagonal holds the positive-pair similarities and the off-diagonal the
    negatives -- exactly the quantities the NT-Xent numerator and denominator use.
    """
    return [[cosine_similarity(a, b) for b in second] for a in first]


def build_pretext_views(graphs: Sequence[CADGraph],
                        embed_fn: Callable[[CADGraph], Vector], seed, *,
                        scheme: str = "node", feature_ratio: float = 0.1,
                        structure_ratio: float = 0.1
                        ) -> Tuple[List[Vector], List[Vector]]:
    """Build the two embedding lists ``(z', z'')`` for a batch of CAD graphs.

    Each graph is augmented twice (a positive pair) and each augmented graph is
    embedded with ``embed_fn``. Per-part seeds are derived deterministically from
    ``seed`` so the whole batch is reproducible. Returns ``(first, second)`` where
    ``first[i]`` and ``second[i]`` are the two views of ``graphs[i]``.
    """
    first: List[Vector] = []
    second: List[Vector] = []
    for i, graph in enumerate(graphs):
        g1, g2 = positive_pair(graph, seed=(hash((seed, i)) & 0x7FFFFFFF),
                               scheme=scheme, feature_ratio=feature_ratio,
                               structure_ratio=structure_ratio)
        first.append(list(embed_fn(g1)))
        second.append(list(embed_fn(g2)))
    return first, second


def pretext_loss(graphs: Sequence[CADGraph],
                 embed_fn: Callable[[CADGraph], Vector], seed, *,
                 scheme: str = "node", feature_ratio: float = 0.1,
                 structure_ratio: float = 0.1, temperature: float = 0.5,
                 symmetric: bool = False) -> float:
    """End-to-end deterministic graph-contrastive pretext loss for a batch.

    Augments every graph twice, embeds the views with ``embed_fn``, then returns
    the mean NT-Xent loss. This is the fully reproducible surrogate for GC-CAD's
    contrastive training term given a deterministic embedding function.
    """
    first, second = build_pretext_views(graphs, embed_fn, seed, scheme=scheme,
                                        feature_ratio=feature_ratio,
                                        structure_ratio=structure_ratio)
    return graph_nt_xent(first, second, temperature, symmetric=symmetric)
