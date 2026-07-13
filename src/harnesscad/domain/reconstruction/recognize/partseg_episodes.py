"""Deterministic C-way K-shot episode construction for few-shot part segmentation.

Section II-A of Wang et al. formulates part segmentation as an episodic few-shot
task. The dataset of ``Ntotal`` classes is split into disjoint train/test class
groups; an episode draws ``C`` classes, ``K`` support samples per class, and
``NQ`` query samples, plus a background class, producing a label space of
``C + 1``. The sampling procedure -- although described as "random" -- is fully
deterministic once seeded, and no learning is involved, so it is buildable here.

A sample is any object carrying a point cloud and its per-point class labels;
this module keeps the container abstract (a ``Sample`` dataclass) and focuses on
the episode/label-remapping bookkeeping.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


BACKGROUND = 0  # episode-local label reserved for the background class.


@dataclass(frozen=True)
class Sample:
    """One labelled point cloud belonging to a single (foreground) class."""
    cls: object
    features: tuple
    point_labels: tuple  # per-point original class labels


@dataclass(frozen=True)
class Episode:
    ways: tuple            # the C sampled foreground classes, ordered
    label_map: dict        # original class -> episode-local label (1..C)
    support: tuple         # tuple of Sample
    query: tuple           # tuple of Sample


def split_classes(classes, n_train, seed=0):
    """Deterministically partition classes into train / test groups.

    Mirrors the paper's cross-validation split: ``n_train`` classes are drawn for
    training and the remainder reserved for testing.
    """
    ordered = sorted(set(classes), key=repr)
    if not 0 <= n_train <= len(ordered):
        raise ValueError("n_train out of range")
    rng = random.Random(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    train = tuple(sorted(shuffled[:n_train], key=repr))
    test = tuple(sorted(shuffled[n_train:], key=repr))
    return train, test


def remap_labels(point_labels, label_map):
    """Map original per-point labels to episode-local labels.

    Any label not in ``label_map`` (i.e. not one of the episode's ways) becomes
    the background label 0 -- exactly the paper's treatment of the Plane class as
    background when segmenting the other four features.
    """
    return tuple(label_map.get(lab, BACKGROUND) for lab in point_labels)


def build_episode(dataset, ways, shots, queries, seed=0):
    """Construct one C-way K-shot episode.

    ``dataset`` maps class -> list of ``Sample``. ``ways`` classes are sampled,
    ``shots`` support and ``queries`` query samples are drawn per class (without
    replacement within a class), and a ``label_map`` assigns ways to local labels
    ``1..C`` (background stays 0).
    """
    available = [c for c in sorted(dataset, key=repr) if dataset[c]]
    if ways > len(available):
        raise ValueError("not enough classes for requested ways")
    rng = random.Random(seed)
    chosen = sorted(rng.sample(available, ways), key=repr)
    label_map = {c: i + 1 for i, c in enumerate(chosen)}
    support, query = [], []
    for c in chosen:
        pool = list(dataset[c])
        if shots + queries > len(pool):
            raise ValueError(f"class {c!r} has too few samples")
        picks = rng.sample(range(len(pool)), shots + queries)
        for k, p in enumerate(picks):
            s = pool[p]
            remapped = Sample(s.cls, s.features,
                              remap_labels(s.point_labels, label_map))
            (support if k < shots else query).append(remapped)
    return Episode(tuple(chosen), label_map, tuple(support), tuple(query))


def flatten_support(episode):
    """Concatenate support features/labels into flat point-level sequences.

    Convenient for feeding ``build_prototypes``: returns ``(features, labels)``
    with one entry per support point across all support samples.
    """
    feats, labs = [], []
    for s in episode.support:
        feats.extend(s.features)
        labs.extend(s.point_labels)
    return tuple(feats), tuple(labs)
