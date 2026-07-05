"""Tests for chatcadplus_sphere_retrieval -- sphere-projected KD-tree cosine k-NN."""

from __future__ import annotations

import math
import random

import pytest

from chatcadplus_sphere_retrieval import SphereKDTree, normalise


def _cos(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def test_normalise_unit_length():
    v = normalise([3.0, 4.0])
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0)
    assert math.isclose(v[0], 0.6)
    assert math.isclose(v[1], 0.8)


def test_normalise_zero_vector():
    assert normalise([0.0, 0.0]) == (0.0, 0.0)


def test_cosine_invariant_to_magnitude():
    # A vector and its scaled copy point the same way -> cosine 1.
    tree = SphereKDTree([[1.0, 0.0], [0.0, 1.0]])
    res = tree.query([5.0, 0.0], k=1)
    assert res[0][0] == 0
    assert math.isclose(res[0][1], 1.0)


def test_top_k_ordering():
    vecs = [
        [1.0, 0.0],     # 0: aligned with query
        [0.9, 0.1],     # 1: close
        [0.0, 1.0],     # 2: orthogonal
        [-1.0, 0.0],    # 3: opposite
    ]
    tree = SphereKDTree(vecs)
    res = tree.query([1.0, 0.0], k=3)
    idxs = [i for i, _ in res]
    assert idxs[0] == 0
    assert idxs[1] == 1
    # cosine similarities strictly descending
    sims = [s for _, s in res]
    assert sims == sorted(sims, reverse=True)


def test_matches_brute_force_random():
    rng = random.Random(1234)
    dim = 6
    vecs = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(200)]
    tree = SphereKDTree(vecs)
    for _ in range(50):
        q = [rng.gauss(0, 1) for _ in range(dim)]
        kd = tree.query(q, k=5)
        bf = tree.brute_force(q, k=5)
        assert [i for i, _ in kd] == [i for i, _ in bf]
        for (ik, sk), (ib, sb) in zip(kd, bf):
            assert math.isclose(sk, sb, abs_tol=1e-9)


def test_cosine_value_matches_direct_computation():
    vecs = [[2.0, 1.0], [1.0, 3.0], [-1.0, 2.0]]
    tree = SphereKDTree(vecs)
    q = [1.5, 0.5]
    res = tree.query(q, k=3)
    for idx, sim in res:
        assert math.isclose(sim, _cos(q, vecs[idx]), abs_tol=1e-9)


def test_stable_tie_break_by_index():
    # Two identical vectors -> lower index must come first.
    vecs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    tree = SphereKDTree(vecs)
    res = tree.query([1.0, 0.0], k=2)
    assert [i for i, _ in res] == [0, 1]


def test_k_larger_than_n():
    tree = SphereKDTree([[1.0, 0.0], [0.0, 1.0]])
    res = tree.query([1.0, 0.0], k=10)
    assert len(res) == 2


def test_empty_tree_and_k_zero():
    assert SphereKDTree([]).query([1.0], k=3) == []
    assert SphereKDTree([[1.0, 0.0]]).query([1.0, 0.0], k=0) == []


def test_dim_mismatch_raises():
    tree = SphereKDTree([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        tree.query([1.0, 0.0, 0.0], k=1)


def test_len():
    assert len(SphereKDTree([[1.0, 0.0]] * 5)) == 5
