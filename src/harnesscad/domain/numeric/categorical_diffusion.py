"""Categorical (multinomial) discrete diffusion.

This module implements the *conventional* discrete-diffusion baseline:
multinomial / structured discrete diffusion over a categorical state space.
It serves as the categorical ablation against which richer schemes are
compared. Unlike the learned denoiser, the
categorical *forward* process, its cumulative marginal, and the true posterior
``q(x_{t-1} | x_t, x_0)`` are fully deterministic given a transition schedule,
and all reduce to small matrix algebra over the probability simplex.

For a one-hot state ``x`` over ``K`` categories the forward transition is

    q(x_t | x_{t-1}) = Cat( x_t ; x_{t-1} Q_t )

where ``Q_t`` is a ``K x K`` row-stochastic transition matrix. Two standard
families are provided:

* **Uniform** (Multinomial diffusion): ``Q_t = (1-b_t) I + (b_t/K) 11^T`` --
  with probability ``1-b_t`` the category is kept, otherwise it is resampled
  uniformly. As ``t -> T`` the state converges to the uniform categorical.

* **Absorbing** (absorbing/[MASK]): every category decays with rate
  ``b_t`` into a distinguished absorbing/mask category and stays there.

Iterating the chain gives the cumulative transition ``Qbar_t = Q_1 ... Q_t`` and
the closed-form marginal ``q(x_t | x_0) = x_0 Qbar_t``. Bayes' rule then gives
the categorical posterior used by the reverse process

    q(x_{t-1} | x_t, x_0)  proportional to  (x_t Q_t^T) (x)  (x_0 Qbar_{t-1})

(elementwise product, then renormalised). Everything here is stdlib-only,
operates on plain Python float vectors/matrices, and routes all randomness
through an explicit ``random.Random`` seed.
"""

from __future__ import annotations

import random
from typing import Sequence

Matrix = list[list[float]]
Vector = list[float]


def _validate_beta(b: float) -> float:
    if not 0.0 <= b <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    return float(b)


def uniform_transition_matrix(num_classes: int, beta: float) -> Matrix:
    """Multinomial-diffusion uniform transition ``Q = (1-b) I + (b/K) 11^T``.

    Row-stochastic: with probability ``1-b + b/K`` a category is retained and
    ``b/K`` mass leaks to each of the other categories.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    b = _validate_beta(beta)
    k = num_classes
    off = b / k
    q: Matrix = []
    for i in range(k):
        row = [off] * k
        row[i] = (1.0 - b) + off
        q.append(row)
    return q


def absorbing_transition_matrix(
    num_classes: int, beta: float, absorb_index: int | None = None
) -> Matrix:
    """Absorbing-state transition: mass ``b`` decays into an absorbing state.

    ``absorb_index`` defaults to the last category (the conventional ``[MASK]``
    slot). The absorbing row is the identity row, so once absorbed a state never
    leaves.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    b = _validate_beta(beta)
    k = num_classes
    m = k - 1 if absorb_index is None else absorb_index
    if not 0 <= m < k:
        raise IndexError("absorb_index out of range")
    q: Matrix = []
    for i in range(k):
        row = [0.0] * k
        if i == m:
            row[m] = 1.0
        else:
            row[i] = 1.0 - b
            row[m] += b
        q.append(row)
    return q


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """Plain matrix product ``a @ b``."""
    n, k = len(a), len(a[0])
    if len(b) != k:
        raise ValueError("inner dimensions do not match")
    p = len(b[0])
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        ai = a[i]
        oi = out[i]
        for t in range(k):
            ait = ai[t]
            if ait == 0.0:
                continue
            bt = b[t]
            for j in range(p):
                oi[j] += ait * bt[j]
    return out


def vec_mat(v: Sequence[float], m: Matrix) -> Vector:
    """Row-vector times matrix ``v @ m``."""
    if len(v) != len(m):
        raise ValueError("dimension mismatch")
    cols = len(m[0])
    out = [0.0] * cols
    for i, vi in enumerate(v):
        if vi == 0.0:
            continue
        mi = m[i]
        for j in range(cols):
            out[j] += vi * mi[j]
    return out


def transpose(m: Matrix) -> Matrix:
    return [list(col) for col in zip(*m)]


def cumulative_matrices(transitions: Sequence[Matrix]) -> list[Matrix]:
    """Return ``[Qbar_1, Qbar_2, ...]`` where ``Qbar_t = Q_1 ... Q_t``.

    Index ``i`` of the result corresponds to timestep ``t = i + 1``.
    """
    if not transitions:
        return []
    cum: list[Matrix] = [[row[:] for row in transitions[0]]]
    for q in transitions[1:]:
        cum.append(matmul(cum[-1], q))
    return cum


def one_hot(index: int, num_classes: int) -> Vector:
    if not 0 <= index < num_classes:
        raise IndexError("index out of range")
    v = [0.0] * num_classes
    v[index] = 1.0
    return v


def forward_marginal(x0: Sequence[float], qbar_t: Matrix) -> Vector:
    """``q(x_t | x_0) = x_0 Qbar_t`` -- the one-shot categorical marginal."""
    return vec_mat(x0, qbar_t)


def forward_step(x_prev: Sequence[float], q_t: Matrix) -> Vector:
    """``q(x_t | x_{t-1}) = x_{t-1} Q_t`` -- a single forward transition."""
    return vec_mat(x_prev, q_t)


def _normalise(v: Sequence[float]) -> Vector:
    s = sum(v)
    if s <= 0.0:
        raise ValueError("cannot normalise a non-positive distribution")
    return [x / s for x in v]


def categorical_posterior(
    x_t: Sequence[float],
    x0: Sequence[float],
    q_t: Matrix,
    qbar_prev: Matrix | None,
) -> Vector:
    """True posterior ``q(x_{t-1} | x_t, x_0)``.

    ``proportional to (x_t Q_t^T) elementwise (x_0 Qbar_{t-1})``, renormalised.
    For ``t == 1`` pass ``qbar_prev=None``; the ``Qbar_0 = I`` factor makes the
    second term simply ``x_0``.
    """
    left = vec_mat(x_t, transpose(q_t))
    if qbar_prev is None:
        right = list(x0)
    else:
        right = vec_mat(x0, qbar_prev)
    prod = [a * b for a, b in zip(left, right)]
    return _normalise(prod)


def sample_categorical(probs: Sequence[float], rng: random.Random) -> int:
    """Sample a class index from a categorical distribution (deterministic)."""
    total = sum(probs)
    if total <= 0.0:
        raise ValueError("probabilities must sum to a positive value")
    r = rng.random() * total
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1


def argmax_decode(probs: Sequence[float]) -> int:
    """Recover the most-likely class index (ties resolved to lowest index)."""
    best_i, best_v = 0, probs[0]
    for i, p in enumerate(probs):
        if p > best_v:
            best_i, best_v = i, p
    return best_i


def diffuse_categorical(
    x0_index: int,
    num_classes: int,
    transitions: Sequence[Matrix],
    t: int,
    rng: random.Random,
) -> int:
    """Sample a categorical index ``x_t`` from ``q(x_t | x_0)`` at step ``t``.

    ``transitions[i]`` is ``Q_{i+1}``; ``t`` runs in ``1 .. len(transitions)``.
    """
    if not 1 <= t <= len(transitions):
        raise IndexError("t out of range for the given transition schedule")
    qbar = cumulative_matrices(transitions[:t])[-1]
    marg = forward_marginal(one_hot(x0_index, num_classes), qbar)
    return sample_categorical(marg, rng)
