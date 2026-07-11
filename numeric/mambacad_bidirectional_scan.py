"""Bidirectional / multi-directional selective-scan ordering (Mamba-CAD).

Mamba-CAD follows the vision-Mamba line of work (VMamba, ViMamba; paper
"Mamba-based models") that runs the selective scan along *several traversal
orders* -- e.g. "perform bi-directional scanning on these two directions" --
and merges the results, so every token can attend to context on both sides
despite the scan being a strictly causal left-to-right recurrence.

The *learned* per-token diagonal kernels are external; the only deterministic
pieces are (a) the **scan-order permutations** (reverse, and general index
orderings) and (b) the **merge** of the per-direction outputs. Those are
implemented here. The underlying causal recurrence itself is **not**
re-implemented: :func:`bidirectional_scan` imports and calls
:func:`numeric.geofusion_state_space.selective_scan` for each direction.

Provided:

* :func:`reverse_seq` -- reverse a sequence (the backward scan order);
* :func:`apply_order` / :func:`invert_order` -- apply an arbitrary index
  permutation to a sequence and undo it (so per-direction outputs realign to
  the original token positions before merging);
* :func:`scan_direction` -- run the forward selective scan under one ordering
  and return outputs *realigned* to the original positions;
* :func:`merge_directions` -- deterministic ``sum`` / ``average`` / ``concat``
  fusion of the per-direction outputs;
* :func:`bidirectional_scan` -- forward + reverse convenience wrapper;
* :func:`multidirectional_scan` -- the general N-ordering version.

Vectors are ``tuple[float, ...]``; sequences are ``tuple`` of such vectors.
Deterministic, stdlib-only.
"""

from __future__ import annotations

from numeric.geofusion_state_space import selective_scan

Vec = tuple[float, ...]
Seq = tuple[Vec, ...]
Order = tuple[int, ...]


def reverse_seq(seq: Seq) -> Seq:
    """Return ``seq`` reversed (the right-to-left backward scan ordering)."""
    return tuple(reversed(seq))


def apply_order(seq: Seq, order: Order) -> Seq:
    """Reorder ``seq`` by ``order``: result[i] = seq[order[i]]."""
    n = len(seq)
    if sorted(order) != list(range(n)):
        raise ValueError("order must be a permutation of range(len(seq))")
    return tuple(seq[i] for i in order)


def invert_order(seq: Seq, order: Order) -> Seq:
    """Undo :func:`apply_order`: given a sequence produced in ``order``, place
    each element back at its original index. ``result[order[i]] = seq[i]``."""
    n = len(seq)
    if sorted(order) != list(range(n)):
        raise ValueError("order must be a permutation of range(len(seq))")
    out: list[Vec | None] = [None] * n
    for i, dst in enumerate(order):
        out[dst] = seq[i]
    return tuple(v for v in out)  # type: ignore[misc]


def scan_direction(z_seq: Seq, a_seq: Seq, b_seq: Seq, c_seq: Seq, g_seq: Seq,
                   order: Order, h0: Vec | None = None) -> Seq:
    """Run the (imported) forward selective scan under a single traversal
    ``order`` and realign the outputs back to the original token positions.

    All kernel sequences are given in *original* token order; they are permuted
    by ``order`` alongside the input, scanned, then the outputs are inverted so
    ``result[k]`` corresponds to original position ``k`` (ready to be merged
    with other directions position-by-position).
    """
    zo = apply_order(z_seq, order)
    ao = apply_order(a_seq, order)
    bo = apply_order(b_seq, order)
    co = apply_order(c_seq, order)
    go = apply_order(g_seq, order)
    outs, _ = selective_scan(zo, ao, bo, co, go, h0=h0)
    return invert_order(outs, order)


def merge_directions(outputs: tuple[Seq, ...], mode: str = "sum") -> Seq:
    """Merge several equal-length per-direction output sequences.

    ``mode``:

    * ``"sum"``     -- element-wise sum across directions;
    * ``"average"`` -- element-wise mean across directions;
    * ``"concat"``  -- concatenate the per-direction vectors at each position.
    """
    if not outputs:
        raise ValueError("need at least one direction to merge")
    length = len(outputs[0])
    for o in outputs:
        if len(o) != length:
            raise ValueError("all directions must have the same length")
    if length == 0:
        return ()
    d = len(outputs[0][0])
    merged: list[Vec] = []
    for k in range(length):
        vecs = [o[k] for o in outputs]
        for v in vecs:
            if len(v) != d:
                raise ValueError("inconsistent feature width across directions")
        if mode == "concat":
            merged.append(tuple(x for v in vecs for x in v))
        elif mode in ("sum", "average"):
            acc = [0.0] * d
            for v in vecs:
                for ch in range(d):
                    acc[ch] += v[ch]
            if mode == "average":
                n = len(vecs)
                acc = [x / n for x in acc]
            merged.append(tuple(acc))
        else:
            raise ValueError(f"unknown merge mode {mode!r}")
    return tuple(merged)


def bidirectional_scan(z_seq: Seq, a_seq: Seq, b_seq: Seq, c_seq: Seq,
                       g_seq: Seq, mode: str = "sum",
                       h0: Vec | None = None) -> Seq:
    """Forward + reverse bidirectional selective scan, merged by ``mode``.

    Runs the forward scan (identity order) and the backward scan (reversed
    order), realigns both to original positions, and merges them.
    """
    length = len(z_seq)
    fwd_order = tuple(range(length))
    rev_order = tuple(reversed(range(length)))
    fwd = scan_direction(z_seq, a_seq, b_seq, c_seq, g_seq, fwd_order, h0=h0)
    rev = scan_direction(z_seq, a_seq, b_seq, c_seq, g_seq, rev_order, h0=h0)
    return merge_directions((fwd, rev), mode=mode)


def multidirectional_scan(z_seq: Seq, a_seq: Seq, b_seq: Seq, c_seq: Seq,
                          g_seq: Seq, orders: tuple[Order, ...],
                          mode: str = "sum", h0: Vec | None = None) -> Seq:
    """General multi-directional scan over an arbitrary set of ``orders``.

    Each order in ``orders`` is a permutation of ``range(len(z_seq))``; the
    per-direction outputs (realigned to original positions) are merged by
    ``mode``.
    """
    if not orders:
        raise ValueError("need at least one scan order")
    dirs = tuple(
        scan_direction(z_seq, a_seq, b_seq, c_seq, g_seq, order, h0=h0)
        for order in orders
    )
    return merge_directions(dirs, mode=mode)
