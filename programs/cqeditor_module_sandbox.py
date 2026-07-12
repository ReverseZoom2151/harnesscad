"""Snapshot/restore sandbox for a module registry.

CQ-editor re-executes an edited CAD script repeatedly in the same process.  To
stop stale, previously-imported user modules from shadowing fresh edits, it
wraps each run in a context manager that records the set of loaded modules on
entry and deletes every module imported during the run on exit.

This module generalises that idea into a Qt-free, deterministic helper that
works on any mutable mapping (``sys.modules`` by default): it computes the
newly-added keys against an entry snapshot and removes them on exit, so a fresh
run always re-imports.  It is deliberately decoupled from ``sys`` so it can be
exercised on a plain ``dict`` in tests.
"""

import sys
from contextlib import contextmanager


def newly_added_keys(before, after):
    """Return the keys present in *after* but not in *before*, sorted.

    Sorting makes the result deterministic regardless of the mappings' internal
    ordering, which matters when the caller logs or acts on the difference.

    :param before: mapping (or key iterable) representing the entry snapshot.
    :param after: mapping (or key iterable) representing the current state.
    :returns: a sorted ``list`` of the added keys.
    """
    before_keys = set(before)
    after_keys = set(after)
    return sorted(after_keys - before_keys)


@contextmanager
def module_sandbox(registry=None):
    """Unload any modules added to *registry* while the context is active.

    On exit, every key that appeared in *registry* during the ``with`` block is
    deleted, so re-running the guarded code re-imports those modules fresh.
    Keys present before entry are always preserved, even if the guarded code
    replaced their values.

    :param registry: the mutable mapping to guard; defaults to ``sys.modules``.
    :yields: the sorted list of keys that will be unloaded on exit; it is
        updated in place as the block runs, so inspecting it after the block
        reflects what was removed.
    """
    if registry is None:
        registry = sys.modules

    snapshot = set(registry.keys())
    added = []
    try:
        yield added
    finally:
        current = set(registry.keys())
        for name in sorted(current - snapshot):
            added.append(name)
            del registry[name]


def prune_new_modules(registry, snapshot_keys):
    """Delete keys in *registry* that are not in *snapshot_keys*.

    A non-context-manager form of the same operation, useful when the entry
    snapshot was captured separately.  Mutates *registry* in place.

    :param registry: the mutable mapping to prune.
    :param snapshot_keys: iterable of keys to keep.
    :returns: a sorted ``list`` of the removed keys.
    """
    keep = set(snapshot_keys)
    removed = sorted(k for k in list(registry.keys()) if k not in keep)
    for k in removed:
        del registry[k]
    return removed
