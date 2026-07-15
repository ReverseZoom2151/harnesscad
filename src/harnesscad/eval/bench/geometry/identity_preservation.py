"""Edit identity-preservation and locality metrics (Liu et al., 2026, "B-repLer:
Language-guided Editing of CAD Models").

B-repLer performs text-driven B-rep editing that "preserves identities in
unchanged areas" -- a good edit modifies only the region the instruction targets
and leaves everything else intact. The latent editing model is out of scope, but
the *evaluation* of an edit -- did the unchanged region survive, and did the
change stay local -- is deterministic set/geometry book-keeping over entity ids
(faces/edges) before and after an edit:

* :func:`identity_preservation` -- of the entities *outside* the intended edit
  region, what fraction are preserved unchanged (Jaccard-style). 1.0 means the
  edit touched nothing it should not have.
* :func:`edit_locality` -- of the entities that actually changed (added / removed
  / modified), what fraction lie inside the intended region. 1.0 means the edit
  is fully localised.
* :func:`edit_report` -- combines both into an identity score plus the raw
  added/removed/modified id sets, mirroring B-repLer's "functional change with
  identity preserved" evaluation.

Deterministic, stdlib-only. Entities are identified by hashable ids; a
``modified`` set (ids present before and after but geometrically changed) is
supplied by the caller (e.g. from a geometric diff).
"""

from __future__ import annotations

from typing import Dict, Iterable, Set

__all__ = ["identity_preservation", "edit_locality", "edit_report"]


def _as_set(ids: Iterable) -> Set:
    return set(ids)


def identity_preservation(
    before: Iterable,
    after: Iterable,
    intended_region: Iterable,
    modified: Iterable = (),
) -> float:
    """Fraction of out-of-region entities that are preserved unchanged (0..1).

    An out-of-region entity is *preserved* iff it exists both before and after
    and is not in ``modified``. If there are no out-of-region entities the edit
    trivially preserves identity (returns 1.0).
    """
    b, a = _as_set(before), _as_set(after)
    region = _as_set(intended_region)
    mod = _as_set(modified)
    outside = b - region
    if not outside:
        return 1.0
    preserved = {e for e in outside if e in a and e not in mod}
    return len(preserved) / len(outside)


def edit_locality(
    before: Iterable,
    after: Iterable,
    intended_region: Iterable,
    modified: Iterable = (),
) -> float:
    """Fraction of actually-changed entities that lie inside the intended region.

    Changed entities are additions ``(after - before)``, removals
    ``(before - after)``, and ``modified``. If nothing changed, locality is 1.0
    (a vacuously local no-op).
    """
    b, a = _as_set(before), _as_set(after)
    region = _as_set(intended_region)
    changed = (a - b) | (b - a) | _as_set(modified)
    if not changed:
        return 1.0
    inside = {e for e in changed if e in region}
    return len(inside) / len(changed)


def edit_report(
    before: Iterable,
    after: Iterable,
    intended_region: Iterable,
    modified: Iterable = (),
    w_preservation: float = 0.5,
) -> Dict[str, object]:
    """Combined identity report: preservation, locality, and a blended score.

    ``identity_score = w_preservation * preservation + (1 - w_preservation) *
    locality``. Also returns the raw ``added`` / ``removed`` / ``modified`` id
    sets for inspection.
    """
    if not 0.0 <= w_preservation <= 1.0:
        raise ValueError("w_preservation must be in [0, 1]")
    b, a = _as_set(before), _as_set(after)
    pres = identity_preservation(before, after, intended_region, modified)
    loc = edit_locality(before, after, intended_region, modified)
    return {
        "preservation": pres,
        "locality": loc,
        "identity_score": w_preservation * pres + (1.0 - w_preservation) * loc,
        "added": a - b,
        "removed": b - a,
        "modified": _as_set(modified),
    }
