"""Persistent topological-element naming and query-based entity references.

Two papers argue that the missing ingredient in text-to-CAD command sequences is a
stable way to *refer* to a geometric entity (an edge/face) created by an earlier
feature, so that a later feature (a chamfer, a fillet, "extrude up to that face") keeps
pointing at the right thing when the history is edited -- the classic *topological
naming problem* (TNP):

* **WHUCAD / "AI+CAD Data Representation Architecture"** (Fan, He et al., 2026),
  Sec. 1.3 & 2.3. A feature reference such as a chamfer's edge must carry a *native
  permanent name* built from three things: the modeling history up to that point, the
  topological information, and the geometric information. "Modeling history,
  topological information, and geometric information together construct the native
  permanent naming of this topological element." Flat sketch+extrude command
  sequences lack this, so their references cannot survive an edit.

* **CADFS** (Pyatov et al., 2026), Sec. 3.1 -- Onshape FeatureScript's ``makeQuery``
  addressing scheme: a reference is a 4-tuple ``(operation_id, query_type,
  entity_type, disambiguation)`` -- the creating operation scopes the query, the query
  type is the entity's topological role (e.g. ``SWEPT_EDGE``), the entity type is
  vertex/edge/face/body, and disambiguation resolves ties by ancestors or neighbours.
  "This representation mirrors the way a human would verbally identify a geometric
  feature, i.e. by its origin, semantic role, categorical type, and distinguishing
  attributes."

This module unifies both into a deterministic, hashable :class:`PersistentName` and a
:class:`EntityQuery` resolver over a lightweight feature-history model. It is a
reference-stability layer that sits above the CISP op stream (which addresses entities
by ephemeral index) -- letting the harness detect *dangling* references after an edit
and re-resolve a query against a rebuilt topology. Stdlib only, deterministic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

__all__ = [
    "ENTITY_TYPES",
    "PersistentName",
    "EntityRecord",
    "EntityQuery",
    "make_persistent_name",
    "resolve_query",
    "reference_survives",
]

ENTITY_TYPES: tuple[str, ...] = ("vertex", "edge", "face", "body")


@dataclass(frozen=True)
class PersistentName:
    """WHUCAD native permanent name = history x topology x geometry, hashed stably.

    ``digest`` is a content hash over the three provenance components, so two entities
    with identical creation provenance collide (they are the same entity) and any
    change to history/topology/geometry yields a different name.
    """

    entity_type: str
    creating_op: str
    topological_role: str
    history: tuple[str, ...]
    geometry_key: str
    digest: str

    def __post_init__(self):
        if self.entity_type not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type {self.entity_type!r}")


def make_persistent_name(
    entity_type: str,
    creating_op: str,
    topological_role: str,
    history: Sequence[str],
    geometry_key: str = "",
) -> PersistentName:
    """Construct a :class:`PersistentName` (WHUCAD Sec. 1.3).

    ``history`` is the ordered list of feature-operation ids preceding and including the
    one that created the entity; ``topological_role`` is the entity's role in that
    creating op (CADFS query type, e.g. ``"SWEPT_EDGE"``); ``geometry_key`` is an
    optional stable geometric discriminator (e.g. a rounded centroid string).
    """
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"unknown entity type {entity_type!r}")
    payload = "|".join((
        entity_type, creating_op, topological_role,
        ">".join(history), geometry_key,
    ))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return PersistentName(
        entity_type=entity_type,
        creating_op=creating_op,
        topological_role=topological_role,
        history=tuple(history),
        geometry_key=geometry_key,
        digest=digest,
    )


@dataclass(frozen=True)
class EntityRecord:
    """One resolvable topological entity in a rebuilt feature history."""

    entity_id: str
    entity_type: str
    creating_op: str
    topological_role: str
    ancestors: tuple[str, ...] = ()   # creating ops of parent entities (CADFS disambiguation)
    neighbours: tuple[str, ...] = ()   # adjacent entity ids (topology disambiguation)
    geometry_key: str = ""


@dataclass(frozen=True)
class EntityQuery:
    """CADFS ``makeQuery`` reference: (operation_id, query_type, entity_type, disambig).

    ``disambiguation`` narrows multiple candidates by required ancestor creating-ops
    and/or a required geometry key.
    """

    operation_id: str
    query_type: str
    entity_type: str
    require_ancestors: tuple[str, ...] = ()
    require_geometry_key: Optional[str] = None

    def __post_init__(self):
        if self.entity_type not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type {self.entity_type!r}")


def _matches(query: EntityQuery, rec: EntityRecord) -> bool:
    if rec.creating_op != query.operation_id:
        return False
    if rec.entity_type != query.entity_type:
        return False
    if rec.topological_role != query.query_type:
        return False
    if query.require_ancestors and not set(query.require_ancestors).issubset(rec.ancestors):
        return False
    if query.require_geometry_key is not None and rec.geometry_key != query.require_geometry_key:
        return False
    return True


def resolve_query(query: EntityQuery, records: Iterable[EntityRecord]) -> tuple[str, ...]:
    """Resolve a query against a rebuilt topology; returns matching entity ids.

    A well-posed reference resolves to exactly one entity. Returning zero entities means
    the reference dangled (the creating op or role no longer exists after an edit);
    more than one means the disambiguation was insufficient.
    """
    return tuple(r.entity_id for r in records if _matches(query, r))


def reference_survives(query: EntityQuery, records: Iterable[EntityRecord]) -> bool:
    """True iff the query resolves to exactly one entity after a rebuild (TNP check)."""
    return len(resolve_query(query, records)) == 1
