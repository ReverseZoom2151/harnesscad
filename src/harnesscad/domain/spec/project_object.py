"""Versioned-namespace project object.

``ProjectObject`` exposes a document through dotted, versioned namespaces
(``product.electrical``, ``project.docs``, ...) with projected payloads and
per-attribute/item metadata (identity, label, kind, and source path). It uses
dataclasses and explicit validation over a plain ``dict`` document, supports
payload redaction for LLM safety, and increments only the namespace touched by
a targeted edit.

Harness gap filled: the harness has rich flat specs (the MGC in
``harnesscad.domain.spec.contract``, the spec registry in
``harnesscad.domain.spec.registry``) but no *addressable, versioned view* of a
whole project document -- nothing an agent can ask "give me product.geometry
at its current version" of. This module supplies that view layer. It
complements (does not duplicate) ``harnesscad.domain.spec.contract``: the MGC
holds measurable acceptance predicates for one part; the project object holds
the navigable namespace decomposition of the whole evolving document that
iteration engines (see ``harnesscad.agents.agent.project_iteration``) target.

Deterministic and stdlib-only: no wall clock (callers pass ``updated_at``),
no randomness, no uuid4 -- the object id falls back to ``"project"`` or a
caller-supplied value in ``metadata["project_id"]``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional


PROJECT_OBJECT_TYPE = "harnesscad.project"
PROJECT_NAMESPACE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(\.[a-z][a-z0-9_-]*)+$")
DEFAULT_MAX_STRING_CHARS = 4000


def normalize_project_namespace(value: Optional[str]) -> Optional[str]:
    """Lowercase and validate a dotted namespace; None/blank passes through as None.

    Raises ValueError when a non-blank value does not match the dotted-lowercase
    grammar (for example ``product.geometry`` or ``project.docs``).
    """
    if value is None:
        return None
    namespace = value.strip().lower()
    if not namespace:
        return None
    if not PROJECT_NAMESPACE_PATTERN.match(namespace):
        raise ValueError(
            "Project namespace must be dotted lowercase, "
            "for example product.geometry or project.docs."
        )
    return namespace


@dataclass(frozen=True)
class ProjectNamespaceDescriptor:
    """Static description of one dotted namespace."""

    name: str
    label: str
    description: str
    scope: str

    def __post_init__(self) -> None:
        normalized = normalize_project_namespace(self.name)
        if normalized is None:
            raise ValueError("Namespace descriptor requires a dotted name.")
        object.__setattr__(self, "name", normalized)
        if not self.scope:
            object.__setattr__(self, "scope", normalized.split(".", 1)[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "scope": self.scope,
        }


class ProjectNamespaceRegistry:
    """Lookup table of known namespace descriptors."""

    def __init__(self, descriptors: Iterable[ProjectNamespaceDescriptor]) -> None:
        self.descriptors = tuple(descriptors)
        self.names = tuple(descriptor.name for descriptor in self.descriptors)

    def get(self, namespace: str) -> Optional[ProjectNamespaceDescriptor]:
        normalized = normalize_project_namespace(namespace)
        if normalized is None:
            return None
        return next(
            (item for item in self.descriptors if item.name == normalized), None
        )

    def contains(self, namespace: str) -> bool:
        return self.get(namespace) is not None


DEFAULT_PROJECT_NAMESPACES = ProjectNamespaceRegistry(
    (
        ProjectNamespaceDescriptor(
            name="project.meta",
            label="Project Metadata",
            description="Object identity, runtime metadata, and workspace-level state.",
            scope="project",
        ),
        ProjectNamespaceDescriptor(
            name="project.docs",
            label="Project Documentation",
            description="Build docs, notes, constraints, and exported guidance.",
            scope="project",
        ),
        ProjectNamespaceDescriptor(
            name="project.history",
            label="Project History",
            description="Version history, iteration decisions, and revision lineage.",
            scope="project",
        ),
        ProjectNamespaceDescriptor(
            name="product.overview",
            label="Product Overview",
            description="Product intent, requirements, constraints, and top-level description.",
            scope="product",
        ),
        ProjectNamespaceDescriptor(
            name="product.geometry",
            label="Product Geometry",
            description="Modeling ops, sketches, and features that define the part geometry.",
            scope="product",
        ),
        ProjectNamespaceDescriptor(
            name="product.mech",
            label="Product Mechanical",
            description="Dimensions, placements, mechanical constraints, and fit.",
            scope="product",
        ),
        ProjectNamespaceDescriptor(
            name="product.fabrication",
            label="Product Fabrication",
            description="Process selection, material, and fabrication notes.",
            scope="product",
        ),
        ProjectNamespaceDescriptor(
            name="product.assembly",
            label="Product Assembly",
            description="Step-by-step physical assembly and build workflow.",
            scope="product",
        ),
        ProjectNamespaceDescriptor(
            name="product.validation",
            label="Product Validation",
            description="Geometry validation, checks, and operation statuses.",
            scope="product",
        ),
        ProjectNamespaceDescriptor(
            name="product.visuals",
            label="Product Visuals",
            description="Generated imagery, render metadata, and presentation assets.",
            scope="product",
        ),
    )
)


def is_known_project_namespace(value: str) -> bool:
    try:
        return DEFAULT_PROJECT_NAMESPACES.contains(value)
    except ValueError:
        return False


def project_namespace_descriptor(namespace: str) -> ProjectNamespaceDescriptor:
    """Descriptor for a namespace, synthesizing one for unknown-but-valid names."""
    normalized = normalize_project_namespace(namespace)
    if normalized is None:
        raise ValueError("Project namespace is required.")
    descriptor = DEFAULT_PROJECT_NAMESPACES.get(normalized)
    if descriptor is not None:
        return descriptor
    label = " ".join(
        part.replace("-", " ").replace("_", " ").title()
        for part in normalized.split(".")
    )
    return ProjectNamespaceDescriptor(
        name=normalized,
        label=label,
        description=f"Custom project object namespace {normalized}.",
        scope=normalized.split(".", 1)[0],
    )


def list_project_namespaces() -> list[ProjectNamespaceDescriptor]:
    return list(DEFAULT_PROJECT_NAMESPACES.descriptors)


# ---------------------------------------------------------------------------
# Attribute / item object model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectAttributeMeta:
    """Metadata for one attribute (top-level payload key) in a namespace."""

    namespace: str
    attribute: str
    label: str
    source_path: str
    value_type: str
    version: int = 1
    item_kind: Optional[str] = None
    item_count: int = 0

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("Attribute version must be >= 1.")
        if self.item_count < 0:
            raise ValueError("Attribute item_count must be >= 0.")


@dataclass(frozen=True)
class ProjectAttributeItemMeta:
    """Metadata for one list item inside an attribute."""

    namespace: str
    attribute: str
    index: int
    item_id: str
    label: str
    source_path: str
    value_type: str
    item_kind: str
    ref_des: Optional[str] = None
    category: Optional[str] = None
    part_number: Optional[str] = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Item index must be >= 0.")


@dataclass
class ProjectAttributeItemObject:
    meta: ProjectAttributeItemMeta
    value: Any = None

    @property
    def item_id(self) -> str:
        return self.meta.item_id

    @property
    def label(self) -> str:
        return self.meta.label

    @property
    def item_kind(self) -> str:
        return self.meta.item_kind


@dataclass
class ProjectAttributeObject:
    meta: ProjectAttributeMeta
    value: Any = None
    items: list[ProjectAttributeItemObject] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.meta.attribute

    @property
    def label(self) -> str:
        return self.meta.label

    def get_item(self, item_id: str) -> Optional[ProjectAttributeItemObject]:
        normalized = str(item_id).strip().lower()
        if not normalized:
            return None
        return next(
            (item for item in self.items if item.item_id.lower() == normalized), None
        )


@dataclass
class ProjectNamespaceObject:
    name: str
    label: str
    description: str
    scope: str
    version: int = 1
    payload: dict[str, Any] = field(default_factory=dict)
    attributes: list[ProjectAttributeObject] = field(default_factory=list)

    def get_attribute(self, attribute: str) -> Optional[ProjectAttributeObject]:
        normalized = str(attribute).strip()
        if not normalized:
            return None
        return next(
            (item for item in self.attributes if item.name == normalized), None
        )


@dataclass
class ProjectObject:
    """The whole-document namespace decomposition."""

    object_id: str
    version: int = 1
    object_type: str = PROJECT_OBJECT_TYPE
    namespaces: list[ProjectNamespaceObject] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_namespace(self, namespace: str) -> Optional[ProjectNamespaceObject]:
        normalized = normalize_project_namespace(namespace)
        if normalized is None:
            return None
        return next(
            (item for item in self.namespaces if item.name == normalized), None
        )

    def get_attribute(
        self, namespace: str, attribute: str
    ) -> Optional[ProjectAttributeObject]:
        namespace_object = self.get_namespace(namespace)
        if namespace_object is None:
            return None
        return namespace_object.get_attribute(attribute)

    def get_item(
        self, namespace: str, attribute: str, item_id: str
    ) -> Optional[ProjectAttributeItemObject]:
        attribute_object = self.get_attribute(namespace, attribute)
        if attribute_object is None:
            return None
        return attribute_object.get_item(item_id)


# ---------------------------------------------------------------------------
# Redaction and typing helpers
# ---------------------------------------------------------------------------


def redact_payload_value(
    value: Any, *, key: str = "", max_string_chars: int = DEFAULT_MAX_STRING_CHARS
) -> Any:
    """Recursively redact data URLs and truncate oversized strings.

    Strings that begin with ``data:`` under keys containing image/data/visual
    become ``<redacted data url: N chars>``; any string longer than
    ``max_string_chars`` is truncated with a ``...<truncated N chars>`` suffix.
    """
    lowered_key = key.lower()
    if isinstance(value, dict):
        return {
            item_key: redact_payload_value(
                item_value, key=str(item_key), max_string_chars=max_string_chars
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            redact_payload_value(item, max_string_chars=max_string_chars)
            for item in value
        ]
    if isinstance(value, str):
        if (
            "image" in lowered_key or "data" in lowered_key or "visual" in lowered_key
        ) and value.startswith("data:"):
            return f"<redacted data url: {len(value)} chars>"
        if len(value) > max_string_chars:
            overflow = len(value) - max_string_chars
            return value[:max_string_chars] + f"...<truncated {overflow} chars>"
    return value


def json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return value.__class__.__name__


# ---------------------------------------------------------------------------
# Item identity / label / kind inference
# ---------------------------------------------------------------------------

ITEM_IDENTITY_KEYS = (
    "ref_des",
    "net_id",
    "op",
    "id",
    "step_num",
    "part_number",
    "name",
    "title",
)

ITEM_KIND_BY_ATTRIBUTE = {
    "assembly": "assembly_step",
    "components": "component",
    "constraints": "constraint",
    "critical": "validation_issue",
    "fabrication_notes": "fabrication_note",
    "features": "feature",
    "history": "history_entry",
    "info": "validation_issue",
    "issues": "validation_issue",
    "ops": "op",
    "requirements": "requirement",
    "sketches": "sketch",
    "steps": "step",
    "warning": "validation_issue",
    "warnings": "validation_issue",
}


def _title_from_key(value: str) -> str:
    return " ".join(
        piece for piece in value.replace("-", "_").split("_") if piece
    ).title()


def attribute_item_kind(attribute: str, item: Any) -> str:
    if attribute in ITEM_KIND_BY_ATTRIBUTE:
        return ITEM_KIND_BY_ATTRIBUTE[attribute]
    if isinstance(item, Mapping) and item.get("ref_des"):
        return "component"
    if attribute.endswith("s") and len(attribute) > 1:
        return attribute[:-1]
    return "item"


def item_identity(attribute: str, index: int, item: Any) -> str:
    if isinstance(item, Mapping):
        for key in ITEM_IDENTITY_KEYS:
            value = item.get(key)
            if value not in (None, ""):
                return str(value)
    return f"{attribute}_{index + 1}"


def item_label(attribute: str, item_id: str, item: Any) -> str:
    if isinstance(item, Mapping):
        for key in ("label", "name", "title", "description", "part_number", "op"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:120]
        if item.get("ref_des"):
            return str(item.get("ref_des"))
    if isinstance(item, str) and item.strip():
        return item.strip()[:120]
    return item_id


def _item_meta(
    namespace: str, attribute: str, item: Any, index: int
) -> ProjectAttributeItemMeta:
    item_id = item_identity(attribute, index, item)
    is_mapping = isinstance(item, Mapping)
    return ProjectAttributeItemMeta(
        namespace=namespace,
        attribute=attribute,
        index=index,
        item_id=item_id,
        label=item_label(attribute, item_id, item),
        source_path=f"{namespace}.{attribute}[{index}]",
        value_type=json_value_type(item),
        item_kind=attribute_item_kind(attribute, item),
        ref_des=str(item.get("ref_des")) if is_mapping and item.get("ref_des") else None,
        category=str(item.get("category")) if is_mapping and item.get("category") else None,
        part_number=(
            str(item.get("part_number"))
            if is_mapping and item.get("part_number")
            else None
        ),
    )


def _attribute_items(
    namespace: str, attribute: str, value: Any
) -> list[ProjectAttributeItemObject]:
    if not isinstance(value, list):
        return []
    return [
        ProjectAttributeItemObject(
            meta=_item_meta(namespace, attribute, item, index), value=item
        )
        for index, item in enumerate(value)
    ]


def build_project_attribute_objects(
    namespace: str, payload: Mapping[str, Any], *, version: int = 1
) -> list[ProjectAttributeObject]:
    attributes: list[ProjectAttributeObject] = []
    for attribute, value in payload.items():
        attribute = str(attribute)
        items = _attribute_items(namespace, attribute, value)
        attributes.append(
            ProjectAttributeObject(
                meta=ProjectAttributeMeta(
                    namespace=namespace,
                    attribute=attribute,
                    label=_title_from_key(attribute),
                    source_path=f"{namespace}.{attribute}",
                    value_type=json_value_type(value),
                    version=version,
                    item_kind=items[0].item_kind if items else None,
                    item_count=len(items),
                ),
                value=value,
                items=items,
            )
        )
    return attributes


# ---------------------------------------------------------------------------
# Namespace payload projection
# ---------------------------------------------------------------------------

Projector = Callable[[Mapping[str, Any]], dict[str, Any]]


def _document_metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _project_meta_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _document_metadata(document)
    return {
        "metadata": metadata,
        "project_id": metadata.get("project_id"),
        "revision": metadata.get("revision"),
        "workflow": metadata.get("workflow"),
        "source_usage": metadata.get("source_usage"),
    }


def _project_docs_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "docs": document.get("docs") or [],
        "notes": document.get("notes") or [],
        "fabrication_notes": document.get("fabrication_notes") or [],
        "constraints": document.get("constraints") or [],
    }


def _project_history_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    return {"history": document.get("history") or []}


def _product_overview_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "overview": document.get("overview"),
        "requirements": document.get("requirements") or [],
        "constraints": document.get("constraints") or [],
    }


def _product_geometry_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ops": document.get("ops") or [],
        "sketches": document.get("sketches") or [],
        "features": document.get("features") or [],
    }


def _product_mech_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _document_metadata(document)
    return {
        "mechanical": document.get("mechanical"),
        "dimensions": document.get("dimensions") or {},
        "placements": document.get("placements") or [],
        "render_dimensions": metadata.get("render_dimensions"),
    }


def _product_fabrication_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fabrication": document.get("fabrication"),
        "fabrication_notes": document.get("fabrication_notes") or [],
        "process": document.get("process"),
        "material": document.get("material"),
    }


def _product_assembly_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    return {"assembly": document.get("assembly") or []}


def _product_validation_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _document_metadata(document)
    return {
        "validation": document.get("validation") or {},
        "is_valid": document.get("is_valid"),
        "operation_statuses": metadata.get("operation_statuses") or [],
        "operation_summary": metadata.get("operation_summary"),
    }


def _product_visuals_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _document_metadata(document)
    fragments = ("image", "visual", "render")
    return {
        key: value
        for key, value in metadata.items()
        if any(fragment in str(key).lower() for fragment in fragments)
    }


DEFAULT_NAMESPACE_PROJECTORS: dict[str, Projector] = {
    "project.meta": _project_meta_payload,
    "project.docs": _project_docs_payload,
    "project.history": _project_history_payload,
    "product.overview": _product_overview_payload,
    "product.geometry": _product_geometry_payload,
    "product.mech": _product_mech_payload,
    "product.fabrication": _product_fabrication_payload,
    "product.assembly": _product_assembly_payload,
    "product.validation": _product_validation_payload,
    "product.visuals": _product_visuals_payload,
}


def namespace_payload(
    document: Mapping[str, Any],
    namespace: str,
    projectors: Optional[Mapping[str, Projector]] = None,
) -> dict[str, Any]:
    """Project the slice of ``document`` owned by ``namespace``, redacted.

    Precedence: explicit ``metadata["namespace_payloads"][namespace]`` dict in
    the document, then a caller-supplied projector, then the default CAD
    projector set, then an empty payload.
    """
    normalized = normalize_project_namespace(namespace)
    if normalized is None:
        raise ValueError("Project namespace is required.")
    if not isinstance(document, Mapping):
        raise TypeError("Project document must be a mapping.")

    metadata = _document_metadata(document)
    custom_payloads = metadata.get("namespace_payloads")
    if isinstance(custom_payloads, dict) and isinstance(
        custom_payloads.get(normalized), dict
    ):
        return redact_payload_value(custom_payloads[normalized])

    projector: Optional[Projector] = None
    if projectors is not None:
        projector = projectors.get(normalized)
    if projector is None:
        projector = DEFAULT_NAMESPACE_PROJECTORS.get(normalized)
    selected = projector(document) if projector is not None else {}
    if not isinstance(selected, dict):
        raise TypeError(
            f"Namespace projector for {normalized!r} must return a dict, "
            f"got {type(selected).__name__}."
        )
    return redact_payload_value(selected)


# ---------------------------------------------------------------------------
# Project object assembly and metadata attachment
# ---------------------------------------------------------------------------


def project_object_version(document: Mapping[str, Any]) -> int:
    """Current revision: metadata["revision"], else history length, min 1."""
    metadata = _document_metadata(document)
    raw_value = metadata.get("revision")
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        history = document.get("history")
        return max(1, len(history) if isinstance(history, list) else 0)


def _canonical_object_id(value: Any) -> str:
    text = str(value).strip() if value not in (None, "") else ""
    return text or "project"


def _project_object_metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _document_metadata(document)
    raw_object = metadata.get("project_object")
    return dict(raw_object) if isinstance(raw_object, dict) else {}


def _namespace_names_for_document(
    document: Mapping[str, Any], target_namespace: Optional[str] = None
) -> list[str]:
    object_metadata = _project_object_metadata(document)
    raw_versions = object_metadata.get("namespace_versions")
    previous_names = list(raw_versions.keys()) if isinstance(raw_versions, dict) else []
    names = [*DEFAULT_PROJECT_NAMESPACES.names, *previous_names]
    normalized_target = normalize_project_namespace(target_namespace)
    if normalized_target:
        names.append(normalized_target)
    return sorted(dict.fromkeys(names))


def _namespace_versions(
    document: Mapping[str, Any], namespaces: Iterable[str]
) -> dict[str, int]:
    object_metadata = _project_object_metadata(document)
    raw_versions = object_metadata.get("namespace_versions")
    previous_versions = raw_versions if isinstance(raw_versions, dict) else {}
    project_version = project_object_version(document)
    versions: dict[str, int] = {}
    for namespace in namespaces:
        raw_value = previous_versions.get(namespace)
        try:
            versions[namespace] = max(1, int(raw_value))
        except (TypeError, ValueError):
            versions[namespace] = project_version
    return versions


def build_project_object(
    document: Mapping[str, Any],
    *,
    target_namespace: Optional[str] = None,
    projectors: Optional[Mapping[str, Projector]] = None,
) -> ProjectObject:
    """Decompose a plain-dict project document into a ProjectObject."""
    if not isinstance(document, Mapping):
        raise TypeError("Project document must be a mapping.")
    namespace_names = _namespace_names_for_document(
        document, target_namespace=target_namespace
    )
    versions = _namespace_versions(document, namespace_names)
    metadata = _document_metadata(document)
    object_id = _canonical_object_id(metadata.get("project_id"))
    project_version = project_object_version(document)

    namespaces: list[ProjectNamespaceObject] = []
    for namespace in namespace_names:
        descriptor = project_namespace_descriptor(namespace)
        version = versions[namespace]
        payload = namespace_payload(document, namespace, projectors=projectors)
        namespaces.append(
            ProjectNamespaceObject(
                name=namespace,
                label=descriptor.label,
                description=descriptor.description,
                scope=descriptor.scope,
                version=version,
                payload=payload,
                attributes=build_project_attribute_objects(
                    namespace, payload, version=version
                ),
            )
        )

    return ProjectObject(
        object_id=object_id,
        version=project_version,
        namespaces=namespaces,
        metadata={
            "project_id": object_id,
            "revision": project_version,
            "updated_at": metadata.get("iterated_at")
            or metadata.get("generated_at")
            or "",
            "namespace_versions": versions,
        },
    )


def attach_project_object_metadata(
    document: Mapping[str, Any],
    *,
    target_namespace: Optional[str] = None,
    updated_at: str = "",
) -> dict[str, Any]:
    """Return a shallow copy of ``document`` with ``metadata["project_object"]``.

    The block records object_id (caller-supplied ``metadata["project_id"]`` or
    the deterministic fallback ``"project"``), the current revision, the sorted
    namespace list, and per-namespace versions: with a target namespace only
    that namespace is bumped to the current revision (others carry their
    previous version), without one every namespace is stamped at the revision.
    ``updated_at`` is caller-injected; no wall clock is read here.
    """
    if not isinstance(document, Mapping):
        raise TypeError("Project document must be a mapping.")
    normalized_target = normalize_project_namespace(target_namespace)
    namespace_names = _namespace_names_for_document(
        document, target_namespace=normalized_target
    )
    project_version = project_object_version(document)
    if normalized_target:
        versions = _namespace_versions(document, namespace_names)
        versions[normalized_target] = project_version
    else:
        versions = {namespace: project_version for namespace in namespace_names}

    revised = dict(document)
    metadata = _document_metadata(document)
    object_id = _canonical_object_id(metadata.get("project_id"))
    metadata["project_id"] = object_id
    metadata["project_object"] = {
        "object_type": PROJECT_OBJECT_TYPE,
        "object_id": object_id,
        "version": project_version,
        "namespaces": namespace_names,
        "namespace_versions": versions,
        "target_namespace": normalized_target,
        "updated_at": updated_at,
    }
    revised["metadata"] = metadata
    return revised


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------


def _sample_document() -> dict[str, Any]:
    return {
        "overview": {"title": "Bracket", "description": "L-bracket with slots"},
        "ops": [
            {"op": "sketch_rect", "id": "op1", "w": 40, "h": 20},
            {"op": "extrude", "id": "op2", "depth": 5},
        ],
        "sketches": [{"name": "base_profile"}],
        "features": [{"name": "slot", "count": 2}],
        "assembly": [{"step_num": 1, "title": "Deburr edges"}],
        "validation": {"critical": [], "warning": []},
        "is_valid": True,
        "history": [{"version": "0.1", "revision": 1, "description": "initial"}],
        "metadata": {
            "revision": 2,
            "thumbnail_image": "data:image/png;base64," + "A" * 64,
        },
    }


def _run_selfcheck() -> int:
    # Namespace normalization.
    assert normalize_project_namespace(" Product.Geometry ") == "product.geometry"
    assert normalize_project_namespace(None) is None
    assert normalize_project_namespace("  ") is None
    try:
        normalize_project_namespace("NoDots")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for undotted namespace")
    assert is_known_project_namespace("product.geometry")
    assert not is_known_project_namespace("product.unknown_zone")

    # Redaction.
    redacted = redact_payload_value(
        {"thumbnail_image": "data:image/png;base64,AAAA", "note": "x" * 10},
        max_string_chars=4,
    )
    assert redacted["thumbnail_image"] == "<redacted data url: 26 chars>"
    assert redacted["note"] == "xxxx...<truncated 6 chars>"
    assert json_value_type([]) == "array" and json_value_type(True) == "boolean"

    # Build a project object over the sample CAD document.
    document = _sample_document()
    project = build_project_object(document)
    assert project.object_id == "project"  # deterministic fallback, no uuid4
    assert project.version == 2
    geometry = project.get_namespace("product.geometry")
    assert geometry is not None and geometry.version == 2
    ops = geometry.get_attribute("ops")
    assert ops is not None and ops.meta.item_count == 2
    assert ops.meta.item_kind == "op"
    first = ops.get_item("sketch_rect")
    assert first is not None and first.meta.source_path == "product.geometry.ops[0]"
    step = project.get_item("product.assembly", "assembly", "1")
    assert step is not None and step.item_kind == "assembly_step"
    meta_ns = project.get_namespace("project.meta")
    assert meta_ns is not None
    thumb = meta_ns.payload["metadata"]["thumbnail_image"]
    assert thumb.startswith("<redacted data url:")

    # Metadata attach + targeted version bump.
    stamped = attach_project_object_metadata(
        document, target_namespace="product.geometry", updated_at="2026-01-01T00:00:00Z"
    )
    block = stamped["metadata"]["project_object"]
    assert block["object_id"] == "project"
    assert block["version"] == 2
    assert block["namespace_versions"]["product.geometry"] == 2
    assert block["updated_at"] == "2026-01-01T00:00:00Z"
    assert "product.geometry" in block["namespaces"]

    # Second attach after a revision bump: only the target moves.
    stamped["metadata"]["revision"] = 3
    restamped = attach_project_object_metadata(
        stamped, target_namespace="product.geometry", updated_at="2026-01-02T00:00:00Z"
    )
    reblock = restamped["metadata"]["project_object"]
    assert reblock["namespace_versions"]["product.geometry"] == 3
    assert reblock["namespace_versions"]["project.docs"] == 2

    # Pluggable projector override.
    payload = namespace_payload(
        document,
        "product.geometry",
        projectors={"product.geometry": lambda doc: {"ops": doc.get("ops") or []}},
    )
    assert list(payload.keys()) == ["ops"]

    print(
        "PASS project_object selfcheck: namespaces=%d version=%d object_id=%s"
        % (len(project.namespaces), project.version, project.object_id)
    )
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad.domain.spec.project_object",
        description="Versioned-namespace project object.",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="run deterministic assertions over a sample CAD document.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the registry descriptors as JSON.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.json:
        print(
            json.dumps(
                [descriptor.as_dict() for descriptor in list_project_namespaces()],
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _run_selfcheck()
    except AssertionError as exc:
        print(f"SELFCHECK FAILED: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "DEFAULT_NAMESPACE_PROJECTORS",
    "DEFAULT_PROJECT_NAMESPACES",
    "ITEM_IDENTITY_KEYS",
    "ITEM_KIND_BY_ATTRIBUTE",
    "PROJECT_NAMESPACE_PATTERN",
    "PROJECT_OBJECT_TYPE",
    "ProjectAttributeItemMeta",
    "ProjectAttributeItemObject",
    "ProjectAttributeMeta",
    "ProjectAttributeObject",
    "ProjectNamespaceDescriptor",
    "ProjectNamespaceObject",
    "ProjectNamespaceRegistry",
    "ProjectObject",
    "attach_project_object_metadata",
    "attribute_item_kind",
    "build_project_attribute_objects",
    "build_project_object",
    "is_known_project_namespace",
    "item_identity",
    "item_label",
    "json_value_type",
    "list_project_namespaces",
    "main",
    "namespace_payload",
    "normalize_project_namespace",
    "project_namespace_descriptor",
    "project_object_version",
    "redact_payload_value",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
