"""Design-artifact / parameter-patch contract for an external correction loop.

OpenCAD ships a versioned JSON handshake (``opencad/design_artifact.py`` plus the
``caid-design-artifact-v1`` / ``caid-design-patch-v1`` JSON schemas) between the
CAD side and a downstream simulator ("SimCorrect"): the CAD side exports a
*design artifact* -- feature tree, named design parameters, and *simulation
tags* mapping design names to simulator names -- and the simulator sends back a
*design patch*: structured parameter changes against those names, never raw
mesh or geometry edits. The correction loop therefore operates on the
parametric model's own vocabulary ("forearm_length: 0.25 -> 0.30"), which is
exactly the kind of feedback the harness's PDD gate wants to consume.

Three rules make the handshake sound, and all three are enforced here:

* **identity** -- a patch names the ``artifact_id`` it was computed against and
  is refused for any other artifact, so a correction cannot land on the wrong
  design;
* **compare-and-swap** -- a patch item may carry ``old_value``; when present it
  must equal the artifact's current value or the patch is refused. This is the
  staleness guard: a correction computed against design state N cannot silently
  overwrite state N+1;
* **immutability** -- applying a patch returns a NEW artifact; the input object
  is never mutated, so a refused multi-item patch leaves no half-applied state.

Simulation tags are the aliasing half: the simulator reports faults against ITS
names (``link2_length``); ``resolve_simulation_target`` walks the
``kind="parameter"`` tags back to the design name (``forearm_length``) so no
alias table ever lives in diagnosis code.

Relation to neighbours: :mod:`harnesscad.domain.spec.contract` (the Measured
Geometric Contract) states what a part must MEASURE as; this module is the
transport for what an external verifier says the parameters should BECOME. The
feature tree payload is carried opaquely (any JSON object with ``root_id`` +
``nodes``), so it composes with :mod:`harnesscad.core.state.feature_tree` without
depending on it.

Stdlib-only, deterministic: JSON serialisation is sorted-key; ``created_at`` is
caller-supplied (no wall clock) and defaults to a fixed epoch string.

Public API
----------
``DesignParameter``, ``SimulationTag``, ``DesignArtifact``
``ParameterPatch``, ``DesignPatch``
``build_artifact``, ``apply_patch``, ``resolve_simulation_target``
``validate_artifact_payload``, ``validate_patch_payload``
``artifact_to_payload``, ``artifact_from_payload``
``patch_to_payload``, ``patch_from_payload``
``ArtifactError``, ``PatchError``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "SIMULATION_TAG_KINDS",
    "ArtifactError",
    "PatchError",
    "DesignParameter",
    "SimulationTag",
    "DesignArtifact",
    "ParameterPatch",
    "DesignPatch",
    "build_artifact",
    "apply_patch",
    "resolve_simulation_target",
    "validate_artifact_payload",
    "validate_patch_payload",
    "artifact_to_payload",
    "artifact_from_payload",
    "patch_to_payload",
    "patch_from_payload",
]

ARTIFACT_SCHEMA_VERSION = 1

#: Simulator entity kinds a tag may target. ``parameter`` is the one the patch
#: loop resolves through; the body/joint/geom/site kinds label geometry for the
#: simulator's scene description.
SIMULATION_TAG_KINDS = ("body", "joint", "geom", "site", "parameter")

_ARTIFACT_REQUIRED = (
    "schema_version", "artifact_id", "producer", "created_at",
    "feature_tree", "parameters", "simulation_tags",
)
_PATCH_REQUIRED = ("schema_version", "artifact_id", "source", "parameter_patches")

_EPOCH = "1970-01-01T00:00:00Z"

ParameterValue = Union[bool, int, float, str]


class ArtifactError(ValueError):
    """The artifact payload is malformed."""


class PatchError(ValueError):
    """The patch is malformed or cannot be applied to this artifact."""


def _check_value(value: object, where: str) -> ParameterValue:
    if isinstance(value, (bool, int, float, str)):
        return value
    raise ArtifactError(
        "%s must be a bool/int/float/str, got %s" % (where, type(value).__name__)
    )


@dataclass(frozen=True)
class DesignParameter:
    """A named, patchable design-level value.

    ``role`` is a free label ("geometry", "material", ...); ``feature_id``
    optionally anchors the parameter to the feature-tree node it drives.
    """

    name: str
    value: ParameterValue
    unit: Optional[str] = None
    role: Optional[str] = None
    feature_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ArtifactError("DesignParameter requires a non-empty name.")
        _check_value(self.value, "DesignParameter '%s' value" % self.name)


@dataclass(frozen=True)
class SimulationTag:
    """Maps a design name to a simulator name, so no alias table is hardcoded."""

    name: str
    kind: str
    target: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.target:
            raise ArtifactError("SimulationTag requires name and target.")
        if self.kind not in SIMULATION_TAG_KINDS:
            raise ArtifactError(
                "SimulationTag kind %r not in %s" % (self.kind, SIMULATION_TAG_KINDS)
            )


@dataclass(frozen=True)
class DesignArtifact:
    """The versioned handoff object the CAD side exports.

    ``feature_tree`` is an opaque JSON object required to carry ``root_id`` and
    ``nodes``; the contract does not interpret it further.
    """

    artifact_id: str
    feature_tree: Mapping[str, object]
    parameters: Mapping[str, DesignParameter] = field(default_factory=dict)
    simulation_tags: Tuple[SimulationTag, ...] = ()
    producer: Mapping[str, str] = field(
        default_factory=lambda: {"name": "harnesscad", "version": "0"}
    )
    created_at: str = _EPOCH
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactError(
                "Unsupported artifact schema_version %r (supported: %d)."
                % (self.schema_version, ARTIFACT_SCHEMA_VERSION)
            )
        if not self.artifact_id:
            raise ArtifactError("Artifact requires a non-empty artifact_id.")
        if not isinstance(self.feature_tree, Mapping):
            raise ArtifactError("feature_tree must be a JSON object.")
        for key in ("root_id", "nodes"):
            if key not in self.feature_tree:
                raise ArtifactError("feature_tree missing required key '%s'." % key)
        for name, parameter in self.parameters.items():
            if parameter.name != name:
                raise ArtifactError(
                    "Parameter map key '%s' does not match parameter name '%s'."
                    % (name, parameter.name)
                )

    def parameter_values(self) -> Dict[str, ParameterValue]:
        return {name: p.value for name, p in sorted(self.parameters.items())}


@dataclass(frozen=True)
class ParameterPatch:
    """One parameter change: name, new value, optional CAS guard and reason."""

    name: str
    value: ParameterValue
    old_value: Optional[ParameterValue] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise PatchError("ParameterPatch requires a non-empty name.")
        if not isinstance(self.value, (bool, int, float, str)):
            raise PatchError(
                "ParameterPatch '%s' value must be bool/int/float/str." % self.name
            )


@dataclass(frozen=True)
class DesignPatch:
    """A structured correction against a named artifact's parameters."""

    artifact_id: str
    parameter_patches: Tuple[ParameterPatch, ...]
    source: str = "external"
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise PatchError(
                "Unsupported patch schema_version %r (supported: %d)."
                % (self.schema_version, ARTIFACT_SCHEMA_VERSION)
            )
        if not self.artifact_id:
            raise PatchError("Patch requires a non-empty artifact_id.")
        if not self.source:
            raise PatchError("Patch requires a non-empty source.")
        if not self.parameter_patches:
            raise PatchError("Patch requires at least one parameter patch.")


def build_artifact(
    *,
    artifact_id: str,
    feature_tree: Mapping[str, object],
    parameters: Optional[Mapping[str, object]] = None,
    simulation_tags: Optional[Iterable[Mapping[str, object]]] = None,
    producer: Optional[Mapping[str, str]] = None,
    created_at: str = _EPOCH,
) -> DesignArtifact:
    """Build an artifact from loose payloads.

    ``parameters`` accepts either bare values (``{"w": 30.0}``) or parameter
    dicts (``{"w": {"value": 30.0, "unit": "mm"}}``); the map key supplies the
    name in both forms.
    """
    mapped: Dict[str, DesignParameter] = {}
    for name, raw in (parameters or {}).items():
        if isinstance(raw, Mapping):
            payload = dict(raw)
            payload.setdefault("name", name)
            mapped[str(name)] = DesignParameter(
                name=str(payload["name"]),
                value=_check_value(payload.get("value"), "parameter '%s'" % name),
                unit=payload.get("unit"),
                role=payload.get("role"),
                feature_id=payload.get("feature_id"),
            )
        else:
            mapped[str(name)] = DesignParameter(
                name=str(name), value=_check_value(raw, "parameter '%s'" % name)
            )

    tags: List[SimulationTag] = []
    for tag in simulation_tags or []:
        tags.append(SimulationTag(
            name=str(tag["name"]),
            kind=str(tag["kind"]),
            target=str(tag["target"]),
            metadata=dict(tag.get("metadata", {})),
        ))

    kwargs: Dict[str, object] = dict(
        artifact_id=artifact_id,
        feature_tree=dict(feature_tree),
        parameters=mapped,
        simulation_tags=tuple(tags),
        created_at=created_at,
    )
    if producer is not None:
        kwargs["producer"] = dict(producer)
    return DesignArtifact(**kwargs)  # type: ignore[arg-type]


def apply_patch(
    artifact: DesignArtifact, patch: Union[DesignPatch, Mapping[str, object]]
) -> DesignArtifact:
    """Apply a patch, returning a NEW artifact; refuse rather than half-apply.

    Refusals (all :class:`PatchError`): wrong ``artifact_id``; an unknown
    parameter name; a stale ``old_value`` that does not match the artifact's
    current value. Every item is checked before anything is written, so a
    refused patch changes nothing.
    """
    if isinstance(patch, Mapping):
        patch = patch_from_payload(patch)
    if patch.artifact_id != artifact.artifact_id:
        raise PatchError(
            "Patch targets '%s', not '%s'."
            % (patch.artifact_id, artifact.artifact_id)
        )

    # Validate every item first (all-or-nothing).
    for item in patch.parameter_patches:
        current = artifact.parameters.get(item.name)
        if current is None:
            raise PatchError("Unknown design parameter '%s'." % item.name)
        if item.old_value is not None and item.old_value != current.value:
            raise PatchError(
                "Patch for '%s' expected old value %r, but artifact has %r."
                % (item.name, item.old_value, current.value)
            )

    parameters: Dict[str, DesignParameter] = dict(artifact.parameters)
    for item in patch.parameter_patches:
        current = parameters[item.name]
        parameters[item.name] = DesignParameter(
            name=current.name,
            value=item.value,
            unit=current.unit,
            role=current.role,
            feature_id=current.feature_id,
        )

    return DesignArtifact(
        artifact_id=artifact.artifact_id,
        feature_tree=artifact.feature_tree,
        parameters=parameters,
        simulation_tags=artifact.simulation_tags,
        producer=artifact.producer,
        created_at=artifact.created_at,
        schema_version=artifact.schema_version,
    )


def resolve_simulation_target(
    artifact: DesignArtifact, simulator_name: str
) -> Optional[str]:
    """Resolve a simulator-side parameter name back to the design name.

    Walks ``kind="parameter"`` tags whose ``target`` equals *simulator_name*;
    returns the design-side ``name``, or ``None`` when no tag maps it. Multiple
    tags claiming the same target are a modelling error and are refused.
    """
    hits = [
        tag.name for tag in artifact.simulation_tags
        if tag.kind == "parameter" and tag.target == simulator_name
    ]
    if len(hits) > 1:
        raise ArtifactError(
            "Simulator target '%s' is claimed by %d tags: %s"
            % (simulator_name, len(hits), ", ".join(sorted(hits)))
        )
    return hits[0] if hits else None


# ── payload (JSON) round trip ───────────────────────────────────────


def validate_artifact_payload(payload: object) -> None:
    """Structural check on a raw artifact payload; raises :class:`ArtifactError`."""
    if not isinstance(payload, Mapping):
        raise ArtifactError("Design artifact must be a JSON object.")
    missing = sorted(set(_ARTIFACT_REQUIRED) - set(payload.keys()))
    if missing:
        raise ArtifactError(
            "Design artifact missing required key(s): %s." % ", ".join(missing)
        )
    producer = payload["producer"]
    if not isinstance(producer, Mapping) or not producer.get("name") \
            or not producer.get("version"):
        raise ArtifactError("Producer requires name and version.")


def validate_patch_payload(payload: object) -> None:
    """Structural check on a raw patch payload; raises :class:`PatchError`."""
    if not isinstance(payload, Mapping):
        raise PatchError("Design patch must be a JSON object.")
    missing = sorted(set(_PATCH_REQUIRED) - set(payload.keys()))
    if missing:
        raise PatchError(
            "Design patch missing required key(s): %s." % ", ".join(missing)
        )
    if not isinstance(payload["parameter_patches"], Sequence) \
            or isinstance(payload["parameter_patches"], (str, bytes)):
        raise PatchError("parameter_patches must be an array.")


def artifact_to_payload(artifact: DesignArtifact) -> Dict[str, object]:
    return {
        "schema_version": artifact.schema_version,
        "artifact_id": artifact.artifact_id,
        "producer": dict(artifact.producer),
        "created_at": artifact.created_at,
        "feature_tree": dict(artifact.feature_tree),
        "parameters": {
            name: {
                "name": p.name,
                "value": p.value,
                "unit": p.unit,
                "role": p.role,
                "feature_id": p.feature_id,
            }
            for name, p in sorted(artifact.parameters.items())
        },
        "simulation_tags": [
            {
                "name": t.name,
                "kind": t.kind,
                "target": t.target,
                "metadata": dict(t.metadata),
            }
            for t in artifact.simulation_tags
        ],
    }


def artifact_from_payload(payload: Mapping[str, object]) -> DesignArtifact:
    validate_artifact_payload(payload)
    return build_artifact(
        artifact_id=str(payload["artifact_id"]),
        feature_tree=payload["feature_tree"],  # type: ignore[arg-type]
        parameters=payload["parameters"],  # type: ignore[arg-type]
        simulation_tags=payload["simulation_tags"],  # type: ignore[arg-type]
        producer=payload["producer"],  # type: ignore[arg-type]
        created_at=str(payload["created_at"]),
    ) if int(payload["schema_version"]) == ARTIFACT_SCHEMA_VERSION else _refuse(
        payload["schema_version"]
    )


def _refuse(version: object) -> DesignArtifact:
    raise ArtifactError(
        "Unsupported artifact schema_version %r (supported: %d)."
        % (version, ARTIFACT_SCHEMA_VERSION)
    )


def patch_to_payload(patch: DesignPatch) -> Dict[str, object]:
    return {
        "schema_version": patch.schema_version,
        "artifact_id": patch.artifact_id,
        "source": patch.source,
        "parameter_patches": [
            {
                "name": item.name,
                "value": item.value,
                "old_value": item.old_value,
                "reason": item.reason,
            }
            for item in patch.parameter_patches
        ],
    }


def patch_from_payload(payload: Mapping[str, object]) -> DesignPatch:
    validate_patch_payload(payload)
    items: List[ParameterPatch] = []
    for raw in payload["parameter_patches"]:  # type: ignore[union-attr]
        if not isinstance(raw, Mapping):
            raise PatchError("Each parameter patch must be a JSON object.")
        items.append(ParameterPatch(
            name=str(raw["name"]),
            value=raw["value"],  # type: ignore[arg-type]
            old_value=raw.get("old_value"),
            reason=raw.get("reason"),
        ))
    return DesignPatch(
        artifact_id=str(payload["artifact_id"]),
        parameter_patches=tuple(items),
        source=str(payload["source"]),
        schema_version=int(payload["schema_version"]),  # type: ignore[arg-type]
    )


def dumps(payload: Mapping[str, object]) -> str:
    """Canonical (sorted-key) JSON text for either payload kind."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


# ── selfcheck ───────────────────────────────────────────────────────


def selfcheck(verbose: bool = False) -> bool:
    """Exercise the golden loop: export, patch, refuse-stale, round trip."""
    checks: List[Tuple[str, bool]] = []

    artifact = build_artifact(
        artifact_id="forearm-demo",
        feature_tree={"root_id": "root", "nodes": {}},
        parameters={
            "forearm_length": {"value": 0.25, "unit": "m", "role": "geometry"},
            "hole_count": 4,
        },
        simulation_tags=[
            {"name": "forearm_length", "kind": "parameter", "target": "link2_length"},
            {"name": "right_forearm", "kind": "body", "target": "r_forearm"},
        ],
    )
    checks.append(("values map", artifact.parameter_values() == {
        "forearm_length": 0.25, "hole_count": 4,
    }))

    # Simulator alias resolution.
    checks.append((
        "tag resolution",
        resolve_simulation_target(artifact, "link2_length") == "forearm_length",
    ))
    checks.append((
        "unknown target is None",
        resolve_simulation_target(artifact, "nope") is None,
    ))

    # Golden patch: 0.25 -> 0.30, guarded by old_value.
    patch = DesignPatch(
        artifact_id="forearm-demo",
        source="simcorrect",
        parameter_patches=(
            ParameterPatch(name="forearm_length", value=0.30, old_value=0.25,
                           reason="sensitivity analysis"),
        ),
    )
    patched = apply_patch(artifact, patch)
    checks.append(("patched value", patched.parameters["forearm_length"].value == 0.30))
    checks.append(("unit preserved", patched.parameters["forearm_length"].unit == "m"))
    checks.append(("input not mutated",
                   artifact.parameters["forearm_length"].value == 0.25))

    # Stale CAS refusal: the same patch no longer applies to the new state.
    try:
        apply_patch(patched, patch)
        checks.append(("stale old_value refused", False))
    except PatchError:
        checks.append(("stale old_value refused", True))

    # Wrong artifact refusal.
    try:
        apply_patch(artifact, DesignPatch(
            artifact_id="other", source="x",
            parameter_patches=(ParameterPatch(name="forearm_length", value=1.0),),
        ))
        checks.append(("wrong artifact refused", False))
    except PatchError:
        checks.append(("wrong artifact refused", True))

    # Unknown parameter refusal, and all-or-nothing: the valid first item of a
    # mixed patch must NOT land.
    try:
        apply_patch(artifact, DesignPatch(
            artifact_id="forearm-demo", source="x",
            parameter_patches=(
                ParameterPatch(name="forearm_length", value=0.5),
                ParameterPatch(name="ghost", value=1.0),
            ),
        ))
        checks.append(("unknown parameter refused", False))
    except PatchError:
        checks.append(("unknown parameter refused", True))
    checks.append(("all-or-nothing",
                   artifact.parameters["forearm_length"].value == 0.25))

    # Payload round trips (canonical text stable across a re-dump).
    payload = artifact_to_payload(artifact)
    validate_artifact_payload(payload)
    back = artifact_from_payload(json.loads(dumps(payload)))
    checks.append(("artifact round trip",
                   artifact_to_payload(back) == payload))
    ppayload = patch_to_payload(patch)
    validate_patch_payload(ppayload)
    pback = patch_from_payload(json.loads(dumps(ppayload)))
    checks.append(("patch round trip", patch_to_payload(pback) == ppayload))
    checks.append(("canonical dump stable",
                   dumps(payload) == dumps(artifact_to_payload(back))))

    # Structural refusals.
    try:
        validate_artifact_payload({"artifact_id": "x"})
        checks.append(("missing keys refused", False))
    except ArtifactError:
        checks.append(("missing keys refused", True))
    try:
        DesignPatch(artifact_id="a", source="s", parameter_patches=())
        checks.append(("empty patch refused", False))
    except PatchError:
        checks.append(("empty patch refused", True))
    try:
        SimulationTag(name="n", kind="wormhole", target="t")
        checks.append(("bad tag kind refused", False))
    except ArtifactError:
        checks.append(("bad tag kind refused", True))

    ok = all(passed for _, passed in checks)
    if verbose:
        for name, passed in checks:
            print("  %-28s %s" % (name, "ok" if passed else "FAIL"))
        print("design_patch selfcheck: %s" % ("ok" if ok else "FAILED"))
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.spec.design_patch",
        description="Design-artifact / parameter-patch contract for an "
                    "external correction loop.",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the golden-loop self-check (no real data)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.selfcheck:
        return 0 if selfcheck(verbose=True) else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
