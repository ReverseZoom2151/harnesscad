"""CAID design-artifact and design-patch contract.

The "CAID design artifact" is the stable JSON boundary between a CAD tool that
owns the design and a simulation-driven corrector that identifies faulty
parameters and sends back patches. The contract is worth adopting wholesale for
a text-to-CAD harness because it solves a problem the harness will hit with any
external verifier or simulator:

  * the artifact carries an ``artifact_id`` identity so a patch produced against
    one design revision can never be applied to another;
  * every patch item may carry ``old_value``, and a non-null ``old_value`` that
    does not match the artifact's current value is a hard error -- the strict
    stale-write check that catches a correction computed against an outdated
    design before it silently overwrites newer state;
  * ``simulation_tags`` map internal simulator names (``link2_length``) to
    design-facing parameter names (``forearm_length``) so diagnosis code never
    hardcodes aliases;
  * patch application never mutates the input artifact.

This module implements the full contract: artifact/patch validation, parameter
lookup, bidirectional name resolution through simulation tags, patch
construction (including directly from an identification result), and
non-mutating patch application both to the artifact and to a flat simulation
parameter dict.

stdlib-only, deterministic, absolute imports. No simulator anywhere.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

__all__ = [
    "CAID_SCHEMA_VERSION",
    "ARTIFACT_REQUIRED_KEYS",
    "PATCH_REQUIRED_KEYS",
    "SIMULATION_TAG_KINDS",
    "CaidContractError",
    "load_artifact",
    "validate_artifact",
    "validate_patch",
    "get_parameter",
    "resolve_parameter_name",
    "simulation_target_for_parameter",
    "make_parameter_patch",
    "make_patch_from_identification",
    "apply_parameter_patch",
    "apply_patch_to_simulation_params",
    "main",
]

CAID_SCHEMA_VERSION = 1

ARTIFACT_REQUIRED_KEYS = frozenset({
    "schema_version", "artifact_id", "producer", "created_at",
    "feature_tree", "parameters", "simulation_tags",
})
PATCH_REQUIRED_KEYS = frozenset({
    "schema_version", "artifact_id", "source", "parameter_patches",
})
SIMULATION_TAG_KINDS = frozenset({"body", "joint", "geom", "site", "parameter"})
_PARAMETER_VALUE_TYPES = (bool, int, float, str)

_JsonSource = Union[str, Path, Dict[str, Any]]


class CaidContractError(ValueError):
    """Raised when an artifact or patch violates the CAID contract."""


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def load_artifact(source: _JsonSource) -> Dict[str, Any]:
    """Load and validate a design artifact from a path or an in-memory dict."""
    artifact = _load_json(source)
    validate_artifact(artifact)
    return artifact


def validate_artifact(payload: Dict[str, Any]) -> None:
    """Raise :class:`CaidContractError` on any contract violation."""
    _require_object(payload, "CAID artifact")
    _require_keys(payload, ARTIFACT_REQUIRED_KEYS, "CAID artifact")
    _require_version(payload)
    if not isinstance(payload.get("artifact_id"), str) or not payload["artifact_id"]:
        raise CaidContractError("CAID artifact must contain a non-empty artifact_id.")
    _require_producer(payload["producer"])
    _require_feature_tree(payload["feature_tree"])
    if not isinstance(payload.get("parameters"), dict):
        raise CaidContractError("CAID artifact must contain a parameters object.")
    for name, parameter in payload["parameters"].items():
        _require_parameter(name, parameter)
    if not isinstance(payload["simulation_tags"], list):
        raise CaidContractError("CAID artifact simulation_tags must be a list.")
    for tag in payload["simulation_tags"]:
        _require_simulation_tag(tag)


def validate_patch(payload: Dict[str, Any]) -> None:
    """Raise :class:`CaidContractError` on any patch contract violation."""
    _require_object(payload, "CAID patch")
    _require_keys(payload, PATCH_REQUIRED_KEYS, "CAID patch")
    _require_version(payload)
    if not isinstance(payload.get("artifact_id"), str) or not payload["artifact_id"]:
        raise CaidContractError("CAID patch must contain a non-empty artifact_id.")
    if not isinstance(payload.get("source"), str) or not payload["source"]:
        raise CaidContractError("CAID patch must contain a non-empty source.")
    patches = payload.get("parameter_patches")
    if not isinstance(patches, list) or not patches:
        raise CaidContractError("CAID patch must contain at least one parameter patch.")
    for item in patches:
        _require_patch_item(item)


# --------------------------------------------------------------------------- #
# Lookup and name resolution
# --------------------------------------------------------------------------- #
def get_parameter(artifact: Dict[str, Any], name: str) -> Dict[str, Any]:
    """The parameter record for ``name``, or a contract error."""
    validate_artifact(artifact)
    try:
        parameter = artifact["parameters"][name]
    except KeyError as exc:
        raise CaidContractError(f"Unknown design parameter '{name}'.") from exc
    if not isinstance(parameter, dict) or "value" not in parameter:
        raise CaidContractError(f"Design parameter '{name}' is malformed.")
    return parameter


def resolve_parameter_name(artifact: Dict[str, Any], name_or_target: str) -> str:
    """Resolve either a design name or a simulation target to the design name.

    A simulation target such as ``link2_length`` resolves through a
    ``kind="parameter"`` simulation tag to its design parameter
    (``forearm_length``). A plain design name resolves to itself.
    """
    validate_artifact(artifact)
    if name_or_target in artifact["parameters"]:
        return name_or_target
    for tag in artifact.get("simulation_tags", []):
        if tag.get("kind") == "parameter" and tag.get("target") == name_or_target:
            name = tag.get("name")
            if name in artifact["parameters"]:
                return name
    raise CaidContractError(
        f"Unknown design parameter or simulation target '{name_or_target}'."
    )


def simulation_target_for_parameter(artifact: Dict[str, Any], name: str) -> str:
    """The simulator-facing target name for a design parameter (or itself)."""
    validate_artifact(artifact)
    get_parameter(artifact, name)
    for tag in artifact.get("simulation_tags", []):
        if tag.get("kind") == "parameter" and tag.get("name") == name:
            target = tag.get("target")
            if isinstance(target, str) and target:
                return target
            raise CaidContractError(f"Simulation tag for '{name}' is missing a target.")
    return name


# --------------------------------------------------------------------------- #
# Patch construction
# --------------------------------------------------------------------------- #
def make_parameter_patch(
    artifact: Dict[str, Any],
    name: str,
    value: Union[bool, int, float, str],
    *,
    reason: Optional[str] = None,
    source: str = "harnesscad",
) -> Dict[str, Any]:
    """A single-item patch that records the artifact's current value as old_value."""
    parameter = get_parameter(artifact, name)
    return {
        "schema_version": CAID_SCHEMA_VERSION,
        "artifact_id": artifact["artifact_id"],
        "source": source,
        "parameter_patches": [{
            "name": name,
            "old_value": parameter["value"],
            "value": value,
            "reason": reason,
        }],
    }


def make_patch_from_identification(
    artifact: Dict[str, Any],
    identification: Dict[str, Any],
    *,
    source: str = "harnesscad",
) -> Dict[str, Any]:
    """Build a patch from an identification result.

    The identification result must carry ``identified_parameter`` (a design
    name or a simulation target) and ``proposed_value``; ``method`` is folded
    into the recorded reason when present.
    """
    missing = [k for k in ("identified_parameter", "proposed_value") if k not in identification]
    if missing:
        raise CaidContractError(
            f"Identification result missing required key(s): {', '.join(missing)}."
        )
    name = resolve_parameter_name(artifact, identification["identified_parameter"])
    method = identification.get("method", "parameter_identification")
    return make_parameter_patch(
        artifact,
        name,
        identification["proposed_value"],
        reason=f"{method} identified {identification['identified_parameter']}.",
        source=source,
    )


# --------------------------------------------------------------------------- #
# Patch application
# --------------------------------------------------------------------------- #
def apply_parameter_patch(artifact: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """A new artifact with the patch applied. The input is never mutated."""
    validate_artifact(artifact)
    validate_patch(patch)
    _require_patch_targets_artifact(artifact, patch)
    updated = deepcopy(artifact)
    for item in patch["parameter_patches"]:
        parameter = get_parameter(updated, item["name"])
        _require_current_value(parameter, item)
        parameter["value"] = item["value"]
    return updated


def apply_patch_to_simulation_params(
    artifact: Dict[str, Any],
    patch: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """A new flat simulation-parameter dict with the patch routed through tags.

    Each design-name patch item is translated to its simulation target before
    being written, so simulator code keeps its own vocabulary.
    """
    validate_artifact(artifact)
    validate_patch(patch)
    _require_patch_targets_artifact(artifact, patch)
    updated = dict(params)
    for item in patch["parameter_patches"]:
        parameter = get_parameter(artifact, item["name"])
        _require_current_value(parameter, item)
        target = simulation_target_for_parameter(artifact, item["name"])
        if target not in updated:
            raise CaidContractError(
                f"Simulation parameter '{target}' is not present in current params."
            )
        updated[target] = item["value"]
    return updated


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _load_json(source: _JsonSource) -> Dict[str, Any]:
    if isinstance(source, dict):
        return deepcopy(source)
    if not isinstance(source, (str, Path)):
        raise CaidContractError("CAID JSON source must be a path or object.")
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CaidContractError("CAID JSON payload must be an object.")
    return payload


def _require_version(payload: Dict[str, Any]) -> None:
    if payload.get("schema_version") != CAID_SCHEMA_VERSION:
        raise CaidContractError(
            f"Unsupported CAID schema_version: {payload.get('schema_version')!r}."
        )


def _require_object(payload: Any, label: str) -> None:
    if not isinstance(payload, dict):
        raise CaidContractError(f"{label} must be an object.")


def _require_keys(payload: Dict[str, Any], required: frozenset, label: str) -> None:
    missing = sorted(required - payload.keys())
    if missing:
        raise CaidContractError(f"{label} missing required key(s): {', '.join(missing)}.")


def _require_producer(producer: Any) -> None:
    if not isinstance(producer, dict):
        raise CaidContractError("CAID artifact producer must be an object.")
    for key in ("name", "version"):
        if not isinstance(producer.get(key), str) or not producer[key]:
            raise CaidContractError(f"CAID artifact producer must contain a non-empty {key}.")


def _require_feature_tree(feature_tree: Any) -> None:
    if not isinstance(feature_tree, dict):
        raise CaidContractError("CAID artifact feature_tree must be an object.")
    if not isinstance(feature_tree.get("root_id"), str) or not feature_tree["root_id"]:
        raise CaidContractError("CAID artifact feature_tree must contain a non-empty root_id.")
    if not isinstance(feature_tree.get("nodes"), dict):
        raise CaidContractError("CAID artifact feature_tree must contain a nodes object.")


def _require_parameter(name: Any, parameter: Any) -> None:
    if not isinstance(name, str) or not name:
        raise CaidContractError("CAID artifact parameter keys must be non-empty strings.")
    if not isinstance(parameter, dict):
        raise CaidContractError(f"Design parameter '{name}' must be an object.")
    if parameter.get("name") != name:
        raise CaidContractError(
            f"Design parameter key '{name}' does not match parameter name "
            f"'{parameter.get('name')}'."
        )
    if "value" not in parameter:
        raise CaidContractError(f"Design parameter '{name}' is missing value.")
    if not isinstance(parameter["value"], _PARAMETER_VALUE_TYPES):
        raise CaidContractError(f"Design parameter '{name}' has unsupported value type.")
    for optional in ("unit", "role", "feature_id"):
        if optional in parameter and parameter[optional] is not None \
                and not isinstance(parameter[optional], str):
            raise CaidContractError(
                f"Design parameter '{name}' field '{optional}' must be a string when present."
            )


def _require_simulation_tag(tag: Any) -> None:
    if not isinstance(tag, dict):
        raise CaidContractError("CAID artifact simulation tags must be objects.")
    if not isinstance(tag.get("name"), str) or not tag["name"]:
        raise CaidContractError("CAID artifact simulation tag must contain a non-empty name.")
    if tag.get("kind") not in SIMULATION_TAG_KINDS:
        raise CaidContractError(
            f"CAID artifact simulation tag has unsupported kind: {tag.get('kind')!r}."
        )
    if not isinstance(tag.get("target"), str) or not tag["target"]:
        raise CaidContractError("CAID artifact simulation tag must contain a non-empty target.")
    if "metadata" in tag and not isinstance(tag["metadata"], dict):
        raise CaidContractError(
            "CAID artifact simulation tag metadata must be an object when present."
        )


def _require_patch_item(item: Any) -> None:
    if not isinstance(item, dict):
        raise CaidContractError("Each parameter patch must be an object.")
    if not isinstance(item.get("name"), str) or not item["name"]:
        raise CaidContractError("Each parameter patch must contain a non-empty name.")
    if "value" not in item:
        raise CaidContractError(f"Parameter patch '{item.get('name')}' is missing value.")
    if not isinstance(item["value"], _PARAMETER_VALUE_TYPES):
        raise CaidContractError(
            f"Parameter patch '{item['name']}' has unsupported value type."
        )
    if "old_value" in item and item["old_value"] is not None \
            and not isinstance(item["old_value"], _PARAMETER_VALUE_TYPES):
        raise CaidContractError(
            f"Parameter patch '{item['name']}' has unsupported old_value type."
        )
    if "reason" in item and item["reason"] is not None and not isinstance(item["reason"], str):
        raise CaidContractError(
            f"Parameter patch '{item['name']}' reason must be a string when present."
        )


def _require_patch_targets_artifact(artifact: Dict[str, Any], patch: Dict[str, Any]) -> None:
    if patch.get("artifact_id") != artifact.get("artifact_id"):
        raise CaidContractError("Patch artifact_id does not match artifact.")


def _require_current_value(parameter: Dict[str, Any], item: Dict[str, Any]) -> None:
    if item.get("old_value") is not None and item["old_value"] != parameter["value"]:
        raise CaidContractError(
            f"Patch for '{item['name']}' expected old value {item['old_value']!r}, "
            f"but artifact has {parameter['value']!r}."
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _demo_artifact() -> Dict[str, Any]:
    return {
        "schema_version": CAID_SCHEMA_VERSION,
        "artifact_id": "selfcheck-forearm",
        "producer": {"name": "harnesscad", "version": "0"},
        "created_at": "2026-01-01T00:00:00Z",
        "feature_tree": {"root_id": "root", "nodes": {}},
        "parameters": {
            "forearm_length": {"name": "forearm_length", "value": 0.25, "unit": "m"},
        },
        "simulation_tags": [
            {"name": "forearm_length", "kind": "parameter", "target": "link2_length"},
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.spec.caid_artifact",
        description="CAID design-artifact and design-patch contract (SimCorrect).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the golden identification-to-patch loop on a "
                             "synthetic artifact and a stale-write rejection.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    artifact = load_artifact(_demo_artifact())
    identification = {
        "identified_parameter": "link2_length",
        "proposed_value": 0.30,
        "method": "sensitivity_analysis",
    }
    patch = make_patch_from_identification(artifact, identification)
    corrected = apply_parameter_patch(artifact, patch)
    params = apply_patch_to_simulation_params(
        artifact, patch, {"link1_length": 0.30, "link2_length": 0.22})
    assert corrected["parameters"]["forearm_length"]["value"] == 0.30
    assert artifact["parameters"]["forearm_length"]["value"] == 0.25
    assert params["link2_length"] == 0.30
    print("[selfcheck] golden loop: forearm_length 0.25 -> 0.30 via target link2_length")

    stale = make_parameter_patch(artifact, "forearm_length", 0.31)
    stale["parameter_patches"][0]["old_value"] = 0.99
    try:
        apply_parameter_patch(artifact, stale)
    except CaidContractError as exc:
        print(f"[selfcheck] stale write rejected: {exc}")
    else:
        raise AssertionError("stale patch must be rejected")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
