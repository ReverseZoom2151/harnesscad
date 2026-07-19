"""Conditional factorization of a sketch-extrude CAD model.

This is the deterministic core of *VLM-assisted conditional factorization*. The
image-to-CAD task is factored into two conditional sub-problems::

    P(CAD | image) = P(structure | image) * P(attributes | structure, image)

    Stage 1 (discrete):   the global base structure -- the part decomposition and
                          the ordered CAD command *types* per part, with semantic
                          part labels. (Predicted by a VLM.)
    Stage 2 (continuous): the continuous attribute values for every command,
                          conditioned on the Stage-1 structure. (Predicted by a
                          flow-matching network.)

This module implements the deterministic pieces around that factorization: the
factored representation itself, splitting a full CAD model into its discrete
structure + continuous attribute tensor, the deterministic assembly of factored
predictions back into a CAD model, and a canonical structure signature used to
match / deduplicate base structures across shapes. The learned VLM and
attribute-prediction network are out of scope.

A CAD model is represented as a list of parts::

    part = {"label": str, "commands": [{"type": str, "attrs": [float, ...]}, ...]}

The two-stage factored form is::

    structure = [{"label": str, "command_types": [str, ...]}, ...]
    attributes = [[float, ...], ...]   # one vector per command, model order
"""

from __future__ import annotations

from harnesscad.domain.reconstruction.sequences.sketch_extrude_schema import (
    attribute_dim,
    is_command_type,
    validate_attribute_vector,
)


def normalize_model(model):
    """Validate a CAD model and return a canonical copy (floats, checked arity)."""
    out = []
    for pi, part in enumerate(model):
        label = str(part["label"])
        cmds = []
        for ci, cmd in enumerate(part["commands"]):
            ctype = cmd["type"]
            if not is_command_type(ctype):
                raise ValueError(f"part {pi} cmd {ci}: unknown type {ctype!r}")
            attrs = [float(a) for a in cmd["attrs"]]
            validate_attribute_vector(ctype, attrs)
            cmds.append({"type": ctype, "attrs": attrs})
        out.append({"label": label, "commands": cmds})
    return out


def factorize(model):
    """Split a CAD model into (discrete structure, continuous attributes).

    Returns ``(structure, attributes)`` where ``structure`` carries only labels
    and command *types* (Stage-1) and ``attributes`` is the flat, model-ordered
    list of per-command attribute vectors (Stage-2). Deterministic and lossless.
    """
    model = normalize_model(model)
    structure = []
    attributes = []
    for part in model:
        structure.append({
            "label": part["label"],
            "command_types": [c["type"] for c in part["commands"]],
        })
        for cmd in part["commands"]:
            attributes.append(list(cmd["attrs"]))
    return structure, attributes


def structure_command_count(structure) -> int:
    """Total number of commands described by a discrete structure."""
    return sum(len(p["command_types"]) for p in structure)


def structure_attribute_dim(structure) -> int:
    """Total continuous-attribute dimensionality implied by a structure."""
    total = 0
    for part in structure:
        for ctype in part["command_types"]:
            total += attribute_dim(ctype)
    return total


def validate_structure(structure) -> None:
    """Raise ValueError if a Stage-1 structure is malformed."""
    for pi, part in enumerate(structure):
        if "label" not in part or "command_types" not in part:
            raise ValueError(f"part {pi}: missing 'label'/'command_types'")
        for ctype in part["command_types"]:
            if not is_command_type(ctype):
                raise ValueError(f"part {pi}: unknown command type {ctype!r}")


def assemble(structure, attributes):
    """Deterministically re-assemble a CAD model from factored predictions.

    Inverse of :func:`factorize`: given a Stage-1 discrete structure and the
    Stage-2 flat attribute list (model order), rebuild the full CAD model. Raises
    ValueError if the number/arity of attribute vectors disagrees with the
    structure -- this is exactly the consistency the factorization guarantees.
    """
    validate_structure(structure)
    expected = structure_command_count(structure)
    if len(attributes) != expected:
        raise ValueError(
            f"structure needs {expected} attribute vectors, got {len(attributes)}"
        )
    model = []
    idx = 0
    for part in structure:
        cmds = []
        for ctype in part["command_types"]:
            attrs = [float(a) for a in attributes[idx]]
            validate_attribute_vector(ctype, attrs)
            cmds.append({"type": ctype, "attrs": attrs})
            idx += 1
        model.append({"label": str(part["label"]), "commands": cmds})
    return model


def round_trip(model):
    """factorize then assemble; returns the reconstructed normalized model."""
    structure, attributes = factorize(model)
    return assemble(structure, attributes)


def structure_signature(structure) -> str:
    """Canonical string signature of a discrete base structure.

    Two shapes with the same part labels and per-part command-type sequences share
    a signature -- the basis for the observation that many objects share
    sub-structures (e.g. four-leg chairs). Order-sensitive within a part; parts are
    taken in given order.
    """
    parts = []
    for part in structure:
        parts.append(part["label"] + ":" + ",".join(part["command_types"]))
    return "|".join(parts)


def part_signature(part_structure) -> str:
    """Signature of a single part (label + command-type sequence)."""
    return part_structure["label"] + ":" + ",".join(part_structure["command_types"])
