"""Catalogue of a CAD command-line tool's command surface.

Static reference data covers a CLI verb surface. This complements the two
language and engine-surface modules already in the harness:

*   :mod:`harnesscad.domain.spec.kcl_grammar` models the CAD language;
*   :mod:`harnesscad.domain.spec.zoo_catalog` models the engine op set,
    standard library, and file-conversion matrix;
*   this module models the *CLI verb surface* -- the commands a user or an agent
    driving the CAD CLI binary can actually invoke, which neither of the other
    two captured.

The load-bearing part for a text-to-CAD harness is the set of **geometry
commands**: geometry-property and conversion operations run a headless engine
query against a CAD program or imported file and return scalar/vector geometric
properties. Generative endpoints are represented separately.

Everything here is inert data plus pure query helpers; nothing shells out and no
CLI is invoked.  It exists so an agent that wants to *use* the CLI (or a
harness author wiring a subprocess backend) has one checked place to read the
command tree, the geometry-query verbs, and which verbs accept a KCL program vs.
an imported file.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

__all__ = [
    "COMMANDS",
    "GEOMETRY_QUERIES",
    "top_level_commands",
    "subcommands",
    "is_geometry_query",
    "command_path_exists",
    "geometry_query_commands",
]

# ---------------------------------------------------------------------------
# Command tree: top-level verb -> ordered tuple of subcommand names.
# An empty tuple means the command takes no subcommands (a leaf verb).
# Subcommand names use the CLI spelling (kebab-case), including clap aliases
# where the enum declares them (e.g. kcl "format" is invoked as "fmt").
# ---------------------------------------------------------------------------

COMMANDS: Dict[str, Tuple[str, ...]] = {
    "alias": ("set", "delete", "list"),
    "api": (),
    "api-call": (),
    "app": (),
    "auth": ("login", "logout", "status"),
    "completion": (),
    "config": ("set", "list", "get"),
    "file": (
        "convert",
        "snapshot",
        "volume",
        "mass",
        "center-of-mass",
        "density",
        "surface-area",
    ),
    "generate": ("markdown",),
    "kcl": (
        "export",
        "format",
        "snapshot",
        "view",
        "analyze",
        "volume",
        "mass",
        "center-of-mass",
        "density",
        "surface-area",
        "lint",
        "bounding-box",
    ),
    "ml": ("text-to-cad", "kcl"),
    "org": ("dataset",),
    "say": (),
    "start-session": (),
    "open": (),
    "update": (),
    "user": (),
    "version": (),
}

# Nested subcommands two levels deep (verb -> subverb -> sub-subcommands).
ML_SUBCOMMANDS: Dict[str, Tuple[str, ...]] = {
    "text-to-cad": ("export", "snapshot", "view"),
    "kcl": ("edit", "copilot"),
}

# The geometry-property query verbs shared by the `kcl` and `file` groups.
# Each runs a headless engine measurement and returns the named property.
GEOMETRY_QUERIES: Tuple[str, ...] = (
    "volume",
    "mass",
    "center-of-mass",
    "density",
    "surface-area",
    "bounding-box",  # kcl-only (file has no bounding-box verb)
)


def top_level_commands() -> List[str]:
    """All top-level CLI verbs, sorted."""
    return sorted(COMMANDS)


def subcommands(command: str) -> Tuple[str, ...]:
    """Subcommands of ``command`` (empty tuple for a leaf verb / unknown)."""
    return COMMANDS.get(command, ())


def is_geometry_query(subcommand: str) -> bool:
    """True if ``subcommand`` is a geometry-property measurement verb."""
    return subcommand in GEOMETRY_QUERIES


def command_path_exists(command: str, subcommand: str = "") -> bool:
    """True if ``<command> [<subcommand>]`` is a valid invocation path."""
    if command not in COMMANDS:
        return False
    if not subcommand:
        return True
    return subcommand in COMMANDS[command]


def geometry_query_commands() -> List[Tuple[str, str]]:
    """Every ``(command, subcommand)`` pair that runs a geometry query.

    e.g. ``("kcl", "volume")`` and ``("file", "volume")``.
    """
    out: List[Tuple[str, str]] = []
    for command in ("kcl", "file"):
        for sub in COMMANDS[command]:
            if sub in GEOMETRY_QUERIES:
                out.append((command, sub))
    return out
