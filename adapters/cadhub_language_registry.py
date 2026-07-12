"""Code-CAD language capability registry (deterministic, stdlib-only).

Ported from CadHub (``app/web/src/helpers/cadPackages/index.ts`` + the per-language
controllers and the ``app/api/src/docker/*`` lambda runners). CadHub's transferable
core is not its web app but its *multi-language code-CAD abstraction layer*: every
supported language (OpenSCAD, CadQuery, JSCAD, curv) is described by the same
adapter contract -- a file extension, a fixed entry-point filename, a set of
producible artifact kinds, an execution model (out-of-process CLI vs in-process
worker), whether it can export a mesh, and how its parameters are supplied.

The harness had adapter *protocols* (``adapters/base.py``, for live CAD apps such
as Rhino) and per-language checkers (``programs/scadlm_*``, ``programs/t2cq_*``),
but no registry that answers cross-language questions: "which languages can export
STL?", "what entry file do I write for CadQuery?", "does curv support a parameter
panel?", "given ``part.jscad``, which toolchain?". A router/planner needs exactly
that capability matrix before it dispatches generated code to a backend.

Deterministic: a frozen table, pure lookups, sorted outputs. No I/O, no clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Artifact kinds a language's render pipeline can hand back (CadHub's ArtifactTypes).
ARTIFACT_MESH = "mesh"  # tessellated geometry loaded from an STL
ARTIFACT_IMAGE = "image"  # server-rendered raster preview (png)
ARTIFACT_PRIMITIVES = "primitives"  # in-memory primitive/CSG array (no file round trip)

ARTIFACT_KINDS = (ARTIFACT_MESH, ARTIFACT_IMAGE, ARTIFACT_PRIMITIVES)

# How the language is executed.
EXEC_CLI = "cli"  # external binary invoked on a temp directory
EXEC_WORKER = "worker"  # evaluated in-process by a sandboxed interpreter

# Where parameter definitions come from.
PARAMS_NONE = "none"  # language has no parameter concept in this pipeline
PARAMS_SOURCE_ANNOTATION = "source-annotation"  # parsed out of the source comments
PARAMS_TOOL_EXPORT = "tool-export"  # the toolchain emits a param manifest
PARAMS_SCRIPT_CALLBACK = "script-callback"  # the script declares them at eval time


class UnknownLanguage(KeyError):
    """Raised when a language / extension is not in the registry."""


@dataclass(frozen=True)
class LanguageSpec:
    """One row of the code-CAD capability matrix."""

    name: str
    extension: str  # canonical source extension, with leading dot
    entry_file: str  # entry-point filename written into the working dir
    execution: str  # EXEC_CLI | EXEC_WORKER
    artifacts: Tuple[str, ...]  # producible artifact kinds, canonical order
    mesh_export: bool  # can produce a mesh file (STL) for download
    params: str  # PARAMS_*
    param_file: Optional[str] = None  # file used to feed parameter values, if any
    diagnostic_dialect: str = "generic"  # key into the diagnostic parser
    display_name: str = ""
    aliases: Tuple[str, ...] = field(default_factory=tuple)

    def supports(self, artifact: str) -> bool:
        return artifact in self.artifacts

    def as_row(self) -> Dict[str, object]:
        """Flat dict, suitable for a deterministic table dump."""
        return {
            "name": self.name,
            "extension": self.extension,
            "entry_file": self.entry_file,
            "execution": self.execution,
            "artifacts": list(self.artifacts),
            "mesh_export": self.mesh_export,
            "params": self.params,
            "param_file": self.param_file,
            "diagnostic_dialect": self.diagnostic_dialect,
        }


# ---------------------------------------------------------------------------
# The matrix (mirrors CadHub's four shipped integrations)
# ---------------------------------------------------------------------------

_SPECS: Tuple[LanguageSpec, ...] = (
    LanguageSpec(
        name="openscad",
        display_name="OpenSCAD",
        extension=".scad",
        entry_file="main.scad",
        execution=EXEC_CLI,
        # preview is a png; the STL path is a separate CLI invocation
        artifacts=(ARTIFACT_IMAGE, ARTIFACT_MESH),
        mesh_export=True,
        params=PARAMS_SOURCE_ANNOTATION,
        param_file="params.json",
        diagnostic_dialect="openscad",
    ),
    LanguageSpec(
        name="cadquery",
        display_name="CadQuery",
        extension=".py",
        entry_file="main.py",
        execution=EXEC_CLI,
        artifacts=(ARTIFACT_MESH,),
        mesh_export=True,
        params=PARAMS_TOOL_EXPORT,
        param_file="params.json",
        diagnostic_dialect="cadquery",
        aliases=("cq",),
    ),
    LanguageSpec(
        name="jscad",
        display_name="JSCAD",
        extension=".jscad",
        entry_file="main.jscad.js",
        execution=EXEC_WORKER,
        artifacts=(ARTIFACT_PRIMITIVES,),
        mesh_export=False,  # renders in-process; no server STL route in CadHub
        params=PARAMS_SCRIPT_CALLBACK,
        param_file=None,
        diagnostic_dialect="jscad",
        aliases=("openjscad",),
    ),
    LanguageSpec(
        name="curv",
        display_name="curv",
        extension=".curv",
        entry_file="main.curv",
        execution=EXEC_CLI,
        artifacts=(ARTIFACT_IMAGE, ARTIFACT_MESH),
        mesh_export=True,
        params=PARAMS_NONE,
        param_file=None,
        diagnostic_dialect="curv",
    ),
)

_BY_NAME: Dict[str, LanguageSpec] = {}
for _spec in _SPECS:
    _BY_NAME[_spec.name] = _spec
    for _alias in _spec.aliases:
        _BY_NAME[_alias] = _spec


def language_names() -> List[str]:
    """Canonical language names, sorted."""
    return sorted(spec.name for spec in _SPECS)


def get(name: str) -> LanguageSpec:
    """Look a language up by name or alias (case-insensitive)."""
    try:
        return _BY_NAME[name.strip().lower()]
    except KeyError as exc:  # pragma: no cover - message path
        raise UnknownLanguage(name) from exc


def has(name: str) -> bool:
    return name.strip().lower() in _BY_NAME


def for_extension(path_or_ext: str) -> LanguageSpec:
    """Resolve a source path or bare extension to its language.

    ``.py`` is ambiguous in general but unambiguous inside this registry
    (CadQuery is the only Python code-CAD integration).
    """
    text = path_or_ext.strip().lower()
    dot = text.rfind(".")
    ext = text[dot:] if dot >= 0 else "." + text
    # longest-suffix match so ``main.jscad.js`` resolves via ``.jscad`` too
    for spec in _SPECS:
        if text.endswith(spec.extension):
            return spec
    for spec in _SPECS:
        if spec.extension == ext:
            return spec
    raise UnknownLanguage(path_or_ext)


def entry_file(name: str) -> str:
    return get(name).entry_file


def languages_with_artifact(artifact: str) -> List[str]:
    """Names of languages that can produce ``artifact``, sorted."""
    if artifact not in ARTIFACT_KINDS:
        raise ValueError("unknown artifact kind: %s" % artifact)
    return sorted(spec.name for spec in _SPECS if spec.supports(artifact))


def mesh_exporters() -> List[str]:
    return sorted(spec.name for spec in _SPECS if spec.mesh_export)


def parametric_languages() -> List[str]:
    """Languages that expose a parameter panel of any flavour."""
    return sorted(spec.name for spec in _SPECS if spec.params != PARAMS_NONE)


def capability_matrix() -> List[Dict[str, object]]:
    """The full table, one row per language, sorted by name."""
    return [get(n).as_row() for n in language_names()]


def missing_capabilities(name: str) -> List[str]:
    """Capabilities present in at least one other language but absent here.

    Used by a planner to explain *why* a target language cannot serve a request
    (e.g. jscad cannot hand back an STL).
    """
    spec = get(name)
    gaps: List[str] = []
    for artifact in ARTIFACT_KINDS:
        others = languages_with_artifact(artifact)
        if others and not spec.supports(artifact):
            gaps.append("artifact:" + artifact)
    if not spec.mesh_export:
        gaps.append("mesh_export")
    if spec.params == PARAMS_NONE:
        gaps.append("params")
    return sorted(set(gaps))


def select(
    *,
    artifact: Optional[str] = None,
    mesh_export: Optional[bool] = None,
    execution: Optional[str] = None,
    params: Optional[bool] = None,
) -> List[str]:
    """Names of languages matching every supplied constraint, sorted."""
    out: List[str] = []
    for spec in _SPECS:
        if artifact is not None and not spec.supports(artifact):
            continue
        if mesh_export is not None and spec.mesh_export is not mesh_export:
            continue
        if execution is not None and spec.execution != execution:
            continue
        if params is not None and (spec.params != PARAMS_NONE) is not params:
            continue
        out.append(spec.name)
    return sorted(out)


def working_files(name: str) -> Sequence[str]:
    """Files a runner must materialise for this language, in write order."""
    spec = get(name)
    files = [spec.entry_file]
    if spec.param_file:
        files.append(spec.param_file)
    return tuple(files)
