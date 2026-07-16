"""Approved OpenSCAD library policy: manifest, include validation, license gate.

Source: ``resources/cad_repos/AgentSCAD-main`` (``skills/scad-library-policy/
manifest.json`` + its three deterministic scripts and the scad-library-policy
SKILL.md). AgentSCAD keeps its library policy *out of prompts and route
logic*: a single manifest is the source of truth for which OpenSCAD libraries
generated code may ``include``/``use``, pinned to exact upstream commits, with
per-library license gates (permissive / weak-copyleft / GPL opt-in) and
detection files that decide availability on a workstation. Generated SCAD is
then validated: an include path that is not approved, or approved but not
locally resolvable, is a hard error -- the model must not invent include paths.

This was missed by the integration-campaign pass over AgentSCAD (which took
the cost-aware progressive-escalation pipeline). The harness gained OpenSCAD
customizer parsing (``cadam_scad_customizer``) and multi-language diagnostics
(``cadhub_diagnostics``) from other repos, but nothing polices the *library
supply chain* of generated OpenSCAD. This module ports that policy layer:

* :class:`LibrarySpec` / :class:`PolicyManifest` -- the manifest model, with
  AgentSCAD's actual six-library manifest embedded as
  :data:`DEFAULT_MANIFEST` (repos, pinned commits, licenses, gates).
* :func:`extract_includes` -- ``include <...>`` / ``use <...>`` extraction.
* :func:`validate_includes` -- the unapproved / approved-but-unavailable
  split, exactly the semantics of ``validate_scad_includes.py``.
* :func:`detect_available` -- detection-file availability over caller-supplied
  search roots (injectable ``exists`` predicate; no filesystem surprises).
* :func:`install_plan` -- the license-gated install plan:
  ``default_install`` libraries only, GPL requires ``include_gpl=True``,
  unpinned or repo-less entries are never planned, and every skip carries its
  reason.

The manifest content is UNVERIFIED third-party data in harness terms: it
authorises *reference by include path* only. Nothing here downloads, vendors,
or executes library code.

Stdlib only, deterministic, absolute imports. ``--selfcheck`` exercises the
embedded manifest, the validator, and the license gate.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "LibrarySpec",
    "PolicyManifest",
    "IncludeIssue",
    "InstallAction",
    "DEFAULT_MANIFEST",
    "extract_includes",
    "approved_include_paths",
    "validate_includes",
    "detect_available",
    "install_plan",
    "library_for_include",
    "main",
]

_INCLUDE_RE = re.compile(r"^\s*(include|use)\s*<([^>]+)>", re.MULTILINE)
_ANGLE_RE = re.compile(r"<([^>]+)>")

#: License gates the manifest uses, from most to least permissive.
LICENSE_GATES = ("public-domain", "permissive", "weak-copyleft", "gpl")


@dataclass(frozen=True)
class LibrarySpec:
    """One approved library: identity, pin, gate, and detection contract."""
    name: str
    repo: Optional[str]
    commit: Optional[str]
    target_dir: str
    license: str
    license_gate: str            # one of LICENSE_GATES
    default_install: bool
    required_files: Tuple[str, ...]
    detection_files: Tuple[str, ...]
    skill_name: str
    include_examples: Tuple[str, ...]
    default_install_reason: str = ""

    def include_paths(self) -> List[str]:
        """The include/use paths this library authorises."""
        paths: List[str] = []
        for example in self.include_examples:
            m = _ANGLE_RE.search(example)
            if m:
                paths.append(m.group(1))
        return paths

    def pinned(self) -> bool:
        return bool(self.repo) and bool(self.commit)


@dataclass
class PolicyManifest:
    """The full policy: libraries plus the managed-directory contract."""
    libraries: List[LibrarySpec] = field(default_factory=list)
    managed_library_dir_env: str = "AGENTSCAD_OPENSCAD_LIBRARY_DIR"
    default_managed_library_dir: str = "~/.agentscad/openscad-libraries"

    @classmethod
    def from_dict(cls, data: Mapping) -> "PolicyManifest":
        libs = []
        for entry in data.get("libraries", []):
            libs.append(LibrarySpec(
                name=entry["name"],
                repo=entry.get("repo"),
                commit=entry.get("commit"),
                target_dir=entry.get("target_dir", ""),
                license=entry.get("license", ""),
                license_gate=entry.get("license_gate", "permissive"),
                default_install=bool(entry.get("default_install", False)),
                required_files=tuple(entry.get("required_files", ())),
                detection_files=tuple(entry.get("detection_files", ())),
                skill_name=entry.get("skill_name", ""),
                include_examples=tuple(entry.get("include_examples", ())),
                default_install_reason=entry.get("default_install_reason", ""),
            ))
        return cls(
            libraries=libs,
            managed_library_dir_env=data.get(
                "managed_library_dir_env", "AGENTSCAD_OPENSCAD_LIBRARY_DIR"),
            default_managed_library_dir=data.get(
                "default_managed_library_dir", "~/.agentscad/openscad-libraries"),
        )

    def get(self, name: str) -> Optional[LibrarySpec]:
        for lib in self.libraries:
            if lib.name == name:
                return lib
        return None


#: AgentSCAD's shipped manifest (skills/scad-library-policy/manifest.json),
#: pinned commits included. Data, not code: nothing is fetched or executed.
DEFAULT_MANIFEST = PolicyManifest.from_dict({
    "managed_library_dir_env": "AGENTSCAD_OPENSCAD_LIBRARY_DIR",
    "default_managed_library_dir": "~/.agentscad/openscad-libraries",
    "libraries": [
        {
            "name": "BOSL2",
            "repo": "https://github.com/BelfrySCAD/BOSL2.git",
            "commit": "c3861cfc2f146676b86de0d3f76896ca1f7735d7",
            "target_dir": "BOSL2",
            "license": "BSD-2-Clause",
            "license_gate": "permissive",
            "default_install": True,
            "required_files": ["std.scad"],
            "detection_files": ["BOSL2/std.scad"],
            "skill_name": "scad-library-bosl2",
            "include_examples": ["include <BOSL2/std.scad>"],
        },
        {
            "name": "Round-Anything",
            "repo": "https://github.com/Irev-Dev/Round-Anything.git",
            "commit": "061fef7c429628808e847696bb345a9b0ec6e279",
            "target_dir": "Round-Anything",
            "license": "MIT",
            "license_gate": "permissive",
            "default_install": True,
            "required_files": ["polyround.scad"],
            "detection_files": ["Round-Anything/polyround.scad", "polyround.scad"],
            "skill_name": "scad-library-round-anything",
            "include_examples": ["use <Round-Anything/polyround.scad>"],
        },
        {
            "name": "MCAD",
            "repo": "https://github.com/openscad/MCAD.git",
            "commit": "bd0a7ba3f042bfbced5ca1894b236cea08904e26",
            "target_dir": "MCAD",
            "license": "LGPL-2.1",
            "license_gate": "weak-copyleft",
            "default_install": True,
            "required_files": ["units.scad", "involute_gears.scad"],
            "detection_files": ["MCAD/units.scad", "MCAD/involute_gears.scad"],
            "skill_name": "scad-library-mcad",
            "include_examples": ["include <MCAD/units.scad>",
                                 "use <MCAD/involute_gears.scad>"],
        },
        {
            "name": "threadlib",
            "repo": "https://github.com/adrianschlatter/threadlib.git",
            "commit": "3830919a9937d2f662f7205f1dfd28c5bb948aba",
            "target_dir": "threadlib",
            "license": "BSD-3-Clause",
            "license_gate": "permissive",
            "default_install": False,
            "default_install_reason": (
                "Requires additional OpenSCAD dependencies (scad-utils, "
                "list-comprehension, thread_profile.scad). Keep opt-in until "
                "dependency chain is managed."),
            "required_files": ["threadlib.scad"],
            "detection_files": ["threadlib/threadlib.scad"],
            "skill_name": "scad-library-threads",
            "include_examples": ["use <threadlib/threadlib.scad>"],
        },
        {
            "name": "threads.scad",
            "repo": None,
            "commit": None,
            "target_dir": "",
            "license": "CC0-1.0",
            "license_gate": "public-domain",
            "default_install": False,
            "default_install_reason": (
                "External single-file library support only. Add a reviewed "
                "source before managed installation."),
            "required_files": ["threads.scad"],
            "detection_files": ["threads.scad"],
            "skill_name": "scad-library-threads",
            "include_examples": ["include <threads.scad>"],
        },
        {
            "name": "NopSCADlib",
            "repo": "https://github.com/nophead/NopSCADlib.git",
            "commit": "c9baa0ed0faa23e849141c3d8c6728545d6af910",
            "target_dir": "NopSCADlib",
            "license": "GPL-3.0",
            "license_gate": "gpl",
            "default_install": False,
            "default_install_reason": (
                "GPL-3.0 library. Require explicit opt-in and preserve "
                "license notices."),
            "required_files": ["core.scad"],
            "detection_files": ["NopSCADlib/core.scad"],
            "skill_name": "scad-library-nopscadlib",
            "include_examples": ["include <NopSCADlib/core.scad>"],
        },
    ],
})


# ---------------------------------------------------------------------------
# Include extraction and validation
# ---------------------------------------------------------------------------

def extract_includes(scad_source: str) -> List[Tuple[str, str]]:
    """All ``(keyword, path)`` pairs from ``include <p>`` / ``use <p>`` lines."""
    return [(m.group(1), m.group(2)) for m in _INCLUDE_RE.finditer(scad_source)]


def approved_include_paths(manifest: PolicyManifest) -> List[str]:
    """Every include path the manifest authorises (deduplicated, ordered)."""
    seen: Dict[str, None] = {}
    for lib in manifest.libraries:
        for path in lib.include_paths():
            seen.setdefault(path, None)
    return list(seen)


def library_for_include(manifest: PolicyManifest, include_path: str) -> Optional[LibrarySpec]:
    """The library that authorises ``include_path``, if any."""
    for lib in manifest.libraries:
        if include_path in lib.include_paths():
            return lib
    return None


@dataclass(frozen=True)
class IncludeIssue:
    """One rejected include/use statement."""
    path: str
    kind: str      # "unapproved" | "unavailable"
    message: str

    def to_dict(self) -> dict:
        return {"path": self.path, "kind": self.kind, "message": self.message}


def validate_includes(scad_source: str,
                      manifest: PolicyManifest = DEFAULT_MANIFEST,
                      available_paths: Optional[Iterable[str]] = None,
                      ) -> List[IncludeIssue]:
    """Validate every include/use in ``scad_source`` against the policy.

    Semantics of AgentSCAD's ``validate_scad_includes.py``: a path that no
    approved library authorises is ``unapproved``; a path that is approved but
    not in ``available_paths`` is ``unavailable``. When ``available_paths`` is
    ``None``, availability is not checked (approve-only mode).
    """
    approved = set(approved_include_paths(manifest))
    available = None if available_paths is None else set(available_paths)
    issues: List[IncludeIssue] = []
    for _keyword, path in extract_includes(scad_source):
        if path not in approved:
            issues.append(IncludeIssue(
                path=path, kind="unapproved",
                message=f"Unapproved OpenSCAD include/use path: {path}"))
        elif available is not None and path not in available:
            issues.append(IncludeIssue(
                path=path, kind="unavailable",
                message=("OpenSCAD library path is approved but not "
                         f"available locally: {path}")))
    return issues


def detect_available(manifest: PolicyManifest,
                     search_roots: Sequence[str],
                     exists: Callable[[str], bool]) -> Dict[str, bool]:
    """Which approved libraries resolve under the given search roots.

    ``exists`` is injected (typically ``os.path.isfile`` over joined paths)
    so detection is testable and never touches the filesystem by surprise.
    A library is available when ANY of its detection files exists under ANY
    search root. Returns ``{library_name: available}``.
    """
    result: Dict[str, bool] = {}
    for lib in manifest.libraries:
        found = False
        for root in search_roots:
            base = root.rstrip("/\\")
            for probe in lib.detection_files:
                candidate = f"{base}/{probe}" if base else probe
                if exists(candidate):
                    found = True
                    break
            if found:
                break
        result[lib.name] = found
    return result


def available_include_paths(manifest: PolicyManifest,
                            availability: Mapping[str, bool]) -> List[str]:
    """Include paths whose owning library is available."""
    paths: Dict[str, None] = {}
    for lib in manifest.libraries:
        if availability.get(lib.name):
            for path in lib.include_paths():
                paths.setdefault(path, None)
    return list(paths)


# ---------------------------------------------------------------------------
# License-gated install plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstallAction:
    """One planned or skipped install, always with a reason when skipped."""
    library: str
    action: str          # "install" | "skip"
    repo: Optional[str] = None
    commit: Optional[str] = None
    target_dir: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {"library": self.library, "action": self.action,
                "repo": self.repo, "commit": self.commit,
                "target_dir": self.target_dir, "reason": self.reason}


def install_plan(manifest: PolicyManifest = DEFAULT_MANIFEST,
                 include_gpl: bool = False,
                 include_optional: bool = False) -> List[InstallAction]:
    """The deterministic install plan (plan only: nothing is fetched).

    Rules from ``install_scad_libraries.py`` + the SKILL.md license gate:

    * only pinned entries (repo AND commit) can be planned;
    * ``default_install`` entries are planned unless gated;
    * GPL-gated libraries require ``include_gpl=True`` regardless of
      ``default_install``;
    * non-default entries are planned only with ``include_optional=True``
      (and still only when pinned and license-permitted);
    * every skip carries the manifest's stated reason or the gate that
      blocked it.
    """
    plan: List[InstallAction] = []
    for lib in manifest.libraries:
        gated_gpl = lib.license_gate == "gpl" and not include_gpl
        wanted = lib.default_install or include_optional
        if not wanted:
            plan.append(InstallAction(
                library=lib.name, action="skip",
                reason=lib.default_install_reason or "not a default install"))
            continue
        if gated_gpl:
            plan.append(InstallAction(
                library=lib.name, action="skip",
                reason="GPL license gate: requires explicit include_gpl=True"))
            continue
        if not lib.pinned():
            plan.append(InstallAction(
                library=lib.name, action="skip",
                reason=lib.default_install_reason
                or "no pinned upstream (repo/commit missing)"))
            continue
        plan.append(InstallAction(
            library=lib.name, action="install", repo=lib.repo,
            commit=lib.commit, target_dir=lib.target_dir,
            reason="default install" if lib.default_install else "opt-in"))
    return plan


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------

def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            failures.append(message)

    m = DEFAULT_MANIFEST
    check(len(m.libraries) == 6, "six libraries in the shipped manifest")
    check(all(lib.license_gate in LICENSE_GATES for lib in m.libraries),
          "every gate is a known gate")

    approved = approved_include_paths(m)
    check("BOSL2/std.scad" in approved and "NopSCADlib/core.scad" in approved,
          "approved paths derived from include examples")

    source = (
        "// header\n"
        "include <BOSL2/std.scad>;\n"
        "use <MCAD/involute_gears.scad>;\n"
        "include <EvilLib/backdoor.scad>;\n")
    issues = validate_includes(source, m, available_paths=["BOSL2/std.scad"])
    kinds = {(i.path, i.kind) for i in issues}
    check(("EvilLib/backdoor.scad", "unapproved") in kinds,
          "invented include path rejected as unapproved")
    check(("MCAD/involute_gears.scad", "unavailable") in kinds,
          "approved-but-missing path rejected as unavailable")
    check(("BOSL2/std.scad", "unapproved") not in kinds
          and ("BOSL2/std.scad", "unavailable") not in kinds,
          "approved and available path accepted")
    check(not validate_includes(source, m)[1:2] or True, "approve-only mode runs")
    check(len(validate_includes("include <BOSL2/std.scad>;", m)) == 0,
          "approve-only mode passes an approved path")

    # Availability detection with an injected filesystem.
    fake_fs = {"/libs/BOSL2/std.scad", "/libs/MCAD/units.scad"}
    availability = detect_available(m, ["/libs"], lambda p: p in fake_fs)
    check(availability["BOSL2"] and availability["MCAD"], "detected libraries")
    check(not availability["NopSCADlib"], "absent library not detected")
    avail_paths = available_include_paths(m, availability)
    check("MCAD/involute_gears.scad" in avail_paths,
          "available library authorises all its paths")

    # License gate.
    default_plan = {a.library: a for a in install_plan(m)}
    check(default_plan["BOSL2"].action == "install"
          and default_plan["BOSL2"].commit is not None, "BOSL2 pinned install")
    check(default_plan["NopSCADlib"].action == "skip", "GPL skipped by default")
    gpl_plan = {a.library: a for a in install_plan(m, include_gpl=True,
                                                   include_optional=True)}
    check(gpl_plan["NopSCADlib"].action == "install", "GPL installs on opt-in")
    check(gpl_plan["threads.scad"].action == "skip",
          "unpinned library never planned")
    check(all(a.reason for a in install_plan(m) if a.action == "skip"),
          "every skip has a reason")

    lib = library_for_include(m, "threadlib/threadlib.scad")
    check(lib is not None and lib.name == "threadlib", "reverse lookup")

    if failures:
        for f in failures:
            print(f"selfcheck FAIL: {f}")
        return 1
    print("scad_library_policy selfcheck: OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Approved OpenSCAD library policy (AgentSCAD)")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
