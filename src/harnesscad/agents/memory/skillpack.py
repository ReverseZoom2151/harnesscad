"""Skill packs -- a deterministic, file-based CAD skill format and loader.

Mined from earthtojake/text-to-cad (resources/cad_repos/text-to-cad-main),
which ships its CAD knowledge as *agent skills*: a ``skills/<name>/`` directory
per skill containing a ``SKILL.md`` (YAML frontmatter ``name``/``description``
plus Markdown sections -- workflow, safety rules, defaults, validation) and a
``references/*.md`` set of progressive-disclosure knowledge files. A
``.claude-plugin/plugin.json`` manifest points a coding agent at the pack.
That repo treats the skill files themselves as the product ("Treat ``skills/``
as the product", its AGENTS.md).

This module gives HarnessCAD the same packaging, on harness terms:

* :class:`PackSkill` -- one parsed skill: name, description, trigger
  conditions, ordered workflow recipe, defaults, safety rules, verification
  criteria, and full reference texts, with provenance.
* :class:`SkillPack` -- a named, versioned collection of :class:`PackSkill`
  with deterministic JSON (de)serialisation via the shared persistence layer.
* :func:`import_skill_dir` / :func:`import_pack` -- the importer: ingest a
  real text-to-cad style skill tree (``SKILL.md`` + ``references/``) into a
  pack. The shipped corpus at :func:`default_pack_path` was produced by this
  importer over the text-to-cad skill set.
* :func:`register_pack` -- the bridge into the execution-verified
  :class:`~harnesscad.agents.memory.skills.SkillLibrary`.

VERIFICATION-FIRST INVARIANT
----------------------------
An imported skill is a *recipe written by someone else*: plausible text, not
executed geometry. It therefore enters the library **UNVERIFIED**
(``Skill.verified == False``) and can only be promoted through the existing
Voyager gate, :meth:`SkillLibrary.add_verified`, once an op-template expander
exists and its expansion actually builds. Nothing in this module ever flips
``verified`` itself, and :func:`verified_prompt_lines` -- the only helper that
formats skills for a model prompt -- refuses to surface unverified entries.
Unverified recipes are for the *planner's* retrieval and for human review;
they are never injected into the model's prompt as trusted construction
knowledge.

Stdlib-only, deterministic, absolute imports. ``--selfcheck`` exercises the
frontmatter parser, section extraction, JSON round-trip, and the
verification-first bridge on a synthetic in-memory skill tree.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import harnesscad.agents.memory.persistence as persistence
from harnesscad.agents.memory.skills import Expander, Skill, SkillLibrary

__all__ = [
    "SKILLPACK_VERSION",
    "PackSkill",
    "SkillPack",
    "parse_skill_md",
    "import_skill_dir",
    "import_pack",
    "register_pack",
    "verified_prompt_lines",
    "unverified_names",
    "default_pack_path",
    "main",
]

SKILLPACK_VERSION = 1

# Section-heading keywords -> PackSkill field. Matched case-insensitively
# against each `## ` heading of a SKILL.md body; first hit wins per section.
_TRIGGER_HEADINGS = ("use this skill when", "when to use")
_WORKFLOW_HEADINGS = ("workflow",)
_SAFETY_HEADINGS = ("safety", "non-negotiables", "hard rules", "core rules")
_DEFAULT_HEADINGS = ("default",)
_VERIFY_HEADINGS = ("validation", "verification", "inspection")


@dataclass
class PackSkill:
    """One file-based skill: a named recipe with triggers and verification.

    ``workflow`` is the ordered recipe (the numbered steps of the source
    skill); ``verification`` is what must be checked before success may be
    claimed; ``references`` maps a reference name to its full Markdown text
    (the progressive-disclosure knowledge files). ``sections`` keeps every
    SKILL.md section verbatim under its heading so no source knowledge is
    dropped by the field extraction.
    """

    name: str
    description: str
    triggers: List[str] = field(default_factory=list)
    workflow: List[str] = field(default_factory=list)
    defaults: List[str] = field(default_factory=list)
    safety_rules: List[str] = field(default_factory=list)
    verification: List[str] = field(default_factory=list)
    references: Dict[str, str] = field(default_factory=dict)
    sections: Dict[str, str] = field(default_factory=dict)
    provenance: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "triggers": list(self.triggers),
            "workflow": list(self.workflow),
            "defaults": list(self.defaults),
            "safety_rules": list(self.safety_rules),
            "verification": list(self.verification),
            "references": dict(self.references),
            "sections": dict(self.sections),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PackSkill":
        return cls(
            name=str(d["name"]),
            description=str(d.get("description", "")),
            triggers=[str(t) for t in d.get("triggers", [])],
            workflow=[str(s) for s in d.get("workflow", [])],
            defaults=[str(s) for s in d.get("defaults", [])],
            safety_rules=[str(s) for s in d.get("safety_rules", [])],
            verification=[str(s) for s in d.get("verification", [])],
            references={str(k): str(v) for k, v in d.get("references", {}).items()},
            sections={str(k): str(v) for k, v in d.get("sections", {}).items()},
            provenance={str(k): str(v) for k, v in d.get("provenance", {}).items()},
        )


@dataclass
class SkillPack:
    """A named collection of :class:`PackSkill` with provenance."""

    name: str
    description: str = ""
    provenance: Dict[str, str] = field(default_factory=dict)
    skills: List[PackSkill] = field(default_factory=list)

    def names(self) -> List[str]:
        return [s.name for s in self.skills]

    def get(self, name: str) -> PackSkill:
        for s in self.skills:
            if s.name == name:
                return s
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {
            "version": SKILLPACK_VERSION,
            "name": self.name,
            "description": self.description,
            "provenance": dict(self.provenance),
            "skills": [s.to_dict() for s in self.skills],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkillPack":
        version = d.get("version", SKILLPACK_VERSION)
        if version != SKILLPACK_VERSION:
            raise ValueError(
                f"unsupported skillpack version {version!r} "
                f"(this loader understands {SKILLPACK_VERSION})")
        return cls(
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            provenance={str(k): str(v) for k, v in d.get("provenance", {}).items()},
            skills=[PackSkill.from_dict(s) for s in d.get("skills", [])],
        )

    def save(self, path: str) -> None:
        persistence.dump_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str) -> "SkillPack":
        return cls.from_dict(persistence.load_json(path))


# ---------------------------------------------------------------------------
# SKILL.md parsing (deterministic; a minimal frontmatter subset, no YAML dep)
# ---------------------------------------------------------------------------
def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Split ``---`` YAML-style frontmatter from a Markdown body.

    Understands the ``key: value`` single-line subset that text-to-cad's
    SKILL.md files actually use (``name`` and a one-line ``description``).
    A file without frontmatter yields an empty meta dict and the full body.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: Dict[str, str] = {}
    body_start = len(lines)
    current_key: Optional[str] = None
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == "---":
            body_start = i + 1
            break
        if line[:1] in (" ", "\t") and current_key:
            # folded continuation of the previous value
            meta[current_key] = (meta[current_key] + " " + line.strip()).strip()
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            current_key = key.strip()
            meta[current_key] = value.strip().strip("\"'")
    body = "\n".join(lines[body_start:])
    return meta, body


def _split_sections(body: str) -> List[Tuple[str, str]]:
    """Split a Markdown body into ``(heading, text)`` pairs on ``## `` headings.

    Text before the first ``## `` heading is returned under the ``""`` heading.
    Code fences are respected: a ``## `` line inside a fence is body text.
    """
    sections: List[Tuple[str, str]] = []
    heading = ""
    buf: List[str] = []
    in_fence = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("## "):
            sections.append((heading, "\n".join(buf).strip()))
            heading = line[3:].strip()
            buf = []
            continue
        buf.append(line)
    sections.append((heading, "\n".join(buf).strip()))
    return [(h, t) for h, t in sections if t or h]


def _strip_fences(text: str) -> str:
    """Drop fenced code blocks; extraction fields keep prose, not shell."""
    out: List[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


def _bullet_items(text: str) -> List[str]:
    """Collect ``- `` bullet items (with hanging-indent continuations)."""
    items: List[str] = []
    for line in _strip_fences(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
        elif items and line[:1] in (" ", "\t") and stripped:
            items[-1] = items[-1] + " " + stripped
    return items


def _numbered_items(text: str) -> List[str]:
    """Collect ``1.``-style numbered items (with continuations)."""
    items: List[str] = []
    for line in _strip_fences(text).splitlines():
        stripped = line.strip()
        head = stripped.split(".", 1)
        if len(head) == 2 and head[0].isdigit() and head[1][:1] == " ":
            items.append(head[1].strip())
        elif items and stripped and line[:1] in (" ", "\t"):
            items[-1] = items[-1] + " " + stripped
    return items


def _sentences(text: str) -> List[str]:
    flat = " ".join(_strip_fences(text).split())
    out = []
    for raw in flat.split(". "):
        s = raw.strip().rstrip(".")
        if s:
            out.append(s + ".")
    return out


def _heading_matches(heading: str, keywords: Sequence[str]) -> bool:
    low = heading.lower()
    return any(k in low for k in keywords)


def parse_skill_md(text: str, provenance: Optional[Dict[str, str]] = None) -> PackSkill:
    """Parse one SKILL.md text into a :class:`PackSkill`.

    Field extraction is heuristic but deterministic: sections are routed by
    heading keywords, and every section is additionally kept verbatim in
    ``sections`` so nothing is lost when a heading matches no field.
    """
    meta, body = _parse_frontmatter(text)
    name = meta.get("name", "").strip()
    description = meta.get("description", "").strip()
    sections = _split_sections(body)
    skill = PackSkill(name=name, description=description,
                      provenance=dict(provenance or {}))
    if not name:
        # fall back to the first `# ` title
        for line in body.splitlines():
            if line.startswith("# "):
                skill.name = line[2:].strip().lower().replace(" ", "-")
                break
    for heading, sec_text in sections:
        if heading:
            skill.sections[heading] = sec_text
        if _heading_matches(heading, _TRIGGER_HEADINGS):
            skill.triggers.extend(_sentences(sec_text))
        elif _heading_matches(heading, _WORKFLOW_HEADINGS):
            if not skill.workflow:
                skill.workflow = _numbered_items(sec_text) or _bullet_items(sec_text)
        elif _heading_matches(heading, _SAFETY_HEADINGS):
            skill.safety_rules.extend(
                _bullet_items(sec_text) or _numbered_items(sec_text))
        elif _heading_matches(heading, _DEFAULT_HEADINGS):
            skill.defaults.extend(_bullet_items(sec_text))
        elif _heading_matches(heading, _VERIFY_HEADINGS):
            skill.verification.extend(
                _bullet_items(sec_text) or _sentences(sec_text))
    if not skill.triggers and description:
        # the frontmatter description doubles as the trigger condition
        skill.triggers = [description]
    return skill


# ---------------------------------------------------------------------------
# Importer over a real skill tree (SKILL.md + references/*.md per directory)
# ---------------------------------------------------------------------------
def import_skill_dir(path: Path,
                     provenance: Optional[Dict[str, str]] = None,
                     include_references: bool = True) -> PackSkill:
    """Ingest one ``<skill>/SKILL.md`` (+ ``references/*.md``) directory."""
    path = Path(path)
    skill_md = path / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"no SKILL.md under {path}")
    prov = dict(provenance or {})
    prov.setdefault("path", path.name)
    skill = parse_skill_md(skill_md.read_text(encoding="utf-8", errors="replace"),
                           provenance=prov)
    if not skill.name:
        skill.name = path.name
    if include_references:
        refs_dir = path / "references"
        if refs_dir.is_dir():
            for ref in sorted(refs_dir.glob("*.md")):
                skill.references[ref.stem] = ref.read_text(
                    encoding="utf-8", errors="replace")
    return skill


def import_pack(root: Path, name: str, description: str = "",
                provenance: Optional[Dict[str, str]] = None,
                include_references: bool = True) -> SkillPack:
    """Ingest every ``<root>/<skill>/SKILL.md`` into one :class:`SkillPack`."""
    root = Path(root)
    pack = SkillPack(name=name, description=description,
                     provenance=dict(provenance or {}))
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "SKILL.md").is_file():
            prov = dict(pack.provenance)
            pack.skills.append(import_skill_dir(
                child, provenance=prov, include_references=include_references))
    if not pack.skills:
        raise ValueError(f"no skill directories with SKILL.md under {root}")
    return pack


# ---------------------------------------------------------------------------
# Bridge into the execution-verified SkillLibrary (verification-first)
# ---------------------------------------------------------------------------
def _recipe_description(ps: PackSkill) -> str:
    """A retrieval-friendly one-string description: description + triggers."""
    parts = [ps.description] if ps.description else []
    parts.extend(ps.triggers[:3])
    return " ".join(parts)[:2000]


def _no_expander(name: str) -> Expander:
    def _raise(**_params: Any) -> list:
        raise RuntimeError(
            f"imported pack skill '{name}' has no op-template expander yet; "
            f"it is a recipe, not executable geometry. Attach an expander and "
            f"promote it through SkillLibrary.add_verified before expanding.")
    return _raise


def register_pack(library: SkillLibrary, pack: SkillPack,
                  expanders: Optional[Dict[str, Expander]] = None,
                  overwrite: bool = False) -> List[str]:
    """Register a pack's skills into ``library`` as UNVERIFIED entries.

    Each imported skill becomes a :class:`Skill` with ``verified=False``. When
    ``expanders`` supplies an op-template for a name it is attached, but the
    skill is still NOT verified here -- promotion happens only through
    :meth:`SkillLibrary.add_verified`, which executes the expansion on a fresh
    session and admits it only if the geometry builds. Names already present
    in the library are skipped unless ``overwrite`` is set, so an import can
    never displace an execution-verified skill by accident.

    Returns the names actually registered.
    """
    expanders = expanders or {}
    added: List[str] = []
    for ps in pack.skills:
        if ps.name in library and not overwrite:
            continue
        library.register(Skill(
            name=ps.name,
            description=_recipe_description(ps),
            template=expanders.get(ps.name, _no_expander(ps.name)),
            params={},
            sample_params={},
            verified=False,
        ))
        added.append(ps.name)
    return added


def verified_prompt_lines(library: SkillLibrary, query: Optional[str] = None,
                          k: int = 5) -> List[str]:
    """Format skills for injection into a model prompt -- VERIFIED ONLY.

    This is the single prompt-facing accessor for imported skills and it
    hard-filters on ``verified``: an unverified recipe is never surfaced to
    the model, no matter how well it matches the query. Retrieval over
    unverified recipes is a planner/human concern (use the pack directly).
    """
    candidates = (library.find(query, k=max(k, len(library.names())))
                  if query else [library.get(n) for n in library.names()])
    lines: List[str] = []
    for sk in candidates:
        if not sk.verified:
            continue
        lines.append(f"- {sk.name}: {sk.description}")
        if len(lines) >= k:
            break
    return lines


def unverified_names(library: SkillLibrary) -> List[str]:
    """Names registered but not yet execution-verified (for reporting)."""
    return [n for n in library.names() if not library.get(n).verified]


def default_pack_path() -> Path:
    """The shipped imported corpus (text-to-cad's skill set)."""
    return (Path(__file__).resolve().parents[2]
            / "data" / "skillpacks" / "text_to_cad.json")


# ---------------------------------------------------------------------------
# selfcheck + CLI
# ---------------------------------------------------------------------------
_SAMPLE_SKILL_MD = """\
---
name: sample-plate
description: Create validated rectangular plates. Use for plate briefs.
---

# Sample plate

## Use this skill when

Use this skill when the user asks for a flat plate. Also use it for panels.

## Default assumptions

- Units: millimeters.
- M3/M4/M5 normal clearance holes: 3.4/4.5/5.5 mm unless another standard is requested.

## Workflow

1. Write a brief.
2. Plan before coding.

```bash
1. this numbered line is inside a fence and must be ignored
```

3. Validate geometrically.

## Non-negotiables

- Report only checks that actually ran.

## Validation

- One positive-volume solid.
"""


def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            failures.append(message)

    ps = parse_skill_md(_SAMPLE_SKILL_MD, provenance={"repo": "selfcheck"})
    check(ps.name == "sample-plate", "frontmatter name parsed")
    check("plate briefs" in ps.description.lower(), "frontmatter description parsed")
    check(len(ps.triggers) == 2, f"two trigger sentences, got {ps.triggers!r}")
    check(ps.workflow == ["Write a brief.", "Plan before coding.",
                          "Validate geometrically."],
          f"workflow steps parsed, fence skipped: {ps.workflow!r}")
    check(any("3.4/4.5/5.5" in d for d in ps.defaults), "defaults captured")
    check(ps.safety_rules == ["Report only checks that actually ran."],
          f"non-negotiables routed to safety rules: {ps.safety_rules!r}")
    check(ps.verification == ["One positive-volume solid."],
          "verification criteria captured")
    check("Workflow" in ps.sections, "sections kept verbatim")

    pack = SkillPack(name="selfcheck-pack", description="synthetic",
                     provenance={"repo": "selfcheck"}, skills=[ps])
    pack2 = SkillPack.from_dict(pack.to_dict())
    check(pack2.to_dict() == pack.to_dict(), "JSON round-trip is exact")

    lib = SkillLibrary()
    added = register_pack(lib, pack)
    check(added == ["sample-plate"], "pack skill registered")
    check("sample-plate" in lib, "registered into library")
    check(lib.get("sample-plate").verified is False,
          "imported skill enters UNVERIFIED")
    check(unverified_names(lib) == ["sample-plate"], "unverified reported")
    check(verified_prompt_lines(lib, query="plate") == [],
          "unverified recipe never reaches the prompt")
    try:
        lib.expand("sample-plate")
        check(False, "expanding a recipe without an expander must raise")
    except RuntimeError:
        pass
    # re-registration cannot displace an existing entry without overwrite
    check(register_pack(lib, pack) == [], "no silent overwrite on re-import")

    # promotion path: attach a real expander and go through add_verified
    from harnesscad.agents.memory.skills import plate_ops

    class _OkResult:
        ok = True

    class _Session:
        def apply_ops(self, ops: list) -> "_OkResult":
            return _OkResult() if ops else _OkResult()

    promoted = Skill(name="sample-plate", description=ps.description,
                     template=plate_ops, params={}, sample_params={})
    check(lib.add_verified(promoted, lambda: _Session()) is True,
          "promotion through add_verified succeeds with a real expander")
    check(lib.get("sample-plate").verified is True, "promoted skill verified")
    check(verified_prompt_lines(lib, query="plate")
          == [f"- sample-plate: {ps.description}"],
          "verified skill now surfaces to the prompt")

    for message in failures:
        print("selfcheck FAIL: " + message)
    print("selfcheck: %s" % ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="skillpack",
        description="deterministic file-based CAD skill packs "
                    "(import, inspect, verification-first library bridge)")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the built-in self-test and exit")
    parser.add_argument("--import-dir", metavar="ROOT",
                        help="import every ROOT/<skill>/SKILL.md into a pack")
    parser.add_argument("--name", default="imported-pack",
                        help="pack name for --import-dir")
    parser.add_argument("--repo", default="",
                        help="source repository recorded as provenance")
    parser.add_argument("--no-references", action="store_true",
                        help="skip ingesting references/*.md texts")
    parser.add_argument("-o", "--out", metavar="PACK.json",
                        help="write the imported pack here")
    parser.add_argument("--show", metavar="PACK.json",
                        help="summarise an existing pack file")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return _selfcheck()
    if args.import_dir:
        prov = {"repo": args.repo} if args.repo else {}
        pack = import_pack(Path(args.import_dir), name=args.name,
                           provenance=prov,
                           include_references=not args.no_references)
        if args.out:
            pack.save(args.out)
            print(f"wrote {args.out}: {len(pack.skills)} skills "
                  f"({', '.join(pack.names())})")
        else:
            for ps in pack.skills:
                print(f"{ps.name}: {len(ps.workflow)} steps, "
                      f"{len(ps.references)} references")
        return 0
    if args.show:
        pack = SkillPack.load(args.show)
        print(f"pack {pack.name!r} (provenance {pack.provenance})")
        for ps in pack.skills:
            print(f"  {ps.name}: triggers={len(ps.triggers)} "
                  f"workflow={len(ps.workflow)} defaults={len(ps.defaults)} "
                  f"safety={len(ps.safety_rules)} verify={len(ps.verification)} "
                  f"references={len(ps.references)}")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
