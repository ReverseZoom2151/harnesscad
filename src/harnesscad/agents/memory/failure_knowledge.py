"""AgentSCAD's CAD failure-mode knowledge, as UNVERIFIED skill-pack recipes.

Vendored from **AgentSCAD** (``resources/cad_repos/AgentSCAD-main``,
``cad_knowledge/failures/{floating_parts,missing_holes,non_manifold_boolean}.md``).
Source repo LICENSE: **MIT** -- the symptom / cause / repair-strategy /
prevention content below is reproduced from those files with attribution.

Each failure MD is one diagnosis: a *symptom* a validator can observe, the
*causes* that produce it, an ordered *repair strategy*, and the *prevention*
rules that stop it recurring. That is exactly a
:class:`~harnesscad.agents.memory.skillpack.PackSkill`: triggers (the symptom),
workflow (the repair strategy), safety rules (prevention), verification (what
must be re-checked before success may be claimed).

VERIFICATION-FIRST
------------------
This is a recipe written by someone else -- plausible text, not executed
geometry -- and the pack convention is verification-first. So
:func:`register_failure_knowledge` routes through
:func:`~harnesscad.agents.memory.skillpack.register_pack`, which admits every
entry with ``verified=False`` and no expander. Nothing here flips ``verified``;
only :meth:`~harnesscad.agents.memory.skills.SkillLibrary.add_verified` can,
and only by executing an expansion that actually builds. Until then the
knowledge is for the planner's retrieval and for human review --
``verified_prompt_lines`` will not surface it to a model, which is also why the
source's OpenSCAD-specific numbers (``$fn``, ``_merge_tol``) are safe to carry
verbatim: they are recorded provenance, not injected instruction.

The prose is *reference* text (Category B exemplar corpus), never a prompt.

Stdlib-only, deterministic, absolute imports. ``--selfcheck`` builds the pack,
registers it into a real :class:`SkillLibrary`, and proves every entry lands
unverified, is refused by the expander, and is invisible to the prompt
accessor.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from harnesscad.agents.memory.skillpack import (
    PackSkill,
    SkillPack,
    register_pack,
)
from harnesscad.agents.memory.skills import SkillLibrary

__all__ = [
    "FAILURE_PACK_NAME",
    "build_failure_pack",
    "register_failure_knowledge",
    "main",
]

FAILURE_PACK_NAME = "agentscad_failures"

_PROVENANCE = {
    "repo": "AgentSCAD",
    "path": "cad_knowledge/failures",
    "license": "MIT",
    "status": "unverified-reference",
}


def _skill(name: str, source_md: str, description: str, symptom: str,
           causes: Sequence[str], strategy: Sequence[str],
           prevention: Sequence[str], verification: Sequence[str]) -> PackSkill:
    return PackSkill(
        name=name,
        description=description,
        triggers=[symptom],
        workflow=list(strategy),
        safety_rules=list(prevention),
        verification=list(verification),
        sections={
            "Symptom": symptom,
            "Common Causes": "\n".join(f"{i}. {c}" for i, c in
                                       enumerate(causes, 1)),
            "Repair Strategy": "\n".join(f"{i}. {s}" for i, s in
                                         enumerate(strategy, 1)),
            "Prevention": "\n".join(f"- {p}" for p in prevention),
        },
        provenance=dict(_PROVENANCE, file=source_md),
    )


def build_failure_pack() -> SkillPack:
    """The three AgentSCAD failure diagnoses as a :class:`SkillPack`."""
    return SkillPack(
        name=FAILURE_PACK_NAME,
        description=("AgentSCAD CAD failure modes: symptom, causes, repair "
                     "strategy and prevention per validation failure. "
                     "Unverified reference knowledge."),
        provenance=dict(_PROVENANCE),
        skills=[
            _skill(
                "repair-floating-parts",
                "floating_parts.md",
                "Repair a part whose validation found disconnected components.",
                symptom=("The generated part contains disconnected components; "
                         "validation detects floating geometry that would print "
                         "separately from the main body."),
                causes=[
                    "Missing union(): multiple solids defined at module scope "
                    "without being wrapped, rendering as separate mesh islands.",
                    "Boolean operations with gaps: a difference() severs the "
                    "connection between two regions of the part.",
                    "Translated children not connected: translate() moves a "
                    "child outside the parent body with no connecting element.",
                    "Misaligned components output as separate meshes (e.g. lid "
                    "separate from body without a connecting runner).",
                ],
                strategy=[
                    "Wrap all geometry in a single union() at the top level.",
                    "Add connecting bridges or runners between separated "
                    "components.",
                    "Verify translation offsets -- ensure features are "
                    "positioned within the main body.",
                    "For multi-part assemblies, offset the second part so it is "
                    "clearly separate rather than accidentally detached.",
                ],
                prevention=[
                    "All geometry must be inside a single union() in the "
                    "top-level part module.",
                    "Features (ribs, bosses, standoffs) must penetrate the base "
                    "by the merge tolerance.",
                    "Prefer a library boss helper over a raw cylinder for "
                    "standoffs.",
                    "Verify translations are within expected body bounds.",
                ],
                verification=[
                    "Re-run the connectivity check: the part must be one "
                    "connected component.",
                ],
            ),
            _skill(
                "repair-missing-holes",
                "missing_holes.md",
                "Repair a part whose hole_count validation found fewer holes "
                "than requested.",
                symptom=("Validation detects fewer holes than expected "
                         "(hole_count check fails): N requested, M < N present."),
                causes=[
                    "Hole pattern uses wrong coordinates: the loop positions "
                    "holes outside the part body.",
                    "Boolean subtraction order: holes are subtracted from the "
                    "wrong parent solid.",
                    "Insufficient circle resolution makes circular holes render "
                    "as polygons.",
                    "Hole depth too shallow: the subtracted cylinder does not "
                    "fully penetrate the part.",
                ],
                strategy=[
                    "Count the holes actually generated inside the subtraction "
                    "blocks.",
                    "Verify hole coordinates are inside the part bounding box.",
                    "Verify hole height exceeds the part thickness at the hole "
                    "location.",
                    "Add the missing holes at the correct positions.",
                    "Re-validate with the hole_count check.",
                ],
                prevention=[
                    "Use the standard bolt-pattern / circular-array helpers "
                    "rather than hand-placed holes.",
                    "Always extend hole height past the part surface.",
                    "Verify the edge margin is large enough that holes do not "
                    "break through edges.",
                ],
                verification=[
                    "Re-run hole_count: the rendered part must contain exactly "
                    "the requested number of through-holes.",
                ],
            ),
            _skill(
                "repair-non-manifold-boolean",
                "non_manifold_boolean.md",
                "Repair a mesh that renders but fails the manifold check.",
                symptom=("The part renders but mesh validation reports "
                         "non-manifold geometry: holes, self-intersections, or "
                         "degenerate triangles in the exported mesh."),
                causes=[
                    "Coincident faces: two solids share exactly the same "
                    "surface plane, so inside/outside is undecidable at the "
                    "boundary.",
                    "Zero-thickness geometry: a difference() creates an "
                    "infinitely thin wall where two subtracted volumes meet.",
                    "Degenerate triangles from minkowski sums with tiny radii.",
                    "Non-closed polyhedron: faces do not form a closed solid.",
                ],
                strategy=[
                    "Identify the non-manifold location from the validation "
                    "report.",
                    "Add merge-tolerance overlaps between unioned solids.",
                    "Extend subtracted volumes past the part boundaries.",
                    "Replace problematic patterns: prefer a hull over a "
                    "minkowski for simple roundings; offset coplanar faces.",
                ],
                prevention=[
                    "Always overlap boolean unions by the merge tolerance.",
                    "Extend subtracted cylinders and cubes past part surfaces.",
                    "Prefer the library rounded-box and boss helpers.",
                    "Never position two solids with exact face contact inside a "
                    "union.",
                ],
                verification=[
                    "Re-run the manifold check: every edge must be shared by "
                    "exactly two faces.",
                ],
            ),
        ],
    )


def register_failure_knowledge(library: SkillLibrary,
                               overwrite: bool = False) -> List[str]:
    """Register the failure pack into ``library`` as UNVERIFIED entries.

    Thin wrapper over :func:`register_pack` -- no expanders are supplied, so
    every entry carries the refusing placeholder and can only become verified
    through the execution gate.
    """
    return register_pack(library, build_failure_pack(), overwrite=overwrite)


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="AgentSCAD failure-mode knowledge as unverified skill-pack "
                    "recipes (cad_knowledge/failures/*.md, MIT).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="prove the pack round-trips and every entry lands "
                             "unverified, unexpandable and prompt-invisible.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    from harnesscad.agents.memory.skillpack import (
        unverified_names, verified_prompt_lines,
    )

    pack = build_failure_pack()
    assert len(pack.skills) == 3
    assert pack.names() == ["repair-floating-parts", "repair-missing-holes",
                            "repair-non-manifold-boolean"]
    print(f"[selfcheck] pack '{pack.name}' carries {len(pack.skills)} "
          f"failure diagnoses")

    # 1. Every skill has all four parts of a diagnosis + provenance.
    for ps in pack.skills:
        assert ps.triggers and ps.workflow and ps.safety_rules
        assert ps.verification, ps.name
        assert set(ps.sections) == {"Symptom", "Common Causes",
                                    "Repair Strategy", "Prevention"}, ps.name
        assert ps.provenance["repo"] == "AgentSCAD"
        assert ps.provenance["license"] == "MIT"
        assert ps.provenance["file"].endswith(".md")
    print("[selfcheck] each diagnosis: symptom + causes + strategy + "
          "prevention + verification, with provenance")

    # 2. JSON round-trip is lossless and deterministic.
    assert SkillPack.from_dict(pack.to_dict()).to_dict() == pack.to_dict()
    assert build_failure_pack().to_dict() == pack.to_dict()
    print("[selfcheck] deterministic, lossless JSON round-trip")

    # 3. Registration is verification-first.
    library = SkillLibrary()
    added = register_failure_knowledge(library)
    assert added == pack.names(), added
    assert sorted(unverified_names(library)) == sorted(added)
    for name in added:
        assert not library.get(name).verified, name
    print(f"[selfcheck] {len(added)} entries registered UNVERIFIED")

    # 4. An unverified recipe refuses to expand -- it is text, not geometry.
    for name in added:
        try:
            library.get(name).template()
        except RuntimeError as exc:
            assert "no op-template expander" in str(exc)
        else:
            raise AssertionError(f"{name} expanded without a verified template")
    print("[selfcheck] unverified recipes refuse to expand")

    # 5. The prompt accessor will not surface them.
    lines = verified_prompt_lines(library, "non-manifold mesh", k=5)
    assert lines == [], lines
    print("[selfcheck] prompt accessor surfaces nothing unverified")

    # 6. A re-register never displaces an existing entry by accident.
    assert register_failure_knowledge(library) == []
    print("[selfcheck] re-import cannot displace a registered skill")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
