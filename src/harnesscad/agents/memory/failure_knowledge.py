"""CAD failure-mode knowledge, as UNVERIFIED skill-pack recipes.

The three failure modes covered here -- floating parts, missing holes and
non-manifold booleans -- and the four-part diagnosis shape are taken from the
failure notes in **AgentSCAD** (``resources/cad_repos/AgentSCAD-main``,
``cad_knowledge/failures/*.md``; that repo is MIT). ``_PROVENANCE`` records
that origin on every entry. The symptom / cause / strategy / prevention text
below is written for this repo rather than copied: what is borrowed is the
taxonomy, not the sentences.

Each diagnosis is a *symptom* a validator can observe, the
*causes* that produce it, an ordered *repair strategy*, and the *prevention*
rules that stop it recurring. That is exactly a
:class:`~harnesscad.agents.memory.skillpack.PackSkill`: triggers (the symptom),
workflow (the repair strategy), safety rules (prevention), verification (what
must be re-checked before success may be claimed).

VERIFICATION-FIRST
------------------
This is a recipe, not executed geometry -- plausible text that no build has
confirmed -- and the pack convention is verification-first. So
:func:`register_failure_knowledge` routes through
:func:`~harnesscad.agents.memory.skillpack.register_pack`, which admits every
entry with ``verified=False`` and no expander. Nothing here flips ``verified``;
only :meth:`~harnesscad.agents.memory.skills.SkillLibrary.add_verified` can,
and only by executing an expansion that actually builds. Until then the
knowledge is for the planner's retrieval and for human review --
``verified_prompt_lines`` will not surface it to a model.

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

# Origin of the failure taxonomy, recorded on every entry. The prose in this
# module is ours; what these fields credit is where the three failure modes and
# the four-part diagnosis shape came from. ``license`` records the source repo's
# licence so a later reader can check the lineage for themselves.
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
    """The three failure diagnoses as a :class:`SkillPack`."""
    return SkillPack(
        name=FAILURE_PACK_NAME,
        description=("CAD failure modes: symptom, causes, repair "
                     "strategy and prevention per validation failure. "
                     "Unverified reference knowledge."),
        provenance=dict(_PROVENANCE),
        skills=[
            _skill(
                "repair-floating-parts",
                "floating_parts.md",
                "Repair a part whose validation found disconnected components.",
                symptom=("The model splits into more than one body: the "
                         "connectivity check reports islands of geometry that "
                         "would come off the printer as loose pieces."),
                causes=[
                    "No enclosing union(): several solids sit side by side at "
                    "module scope, so each one meshes as its own island.",
                    "A difference() cut all the way through the material that "
                    "used to join two regions of the part.",
                    "A translate() carried a child clear of the parent body "
                    "and nothing was added to span the resulting gap.",
                    "Components that were meant to touch are offset slightly, "
                    "so they mesh apart (a lid sitting above its body with no "
                    "runner between them).",
                ],
                strategy=[
                    "Put the whole part inside one top-level union().",
                    "Span any deliberate separation with a bridge or runner.",
                    "Re-check every translation offset and pull stray features "
                    "back inside the main body.",
                    "If the file really is a multi-part assembly, move the "
                    "second part far enough away that the separation reads as "
                    "intentional instead of as a defect.",
                ],
                prevention=[
                    "Keep every solid inside the single union() of the "
                    "top-level part module.",
                    "Sink ribs, bosses and standoffs into the base by at least "
                    "the merge tolerance so the union has material to fuse.",
                    "Reach for the library's boss helper instead of dropping a "
                    "bare cylinder in for a standoff.",
                    "Check that each translation leaves the feature inside the "
                    "expected body bounds.",
                ],
                verification=[
                    "Re-run the connectivity check: the part must come back as "
                    "exactly one connected component.",
                ],
            ),
            _skill(
                "repair-missing-holes",
                "missing_holes.md",
                "Repair a part whose hole_count validation found fewer holes "
                "than requested.",
                symptom=("The hole_count check comes back short: N holes were "
                         "asked for and only M of them, M < N, are present in "
                         "the rendered part."),
                causes=[
                    "The pattern loop computes coordinates that fall outside "
                    "the body, so those cutters remove nothing.",
                    "The cutters were subtracted from the wrong solid, leaving "
                    "the intended parent untouched.",
                    "Circle resolution is too coarse, so what should be a hole "
                    "renders as a low-sided polygon.",
                    "The cutting cylinder is shorter than the material it has "
                    "to clear, so it leaves a blind pocket instead of a hole.",
                ],
                strategy=[
                    "Count how many cutters the subtraction blocks actually "
                    "emit, not how many the pattern was meant to emit.",
                    "Confirm each hole centre lies inside the part bounding "
                    "box.",
                    "Confirm each cutter is taller than the material at its "
                    "location.",
                    "Place the holes that are missing.",
                    "Run hole_count again.",
                ],
                prevention=[
                    "Reach for the bolt-pattern and circular-array helpers "
                    "instead of placing holes by hand.",
                    "Always run cutters past both faces of the material.",
                    "Leave enough edge margin that a hole cannot break out "
                    "through the side of the part.",
                ],
                verification=[
                    "Re-run hole_count: the rendered part must carry exactly "
                    "the requested number of through-holes.",
                ],
            ),
            _skill(
                "repair-non-manifold-boolean",
                "non_manifold_boolean.md",
                "Repair a mesh that renders but fails the manifold check.",
                symptom=("The part renders on screen, but the mesh check "
                         "rejects it as non-manifold: the export carries gaps, "
                         "self-intersections or degenerate triangles."),
                causes=[
                    "Two solids meet on exactly the same plane, which leaves "
                    "the boolean with no way to decide which side of that "
                    "surface is inside the part.",
                    "Two subtracted volumes meet edge to edge and the material "
                    "left between them has no thickness at all.",
                    "A minkowski sum with a very small radius emits triangles "
                    "of near-zero area.",
                    "A hand-written polyhedron whose faces never close into a "
                    "watertight shell.",
                ],
                strategy=[
                    "Read the offending location off the validation report.",
                    "Overlap the unioned solids by the merge tolerance so they "
                    "have material in common.",
                    "Run the subtracted volumes past the part boundary.",
                    "Swap the pattern that caused it: a hull is safer than a "
                    "minkowski for simple roundings, and coplanar faces can be "
                    "nudged apart.",
                ],
                prevention=[
                    "Give every boolean union a merge-tolerance overlap.",
                    "Run cutting cylinders and cubes past the surfaces they "
                    "pass through.",
                    "Reach for the library's rounded-box and boss helpers.",
                    "Never leave two solids in a union touching face to face "
                    "with no overlap at all.",
                ],
                verification=[
                    "Re-run the manifold check: every edge must belong to "
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
        description="CAD failure-mode knowledge as unverified skill-pack "
                    "recipes.")
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
