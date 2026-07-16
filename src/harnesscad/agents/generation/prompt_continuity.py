"""Continuity-contract reviewer for multi-prompt generation batches.

Ported from Forma-OSS (blueprint_core/prompt_continuity.py: PromptSeed,
PromptContinuityFinding, PromptContinuityReview, PromptContinuityReviewer, and
the supersede-with-child repair pattern of FirecrawlPromptBatchSeeder),
decoupled from the Firecrawl research client and the OpenAI JSONL job queues.
Where the reference minted uuid4 job ids and stamped wall-clock metadata, this
port takes caller-supplied ids and derives deterministic child ids
(f"{job_id}-r1"), so reviews are replayable byte for byte.

Harness gap filled: HarnessCAD generates multi-stage prompt batches (design
plan -> geometry -> fabrication -> docs) but had no reviewer that checks each
prompt carries its continuity obligations -- batch identity, stage, anchor
part, the preserve-earlier-decisions clause -- before the model sees it. This
module complements harnesscad.agents.generation.prompt_evolution (which
mutates and scores prompt VARIANTS for quality) without duplicating it: this
reviewer does not rewrite for quality, it audits a fixed batch for continuity
contract violations and, when a prompt fails, wraps it in an explicit
continuity contract block and issues a deterministic repaired child.

Finding codes (same set as the reference, Firecrawl codes replaced by the
generic source-context code): wrong_model, missing_batch_id,
missing_prompt_index, missing_stage, missing_anchor,
missing_continuity_contract, missing_source_context, and
missing_previous_decisions_clause for prompts with index > 1.

Deterministic: no wall clock, no randomness, no uuid4. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_BATCH_OBJECTIVE = (
    "Iterate a buildable CAD part and its surrounding project artifacts while "
    "preserving continuity across geometry, mechanical, fabrication, "
    "documentation, visual, validation, manufacturing, and release decisions."
)
DEFAULT_CONTINUITY_ANCHOR = "parametric mounting bracket"

_PREVIOUS_DECISIONS_CLAUSE = "preserve decisions from earlier prompts"
_CONTINUITY_CONTRACT_MARKER = "continuity contract"
_SOURCE_CONTEXT_MARKER = "source context"


@dataclass(frozen=True)
class PromptSeed:
    """One prompt in a multi-stage batch."""

    index: int
    stage: str
    prompt: str
    source_query: str


@dataclass(frozen=True)
class ContinuityFinding:
    """A single continuity violation found in a prompt."""

    severity: str
    code: str
    message: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class ContinuityReview:
    """Review outcome for one prompt in a batch."""

    job_id: str
    prompt_index: Optional[int]
    passed: bool
    findings: Tuple[ContinuityFinding, ...]
    revised_prompt: str
    child_job_id: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "job_id": self.job_id,
            "prompt_index": self.prompt_index,
            "passed": self.passed,
            "findings": [finding.to_dict() for finding in self.findings],
            "revised_prompt": self.revised_prompt,
            "child_job_id": self.child_job_id,
        }


class PromptContinuityReviewer:
    """Audits prompts for continuity obligations and repairs the failures.

    The repair follows the Forma-OSS supersede-with-child pattern: the failing
    prompt is never mutated in place; a revised prompt is produced under a
    deterministic child id (f"{job_id}-r1") so callers can record the original
    job as superseded.
    """

    def __init__(self, *, required_anchor: str = DEFAULT_CONTINUITY_ANCHOR) -> None:
        self.required_anchor = required_anchor

    def review(
        self,
        prompt: str,
        *,
        job_id: str,
        prompt_index: int,
        total_prompts: int,
        stage: str,
        batch_id: str,
        objective: str = "",
        anchor: str = "",
        source_context: str = "",
        expected_model: str = "",
        model: str = "",
    ) -> ContinuityReview:
        findings: List[ContinuityFinding] = []
        prompt_lower = (prompt or "").lower()
        anchor_value = anchor or self.required_anchor
        anchor_lower = anchor_value.lower()

        if expected_model and model != expected_model:
            findings.append(
                ContinuityFinding(
                    "error",
                    "wrong_model",
                    f"Expected {expected_model}, got {model or 'unset'}.",
                )
            )
        if not batch_id:
            findings.append(
                ContinuityFinding(
                    "error", "missing_batch_id", "Job metadata is missing batch_id."
                )
            )
        if prompt_index is None:
            findings.append(
                ContinuityFinding(
                    "error",
                    "missing_prompt_index",
                    "Job metadata is missing prompt_index.",
                )
            )
        if not stage:
            findings.append(
                ContinuityFinding(
                    "warning",
                    "missing_stage",
                    "Job metadata is missing stage/namespace.",
                )
            )
        if anchor_lower and anchor_lower not in prompt_lower:
            findings.append(
                ContinuityFinding(
                    "warning",
                    "missing_anchor",
                    "Prompt does not name the continuity anchor.",
                )
            )
        if _CONTINUITY_CONTRACT_MARKER not in prompt_lower:
            findings.append(
                ContinuityFinding(
                    "warning",
                    "missing_continuity_contract",
                    "Prompt does not include an explicit continuity contract.",
                )
            )
        if _SOURCE_CONTEXT_MARKER not in prompt_lower:
            findings.append(
                ContinuityFinding(
                    "warning",
                    "missing_source_context",
                    "Prompt does not include source context.",
                )
            )
        if (
            prompt_index is not None
            and prompt_index > 1
            and _PREVIOUS_DECISIONS_CLAUSE not in prompt_lower
        ):
            findings.append(
                ContinuityFinding(
                    "warning",
                    "missing_previous_decisions_clause",
                    "Prompt does not instruct the model to preserve earlier "
                    "decisions.",
                )
            )

        if not findings:
            return ContinuityReview(
                job_id=job_id,
                prompt_index=prompt_index,
                passed=True,
                findings=(),
                revised_prompt=prompt,
            )

        revised = self.revised_prompt(
            prompt,
            findings=tuple(findings),
            prompt_index=prompt_index,
            total_prompts=total_prompts,
            stage=stage,
            batch_id=batch_id,
            objective=objective,
            anchor=anchor_value,
            source_context=source_context,
        )
        return ContinuityReview(
            job_id=job_id,
            prompt_index=prompt_index,
            passed=False,
            findings=tuple(findings),
            revised_prompt=revised,
            child_job_id=f"{job_id}-r1",
        )

    def revised_prompt(
        self,
        prompt: str,
        *,
        findings: Tuple[ContinuityFinding, ...],
        prompt_index: Optional[int],
        total_prompts: Optional[int],
        stage: str,
        batch_id: str,
        objective: str = "",
        anchor: str = "",
        source_context: str = "",
    ) -> str:
        """Wrap the original prompt in an explicit continuity contract block."""
        anchor_value = anchor or self.required_anchor
        context = (
            source_context
            or "No source context was available; explicitly record that limitation."
        )
        finding_text = "\n".join(
            f"- {finding.code}: {finding.message}" for finding in findings
        )
        previous_clause = (
            "Preserve decisions from earlier prompts in this batch unless a "
            "reviewer finding explicitly requires changing them."
        )
        return (
            f"Continuity contract for {anchor_value}:\n"
            f"- Batch: {batch_id or 'unknown'} prompt "
            f"{prompt_index if prompt_index is not None else '?'} of "
            f"{total_prompts or '?'}.\n"
            f"- Stage: {stage or 'unknown'}.\n"
            f"- Objective: {objective or DEFAULT_BATCH_OBJECTIVE}.\n"
            f"- {previous_clause}\n"
            "- Keep part names, interfaces, datum frames, material choices, "
            "tolerances, and validation criteria consistent across prompts.\n"
            "- Keep the response concise for streaming review: target under 900 "
            "words and avoid exhaustive tables unless essential.\n"
            "- If information is unavailable, say what is missing instead of "
            "inventing contradictory details.\n\n"
            f"Source context:\n{context}\n\n"
            f"Continuity reviewer findings to fix:\n{finding_text}\n\n"
            f"Original prompt:\n{prompt}"
        )

    def review_batch(
        self,
        seeds: Sequence[PromptSeed],
        *,
        batch_id: str,
        job_id_prefix: str = "job",
        objective: str = "",
        anchor: str = "",
        source_context: str = "",
        expected_model: str = "",
        model: str = "",
    ) -> Tuple[ContinuityReview, ...]:
        """Review every seed; repaired prompts get deterministic child ids.

        Job ids are derived from the seed index (f"{job_id_prefix}-{index}"),
        and each failing prompt's repaired child id is f"{job_id}-r1" -- the
        supersede-with-child pattern from Forma-OSS, minus the JSONL queue.
        """
        seeds = tuple(seeds)
        reviews: List[ContinuityReview] = []
        for seed in seeds:
            job_id = f"{job_id_prefix}-{seed.index}"
            reviews.append(
                self.review(
                    seed.prompt,
                    job_id=job_id,
                    prompt_index=seed.index,
                    total_prompts=len(seeds),
                    stage=seed.stage,
                    batch_id=batch_id,
                    objective=objective,
                    anchor=anchor,
                    source_context=source_context,
                    expected_model=expected_model,
                    model=model,
                )
            )
        return tuple(reviews)


def default_cad_prompt_seeds(
    anchor: str = DEFAULT_CONTINUITY_ANCHOR,
) -> Tuple[PromptSeed, ...]:
    """Ten CAD-adapted stages around a named anchor part.

    Adapted from Forma-OSS default_prompt_seeds (which targeted an
    electromechanical monitor); here the stages follow a CAD part from
    overview through geometry, mechanics, fabrication, docs, visuals,
    validation, manufacturing, iteration, and release.
    """
    part = anchor
    return (
        PromptSeed(
            1,
            "product.overview",
            f"Define the design intent for the {part}: user, load case, "
            "envelope, interfaces, constraints, and acceptance criteria.",
            "parametric CAD part design intent load case envelope interfaces",
        ),
        PromptSeed(
            2,
            "product.geometry",
            f"Model the parametric geometry for the {part}: datum frame, "
            "sketch parameters, features, fillets, and hole patterns.",
            "parametric geometry datum frame sketch features hole pattern",
        ),
        PromptSeed(
            3,
            "product.mech",
            f"Analyze the mechanics of the {part}: material, wall thickness, "
            "stress paths, fastener loads, and deflection limits.",
            "bracket mechanics material wall thickness stress fastener load",
        ),
        PromptSeed(
            4,
            "product.fabrication",
            f"Plan fabrication for the {part}: process selection, tolerances, "
            "draft or overhang rules, post-processing, and inspection features.",
            "fabrication process selection tolerances draft overhang inspection",
        ),
        PromptSeed(
            5,
            "project.docs",
            f"Write build documentation for the {part}: drawing notes, "
            "assembly steps, torque values, and acceptance checks.",
            "engineering drawing notes assembly steps torque acceptance checks",
        ),
        PromptSeed(
            6,
            "product.visuals",
            f"Describe render views for the {part}: isometric hero shot, "
            "section view, exploded fasteners, and dimension callouts.",
            "CAD render views isometric section exploded dimension callouts",
        ),
        PromptSeed(
            7,
            "project.validation",
            f"Create a validation plan for the {part}: dimensional checks, "
            "fit tests, load tests, and tolerance stack verification.",
            "part validation dimensional check fit test load test tolerance stack",
        ),
        PromptSeed(
            8,
            "project.manufacturing",
            f"Prepare manufacturing notes for the {part}: batch quantity, "
            "sourcing substitutions, fixturing, QA gates, and packaging.",
            "manufacturing notes batch quantity fixturing QA gates packaging",
        ),
        PromptSeed(
            9,
            "project.iteration",
            f"Review all prior {part} outputs and identify contradictions, "
            "missing interfaces, untestable claims, and prompt updates.",
            "design review continuity contradictions interface checks iteration",
        ),
        PromptSeed(
            10,
            "project.release",
            f"Produce a release-ready summary for the {part}: final spec, "
            "parameter table, known risks, and next iteration backlog.",
            "release checklist final spec parameter table risks backlog",
        ),
    )


# --- selfcheck ---------------------------------------------------------------
def _selfcheck() -> int:
    reviewer = PromptContinuityReviewer(required_anchor="parametric mounting bracket")

    # 1. A raw seed prompt fails: no contract, no source context, and for
    #    index > 1 no previous-decisions clause.
    seeds = default_cad_prompt_seeds()
    assert len(seeds) == 10
    assert seeds[0].stage == "product.overview"
    assert seeds[9].stage == "project.release"

    review2 = reviewer.review(
        seeds[1].prompt,
        job_id="job-2",
        prompt_index=2,
        total_prompts=10,
        stage=seeds[1].stage,
        batch_id="batch-A",
    )
    assert not review2.passed
    codes = {finding.code for finding in review2.findings}
    assert "missing_continuity_contract" in codes
    assert "missing_source_context" in codes
    assert "missing_previous_decisions_clause" in codes
    assert "missing_anchor" not in codes  # seed names the anchor
    assert review2.child_job_id == "job-2-r1"
    assert "Continuity contract for parametric mounting bracket" in review2.revised_prompt
    assert "Original prompt:" in review2.revised_prompt
    assert seeds[1].prompt in review2.revised_prompt

    # 2. The revised prompt passes on re-review (self-healing contract).
    rereview = reviewer.review(
        review2.revised_prompt,
        job_id="job-2-r1",
        prompt_index=2,
        total_prompts=10,
        stage=seeds[1].stage,
        batch_id="batch-A",
    )
    assert rereview.passed, [f.code for f in rereview.findings]
    assert rereview.child_job_id is None
    assert rereview.revised_prompt == review2.revised_prompt

    # 3. Model mismatch and missing batch_id produce error findings.
    bad = reviewer.review(
        "prompt without anything",
        job_id="job-x",
        prompt_index=1,
        total_prompts=1,
        stage="",
        batch_id="",
        expected_model="model-a",
        model="model-b",
    )
    bad_codes = {finding.code for finding in bad.findings}
    assert "wrong_model" in bad_codes
    assert "missing_batch_id" in bad_codes
    assert "missing_stage" in bad_codes
    assert "missing_anchor" in bad_codes
    # index 1: no previous-decisions requirement
    assert "missing_previous_decisions_clause" not in bad_codes
    severities = {f.code: f.severity for f in bad.findings}
    assert severities["wrong_model"] == "error"
    assert severities["missing_stage"] == "warning"

    # 4. review_batch: every seed reviewed, deterministic child ids.
    reviews = reviewer.review_batch(
        seeds, batch_id="batch-A", job_id_prefix="batch-A-job"
    )
    assert len(reviews) == 10
    assert all(not review.passed for review in reviews)  # raw seeds lack contracts
    assert reviews[0].job_id == "batch-A-job-1"
    assert reviews[0].child_job_id == "batch-A-job-1-r1"
    assert reviews[4].child_job_id == "batch-A-job-5-r1"

    # 5. Determinism: identical inputs produce identical reviews.
    reviews_again = reviewer.review_batch(
        seeds, batch_id="batch-A", job_id_prefix="batch-A-job"
    )
    assert reviews == reviews_again
    assert json.dumps([r.to_dict() for r in reviews], sort_keys=True) == json.dumps(
        [r.to_dict() for r in reviews_again], sort_keys=True
    )

    repaired = sum(1 for review in reviews if not review.passed)
    print(
        "PASS prompt_continuity selfcheck: 5 scenarios "
        f"({len(reviews)} seeds reviewed, {repaired} repaired with child ids)"
    )
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Continuity-contract reviewer for multi-prompt batches "
            "(ported from Forma-OSS)."
        )
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="review the default CAD seed batch and assert the finding codes, "
        "repair wrapping, and deterministic child ids.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _selfcheck()
    except AssertionError as exc:
        print(f"SELFCHECK FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
