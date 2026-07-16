"""analysis_pipeline -- staged drawing-analysis pipeline, session record, and
edit audit trail.

Ported from the CAD-Annotator reference repo (artifacts/api-server/src/lib/
pipeline-orchestrator.ts plus the analysis-sessions and annotation-edits DB
schemas in lib/db/src/schema). CAD-Annotator runs a four-stage pipeline
(detection -> re-query -> compliance -> DFM), collects per-stage errors so a
late-stage failure still yields partial results, persists everything as an
analysis session with a completed/partial/failed status, and records every
human edit to an annotation as a previous/new JSONB snapshot pair -- then
re-validates compliance on the edited set (optimistic update with
revalidation).

Harness gap filled: harnesscad gained the building blocks in sibling modules
(harnesscad.domain.drawings.annotation_schema for tolerant parsing,
annotation_set_compliance for set-level rules, dfm_review for
manufacturability findings, requery for the self-correction loop; per-frame
GD&T checks already live in harnesscad.domain.drawings.gdt and are not
duplicated) but had no orchestrator tying them into one session artifact with
stage-level fault isolation and an auditable edit history. All impure steps
(the vision detect call, the optional re-query pass, the optional DFM LLM)
are injected callables; sequence numbers are deterministic and no wall clock
is read -- timestamps are simply omitted.

Pure stdlib.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace as dc_replace
from typing import Callable, List, Optional, Sequence, Tuple

from harnesscad.domain.drawings.annotation_schema import (
    Annotation,
    parse_annotation_response,
)
from harnesscad.domain.drawings.annotation_set_compliance import (
    ComplianceIssue,
    compute_compliance_summary,
    validate_compliance,
)
from harnesscad.domain.drawings.dfm_review import DfmFinding, review_dfm

# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

STAGES = ("detection", "requery", "compliance", "dfm")

STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class StageError:
    """A failure recorded against one pipeline stage."""

    stage: str  # "detection" | "requery" | "compliance" | "dfm"
    message: str

    def to_dict(self) -> dict:
        return {"stage": self.stage, "message": self.message}


@dataclass(frozen=True)
class AnnotationEdit:
    """One audit-trail entry: a previous/new snapshot of an edited annotation.

    Mirrors the annotation_edits table (id/createdAt replaced by a
    deterministic per-session ``sequence`` number; no clock is read).
    """

    session_id: str
    annotation_id: str
    previous_value: dict
    new_value: dict
    sequence: int

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "annotation_id": self.annotation_id,
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class AnalysisSession:
    """A complete pipeline run: annotations, issues, findings, stage errors,
    and the edit audit trail. Mirrors the analysis_sessions table."""

    session_id: str
    image_reference: str
    status: str  # "completed" | "partial" | "failed"
    description: Optional[str]
    views: Tuple[str, ...]
    stage_errors: Tuple[StageError, ...]
    annotations: Tuple[Annotation, ...]
    compliance_issues: Tuple[ComplianceIssue, ...]
    dfm_findings: Tuple[DfmFinding, ...]
    edits: Tuple[AnnotationEdit, ...] = ()

    def to_dict(self) -> dict:
        summary = compute_compliance_summary(self.annotations, self.compliance_issues)
        return {
            "session_id": self.session_id,
            "image_reference": self.image_reference,
            "status": self.status,
            "description": self.description,
            "views": list(self.views),
            "stage_errors": [e.to_dict() for e in self.stage_errors],
            "annotations": [a.to_dict() for a in self.annotations],
            "compliance_issues": [i.to_dict() for i in self.compliance_issues],
            "compliance_summary": summary.to_dict(),
            "dfm_findings": [f.to_dict() for f in self.dfm_findings],
            "edits": [e.to_dict() for e in self.edits],
        }


# --------------------------------------------------------------------------- #
# Status rule (persistSession in pipeline-orchestrator.ts)
# --------------------------------------------------------------------------- #


def session_status(errors: Sequence[StageError]) -> str:
    """failed if a detection-stage error, partial if any error, else completed."""
    if any(e.stage == "detection" for e in errors):
        return STATUS_FAILED
    if errors:
        return STATUS_PARTIAL
    return STATUS_COMPLETED


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run_analysis_pipeline(
    detect: Callable[[], str],
    *,
    requery: Optional[Callable[[List[Annotation]], List[Annotation]]] = None,
    dfm_llm: Optional[Callable[[str], str]] = None,
    session_id: str = "session-1",
    image_reference: str = "",
) -> AnalysisSession:
    """Run the staged analysis pipeline and return an AnalysisSession.

    ``detect`` is a callable returning the raw response text of the detection
    model (stage 1); it is parsed tolerantly via
    annotation_schema.parse_annotation_response. ``requery`` is an optional
    callable mapping the detected annotations to a corrected list (e.g. a
    closure over harnesscad.domain.drawings.requery.requery_low_confidence).
    ``dfm_llm`` is forwarded to dfm_review.review_dfm.

    Error handling mirrors the TS orchestrator: every stage is wrapped in
    try/except and a failure is collected as a StageError while the pipeline
    continues with partial results. A detection failure is fatal but still
    returns a (failed, empty) session rather than raising.
    """
    errors: List[StageError] = []
    annotations: List[Annotation] = []
    compliance_issues: List[ComplianceIssue] = []
    dfm_findings: List[DfmFinding] = []
    views: List[str] = ["View 1"]
    description: Optional[str] = None

    # Stage 1: detection -- fatal on failure, but still returns a session.
    try:
        content = detect()
        annotations, views, description = parse_annotation_response(content)
    except Exception as exc:
        errors.append(
            StageError(stage="detection", message=str(exc) or "Detection stage failed")
        )
        return AnalysisSession(
            session_id=session_id,
            image_reference=image_reference,
            status=session_status(errors),
            description=description,
            views=tuple(views),
            stage_errors=tuple(errors),
            annotations=(),
            compliance_issues=(),
            dfm_findings=(),
        )

    # Re-query stage: continue with original annotations on failure.
    if requery is not None:
        try:
            annotations = list(requery(list(annotations)))
        except Exception as exc:
            errors.append(
                StageError(stage="requery", message=str(exc) or "Re-query stage failed")
            )

    # Stage 2: compliance -- continue with empty issues on failure.
    try:
        compliance_issues = validate_compliance(annotations)
    except Exception as exc:
        errors.append(
            StageError(
                stage="compliance", message=str(exc) or "Compliance stage failed"
            )
        )

    # Stage 3: DFM review -- continue with empty findings on failure.
    # (review_dfm already swallows LLM failures; this guards the stage itself.)
    try:
        dfm_findings = review_dfm(annotations, llm=dfm_llm)
    except Exception as exc:
        errors.append(
            StageError(stage="dfm", message=str(exc) or "DFM stage failed")
        )

    return AnalysisSession(
        session_id=session_id,
        image_reference=image_reference,
        status=session_status(errors),
        description=description,
        views=tuple(views),
        stage_errors=tuple(errors),
        annotations=tuple(annotations),
        compliance_issues=tuple(compliance_issues),
        dfm_findings=tuple(dfm_findings),
    )


# --------------------------------------------------------------------------- #
# Edit audit trail (annotation-edits schema + optimistic update with
# revalidation)
# --------------------------------------------------------------------------- #


def apply_annotation_edit(
    session: AnalysisSession, updated_annotation: Annotation
) -> Tuple[AnalysisSession, AnnotationEdit]:
    """Apply a human edit to one annotation and revalidate the set.

    Snapshots the previous and new values (JSON-ready dicts), replaces the
    annotation with the same id, RE-RUNS validate_compliance on the edited
    set (the optimistic-update-with-revalidation idea from CAD-Annotator's
    review UI), and appends the edit to the session's audit trail with a
    deterministic sequence number. Returns (new_session, edit); the input
    session is not mutated. Raises ValueError when no annotation in the
    session has the updated annotation's id.
    """
    previous = None
    new_annotations: List[Annotation] = []
    for ann in session.annotations:
        if ann.id == updated_annotation.id:
            previous = ann
            new_annotations.append(updated_annotation)
        else:
            new_annotations.append(ann)
    if previous is None:
        raise ValueError(
            'annotation "%s" not found in session "%s"'
            % (updated_annotation.id, session.session_id)
        )

    edit = AnnotationEdit(
        session_id=session.session_id,
        annotation_id=updated_annotation.id,
        previous_value=previous.to_dict(),
        new_value=updated_annotation.to_dict(),
        sequence=len(session.edits) + 1,
    )

    new_session = dc_replace(
        session,
        annotations=tuple(new_annotations),
        compliance_issues=tuple(validate_compliance(new_annotations)),
        edits=session.edits + (edit,),
    )
    return new_session, edit


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_SYNTHETIC_DETECT_RESPONSE = json.dumps(
    {
        "annotations": [
            {"id": "ann_1", "type": "datum", "label": "Datum A", "value": "A",
             "view": "Front View",
             "boundingBox": {"x": 10, "y": 10, "width": 5, "height": 5},
             "confidence": 0.97, "datumLetter": "A"},
            {"id": "ann_2", "type": "datum", "label": "Datum B", "value": "B",
             "view": "Front View",
             "boundingBox": {"x": 20, "y": 10, "width": 5, "height": 5},
             "confidence": 0.95, "datumLetter": "B"},
            {"id": "ann_3", "type": "fcf", "label": "Position 0.05 A D",
             "value": "0.05", "view": "Front View",
             "boundingBox": {"x": 30, "y": 40, "width": 20, "height": 6},
             "confidence": 0.88, "geometricCharacteristic": "position",
             "toleranceValue": 0.05, "materialCondition": None,
             "datumReferences": ["A", "D"]},
        ],
        "views": ["Front View"],
        "description": "Synthetic plate",
    }
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` runs the pipeline on injected doubles
    demonstrating a completed run, a partial run (failing re-query stage),
    and a failed run (failing detection), then applies an annotation edit and
    asserts the compliance revalidation and audit-trail invariants."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.drawings.analysis_pipeline",
        description="Staged drawing-analysis pipeline + session record + edit "
        "audit trail (ported from CAD-Annotator). All model calls injected.",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="run completed/partial/failed pipeline paths on injected doubles "
        "and exercise the edit audit trail with compliance revalidation.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the sessions as JSON."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    # Completed path.
    completed = run_analysis_pipeline(
        lambda: _SYNTHETIC_DETECT_RESPONSE,
        session_id="session-ok",
        image_reference="synthetic://plate.png",
    )

    # Partial path: the re-query stage raises.
    def broken_requery(anns: List[Annotation]) -> List[Annotation]:
        raise RuntimeError("vision service unreachable")

    partial = run_analysis_pipeline(
        lambda: _SYNTHETIC_DETECT_RESPONSE,
        requery=broken_requery,
        session_id="session-partial",
        image_reference="synthetic://plate.png",
    )

    # Failed path: detection raises.
    def broken_detect() -> str:
        raise RuntimeError("model timeout")

    failed = run_analysis_pipeline(
        broken_detect,
        session_id="session-failed",
        image_reference="synthetic://plate.png",
    )

    # Edit path: fix ann_3's dangling datum reference D -> B and revalidate.
    fcf = next(a for a in completed.annotations if a.id == "ann_3")
    fixed_fcf = dc_replace(fcf, datum_references=("A", "B"), label="Position 0.05 A B")
    edited_session, edit = apply_annotation_edit(completed, fixed_fcf)

    if args.json:
        print(
            json.dumps(
                {
                    "completed": completed.to_dict(),
                    "partial": partial.to_dict(),
                    "failed": failed.to_dict(),
                    "edited": edited_session.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for s in (completed, partial, failed, edited_session):
            print(
                "%s: status=%s annotations=%d issues=%d dfm=%d errors=%d edits=%d"
                % (
                    s.session_id,
                    s.status,
                    len(s.annotations),
                    len(s.compliance_issues),
                    len(s.dfm_findings),
                    len(s.stage_errors),
                    len(s.edits),
                )
            )
        print(
            "edit #%d on %s: %s -> %s"
            % (
                edit.sequence,
                edit.annotation_id,
                edit.previous_value.get("datum_references"),
                edit.new_value.get("datum_references"),
            )
        )

    before_rules = sorted(i.rule_id for i in completed.compliance_issues)
    after_rules = sorted(i.rule_id for i in edited_session.compliance_issues)
    ok = (
        completed.status == STATUS_COMPLETED
        and len(completed.annotations) == 3
        and "DATUM_REF_EXISTS" in before_rules  # dangling datum D
        and any(f.category == "datum_scheme_completeness" for f in completed.dfm_findings)
        and partial.status == STATUS_PARTIAL
        and partial.stage_errors[0].stage == "requery"
        and len(partial.annotations) == 3  # continued with original annotations
        and failed.status == STATUS_FAILED
        and failed.annotations == ()
        and failed.stage_errors[0].stage == "detection"
        # revalidation removed the dangling-reference error
        and "DATUM_REF_EXISTS" not in after_rules
        and edit.sequence == 1
        and edited_session.edits == (edit,)
        and edit.previous_value["datum_references"] == ["A", "D"]
        and edit.new_value["datum_references"] == ["A", "B"]
        # original session untouched (immutability)
        and completed.edits == ()
    )
    if not ok:
        print("SELFCHECK FAILED")
        return 1
    print("selfcheck OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
