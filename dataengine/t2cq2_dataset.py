"""Text-to-CadQuery dataset construction: records, prompt templates, retry loop.

Reference implementation of paper 171 -- *Text-to-CadQuery* (repo
``Text-to-CadQuery``): ``data_annotation/gemini_pipeline.py`` (the 170k-sample
annotation pipeline) and ``inference/step1_generate_CadQuery`` (the prompt format
and the ``test_filtered.jsonl`` schema). Paper 171's harness modules translate
DeepCAD commands to CadQuery and analyse the result; this module supplies the
surrounding **dataset layer** the reference repo pins down, all of which is
deterministic string / record handling:

  * **Record schema.** Every train/test sample is one JSONL object with exactly two
    keys, ``input`` (the Text2CAD natural-language description) and ``output`` (the
    CadQuery program) -- :class:`CadQueryRecord`, with strict parsing
    (:func:`parse_jsonl`) and round-tripping (:func:`to_jsonl`).

  * **Prompt format.** Finetuning and inference both render a record as
    ``"### Instruction:\\n{input}\\n\\n### Response:\\n{output}"``
    (:func:`format_prompt` / :func:`format_training_example`), and decoding splits
    the completion back off at the ``### Response:`` marker
    (:func:`split_response`).

  * **Annotation request.** The LLM annotator is asked for CadQuery code from a
    minimal DeepCAD JSON with a fixed instruction that also fixes the export path
    and forbids ``show()`` (:func:`build_annotation_request`).

  * **Two-attempt execution feedback loop** (the repo's key data-quality trick, and
    the reason its annotations run at all): a generated script is executed; on
    failure the **last five stderr lines** are fed back with a fixed retry prompt
    and the model gets exactly one more attempt, after which the sample is written
    to the failed-scripts list. :func:`build_retry_prompt` renders that feedback
    message and :func:`annotation_outcomes` replays a recorded attempt log into
    :class:`AnnotationStats` (successes, failures, failed script paths) without any
    model or subprocess.

  * **UID layout.** DeepCAD UIDs are ``"0001/00010001"`` -- a four-digit bucket plus
    the sample id -- which is how the ground-truth STL of a generated
    ``00010001.stl`` is located (:func:`uid_for_stem` / :func:`ground_truth_path`),
    and the numeric generation index maps back to a UID through the filtered-index
    list (:func:`export_targets`).

Pure stdlib (``json``), deterministic, no network, no model, no subprocess. The
LLM annotator itself is external and out of scope: this module is everything
around it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

INSTRUCTION_HEADER = "### Instruction:"
RESPONSE_MARKER = "### Response:"
UID_BUCKET_LEN = 4
STDERR_TAIL_LINES = 5
MAX_ANNOTATION_ATTEMPTS = 2


@dataclass(frozen=True)
class CadQueryRecord:
    """One dataset sample: a description and its CadQuery program."""

    input: str
    output: str


def parse_jsonl(lines) -> list:
    """Parse JSONL text lines into records; blank lines are skipped."""
    records = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "input" not in obj or "output" not in obj:
            raise ValueError("record must have 'input' and 'output' keys")
        records.append(CadQueryRecord(str(obj["input"]), str(obj["output"])))
    return records


def to_jsonl(records) -> str:
    """Serialise records back to JSONL (stable key order, no trailing newline)."""
    return "\n".join(
        json.dumps({"input": r.input, "output": r.output}, sort_keys=True)
        for r in records
    )


def format_prompt(instruction: str) -> str:
    """The inference-time prompt: instruction header plus an empty response slot."""
    return f"{INSTRUCTION_HEADER}\n{instruction}\n\n{RESPONSE_MARKER}\n"


def format_training_example(record: CadQueryRecord) -> str:
    """The finetuning string: the prompt with the target program appended."""
    return format_prompt(record.input) + record.output


def split_response(decoded: str) -> str:
    """Recover the completion from a decoded output that still carries the prompt."""
    if RESPONSE_MARKER in decoded:
        return decoded.split(RESPONSE_MARKER, 1)[1].strip()
    return decoded.strip()


def build_annotation_request(json_str: str, file_index: str, save_path: str) -> str:
    """The annotator instruction: CadQuery from a DeepCAD JSON, exporting one STL."""
    target = f"{save_path.rstrip('/')}/{file_index}.stl"
    return (
        f"'''Give me CAD query from this CAD sequence: {json_str}. "
        f"The export file name should be {target}. "
        f"In the end, only save stl file, don't need to use show().'''"
    )


def build_retry_prompt(code: str, stderr: str, tail: int = STDERR_TAIL_LINES) -> str:
    """The feedback message: the failing code plus the last ``tail`` stderr lines."""
    error_msg = "\n".join(stderr.splitlines()[-tail:])
    return (
        f"code: {code} has an error: {error_msg}, "
        f"generate it again, only give me python code"
    )


@dataclass(frozen=True)
class AnnotationStats:
    """Outcome of the two-attempt annotation loop over a corpus."""

    success_count: int
    fail_count: int
    failed_scripts: tuple

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.0
        return self.success_count / total


def annotation_outcomes(attempt_log, cq_dir: str) -> AnnotationStats:
    """Replay a recorded annotation log into stats.

    ``attempt_log`` maps ``file_index -> sequence of per-attempt return codes``
    (``0`` = the generated script executed). A sample succeeds if any of its first
    :data:`MAX_ANNOTATION_ATTEMPTS` attempts returned ``0``; otherwise its script
    path is recorded as failed, exactly like the reference pipeline's
    ``failed_scripts`` list.
    """
    ok, failed = 0, []
    for file_index in sorted(attempt_log):
        codes = list(attempt_log[file_index])[:MAX_ANNOTATION_ATTEMPTS]
        if any(int(c) == 0 for c in codes):
            ok += 1
        else:
            failed.append(f"{cq_dir.rstrip('/')}/{file_index}.py")
    return AnnotationStats(ok, len(failed), tuple(failed))


def uid_for_stem(stem: str, bucket_len: int = UID_BUCKET_LEN) -> str:
    """DeepCAD UID of a sample id: ``"00010001"`` -> ``"0001/00010001"``."""
    if len(stem) <= bucket_len:
        raise ValueError(f"sample id too short for a bucketed uid: {stem!r}")
    return f"{stem[:bucket_len]}/{stem}"


def ground_truth_path(stl_name: str, root: str, bucket_len: int = UID_BUCKET_LEN) -> str:
    """Locate the ground-truth STL of a generated one via its bucketed UID."""
    stem = stl_name.split(".")[0]
    bucket = uid_for_stem(stem, bucket_len).split("/")[0]
    return f"{root.rstrip('/')}/{bucket}/{stl_name}"


def export_targets(index_list, out_dir: str) -> dict:
    """Map generation index -> export path, via the filtered-index STL name list."""
    return {
        i: f"{out_dir.rstrip('/')}/{name}" for i, name in enumerate(index_list)
    }
