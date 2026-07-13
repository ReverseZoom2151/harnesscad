"""Deterministic cleaning of raw LLM output into a runnable CadQuery script.

Reference implementation of paper 171 -- *Text-to-CadQuery* (repo
``Text-to-CadQuery``), inference ``step2_clean_run_CadQuery`` (the
``clean_*.ipynb`` notebooks) plus ``data_annotation/gemini_pipeline.py``
(``get_pure_python``). Paper 171's harness modules cover the *content* of
CadQuery programs (:mod:`programs.t2cq_ast`, :mod:`programs.t2cq_validity`,
:mod:`programs.t2cq_analysis`, :mod:`reconstruction.t2cq_translate`); what the
reference repo adds -- and what is implemented here -- is the deterministic
**text pipeline that turns a raw decoder output into an executable file**, run
before any validity/geometry metric is computed. Every generated sample in the
paper's evaluation passes through exactly these steps:

  1. **EOS truncation.** The decoded text is cut at the first ``<|endoftext|>``
     token and right-stripped. A sample with *no* EOS token is a truncated
     generation and is **dropped from the corpus** (the notebook deletes the
     file); this drop is counted, since it feeds the reported invalid rate.

  2. **Response de-prefixing.** Instruction-tuned outputs are decoded with the
     prompt still attached; the text after the first ``### Response:`` marker is
     kept (the split used in ``step1_generate_CadQuery``).

  3. **Fence stripping.** A leading ```` ```python ```` and a trailing ```` ``` ````
     are removed (``get_pure_python`` in the annotation pipeline).

  4. **Export canonicalisation.** The model frequently emits zero, several, or a
     *truncated* ``cq.exporters.export(...)`` call. The repo keeps the first
     complete export's **first argument** (the shape being exported), deletes
     every line mentioning ``cq.exporters.`` (which removes both the surplus
     exports and any incomplete trailing one), and appends a single canonical
     ``cq.exporters.export(<shape>, "<path>")`` naming the evaluation-side STL
     path. A script with no complete export statement cannot produce geometry
     and is rejected.

The result is a :class:`CleanResult` carrying either the runnable source or a
stable rejection ``reason`` (``no_eos`` / ``no_export`` / ``empty``), and
:func:`clean_corpus` aggregates a whole generation batch into :class:`CleanStats`
-- the pass/drop bookkeeping the paper's Invalid-Rate table is built on.

Pure stdlib (``re``), deterministic, no CadQuery/OCCT and no execution: unlike
the notebook, nothing is subprocess-run here. Distinct from
:mod:`programs.t2cq_validity` (static API/arity checking of an *already clean*
script): this module is the upstream normaliser that produces that script.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The EOS marker the paper's finetuned models emit at end of a CadQuery script.
EOS_TOKEN = "<|endoftext|>"

# The instruction-tuning response marker (step1_generate_CadQuery).
RESPONSE_MARKER = "### Response:"

# A *complete* export call: `cq.exporters.export(<shape>, "<something>.stl")`.
# Group 1 is the exported shape expression (the first positional argument).
EXPORT_RE = re.compile(
    r"cq\.exporters\.export\s*\(\s*([^,]+),\s*['\"].*?\.stl['\"].*?\)"
)

# Any mention of the exporters module -- used to strip surplus/truncated exports.
EXPORTER_MENTION_RE = re.compile(r"cq\.exporters\.")

REASON_NO_EOS = "no_eos"
REASON_NO_EXPORT = "no_export"
REASON_EMPTY = "empty"


@dataclass(frozen=True)
class CleanResult:
    """Outcome of cleaning one raw generation."""

    ok: bool
    code: str | None
    reason: str | None = None
    shape: str | None = None
    exports_found: int = 0
    truncated_at_eos: bool = False


@dataclass(frozen=True)
class CleanStats:
    """Aggregate bookkeeping over a batch of generations."""

    total: int
    kept: int
    dropped_no_eos: int
    dropped_no_export: int
    dropped_empty: int

    @property
    def drop_rate(self) -> float:
        """Fraction of generations that never reach the execution stage."""
        if self.total == 0:
            return 0.0
        return (self.total - self.kept) / self.total


def truncate_at_eos(text: str, token: str = EOS_TOKEN) -> str | None:
    """Cut ``text`` at the first EOS token; ``None`` when the token is absent.

    Absence means the generation hit the token budget mid-script, so the sample
    is dropped (the reference notebook deletes such files).
    """
    if token not in text:
        return None
    return text.split(token)[0].rstrip()


def strip_response_prefix(text: str, marker: str = RESPONSE_MARKER) -> str:
    """Keep the text after the first instruction-tuning response marker."""
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def strip_markdown_fence(text: str) -> str:
    """Remove a leading ```` ```python ```` fence and a trailing ```` ``` ````."""
    out = text.strip()
    if out.startswith("```python"):
        out = out[len("```python"):].strip()
    elif out.startswith("```"):
        out = out[len("```"):].strip()
    if out.endswith("```"):
        out = out[: -len("```")].strip()
    return out


def find_export_shapes(code: str) -> list[str]:
    """First positional argument of every *complete* export call, in order."""
    return [m.group(1).strip() for m in EXPORT_RE.finditer(code)]


def strip_export_statements(code: str) -> str:
    """Drop complete export calls, then every remaining ``cq.exporters.`` line.

    The second pass is what removes a *truncated* trailing export (e.g.
    ``cq.exporters.expo``), which the complete-call regex cannot match.
    """
    without_calls = EXPORT_RE.sub("", code)
    kept = [
        line
        for line in without_calls.split("\n")
        if not EXPORTER_MENTION_RE.search(line)
    ]
    return "\n".join(kept)


def canonicalize_export(code: str, export_path: str) -> str | None:
    """Rewrite ``code`` so it ends in exactly one export to ``export_path``.

    Returns ``None`` when the script contains no complete export statement (the
    shape to export is then unknown, so the sample is unusable).
    """
    shapes = find_export_shapes(code)
    if not shapes:
        return None
    body = strip_export_statements(code).strip()
    return f'{body}\n\ncq.exporters.export({shapes[0]}, "{export_path}")'


def clean_output(
    raw: str,
    export_path: str,
    *,
    require_eos: bool = True,
    eos_token: str = EOS_TOKEN,
) -> CleanResult:
    """Run the full step-2 pipeline on one raw decoded generation.

    ``require_eos=False`` skips the truncation gate (used for annotation-time
    outputs, which are produced by a chat model that does not emit the token).
    """
    text = raw
    truncated = False
    if require_eos:
        cut = truncate_at_eos(text, eos_token)
        if cut is None:
            return CleanResult(False, None, REASON_NO_EOS)
        truncated = True
        text = cut
    elif eos_token in text:
        text = truncate_at_eos(text, eos_token) or ""
        truncated = True

    text = strip_markdown_fence(strip_response_prefix(text))
    if not text.strip():
        return CleanResult(False, None, REASON_EMPTY, truncated_at_eos=truncated)

    shapes = find_export_shapes(text)
    code = canonicalize_export(text, export_path)
    if code is None:
        return CleanResult(
            False, None, REASON_NO_EXPORT, truncated_at_eos=truncated
        )
    if not strip_export_statements(text).strip():
        return CleanResult(False, None, REASON_EMPTY, truncated_at_eos=truncated)
    return CleanResult(
        True,
        code,
        None,
        shape=shapes[0],
        exports_found=len(shapes),
        truncated_at_eos=truncated,
    )


def clean_corpus(
    items: list[tuple[str, str]],
    *,
    require_eos: bool = True,
) -> tuple[list[CleanResult], CleanStats]:
    """Clean ``(raw_text, export_path)`` pairs; return results plus statistics."""
    results = [
        clean_output(raw, path, require_eos=require_eos) for raw, path in items
    ]
    counts = {REASON_NO_EOS: 0, REASON_NO_EXPORT: 0, REASON_EMPTY: 0}
    kept = 0
    for res in results:
        if res.ok:
            kept += 1
        elif res.reason in counts:
            counts[res.reason] += 1
    stats = CleanStats(
        total=len(results),
        kept=kept,
        dropped_no_eos=counts[REASON_NO_EOS],
        dropped_no_export=counts[REASON_NO_EXPORT],
        dropped_empty=counts[REASON_EMPTY],
    )
    return results, stats
