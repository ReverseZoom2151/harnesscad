# Deep read — DesignQA (FSAE-rules multimodal QA benchmark)

New repo added today. `resources/cad_repos/design_qa-main/design_qa-main/`
(double-nested). Genuine read of the eval code and every ground-truth CSV; image
volume stated honestly, not sampled. Every "already-covered"/"gap" claim was
checked against `registry.index()` (1602 modules) and targeted greps of
`src/harnesscad/eval`.

Bottom line up front: DesignQA is a **genuinely new benchmark** the harness does
not carry — FSAE-competition-rules multimodal QA with a **committed, fully
machine-checkable grading protocol** (`eval/metrics/metrics.py`, six scorers,
pure-Python, nltk+rouge deps). The harness has ~7 CAD-QA scoring/judge modules
but **all reimplement other papers** (QueryCAD, Query2CAD, Text2CAD-Bench,
textbook-QA, CVCAD dims) and **none uses DesignQA's SQuAD/ScienceQA-style
token-overlap grading**. The corpus itself is **manifest-only** (no LICENSE) and
the highest-value text tasks are **unusable as committed** because the FSAE rules
PDF they depend on is not in the repo. No refusal/cannot-determine ground truth.

---

## design_qa (534 files: 24 py, ~230 committed jpg/png/pdf images, ~50 csv/txt data)

### LICENSE — VERDICT: MANIFEST-ONLY (no license present)
Verified absent: no `LICENSE`/`COPYING`/`NOTICE` at depth ≤4 in either nesting
level; `README.md` has no license section (only a JCISE citation +
`@article{doris2025designqa}`); `requirements.txt` carries no classifiers. Paper
is MIT DECODE Lab / ASME JCISE 2025; code is on GitHub (anniedoris/design_qa) but
**no license is committed here**. → Corpus data (CSVs, images, extracted rule
text) is **manifest-only**: record paths + SHA, resolve at runtime, vendor
nothing. The **grading algorithm is facts-only**: the six metric definitions are
lifted from SQuAD v1.1 and ScienceQA (both openly documented, cited in-file), so
the harness may reimplement the *protocol* deterministically the way it already
reimplements dozens of paper protocols — citing DesignQA — without copying files.

### WHAT IT IS
A multimodal LLM benchmark built with the MIT Motorsport team over the ~200-page
2024 FSAE competition rules. 1451 QA pairs in 3 segments / 6 subsets, each with an
automatic scorer. All ground truth is determinate (yes/no, a rule's verbatim
text, a component name, or a rule-number list) — designed for *automatic* scoring,
no LLM judge. Overall score = unweighted mean of the six subset macro-averages.

### READ (in full)
- `eval/metrics/metrics.py` (600 lines) — the six scorers + cleaning helpers.
- `eval/full_evaluation.py` — CLI harness, score aggregation, results.txt writer.
- `README.md` (189 lines) — task definitions, prompt templates, GT examples.
- All 6 ground-truth CSVs (row/column schema confirmed via pandas — see counts).
- `requirements.txt`, `.gitignore`, dir tree (`find`), refusal-string scan.

### SKIMMED — NOT READ
- 230 committed images (definition/presence/dimension jpg, FP png) — inspected
  paths + counts only, not pixels.
- `scripts/` (24 py + PDFs + raw CSVs): the *dataset-construction* pipeline
  (pdfplumber rule extraction, PPTX→image cropping, QA generators). Read
  filenames + purpose from README; did not read each generator line-by-line —
  they are provenance, not verifier assets.
- The 40+ committed `eval/**/*_{gpt-4,llava-13b,llama-2}*.csv/.txt` reference
  runs — confirmed to exist and to be per-model scored outputs (usable as
  reference values for a grader port); did not read every row.

### FINDINGS (ranked)

**1. The 6-scorer grading protocol — `eval/metrics/metrics.py` (sha256 876aa77c9f6d5cc5)**
*What:* deterministic, dependency-light scorers, one per subset:
  - `eval_retrieval_qa` — SQuAD-style **token-F1 on bag-of-words** after
    `normalize_answer` (lowercase, strip punct, drop a/an/the, collapse ws).
  - `eval_compilation_qa` — **F1 on the rule-number list** (`ast.literal_eval` the
    GT list; split prediction on ", ").
  - `eval_definition_qa` — **F1 on bag-of-characters**, `max` over `;`-separated
    synonyms; buckets by `mentions ∈ {definition, mentioned, none}`.
  - `eval_presence_qa` — yes/no **accuracy** via first-yes/no extraction, else
    the sentinel `noanswer` (scores 0).
  - `eval_dimensions_qa` / `eval_functional_performance_qa` — yes/no accuracy on
    the `Answer:`-tagged span **plus BLEU-2 + ROUGE-L** of the `Explanation:` span
    against a committed reference explanation.
*Why (verifier-first):* a portable, reference-checkable grading harness for
text/QA answers with the exact normalisation rules and the `noanswer` sentinel —
directly reusable for any FSAE-rules or verbatim-retrieval task, and the
per-model `*_llava-13b.txt` files give reference scores to regression-test a port.
*Harness equivalent:* **NONE for this metric family.** Verified: registry has
`eval.bench.protocols.qa_scoring` (textbook 2-of-3 + spatial MC),
`eval.bench.judges.{qa_grade_scale,qa_evidence_grading,vqa_score}` (QueryCAD
Correct/Partial/Wrong; Query2CAD VQAScore), `eval.bench.protocols.vlm_rubric_scorecard`
(Text2CAD-Bench L4 5-question VLM rubric), `eval.verifiers.dimension_qa` (CVCAD
percentage-error). `grep -rE 'rouge|bleu|token_f1|fsae|designqa' src/harnesscad`
→ **only** `vlm_rubric_scorecard.py` (name-match on unrelated words); no SQuAD
token-F1, no bag-of-characters synonym-max, no BLEU/ROUGE text scorer anywhere.
*Disposition:* **facts-only / reimplementable protocol** (metric defs are public
SQuAD/ScienceQA; cite DesignQA). Do not vendor the file (no license).

**2. Committed QA ground truth — 6 CSVs, 1451 pairs.**
Counts verified via pandas:
| subset | file | rows | GT columns |
|---|---|---|---|
| retrieval | `dataset/rule_extraction/rule_retrieval_qa.csv` (81c43f41) | 1192 | question, ground_truth |
| compilation | `dataset/rule_extraction/rule_compilation_qa.csv` (a225bf05) | 30 | question, ground_truth (list) |
| definition | `dataset/rule_comprehension/rule_definition_qa.csv` (f4046838) | 31 | +image, mentions |
| presence | `dataset/rule_comprehension/rule_presence_qa.csv` (146d40e2) | 62 | +image, mentions |
| dimension (context) | `.../rule_dimension_qa/context/..._context.csv` (e1943b68) | 60 | +image, dimension_type, explanation |
| dimension (detailed) | `.../detailed_context/..._detailed_context.csv` (f889f712) | 60 | (same) |
| functional perf | `.../rule_functional_performance_qa/..._qa.csv` (f48a7826) | 16 | +image, explanation |
*Why:* machine-checkable image→answer and text→answer GT with the paired scorer.
The **image-based** subsets (definition/presence/dimension/FP, 229 QAs) are
**self-contained** — images committed under `dataset/`, GT in CSV. 
*Caveat (honest):* the **retrieval + compilation** text subsets (1222 of 1451
QAs) require the **2024 FSAE rules PDF as model input**, and that PDF is **NOT in
the repo** — only pdfplumber-extracted text (`dataset/docs/rules_pdfplumber1.txt`,
`csv_rules/*`, `rule_section_text/*`) is committed as retrieval context. So the
GT answers exist but the task input must be resolved externally.
*Harness equivalent:* NONE (no FSAE corpus; `grep fsae src/` empty).
*Disposition:* **manifest-only** (no license) — paths+SHA above, runtime resolve.

**3. Extracted FSAE rule text + rule-number index — `dataset/docs/`.**
`csv_rules/{all_rules,V,T,F,EV,IC,IN,S,D,VE}_extracted.csv`, `rule_nums/*.txt`,
`rule_section_text/*_rules.txt`: a structured (rule-number → rule-text) mapping of
the whole FSAE ruleset, and the retrieval GT is drawn verbatim from it. Usable as
a known-good "rule N states exactly X" fixture set.
*Disposition:* manifest-only; note the pdfplumber extraction carries mojibake
(e.g. `team�s`), so it is lossy reference text, not canonical.

### REFUSAL / CANNOT-DETERMINE — VERIFIED ABSENT
The task specifically wanted refusal ground truth. **DesignQA has none.** Scanned
all GT columns for `cannot|not determin|n/a|unknown|refuse|insufficient|none of`;
the only hits are those substrings occurring **inside rule body text** (e.g. "a
failure cannot result in…"), never as an answer label. Every GT is determinate.
The `noanswer` token is a **prediction-side sentinel** (model failed to emit
yes/no), not a gradable GT class. So DesignQA does **not** fill the harness's
refusal-corpus gap.

### ALREADY COVERED
- CAD-QA scoring generally: `eval.bench.protocols.qa_scoring`,
  `eval.bench.judges.{qa_grade_scale,qa_evidence_grading,vqa_score}`,
  `eval.bench.protocols.vlm_rubric_scorecard`, `eval.verifiers.dimension_qa`,
  `eval.bench.data.qa_query_schema` — but all are **other benchmarks**; DesignQA's
  metric family and FSAE corpus are the delta, not a duplicate.

### VERDICT
**Genuinely new, partially usable, manifest/facts-only.** Highest yield = the
six-scorer grading protocol (reimplementable as facts, with committed reference
runs to test against) + the 229 self-contained image→answer QAs. The 1222 text
QAs are manifest-only and externally-gated on the uncommitted FSAE PDF. No
refusal GT. No file may be vendored (no license). Recommend: register the metric
protocol as a facts-only DesignQA scorer module and manifest the image subsets;
skip the text subsets until the rules PDF is resolvable.
