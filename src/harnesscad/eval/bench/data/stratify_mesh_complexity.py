"""CADPrompt data-stratification protocol (paper section 4.3).

Deterministic re-implementation of how "Generating CAD Code with Vision-
Language Models for 3D Designs" (Alrashedy et al., ICLR 2025) stratifies its
CADPrompt benchmark to analyse model performance:

  * Mesh complexity -- total (faces + vertices); split at the median into
    "simple" (<= median) and "complex" (> median).
  * Compilation difficulty -- across six attempts (3 models x {zero, few}-shot):
    "easy" if at least four of six produced compilable code, else "hard".
  * Geometric complexity -- an expert 4-level taxonomy (simple / moderate /
    complex / very_complex); the taxonomy and its definitions are encoded here
    (the labels themselves are expert-provided).

Also provides CADPrompt-style per-object statistics (Table 1): vertex/face
counts, description word/sentence counts, and Python code line/token counts.
Stdlib-only, deterministic.
"""
from __future__ import annotations

from statistics import median

# Section 4.3 geometric-complexity taxonomy (expert-assigned labels).
GEOMETRIC_COMPLEXITY_LEVELS = {
    "simple": "basic object with few features; may be one geometric shape",
    "moderate": "moderate detail with a few distinct features or components",
    "complex": "many interconnected parts, fine details, or intricate shapes",
    "very_complex": "highly intricate; many components, detailed textures, or "
                    "unique interlocking geometric features",
}

# Six generation attempts used to define compilation difficulty.
COMPILE_ATTEMPTS = 6
COMPILE_EASY_THRESHOLD = 4


def mesh_complexity_score(vertices, faces):
    """Total face + vertex count (the paper's mesh-complexity measure)."""
    return int(vertices) + int(faces)


def split_mesh_complexity(objects):
    """Label each object simple/complex by the median mesh-complexity score.

    ``objects`` is an iterable of dicts with ``vertices`` and ``faces`` counts.
    Objects at or below the median are "simple", strictly above are "complex".
    Returns a new list with a ``mesh_complexity`` label added to each object.
    """
    rows = [dict(o) for o in objects]
    if not rows:
        return rows
    scores = [mesh_complexity_score(o["vertices"], o["faces"]) for o in rows]
    med = median(scores)
    for o, s in zip(rows, scores):
        o["mesh_complexity_score"] = s
        o["mesh_complexity"] = "simple" if s <= med else "complex"
    return rows


def compilation_difficulty(compile_flags):
    """Classify by how many of six attempts compiled.

    ``compile_flags`` is an iterable of booleans (one per attempt).  Returns
    ("easy"/"hard", num_compiled).  At least :data:`COMPILE_EASY_THRESHOLD` of
    :data:`COMPILE_ATTEMPTS` compiling -> "easy".
    """
    flags = [bool(x) for x in compile_flags]
    if len(flags) != COMPILE_ATTEMPTS:
        raise ValueError("expected exactly %d attempts" % COMPILE_ATTEMPTS)
    n = sum(flags)
    return ("easy" if n >= COMPILE_EASY_THRESHOLD else "hard"), n


def classify_geometric_complexity(label):
    """Validate/normalise an expert geometric-complexity label."""
    key = str(label).strip().lower().replace(" ", "_")
    if key not in GEOMETRIC_COMPLEXITY_LEVELS:
        raise ValueError("unknown geometric complexity: %r" % (label,))
    return key


def _count_words(text):
    return len(str(text).split())


def _count_sentences(text):
    # Sentences ended by . ! or ? ; a trailing fragment without punctuation
    # still counts as one sentence.
    s = str(text)
    count = sum(s.count(ch) for ch in ".!?")
    tail = s
    for ch in ".!?":
        tail = tail.replace(ch, "")
    if count == 0 or tail.strip():
        count = max(count, 1)
    return count


def _count_code_lines(code):
    return sum(1 for ln in str(code).splitlines() if ln.strip())


def _count_code_tokens(code):
    # Whitespace-delimited token approximation (deterministic, dependency-free).
    return len(str(code).split())


def object_statistics(obj):
    """Compute CADPrompt Table-1 statistics for a single annotated object.

    ``obj`` supplies ``vertices``, ``faces``, ``description`` (natural language
    prompt) and ``code`` (ground-truth Python).
    """
    return {
        "vertices": int(obj["vertices"]),
        "faces": int(obj["faces"]),
        "words": _count_words(obj["description"]),
        "sentences": _count_sentences(obj["description"]),
        "code_lines": _count_code_lines(obj["code"]),
        "code_tokens": _count_code_tokens(obj["code"]),
    }


def dataset_statistics(objects):
    """min/max/avg over each Table-1 field for a set of annotated objects."""
    rows = [object_statistics(o) for o in objects]
    if not rows:
        raise ValueError("no objects")
    fields = ("vertices", "faces", "words", "sentences", "code_lines", "code_tokens")
    out = {"datapoints": len(rows)}
    for f in fields:
        vals = [r[f] for r in rows]
        out[f] = {"min": min(vals), "max": max(vals), "avg": sum(vals) / len(vals)}
    return out
