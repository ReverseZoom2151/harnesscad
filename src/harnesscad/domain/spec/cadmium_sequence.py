"""CAD-sequence representation -- parser, normaliser, tokenizer, metric.

This representation treats a CAD model as a *sequence* of parametric
modelling operations rather than a mesh or a B-rep: a natural-language brief is
mapped to an ordered list of feature commands (sketch a profile, extrude it,
place a hole, fillet an edge...), and the model is judged on whether the
*sequence* it emits matches a reference sequence.  The deterministic, learning-
free parts of that idea are:

*   a **normal form** for a CAD sequence -- a canonical, order-and-whitespace
    stable textual/JSON encoding so two equivalent sequences hash the same;
*   a **tokenization scheme** -- the flat token stream a sequence model would be
    trained on (command tokens, parameter-name tokens, quantised numeric tokens),
    plus the inverse (detokenise back to a sequence);
*   an **evaluation metric** -- how close a predicted sequence is to a reference,
    reported as command-level F1, a normalised parameter error, and an ordered
    edit distance, exactly the family of numbers a CAD-sequence benchmark reports.

None of this calls a model or runs geometry.  It is a self-contained, stdlib-only
front end for the *representation*, so the harness can parse, canonicalise, tokenise
and score CAD sequences the same way a sequence-model dataset pipeline does.

Sequence grammar (one operation per line, ``#`` comments and blanks ignored)::

    sketch  plane=XY
    circle  cx=0 cy=0 r=5
    extrude dist=10 op=new
    hole    x=0 y=0 d=3 depth=10
    fillet  edges=all r=1

An operation is ``<command> key=value key=value ...``.  Values are numbers,
identifiers (``XY``, ``new``, ``all``) or quoted strings.  The command set and each
command's known parameters are declared in :data:`COMMAND_SCHEMA`; unknown commands
and unknown parameters are rejected by :func:`parse` (a syntactically valid but
out-of-vocabulary sequence is a real error class a benchmark must catch).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "COMMAND_SCHEMA",
    "CadmiumError",
    "Operation",
    "CadSequence",
    "parse",
    "normalise",
    "tokenize",
    "detokenize",
    "SequenceScore",
    "score",
]


# --------------------------------------------------------------------------- #
# vocabulary                                                                   #
# --------------------------------------------------------------------------- #
# command -> ordered tuple of its recognised parameter names.  The order fixes
# the canonical key ordering used by the normal form and the tokenizer.
COMMAND_SCHEMA: Dict[str, Tuple[str, ...]] = {
    "sketch": ("plane",),
    "line": ("x1", "y1", "x2", "y2"),
    "circle": ("cx", "cy", "r"),
    "rect": ("x", "y", "w", "h"),
    "arc": ("cx", "cy", "r", "start", "end"),
    "extrude": ("dist", "op"),
    "revolve": ("angle", "axis", "op"),
    "hole": ("x", "y", "d", "depth"),
    "fillet": ("edges", "r"),
    "chamfer": ("edges", "d"),
    "shell": ("thickness",),
    "mirror": ("plane",),
    "pattern": ("count", "dx", "dy"),
}

# parameters whose value is a categorical identifier rather than a number.
_CATEGORICAL = {"plane", "op", "axis", "edges"}


class CadmiumError(ValueError):
    """Raised on a malformed or out-of-vocabulary CAD sequence."""


# --------------------------------------------------------------------------- #
# data model                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Operation:
    """A single CAD operation: a command plus an ordered parameter mapping."""

    command: str
    params: Tuple[Tuple[str, object], ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {"command": self.command, "params": dict(self.params)}

    def get(self, key: str, default: object = None) -> object:
        for k, v in self.params:
            if k == key:
                return v
        return default


@dataclass(frozen=True)
class CadSequence:
    """An ordered list of :class:`Operation` -- the model representation."""

    operations: Tuple[Operation, ...] = ()

    def __len__(self) -> int:
        return len(self.operations)

    def commands(self) -> Tuple[str, ...]:
        return tuple(op.command for op in self.operations)

    def as_list(self) -> List[Dict[str, object]]:
        return [op.as_dict() for op in self.operations]


# --------------------------------------------------------------------------- #
# value parsing                                                               #
# --------------------------------------------------------------------------- #
def _parse_value(raw: str) -> object:
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1]
    try:
        if any(c in raw for c in ".eE") and raw.lower() not in ("e", "inf", "nan"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw  # identifier / categorical token


def _canonical_number(value: object) -> object:
    """Collapse ``3.0`` and ``3`` to a single canonical numeric form."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        f = float(value)
        if f.is_integer():
            return int(f)
        return f
    return value


# --------------------------------------------------------------------------- #
# parse                                                                       #
# --------------------------------------------------------------------------- #
def parse(source: str, *, strict: bool = True) -> CadSequence:
    """Parse a CAD-sequence source string into a :class:`CadSequence`.

    ``strict`` (default) rejects unknown commands and unknown parameter names --
    the out-of-vocabulary error class.  With ``strict=False`` unknown commands are
    kept verbatim (useful when scoring a model that hallucinated a command).
    """
    ops: List[Operation] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        parts = text.split()
        command = parts[0]
        if strict and command not in COMMAND_SCHEMA:
            raise CadmiumError(f"line {lineno}: unknown command {command!r}")
        known = COMMAND_SCHEMA.get(command, ())
        params: List[Tuple[str, object]] = []
        seen = set()
        for tok in parts[1:]:
            if "=" not in tok:
                raise CadmiumError(
                    f"line {lineno}: parameter {tok!r} is not key=value")
            key, _, raw = tok.partition("=")
            if not key:
                raise CadmiumError(f"line {lineno}: empty parameter name")
            if strict and known and key not in known:
                raise CadmiumError(
                    f"line {lineno}: command {command!r} has no parameter {key!r}")
            if key in seen:
                raise CadmiumError(
                    f"line {lineno}: duplicate parameter {key!r}")
            seen.add(key)
            params.append((key, _canonical_number(_parse_value(raw))))
        ops.append(Operation(command=command, params=tuple(params)))
    return CadSequence(operations=tuple(ops))


# --------------------------------------------------------------------------- #
# normalise (canonical form)                                                  #
# --------------------------------------------------------------------------- #
def _sorted_params(op: Operation) -> List[Tuple[str, object]]:
    order = COMMAND_SCHEMA.get(op.command)
    if order:
        rank = {k: i for i, k in enumerate(order)}
        return sorted(op.params, key=lambda kv: (rank.get(kv[0], len(order)), kv[0]))
    return sorted(op.params, key=lambda kv: kv[0])


def _fmt_value(value: object) -> str:
    value = _canonical_number(value)
    if isinstance(value, str):
        if value.replace("_", "").isalnum():
            return value
        return json.dumps(value)
    if isinstance(value, float):
        # stable, trailing-zero-free representation.
        return repr(value)
    return str(value)


def normalise(seq: CadSequence) -> str:
    """Return the canonical textual normal form of a sequence.

    Parameters within an operation are reordered to the schema order, numbers are
    canonicalised (``3.0`` -> ``3``), and every line is single-spaced.  Two
    sequences that differ only in parameter order or numeric spelling normalise to
    byte-identical strings, so ``normalise`` is a content hash key.
    """
    lines = []
    for op in seq.operations:
        parts = [op.command]
        for key, value in _sorted_params(op):
            parts.append(f"{key}={_fmt_value(value)}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# tokenizer                                                                    #
# --------------------------------------------------------------------------- #
def tokenize(seq: CadSequence, *, quantum: float = 0.0) -> List[str]:
    """Flatten a sequence into the token stream a CAD-sequence model consumes.

    Tokens: ``<bos>``, then per op ``CMD_<command>``, and per parameter
    ``P_<key>`` followed by either ``C_<value>`` (categorical) or a numeric token.
    Numeric tokens are ``N_<value>``; if ``quantum > 0`` the number is snapped to
    the nearest multiple of ``quantum`` first (the bucketing a discretised numeric
    vocabulary uses).  Every op ends with ``<eop>`` and the stream ends ``<eos>``.
    """
    out: List[str] = ["<bos>"]
    for op in seq.operations:
        out.append(f"CMD_{op.command}")
        for key, value in _sorted_params(op):
            out.append(f"P_{key}")
            if key in _CATEGORICAL or isinstance(value, str):
                out.append(f"C_{value}")
            else:
                num = float(value)
                if quantum > 0.0:
                    num = round(num / quantum) * quantum
                out.append(f"N_{_fmt_value(_canonical_number(num))}")
        out.append("<eop>")
    out.append("<eos>")
    return out


def detokenize(tokens: Sequence[str]) -> CadSequence:
    """Invert :func:`tokenize` (with ``quantum=0``) back into a sequence."""
    ops: List[Operation] = []
    command: Optional[str] = None
    params: List[Tuple[str, object]] = []
    pending_key: Optional[str] = None
    for tok in tokens:
        if tok in ("<bos>", "<eos>"):
            continue
        if tok == "<eop>":
            if command is not None:
                ops.append(Operation(command=command, params=tuple(params)))
            command, params, pending_key = None, [], None
        elif tok.startswith("CMD_"):
            command = tok[4:]
        elif tok.startswith("P_"):
            pending_key = tok[2:]
        elif tok.startswith("C_"):
            if pending_key is None:
                raise CadmiumError("value token with no preceding parameter")
            params.append((pending_key, tok[2:]))
            pending_key = None
        elif tok.startswith("N_"):
            if pending_key is None:
                raise CadmiumError("value token with no preceding parameter")
            params.append((pending_key, _canonical_number(_parse_value(tok[2:]))))
            pending_key = None
        else:
            raise CadmiumError(f"unrecognised token {tok!r}")
    return CadSequence(operations=tuple(ops))


# --------------------------------------------------------------------------- #
# evaluation metric                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SequenceScore:
    """How close a predicted CAD sequence is to a reference."""

    command_precision: float
    command_recall: float
    command_f1: float
    parameter_error: float       # mean normalised abs error over matched numerics
    edit_distance: int           # Levenshtein over command tokens
    exact_match: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "command_precision": self.command_precision,
            "command_recall": self.command_recall,
            "command_f1": self.command_f1,
            "parameter_error": self.parameter_error,
            "edit_distance": self.edit_distance,
            "exact_match": self.exact_match,
        }


def _multiset_prf(pred: Sequence[str], ref: Sequence[str]) -> Tuple[float, float, float]:
    from collections import Counter

    pc, rc = Counter(pred), Counter(ref)
    tp = sum((pc & rc).values())
    p = tp / len(pred) if pred else (1.0 if not ref else 0.0)
    r = tp / len(ref) if ref else (1.0 if not pred else 0.0)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = cur
    return prev[-1]


def _parameter_error(pred: CadSequence, ref: CadSequence) -> float:
    """Mean normalised absolute numeric-parameter error over positionally-aligned
    ops that share a command.  Missing/extra params count as a full unit error."""
    errors: List[float] = []
    for p_op, r_op in zip(pred.operations, ref.operations):
        if p_op.command != r_op.command:
            continue
        r_params = dict(r_op.params)
        p_params = dict(p_op.params)
        keys = set(r_params) | set(p_params)
        for k in keys:
            if k in _CATEGORICAL:
                errors.append(0.0 if p_params.get(k) == r_params.get(k) else 1.0)
                continue
            rv, pv = r_params.get(k), p_params.get(k)
            if isinstance(rv, (int, float)) and isinstance(pv, (int, float)):
                denom = max(abs(float(rv)), 1.0)
                errors.append(min(abs(float(pv) - float(rv)) / denom, 1.0))
            else:
                errors.append(0.0 if pv == rv else 1.0)
    if not errors:
        return 0.0
    return sum(errors) / len(errors)


def score(pred: CadSequence, ref: CadSequence) -> SequenceScore:
    """Score a predicted sequence against a reference (sequence-level metrics)."""
    p_cmds, r_cmds = pred.commands(), ref.commands()
    precision, recall, f1 = _multiset_prf(p_cmds, r_cmds)
    return SequenceScore(
        command_precision=precision,
        command_recall=recall,
        command_f1=f1,
        parameter_error=_parameter_error(pred, ref),
        edit_distance=_edit_distance(p_cmds, r_cmds),
        exact_match=normalise(pred) == normalise(ref),
    )
