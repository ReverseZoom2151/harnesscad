"""Redundancy-based error recovery for terse / errorful CAD commands.

Section 2.3 ("Redundant Encoding of Constraints") and Section 4.2 of "Towards a
Natural Language Interface for CAD" observe that "misspellings and grammatical
deviations pervade the use of natural language interfaces" -- chiefly deletions
of articles and of some prepositions.  Because grammaticality is encoded
*redundantly* (a verb constrains its object and a noun constrains its verb),
Cleopatra can restore such input: "Determiners can be skipped, and the
prepositions of extranuclear constituents can be skipped if no ambiguity
results", so the cryptic

    What is voltage nl 10 ns?

is recovered to

    What is the voltage at nl at 10 ns?

and an unrecognised word triggers a request for a replacement.

This module reconstructs those deterministic recoveries for CAD commands:

* :func:`insert_missing_determiners` -- restore an article dropped between a
  verb and its object noun (``draw circle`` -> ``draw a circle``);
* :func:`insert_missing_prepositions` -- restore the preposition of an
  unambiguous extranuclear locative constituent (``draw circle at radius 5
  origin`` etc.), choosing ``to`` after a motion verb and ``at`` otherwise;
* :func:`recover` -- run both plus unknown-word substitution from a supplied
  replacement map, reporting any word still unknown (the "ask the user for a
  replacement" step);
* :func:`parse_with_recovery` -- recover, then hand the repaired text to
  :func:`spec.nlcad_case_frame.parse_command`.

All transforms are pure and order-stable, so recovery is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from harnesscad.domain.spec.nlcad_case_frame import (
    parse_command, classify, _tokenize, verb_frame,
    _DETERMINERS, _FILLERS, _SHAPE_NOUNS,
)

_PREPOSITIONS = {"at", "on", "in", "of", "with", "to", "by", "after", "before"}
_MOTION_ACTIONS = {"translate", "rotate", "scale"}


@dataclass(frozen=True)
class Repair:
    kind: str            # 'determiner' | 'preposition' | 'replacement'
    inserted: str        # the token added (or the substitute word)
    at: int              # index in the repaired stream
    note: str = ""


@dataclass
class RecoveryResult:
    original: str
    repaired_tokens: List[str] = field(default_factory=list)
    repairs: List[Repair] = field(default_factory=list)
    unknown_words: List[str] = field(default_factory=list)

    @property
    def repaired_text(self) -> str:
        return " ".join(self.repaired_tokens)

    @property
    def needs_replacement(self) -> bool:
        return bool(self.unknown_words)


# --------------------------------------------------------------------------- #
# Individual recoveries
# --------------------------------------------------------------------------- #
def insert_missing_determiners(tokens: List[str]) -> Tuple[List[str], List[Repair]]:
    """Insert ``a`` before an object noun that directly follows its verb."""
    out: List[str] = []
    repairs: List[Repair] = []
    prev_low: Optional[str] = None
    for tok in tokens:
        low = tok.lower()
        if (low in _SHAPE_NOUNS and prev_low is not None
                and verb_frame(prev_low) is not None):
            out.append("a")
            repairs.append(Repair("determiner", "a", len(out) - 1,
                                  "article dropped after verb"))
        out.append(tok)
        prev_low = low
    return out, repairs


def insert_missing_prepositions(
    tokens: List[str], action: Optional[str] = None,
) -> Tuple[List[str], List[Repair]]:
    """Insert the preposition of an ungoverned locative constituent.

    A location word (``origin``) or a coordinate group ``( ... )`` that is not
    already governed by a preposition is unambiguously extranuclear, so its
    preposition is restored: ``to`` for a motion verb, otherwise ``at``.
    """
    prep_word = "to" if action in _MOTION_ACTIONS else "at"
    out: List[str] = []
    repairs: List[Repair] = []
    prep_pending = False
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        low = tok.lower()
        if low in _PREPOSITIONS:
            prep_pending = True
            out.append(tok)
            i += 1
            continue
        if low in _DETERMINERS or low in _FILLERS:
            out.append(tok)
            i += 1
            continue
        if tok == "(":
            if not prep_pending:
                out.append(prep_word)
                repairs.append(Repair("preposition", prep_word, len(out) - 1,
                                      "preposition of coordinate restored"))
            while i < n:                     # copy the whole coordinate group
                out.append(tokens[i])
                if tokens[i] == ")":
                    i += 1
                    break
                i += 1
            prep_pending = False
            continue
        nom = classify(low)
        if nom is not None and nom.feature == "location" and not prep_pending:
            out.append(prep_word)
            repairs.append(Repair("preposition", prep_word, len(out) - 1,
                                  "preposition of location restored"))
        out.append(tok)
        prep_pending = False
        i += 1
    return out, repairs


def _is_known(low: str) -> bool:
    return (low in _DETERMINERS or low in _FILLERS or low in _PREPOSITIONS
            or verb_frame(low) is not None or classify(low) is not None
            or low in ("(", ")", ","))


# --------------------------------------------------------------------------- #
# Combined recovery
# --------------------------------------------------------------------------- #
def recover(text: str, replacements: Optional[Dict[str, str]] = None) -> RecoveryResult:
    """Repair a terse/errorful command, returning the reconstructed stream.

    ``replacements`` maps an unknown surface word to a substitute (the paper's
    "type in one or more words as replacement").  Words still unknown after
    substitution are collected in ``unknown_words``.
    """
    replacements = {k.lower(): v for k, v in (replacements or {}).items()}
    tokens = _tokenize(text)
    result = RecoveryResult(original=text)

    # 1. unknown-word substitution (before structural recovery)
    subbed: List[str] = []
    for tok in tokens:
        low = tok.lower()
        if not _is_known(low):
            if low in replacements:
                sub = replacements[low]
                subbed.append(sub)
                result.repairs.append(
                    Repair("replacement", sub, len(subbed) - 1,
                           f"'{tok}' -> '{sub}'"))
            else:
                result.unknown_words.append(tok)
                subbed.append(tok)
        else:
            subbed.append(tok)

    # 2. determiner restoration
    subbed, dets = insert_missing_determiners(subbed)
    result.repairs.extend(dets)

    # 3. preposition restoration (needs the verb's action for to/at choice)
    action = None
    for tok in subbed:
        vf = verb_frame(tok.lower())
        if vf is not None:
            action = vf.action
            break
    subbed, preps = insert_missing_prepositions(subbed, action)
    result.repairs.extend(preps)

    result.repaired_tokens = subbed
    return result


def parse_with_recovery(text: str, replacements: Optional[Dict[str, str]] = None):
    """Recover ``text`` then parse it; returns ``(ParsedCommand|None, RecoveryResult)``."""
    rec = recover(text, replacements)
    cmd = parse_command(rec.repaired_text)
    return cmd, rec
