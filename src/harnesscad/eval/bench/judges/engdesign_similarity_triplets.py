"""Design-similarity triplet metrics: self-consistency and transitive violations.

Deterministic scoring protocols from Section 3.1 of "From Concept to
Manufacturing". A rater (human or VLM) is shown a triplet of design sketches
(A, B, C) and asked which of B or C is more similar to the anchor A. For n
designs, each design serves as anchor A against every unordered pair of the
remaining n-1 designs, giving n * C(n-1, 2) triplet queries (360 for n=10).

Two quality measures are computed:
  * self-consistency: fraction of repeated triplet queries whose answers agree.
  * transitive violations: number of design triples whose three anchored
    judgements form a cyclic (intransitive) similarity ordering.

The VLM answers are injected; nothing here calls a model. No randomness.
"""

from __future__ import annotations

from itertools import combinations


def enumerate_triplets(n):
    """All anchored triplets (anchor, x, y) with x < y for n designs (0..n-1).

    Each design is the anchor A against every unordered pair {x, y} of the
    other designs. Count is n * C(n-1, 2).
    """
    if n < 3:
        return ()
    designs = range(n)
    out = []
    for anchor in designs:
        others = [d for d in designs if d != anchor]
        for x, y in combinations(others, 2):
            out.append((anchor, x, y))
    return tuple(out)


def triplet_count(n):
    """Number of anchored triplet queries for n designs."""
    return 0 if n < 3 else n * (n - 1) * (n - 2) // 2


def self_consistency(repeated_answers):
    """Fraction of repeated triplet queries that gave a fully consistent answer.

    repeated_answers: mapping triplet_key -> sequence of answers (repeats).
    Only triplets answered more than once contribute. A triplet is consistent
    if every repeat produced the same answer. Returns dict with the rate, the
    per-triplet consistency, and the number of repeated triplets considered.
    """
    per = {}
    considered = consistent = 0
    for key, answers in repeated_answers.items():
        answers = tuple(answers)
        if len(answers) < 2:
            continue
        considered += 1
        ok = len(set(answers)) == 1
        per[key] = ok
        consistent += ok
    rate = (consistent / considered) if considered else None
    return {"self_consistency": rate, "repeated": considered,
            "consistent": consistent, "per_triplet": per}


def _closer(chooser, anchor, x, y):
    """Which of x, y did the rater judge more similar to anchor?"""
    choice = chooser(anchor, x, y)
    if choice not in (x, y):
        raise ValueError("chooser returned %r; expected %r or %r"
                         % (choice, x, y))
    return choice


def _triple_is_intransitive(chooser, a, b, c):
    """True if the three anchored judgements over {a,b,c} form a cycle.

    From the three anchors we derive a strict comparison of the three pairwise
    distances dAB, dAC, dBC. The judgements are intransitive iff they form a
    strict cycle (dAB<dAC, dAC<dBC, dBC<dAB or its reverse), which is
    unsatisfiable for real symmetric distances.
    """
    # anchor a: is dAB < dAC ?  (b chosen => a closer to b => dAB < dAC)
    ab_lt_ac = _closer(chooser, a, b, c) == b
    # anchor b: is dAB < dBC ?  (a chosen => dAB < dBC)
    ab_lt_bc = _closer(chooser, b, a, c) == a
    # anchor c: is dAC < dBC ?  (a chosen => dAC < dBC)
    ac_lt_bc = _closer(chooser, c, a, b) == a
    # Encode as strict order over {AB, AC, BC}; detect a 3-cycle.
    lt = {}
    lt[("AB", "AC")] = ab_lt_ac
    lt[("AC", "AB")] = not ab_lt_ac
    lt[("AB", "BC")] = ab_lt_bc
    lt[("BC", "AB")] = not ab_lt_bc
    lt[("AC", "BC")] = ac_lt_bc
    lt[("BC", "AC")] = not ac_lt_bc
    nodes = ("AB", "AC", "BC")
    # A strict total order over 3 nodes has a unique minimum; a cycle does not.
    for start in nodes:
        smaller_than_all = all(lt[(start, other)]
                               for other in nodes if other != start)
        if smaller_than_all:
            return False
    return True


def transitive_violations(n, chooser):
    """Count design triples with intransitive similarity judgements.

    n: number of designs (0..n-1). chooser(anchor, x, y) returns whichever of x
    or y the rater judged more similar to anchor. Returns dict with the count
    and the sorted tuple of violating triples.
    """
    violations = []
    for a, b, c in combinations(range(n), 3):
        if _triple_is_intransitive(chooser, a, b, c):
            violations.append((a, b, c))
    return {"transitive_violations": len(violations),
            "violating_triples": tuple(violations)}
