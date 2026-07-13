"""CADCodeVerify refinement scaffolding (paper section 3, App. D Table 9).

Deterministic, VLM-free re-implementation of the control flow of CADCodeVerify
from "Generating CAD Code with Vision-Language Models for 3D Designs"
(Alrashedy et al., ICLR 2025).  The learned pieces (a VLM that generates
verification questions, answers them from rendered images, and rewrites code)
are external and are supplied to this module as plain callables; everything
around them is deterministic bookkeeping that the paper specifies exactly:

  * Code-execution repair loop (Eq. 2): re-submit code + compiler error to the
    fixer until it compiles or N iterations are exhausted.
  * Feedback filtering (section 3.3): drop every question answered "Yes"; if
    *all* questions are answered "Yes", no refinement is needed; "Unclear"
    answers are kept and resent for further evaluation.
  * Answer-accuracy accounting (Table 9): of the answers labelled Yes/No, the
    accuracy given a set of ground-truth correctness flags, plus the fraction
    of Unclear answers.

The full two-stage refinement (Eq. 3, M refinement iterations) is orchestrated
with injectable ``question_fn`` / ``answer_fn`` / ``feedback_fn`` / ``refine_fn``
callables so the loop is testable without any model.
"""
from __future__ import annotations

# Canonical answer labels.
YES, NO, UNCLEAR = "Yes", "No", "Unclear"


def repair_until_compiles(code, compile_fn, fix_fn, *, max_iters):
    """Code-execution repair loop, Eq. 2.

    ``compile_fn(code) -> (ok: bool, error: str)``; on failure ``fix_fn(code,
    error) -> new_code`` is invoked.  Returns a record with the final code,
    whether it compiled, and the number of repair attempts made.
    """
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1")
    attempts = 0
    ok, error = compile_fn(code)
    while not ok and attempts < max_iters:
        code = fix_fn(code, error)
        attempts += 1
        ok, error = compile_fn(code)
    return {"code": code, "compiled": bool(ok), "repair_attempts": attempts,
            "last_error": None if ok else error}


def filter_feedback(questions, answers):
    """Apply section-3.3 feedback filtering to a question/answer set.

    Questions answered "Yes" are omitted (resolved); "No" and "Unclear" are
    retained for refinement.  Returns a dict describing which questions remain
    and whether refinement is needed at all.
    """
    qs = list(questions)
    ans = [_norm_label(a) for a in answers]
    if len(qs) != len(ans):
        raise ValueError("questions/answers length mismatch")
    unresolved = [(q, a) for q, a in zip(qs, ans) if a != YES]
    return {
        "unresolved": unresolved,
        "needs_refinement": len(unresolved) > 0,
        "num_yes": sum(1 for a in ans if a == YES),
        "num_no": sum(1 for a in ans if a == NO),
        "num_unclear": sum(1 for a in ans if a == UNCLEAR),
    }


def _norm_label(a):
    key = str(a).strip().lower()
    table = {"yes": YES, "no": NO, "unclear": UNCLEAR}
    if key not in table:
        raise ValueError("invalid answer label: %r" % (a,))
    return table[key]


def answer_accuracy(answers, correctness):
    """Table-9 answer-accuracy accounting.

    ``answers`` are labels (Yes/No/Unclear); ``correctness`` are booleans that
    are only meaningful for Yes/No answers (whether that concrete answer was
    right).  Reports: total answers, accuracy over Yes/No labels only,
    incorrect fraction over all answers, and the Unclear fraction.
    """
    ans = [_norm_label(a) for a in answers]
    if len(ans) != len(correctness):
        raise ValueError("answers/correctness length mismatch")
    total = len(ans)
    if total == 0:
        raise ValueError("no answers")
    labeled = [(a, bool(c)) for a, c in zip(ans, correctness) if a != UNCLEAR]
    n_labeled = len(labeled)
    n_correct = sum(1 for _, c in labeled if c)
    n_incorrect = n_labeled - n_correct
    n_unclear = sum(1 for a in ans if a == UNCLEAR)
    return {
        "total": total,
        "labeled": n_labeled,
        "accuracy_over_labeled": (n_correct / n_labeled) if n_labeled else None,
        "correct_fraction": n_correct / total,
        "incorrect_fraction": n_incorrect / total,
        "unclear_fraction": n_unclear / total,
    }


def run_cadcodeverify(code, description, *, question_fn, answer_fn,
                      feedback_fn, refine_fn, max_refinements=2):
    """Orchestrate the CADCodeVerify refinement loop (Eq. 3).

    Callables (all deterministic in tests):
      * ``question_fn(description) -> [questions]``
      * ``answer_fn(description, questions) -> [labels]``
      * ``feedback_fn(unresolved_pairs) -> feedback_text``
      * ``refine_fn(code, description, feedback_text) -> new_code``

    Stops early when a refinement round leaves no unresolved questions (all
    "Yes"), matching "if all questions are answered Yes we assume no further
    refinement is necessary".  Runs at most ``max_refinements`` rounds.
    """
    if max_refinements < 0:
        raise ValueError("max_refinements must be >= 0")
    history = []
    current = code
    for step in range(max_refinements):
        questions = list(question_fn(description))
        answers = list(answer_fn(description, questions))
        filt = filter_feedback(questions, answers)
        record = {"step": step, "questions": questions, "answers": answers,
                  "num_yes": filt["num_yes"], "num_no": filt["num_no"],
                  "num_unclear": filt["num_unclear"]}
        if not filt["needs_refinement"]:
            record["refined"] = False
            history.append(record)
            break
        feedback = feedback_fn(filt["unresolved"])
        current = refine_fn(current, description, feedback)
        record["refined"] = True
        record["feedback"] = feedback
        history.append(record)
    return {"code": current, "rounds": len(history), "history": history}
