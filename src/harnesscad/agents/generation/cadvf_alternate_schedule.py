"""CADFusion alternate SL/VF training schedule (Section 3.4).

CADFusion (Wang et al., ICML 2025) trains a Text-to-CAD LLM by alternating two
stages so neither degrades the other:

  * Sequential Learning (SL) -- SFT on ground-truth parametric sequences; gives
    logically coherent, well-formatted sequences.
  * Visual Feedback (VF) -- DPO on render-derived preference pairs; makes the
    rendered objects visually natural.

The paper reports that extended VF training alone impairs well-formatted
sequence generation, while prolonged SL weakens visual naturalness. The remedy
(Figure 2(c)) is a fixed alternation: begin with one long initial SL stage,
then run N blocks, each of which does VF *followed by* SL.

Two invariants matter and are enforced here:

  * the schedule STARTS with SL (so the model can produce coherent sequences
    before any preference optimisation);
  * every VF stage is immediately followed by an SL stage within its block (no
    two VF stages run back-to-back), keeping the two objectives balanced.

Reference-model tracking is also made explicit: in the DPO objective (Eq. 2) the
reference distribution ``pref(.)`` is "the reference model from the last round of
sequential learning (f_i^SL)". Each VF stage therefore records, as its
``reference_model``, the model tag produced by the most recent SL stage.

This is a genuinely new piece: the repo has no SL/VF alternation scheduler (the
existing visual-feedback modules build feedback signals and preference data, not
the multi-stage training cadence). Deterministic, stdlib-only; the LLM, renderer
and optimiser are all external.
"""

from __future__ import annotations

from dataclasses import dataclass

# Paper defaults (Section 4.1 / Appendix C): 40-epoch initial SL, then 5 rounds
# of {VF 5 epochs, SL 1 epoch}.
DEFAULT_NUM_ROUNDS = 5
DEFAULT_INIT_SL_EPOCHS = 40
DEFAULT_VF_EPOCHS = 5
DEFAULT_SL_EPOCHS = 1

SL = "SL"
VF = "VF"


@dataclass(frozen=True)
class Stage:
    """One training stage in the alternation.

    ``kind`` is "SL" or "VF". ``round_index`` is 0 for the initial SL stage and
    1..N for each alternation block. ``input_model`` is the tag consumed and
    ``output_model`` the tag produced. ``reference_model`` is the frozen DPO
    reference tag (the last SL output) for VF stages, and None for SL stages.
    """

    kind: str
    round_index: int
    epochs: int
    input_model: str
    output_model: str
    reference_model: str = None


def build_schedule(num_rounds=DEFAULT_NUM_ROUNDS,
                   init_sl_epochs=DEFAULT_INIT_SL_EPOCHS,
                   vf_epochs=DEFAULT_VF_EPOCHS,
                   sl_epochs=DEFAULT_SL_EPOCHS,
                   base_model="pretrained"):
    """Construct the ordered list of Stage objects.

    Produces: initial SL (round 0), then ``num_rounds`` blocks of {VF, SL}. Model
    tags flow linearly: each stage's ``input_model`` is the previous stage's
    ``output_model``. VF stages reference the latest SL output as the DPO
    reference. Raises ValueError on non-positive epoch counts or negative rounds.
    """
    if num_rounds < 0:
        raise ValueError("num_rounds must be non-negative")
    for name, e in (("init_sl_epochs", init_sl_epochs), ("vf_epochs", vf_epochs),
                    ("sl_epochs", sl_epochs)):
        if e <= 0:
            raise ValueError(f"{name} must be positive")

    stages = []
    # Initial sequential-learning stage.
    last_sl_tag = "f0_SL"
    stages.append(Stage(kind=SL, round_index=0, epochs=init_sl_epochs,
                        input_model=base_model, output_model=last_sl_tag))
    current = last_sl_tag

    for i in range(1, num_rounds + 1):
        vf_tag = f"f{i}_VF"
        stages.append(Stage(kind=VF, round_index=i, epochs=vf_epochs,
                            input_model=current, output_model=vf_tag,
                            reference_model=last_sl_tag))
        current = vf_tag
        sl_tag = f"f{i}_SL"
        stages.append(Stage(kind=SL, round_index=i, epochs=sl_epochs,
                            input_model=current, output_model=sl_tag))
        current = sl_tag
        last_sl_tag = sl_tag

    return stages


def validate_schedule(stages):
    """Assert the CADFusion alternation invariants; return True or raise.

    Checks: non-empty; starts with SL; model tags chain (input == previous
    output); no two consecutive VF stages; every VF references the most recent
    SL output.
    """
    if not stages:
        raise ValueError("schedule is empty")
    if stages[0].kind != SL:
        raise ValueError("schedule must start with a sequential-learning stage")

    last_sl_output = None
    prev = None
    for st in stages:
        if st.kind not in (SL, VF):
            raise ValueError(f"unknown stage kind: {st.kind}")
        if prev is not None and st.input_model != prev.output_model:
            raise ValueError("model tags do not chain across stages")
        if st.kind == VF:
            if prev is not None and prev.kind == VF:
                raise ValueError("two visual-feedback stages cannot be adjacent")
            if st.reference_model != last_sl_output:
                raise ValueError("VF stage must reference the last SL output")
        else:  # SL
            last_sl_output = st.output_model
        prev = st
    return True


def total_epochs(stages):
    """Sum of epochs across all stages."""
    return sum(st.epochs for st in stages)


def epochs_by_kind(stages):
    """Total epochs spent in each stage kind, as a {'SL':.., 'VF':..} dict."""
    out = {SL: 0, VF: 0}
    for st in stages:
        out[st.kind] += st.epochs
    return out


def stage_sequence(stages):
    """Compact kind sequence, e.g. ['SL','VF','SL','VF','SL', ...]."""
    return [st.kind for st in stages]
