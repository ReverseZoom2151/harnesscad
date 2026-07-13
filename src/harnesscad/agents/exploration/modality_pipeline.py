"""Deterministic sketch -> text -> image -> mesh intermediate-representation pipeline.

Motivated by "Sketch2Prototype: Rapid Conceptual Design Exploration and
Prototyping with Generative AI" (Edwards, Man, Ahmed). The paper "treats the
Sketch2Prototype problem as a sequence of tasks that move between design
modalities: from sketch-to-text, then from text-to-image(s), and finally from
image-to-3D model" (Sec. 2.1, Fig. 2). Its central methodological claim is that
routing through an INTERMEDIATE TEXT modality (sketch -> text -> image -> 3D)
outperforms a DIRECT sketch-to-3D baseline and a ControlNet image-to-3D baseline
for producing diverse, manufacturable models, and that the text stage is the
place where a human injects feedback (Sec. 3.2, Fig. 5B).

This module builds a small, deterministic representation of that staged
workflow -- the pipeline *schema*, not the generative models themselves:

* Canonical modality ordering ``sketch < text < image < mesh``.
* :class:`Stage` -- one modality hop with a fan-out multiplicity and free-form
  params (e.g. an image stage constrained directly by the sketch, marking a
  ControlNet-style branch).
* :class:`Pipeline` -- an ordered list of stages with:
    - :meth:`Pipeline.validate` -- structural issues (must start at ``sketch``,
      modalities strictly forward, positive fan-outs, no repeated modality).
    - :meth:`Pipeline.classify` -- ``"sketch2prototype"`` (text intermediary
      present), ``"direct_sketch_to_3d"`` (sketch -> mesh, no text), or
      ``"controlnet"`` (image stage bound directly to the sketch), matching the
      paper's three compared frameworks.
    - :meth:`Pipeline.total_artifacts` / :meth:`Pipeline.artifact_counts` --
      how one seed sketch fans out (1 sketch -> 1 text -> N images -> M meshes).
    - :meth:`Pipeline.inject_feedback` -- append a designer-feedback sentence to
      the text stage, returning a NEW pipeline (human-in-the-loop text edit).

No wall clock, no randomness, standard library only; fully deterministic.
Distinct from ``exploration/image_prompt_sweep`` (which enumerates concrete
generator runs over weight/seed grids) -- this models the modality DAG schema
and its baseline classification.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


# Canonical modality progression of the Sketch2Prototype framework.
SKETCH = "sketch"
TEXT = "text"
IMAGE = "image"
MESH = "mesh"

# Forward rank of each modality; a valid forward pipeline never revisits or
# steps backward through this ordering.
_MODALITY_RANK = {SKETCH: 0, TEXT: 1, IMAGE: 2, MESH: 3}


def modality_rank(modality):
    """Return the canonical forward rank of a modality name.

    Raises ValueError for an unknown modality.
    """
    if modality not in _MODALITY_RANK:
        raise ValueError("unknown modality: %r" % (modality,))
    return _MODALITY_RANK[modality]


@dataclass(frozen=True)
class Stage:
    """One modality hop in the pipeline.

    * ``modality``  -- one of SKETCH/TEXT/IMAGE/MESH.
    * ``fanout``    -- how many artifacts this stage emits per input artifact
                       (1 sketch -> 1 text -> N images -> M meshes). Must be >= 1.
    * ``from_sketch`` -- True if this (image) stage is driven directly by the
                       sketch geometry, marking a ControlNet-style constraint.
    * ``params``    -- free-form deterministic metadata (e.g. feedback text).
    """
    modality: str
    fanout: int = 1
    from_sketch: bool = False
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Pipeline:
    """An ordered sequence of modality stages."""
    stages: tuple

    def __init__(self, stages):
        object.__setattr__(self, "stages", tuple(stages))

    def modalities(self):
        """Ordered tuple of the stage modalities."""
        return tuple(s.modality for s in self.stages)

    def validate(self):
        """Return a list of structural issue strings (empty == structurally valid).

        Rules: non-empty; first stage is a sketch; modalities strictly forward
        in canonical rank (no repeats, no back-steps); every fan-out >= 1.
        """
        issues = []
        if not self.stages:
            issues.append("pipeline has no stages")
            return issues
        if self.stages[0].modality != SKETCH:
            issues.append("pipeline must start at the sketch modality")
        prev_rank = None
        for i, st in enumerate(self.stages):
            try:
                rank = modality_rank(st.modality)
            except ValueError as exc:
                issues.append("stage %d: %s" % (i, exc))
                continue
            if st.fanout < 1:
                issues.append("stage %d (%s): fanout must be >= 1" % (i, st.modality))
            if prev_rank is not None and rank <= prev_rank:
                issues.append(
                    "stage %d (%s): modality does not advance past previous stage"
                    % (i, st.modality)
                )
            prev_rank = rank
        return issues

    def is_valid(self):
        """True when :meth:`validate` reports no issues."""
        return not self.validate()

    def has_text_intermediary(self):
        """True if a TEXT stage sits between the sketch and any image/mesh stage."""
        return TEXT in self.modalities()

    def classify(self):
        """Classify the pipeline into one of the paper's three frameworks.

        * ``"direct_sketch_to_3d"`` -- reaches a mesh with NO text stage (the
          "unprocessed sketch into Shap-E" baseline, Sec. 3.1).
        * ``"controlnet"`` -- has an image stage driven directly by the sketch
          (``from_sketch=True``); the paper notes this fixates on the sketch and
          loses diversity.
        * ``"sketch2prototype"`` -- has a text intermediary and no sketch-bound
          image stage (the proposed framework).
        * ``"other"`` -- anything else (e.g. text-only, no mesh).
        """
        mods = self.modalities()
        has_mesh = MESH in mods
        has_text = TEXT in mods
        controlnet = any(s.modality == IMAGE and s.from_sketch for s in self.stages)
        if controlnet:
            return "controlnet"
        if has_mesh and not has_text:
            return "direct_sketch_to_3d"
        if has_text and has_mesh:
            return "sketch2prototype"
        return "other"

    def total_artifacts(self):
        """Number of terminal artifacts one seed sketch produces (product of fan-outs)."""
        total = 1
        for st in self.stages:
            total *= st.fanout
        return total

    def artifact_counts(self):
        """Per-stage cumulative artifact count keyed by modality.

        Each entry is the running product of fan-outs up to and including that
        stage, i.e. how many artifacts exist AT that modality.
        """
        counts = {}
        running = 1
        for st in self.stages:
            running *= st.fanout
            counts[st.modality] = running
        return counts

    def stage_for(self, modality):
        """Return the first stage with the given modality, or None."""
        for st in self.stages:
            if st.modality == modality:
                return st
        return None

    def inject_feedback(self, feedback):
        """Return a NEW pipeline with a feedback sentence appended to the text stage.

        Models the human-in-the-loop text edit (Sec. 3.2): "we add sentences to
        the DALL-E 3 prompt to alter the output image according to designer
        feedback". The text stage's ``params['feedback']`` accumulates the
        appended sentences in order. Raises ValueError if there is no text stage.
        """
        feedback = str(feedback).strip()
        if not feedback:
            raise ValueError("feedback must be a non-empty string")
        new_stages = []
        edited = False
        for st in self.stages:
            if st.modality == TEXT and not edited:
                existing = list(st.params.get("feedback", ()))
                existing.append(feedback)
                new_params = dict(st.params)
                new_params["feedback"] = tuple(existing)
                new_stages.append(replace(st, params=new_params))
                edited = True
            else:
                new_stages.append(st)
        if not edited:
            raise ValueError("pipeline has no text stage to inject feedback into")
        return Pipeline(new_stages)

    def feedback_history(self):
        """Ordered tuple of feedback sentences injected into the text stage."""
        st = self.stage_for(TEXT)
        if st is None:
            return ()
        return tuple(st.params.get("feedback", ()))


def sketch2prototype_pipeline(n_images=4, n_meshes=None):
    """Build the canonical Sketch2Prototype pipeline.

    1 sketch -> 1 text -> ``n_images`` images -> ``n_meshes`` meshes. When
    ``n_meshes`` is None it defaults to ``n_images`` (one mesh per image before
    manual mesh selection). The paper generates 4 images per sketch (Sec. 2.1).
    """
    if n_images < 1:
        raise ValueError("n_images must be >= 1")
    if n_meshes is None:
        n_meshes = n_images
    if n_meshes < 1:
        raise ValueError("n_meshes must be >= 1")
    # Mesh fan-out is expressed relative to the image count so the running
    # product yields exactly n_meshes terminal meshes.
    return Pipeline([
        Stage(SKETCH, fanout=1),
        Stage(TEXT, fanout=1),
        Stage(IMAGE, fanout=n_images),
        Stage(MESH, fanout=1, params={"selected": n_meshes}),
    ])


def direct_sketch_to_3d_pipeline():
    """Build the direct sketch-to-3D baseline (unprocessed sketch -> Shap-E mesh)."""
    return Pipeline([Stage(SKETCH, fanout=1), Stage(MESH, fanout=1)])


def controlnet_pipeline(n_images=4):
    """Build the ControlNet baseline: sketch -> sketch-bound images -> meshes."""
    if n_images < 1:
        raise ValueError("n_images must be >= 1")
    return Pipeline([
        Stage(SKETCH, fanout=1),
        Stage(IMAGE, fanout=n_images, from_sketch=True),
        Stage(MESH, fanout=1),
    ])
