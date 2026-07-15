# A CAD computer-use agent: the synthesis

We mined all 15 computer-use repos and built 21 deterministic modules from them. This
is the design that composes them into a CAD CUA that no web-CUA project can build,
and it rests on two facts unique to CAD.

## The two unfair advantages

1. **CAD toolbars are pixel-stable.** A web page's DOM shifts every load; a CAD ribbon's
   icons are shipped assets at fixed positions. So `io/cua/primitives.py`'s template
   matching (NCC on the icon bitmap) is a free, ~100%-reliable, deterministic grounding
   path for every toolbar/menu action -- no VLM, no cost, no drift. Web CUAs cannot use
   this; we can, for the majority of actions.

2. **CAD has a free exact correctness oracle.** Every other CUA's hardest problem is
   verification -- Fara had to TRAIN a verifier because nobody can auto-check "did it book
   the right flight." We drive the GUI, export, and MEASURE through `io/gate.py` + the
   differential oracle. `agents/cua/verified_trajectory.py` labels a whole trajectory for
   free; `CUAVerifierBench` measures how wrong a fallible judge would be. This is the
   flywheel Fara couldn't have.

## The action stack, highest tier first (already built)

- **Tier 0 -- the app's Python console** (`agents/cua/console_iterate.py`, from
  BabyCommandAGI). FreeCAD and Blender ship a console. Typing into it is a GUI-app agent
  with full TEXT observability -- no pixels, no grounding. THIS SHOULD BE STEP 1. It is
  the cheapest correct path and it sidesteps grounding entirely for the two apps we drive.
- **Tier 1 -- semantic GUI** (`io/cua/uia.py` + `bindings_freecad.py`). The UIA tree is
  isomorphic to our CISP ops (Pad, Pocket, Fillet...); coordinate-free Invoke(). ~100%
  reliable where the tree exposes the control. Template matching (`primitives.py`) is the
  fallback where a control is icon-only.
- **Tier 2 -- computed viewport picks** (`io/cua/viewport.py` + `picks.py`). NEVER a VLM
  guess: we own the B-rep and the camera, so we project the entity to a pixel, click, and
  read the selection back. The 3D viewport is the only place pixels are needed, and even
  there the pick is computed, not guessed.

## The correctness spine (already built, and it is the whole point)

- **Every action returns a verified outcome or refuses** -- TuriX's two-screenshot
  step-eval (`turix.py`) and the hazards checklist (`hazards.py`) enforce that an
  unverified action is not an action. This is the same discipline that caught three
  backends leaking wrong volumes this session.
- **The quantity read-back** (`io/cua/quantity.py`): type 37.5, read it back, parse in the
  app's locale, RAISE on mismatch. The comma-decimal bug (37.5 -> "375,00 mm", a silent 10x
  error) is impossible to commit. No vision agent can catch this; we do.
- **Environment reset** (`io/cua/reset.py`, from E2B): a CAD CUA whose state leaks between
  trials is worthless. Only VM-revert is a reset; sticky tool defaults alone invalidate an
  experiment.
- **Coordinate discipline** (`coords.py`/`coordinate.py`): the model's coordinate space is
  DECLARED, never guessed from magnitude (the anti-pattern we deleted).

## The loop (already built)

`agents/cua/loop.py` reuses AgentHarness. On top:
- **Behavior-Best-of-N** (`best_of_n_trajectory.py`): generate N, and the judge is our
  EXACT oracle, not a model. Agent-S's single largest gain, made exact.
- **Memory** (`experience.py`): a dialog-to-feature store so the agent stops re-learning
  "the Pad dialog needs a sketch selected first" -- gated on the oracle (only verified
  trajectories are remembered), or it fills with plausible garbage as Agent-S's did.
- **Capability router** (`capabilities.py`): any model plugs in regardless of whether it
  natively does predict_step or only predict_click.
- **Falsifiable observation prompt** (`prompts.py`): every turn ends "in order to
  [prediction]", turning the next screenshot into a test of the last action.

## The training flywheel (the thing nobody else can build)

Because the oracle is free and exact:
1. Compile EXPERT trajectories (p=1.0 by construction -- we own the op stream, drive the
   GUI, and the label is the gate's verdict). Not filter agent rollouts from a 1-5% policy.
2. `verified_trajectory.py` is the schema; the grounding corpus (`eval/grounding/corpus.py`)
   already does this for clicks at 942 verified pairs/minute, adjudicated by the app's own
   picker.
3. Per-step reward is exact (after every op the document is fully determined), so credit
   assignment -- the hole in every GUI-RL approach -- is solved for CAD.

## What is honest about the limits

- Tier 0 (console) only exists for apps with a scriptable console (FreeCAD, Blender). For
  SolidWorks/Fusion, it is Tier 1/2 only.
- The oracle is many-to-one (volume+bbox+genus do not pin a part); the shape metric narrows
  but does not close this. A trajectory of verified clicks is verified clicks, not a proof
  the CAD task matches intent.
- Blender is invisible to UIA (draws its own UI in OpenGL) -- we drive it via bpy, not the
  GUI. The GUI path is FreeCAD (proven, 6000.0 exact) and Onshape (built, credential-gated).

## The one-line thesis

Everyone else built a CUA that produces output and cannot tell if it is right. We build the
first CUA whose success signal is MEASURED GEOMETRY, grounded on pixel-stable toolbars and
the app's own console, with a free exact oracle that lets us generate verified training data
no web-CUA project can. The differentiator is not the driving -- it is the knowing.
