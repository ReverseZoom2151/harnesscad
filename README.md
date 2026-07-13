<h1 align="center">
  HarnessCAD
</h1>

<p align="center"><strong>The harness is the product, not the model</strong></p>

Text-to-CAD does not fail because language models cannot write CadQuery. It fails
because nothing checks the geometry before it reaches the kernel. A model emits a
plausible script, the kernel throws, the agent retries the same mistake with
slightly different numbers, and the loop burns tokens until it gives up or ships
a solid that is subtly wrong.

HarnessCAD attacks the loop instead of the model. Geometry is a **typed op
stream**, not generated code. Every op is validated before the kernel sees it,
rejected ops come back as typed diagnostics rather than stack traces, and the
model corrects a specific error instead of resampling. The op stream is
content-digested and event-sourced, so any state can be replayed, diffed, and
edited.

The domain layer is not invented. It is mined from a complete reading of **186
text-to-CAD papers and 70 CAD repositories**, reimplemented as deterministic,
stdlib-only modules. That reading produced a result that shaped the architecture,
described below.

## The loop

A CAD kernel is an expensive, opaque oracle. It says "boolean failed" long after
the mistake was made. So the harness front-runs it: every op is checked against a
typed contract, and a fleet of verifiers inspects the plan before any geometry is
built.

```python
from harnesscad.core.loop import HarnessSession
from harnesscad.io.backends.frep import FRepBackend

session = HarnessSession(FRepBackend(), verify_level="full")
result = session.apply_ops(ops)
result.ok, result.digest, result.diagnostics
```

Take a 20x10x5 plate, fillet it with radius 8, then shell it with thickness 9.
Every op is individually well-formed, and a naive loop accepts all of them. The
kernel will fail, or worse, quietly return something meaningless. The harness
answers before the kernel is called:

```text
[warning] preflight-RADIUS_TOO_LARGE:    fillet radius 8 exceeds half the smallest extent (5)
[warning] preflight-THICKNESS_TOO_LARGE: shell thickness 9 leaves no cavity (smallest extent 5)
[error]   infeasible-plan: shell thickness 9mm >= available stock 5mm; the wall consumes the whole solid
```

23 verifiers run inside that loop, discovered through a capability registry rather
than a hardcoded list. A verifier that throws becomes a diagnostic; it never takes
the run down.

## What reading the field actually taught us

The literature does not agree with itself, and the disagreement is invisible until
you implement it.

**The metrics disagree.** Six chamfer-distance implementations, all published, all
named "chamfer distance". Run them on the *same two point clouds*:

```text
chamfer_unit_sphere             0.0250    unit-sphere normalisation
chamfer_unit_cube             131.14      unit-cube normalisation
chamfer_bbox_judged             0.0382    centroid + max-bbox extent, judge-gated
chamfer_raw                     0.2069    no normalisation
chamfer_scaled_step             0.1005
chamfer_orientation_aligned     0.1443
```

Four orders of magnitude, purely from normalisation choices. Two papers reporting
"chamfer distance" are frequently not comparable, and nothing in either paper says
so.

**The tokenisers disagree.** DeepCAD quantises to 256 levels with round-half-even.
SkexGen truncates to 6 bits, which biases every coordinate downward by half a bin.
HNC-CAD replaces continuous rotation with a 25-frame discrete codebook whose frames
are neither the 24 proper rotations nor orthonormal. Vitruvion floor-quantises but
dequantises at the bin *centre*, making it the only one in the corpus with unbiased
round-trip error. Decode DeepCAD tokens with SkexGen's dequantiser and you get
geometry that is wrong by half a bin everywhere, silently.

**So the architecture refuses to blend them.** This is the load-bearing design
decision in the repository. Rival implementations are selected by name, never
averaged, and the code makes the mistake unrepresentable:

```python
run_suite("deepcad", samples)     # selects chamfer_unit_sphere
run_suite("cadrille", samples)    # selects chamfer_unit_cube

Suite("mine", metrics=["chamfer_unit_sphere", "chamfer_bbox_judged"])
# RivalBlendError: raised at definition time. A blending suite cannot be built.
```

```bash
harnesscad ingest tokens.json --family skexgen
# error: sequence is tagged family 'deepcad' but the 'skexgen' dequantiser was
# requested; quantiser families are mutually incompatible and are never blended
```

Filenames carry the finding too. `chamfer_unit_sphere.py` sits next to
`chamfer_bbox_judged.py`, and `deepcad_quantize.py` next to `skexgen_quantize.py`,
because the difference between them is the point.

## Geometry without a kernel

The default geometry path needs no OCCT, no CadQuery, and no proprietary kernel.
CISP ops compose signed distance fields; marching cubes meshes them; a half-edge
structure proves the result is a closed 2-manifold; the mesh is written as STL.

```bash
harnesscad demo   --backend frep
harnesscad export plate.stl --backend frep     # a real 403KB binary STL
```

A 20x10x5 plate comes out at volume 994.33 against an analytic 1000.0, the error
being grid resolution and reported as such. A boolean cut through it removes
exactly `pi * 9 * 5` of material and the result stays 2-manifold with genus 1.

The supporting numerics are load-bearing rather than decorative. Interval
arithmetic prunes the sampling grid (22,040 field evaluations down to 16,459, with
byte-identical output). Forward-mode autodiff supplies exact surface normals,
matching finite differences to 1.4e-10. Gauss-Legendre quadrature gives the
inertia tensor by the divergence theorem. A BVH cuts hidden-line candidates from
1,592 triangles to 8 when producing an orthographic drawing.

## What it can do today

```bash
harnesscad demo                                  # build and verify a part
harnesscad build "a 20x10x5 plate with a 3mm hole"
harnesscad export part.stl                       # stl glb amf obj step svg xcsg
harnesscad ingest tokens.json --family deepcad   # tokens -> editable CISP ops
harnesscad reconstruct --from point_cloud --to primitives
harnesscad program --lang openscad --validate part.scad
harnesscad bench --suite deepcad --input runs.json
harnesscad report                                # mass, pose, tolerance, DFM
harnesscad dataset --preset flywheel             # a session emits its own training data
harnesscad capabilities --tag sdf
```

Reconstruction runs 114 routes keyed by input and output kind, so "what can turn a
point cloud into CAD?" has a runnable answer. The program surface parses,
validates, emits and reviews seven CAD languages, and dispatching the wrong one
raises rather than guessing. An ISO 10303-11 EXPRESS parser reads 662 of the 664
real ISO schemas in the wild and validates Part-21 files against them.

## Status

**489 of 1,173 modules (42%) are still imported by nothing but their own test.**

That number is published because it is the honest measure of how connected the
system is, and it is measured, not estimated: `harnesscad capabilities --stats`
computes it from the live import graph. It was 76% before the wiring work.

Some of the remainder cannot be connected without lying. Reinforcement-learning
loss functions have no trainer in this repository. Several modules need a renderer
or a pool of human annotators that do not exist here. Roughly 105 benchmark entries
are dataset manifests and judge scaffolding rather than metrics, and were never
going to fit a `score(pred, gold)` seam. Those stay orphaned with a stated reason
instead of receiving a fabricated call site, because a padded number would defeat
the purpose of measuring it.

The default backend is a stub; the OCCT backend and the LLM planner are optional
extras and are not exercised by the suite. Two known correctness questions are
recorded rather than quietly resolved, in
[`docs/corpus/repo-ideas.md`](docs/corpus/repo-ideas.md).

Wiring modules into real call paths found five bugs in code that already existed
and already passed its own unit tests: an STL exporter that wrote binary and read
it back as UTF-8, a format that advertised a codec it does not ship, a mesher that
produced non-manifold output when a face landed on a sample plane, a metric adapter
that errored on every input it was ever given, and a "3D" contourer that only
implements the 2D case. None of them were reachable, so none of them were caught.
That is the argument for wiring, made empirically.

14,704 tests. Stdlib-only, deterministic: no wall clock, seeded randomness.

## Install

```bash
git clone <repo> && cd harnesscad
pip install -e .                 # stdlib-only core, no required dependencies
pip install -e ".[cadquery]"     # OCCT geometry backend
pip install -e ".[llm]"          # LLM planner
pip install -e ".[constraints]"  # SolveSpace sketch solver
```

Python >= 3.10. The core spine has no runtime dependencies. Provider keys are read
from the environment and never stored.

## Documentation

- [`docs/blueprint.md`](docs/blueprint.md): architecture and design rationale
- [`docs/corpus/paper-ideas.md`](docs/corpus/paper-ideas.md): all 186 papers, what was built from each and what was not
- [`docs/corpus/repo-ideas.md`](docs/corpus/repo-ideas.md): all 70 repositories, likewise
- [`audit/`](audit/): mining protocols and machine-readable progress state

Both ledgers record what was skipped and why. Learned, GPU-bound and
kernel-dependent work is marked external rather than faked.

## Contributing

Modules are stdlib-only, deterministic, and use absolute imports; every module has
a `unittest.TestCase` at the mirrored path under `tests/`. A monolithic
`unittest discover` segfaults at OCCT teardown, so count per module:

```bash
python -m unittest tests.domain.geometry.sdf.test_primitives
```

`tests/test_suite_collectable.py` fails loudly if a test file is added that the
canonical runner would not collect. Seven such files once sat in this suite holding
26 assertions that had never executed.

## License and citation

Released under the [MIT License](LICENSE). HarnessCAD does not reproduce a single
paper: its domain layer is drawn from 186 papers and 70 repositories, and each
module's originating work is attributed in the corpus ledgers. Cite the originating
work for the capability you use, not this repository.
